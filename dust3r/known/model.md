# Giải thích `dust3r/inference.py`, `dust3r/model.py`, `dust3r/patch_embed.py`

Tài liệu này mô tả 3 mảnh ghép cốt lõi của DUSt3R khi suy luận (inference):

- `dust3r/patch_embed.py`: biến ảnh (tensor) thành **chuỗi token patches** + **tọa độ positional**; có biến thể xử lý **tỉ lệ ảnh không vuông**.
- `dust3r/model.py`: định nghĩa model `AsymmetricCroCo3DStereo` (2 encoder “siamese”, 2 decoder bất đối xứng, 2 head) và cơ chế “symmetrized forward” để tiết kiệm compute.
- `dust3r/inference.py`: ghép các cặp ảnh thành batch, chạy model theo batch (có thể AMP), gom kết quả về CPU và tiện ích suy ra `pts3d`.

Bạn có thể hiểu pipeline như sau:

1) Ảnh đã được chuẩn hoá thành dict (thường bởi `utils/image.py`) với các key như `img`, `true_shape`, `idx`, ...
2) Danh sách **pairs** (img_i, img_j) được tạo (thường bởi `image_pairs.py`).
3) `inference()` collate các pair thành batch → gọi `model(view1, view2)`.
4) `model` gọi `patch_embed` để lấy token + pos → encoder → decoder bất đối xứng → head → xuất `pts3d`/`depth`/`confidence`.

---

## 1) `dust3r/patch_embed.py`

### Mục tiêu
Trong transformer vision, ảnh được chia thành patch và đưa qua một lớp chiếu tuyến tính (conv) để thành token. File này cung cấp 2 cách embed:

- `PatchEmbedDust3R`: patch embed “chuẩn” (đòi hỏi H và W chia hết patch).
- `ManyAR_PatchEmbed`: patch embed cho **nhiều aspect ratio** (AR = aspect ratio), xử lý landscape/portrait trong cùng một batch bằng cách **hoán đổi trục** cho ảnh portrait.

### `get_patch_embed(patch_embed_cls, img_size, patch_size, enc_embed_dim)`
- Chọn class theo string (`'PatchEmbedDust3R'` hoặc `'ManyAR_PatchEmbed'`) và khởi tạo.
- `eval(patch_embed_cls)` ở đây là cách “factory” dựa trên tên lớp.

Điểm đáng học:
- Viết factory gọn và ràng buộc bằng `assert patch_embed_cls in [...]` để tránh tên sai.

### `PatchEmbedDust3R.forward(x)`
Input: `x` có shape `[B, C, H, W]`.

Luồng xử lý:
1) Assert `H % patch_size == 0` và `W % patch_size == 0`.
2) `x = self.proj(x)` (thường là conv stride = patch_size) → tạo feature map dạng patch-grid.
3) `pos = self.position_getter(B, H_tokens, W_tokens, device)`.
4) Nếu `flatten`: đổi `BCHW → BNC` (N = số patch tokens).
5) `x = self.norm(x)`.

Output:
- `x`: token `[B, N, C_embed]`
- `pos`: positional indices `[B, N, 2]` (thường là (row, col) token coordinate)

### `ManyAR_PatchEmbed.forward(img, true_shape)`
Mục tiêu: hỗ trợ ảnh không vuông và có thể lẫn landscape/portrait.

Ràng buộc quan trọng:
- Assert `W >= H` cho **tensor đầu vào**: nghĩa là ảnh đưa vào embed phải ở landscape “logic” (thường do một wrapper ở tầng trên đảm bảo).
- `true_shape` shape phải là `(B,2)` và thể hiện **kích thước thật** trước khi transpose/crop.

Cách làm:
- Tính số token `n_tokens = (H/patch)*(W/patch)`.
- Dựa trên `true_shape`, phân loại mỗi ảnh là landscape vs portrait:
  - `is_landscape = width >= height`
  - `is_portrait = ~is_landscape`
- Cấp phát output `x` và `pos` dạng zeros.
- Với ảnh landscape:
  - `self.proj(img)` rồi flatten theo patch-grid.
- Với ảnh portrait:
  - `img.swapaxes(-1,-2)` (đổi H/W) trước khi `proj`, để mô hình vẫn làm việc theo quy ước landscape.
- `pos` cũng được tạo tương ứng:
  - landscape: `position_getter(1, H_tokens, W_tokens, ...)`
  - portrait: `position_getter(1, W_tokens, H_tokens, ...)` (đảo lại)

Điểm đáng học:
- Cách xử lý portrait bằng “transpose tại chỗ” + tạo pos tương ứng giúp cùng một backbone làm việc ổn định, tránh phải huấn luyện riêng cho portrait.
- `true_shape` đóng vai trò “ground truth geometry metadata” để quyết định hướng.

---

## 2) `dust3r/model.py`

### Mục tiêu
`AsymmetricCroCo3DStereo` là model chính của DUSt3R. Nó dự đoán 3D theo cách **bất đối xứng**:

- `res1`: dự đoán 3D cho view1 trong hệ quy chiếu view1.
- `res2`: dự đoán 3D của view2 nhưng **biểu diễn trong hệ quy chiếu view1** (nên có key `pts3d_in_other_view`).

Lý do bất đối xứng: giúp ràng buộc 2 view vào một frame tham chiếu, thuận lợi cho ghép nối và tối ưu toàn cục.

### `load_model(model_path, device, verbose=True)`
- Load checkpoint bằng `torch.load(..., weights_only=False)`.
- Lấy `args` trong ckpt để instantiate model bằng `eval(args)`.
- Ép `landscape_only=False` để inference chấp nhận cả portrait/landscape (kết hợp với wrapper/transpose).
- Load state dict với `strict=False`.

Điểm đáng học:
- Checkpoint lưu cả “chuỗi lệnh instantiate” giúp tái lập model đúng cấu hình train (nhưng cũng đòi hỏi kiểm soát chặt chẽ nguồn ckpt vì có `eval`).

### `AsymmetricCroCo3DStereo` (tổng quan kiến trúc)

#### Khởi tạo `__init__`
- Kế thừa `CroCoNet` (từ CroCo).
- Copy `self.dec_blocks` thành `self.dec_blocks2` để có **2 decoder branch**.
- Tạo 2 head downstream (`downstream_head1`, `downstream_head2`) và wrap bằng `transpose_to_landscape(...)`.
- Thiết lập `freeze` (none/mask/encoder).

#### `from_pretrained(...)`
- Nếu `pretrained_model_name_or_path` là file local → dùng `load_model`.
- Nếu không → gọi HuggingFace Hub loader.

#### `_set_patch_embed(...)`
- Lưu `self.patch_size`.
- Dùng `get_patch_embed()` để chọn embed class (`PatchEmbedDust3R` hoặc `ManyAR_PatchEmbed`).

#### `load_state_dict(...)`
- Nếu ckpt không có `dec_blocks2.*` → tự nhân bản từ `dec_blocks.*`.

Điểm đáng học:
- Backward-compatibility: ckpt cũ vẫn load được dù kiến trúc đã thêm decoder2.

### Encoding

#### `_encode_image(image, true_shape)`
- `x, pos = self.patch_embed(image, true_shape=true_shape)`.
- Chạy lần lượt qua `enc_blocks` (transformer encoder).
- `enc_norm`.

#### `_encode_image_pairs(img1, img2, true_shape1, true_shape2)`
- Nếu 2 ảnh cùng kích thước → concat theo batch dimension, encode một lần rồi `chunk(2)`.
- Nếu khác kích thước → encode riêng.

Ý nghĩa:
- Tối ưu compute khi size giống nhau.

#### `_encode_symmetrized(view1, view2)`
- Lấy `img1 = view1['img']`, `img2 = view2['img']`.
- Lấy `true_shape` nếu có, nếu không thì dùng `img.shape[-2:]` giả định là true.

Trường hợp quan trọng: **symmetrized batch**
- Nếu `is_symmetrized(view1, view2)` là True:
  - Chỉ encode **một nửa** batch: `img1[::2]`, `img2[::2]`.
  - Sau đó `interleave(feat1, feat2)` để khôi phục thứ tự đầy đủ.
- Nếu không → encode toàn bộ.

Điểm đáng học:
- Đây là trick hiệu quả: khi batch chứa cặp (A,B) và (B,A), encoder có thể dùng lại.

### Decoder bất đối xứng

#### `_decoder(f1, pos1, f2, pos2)`
- Project qua `decoder_embed`.
- Với mỗi cặp block `(blk1, blk2)` từ `dec_blocks` và `dec_blocks2`:
  - Nhánh view1: `blk1(f1,f2,pos1,pos2)`
  - Nhánh view2: `blk2(f2,f1,pos2,pos1)` (đảo thứ tự input)
- Lưu toàn bộ intermediate outputs để head có thể dùng multi-scale/skip.
- `dec_norm` ở output cuối.

Điểm đáng học:
- Hai decoder “đối ứng” nhưng không chia sẻ tham số, và update theo hướng ngược nhau — đó là một cách encode “quan hệ có hướng” giữa 2 view.

### Head và output

#### `set_downstream_head(...)`
- Assert `img_size` là bội số `patch_size`.
- Tạo 2 head độc lập bằng `head_factory(...)`.
- Wrap bằng `transpose_to_landscape(..., activate=landscape_only)` để tự xử lý portrait/landscape.

#### `forward(view1, view2)`
1) Encode symmetrized → lấy features và pos.
2) Decode → `dec1, dec2`.
3) Tắt autocast cho head để ổn định numerics:
   - `res1 = head1(dec1, shape1)`
   - `res2 = head2(dec2, shape2)`
4) Đổi key:
   - `res2['pts3d_in_other_view'] = res2.pop('pts3d')`

Trả về `(res1, res2)`.

---

## 3) `dust3r/inference.py`

### Mục tiêu
File này là “glue code” để:

- Ghép list `(img1_dict, img2_dict)` thành batch.
- Chuyển tensor lên GPU.
- Gọi forward model.
- Gom kết quả về CPU để dễ hậu xử lý/serialize.
- Cung cấp tiện ích lấy `pts3d` từ output ở nhiều mode.

### `_interleave_imgs(img1, img2)`
Nhận hai dict view có cùng keys, và tạo dict mới theo kiểu xen kẽ:
- Nếu value là tensor: `stack((v1,v2), dim=1).flatten(0,1)`.
- Nếu value là list: xen kẽ phần tử.

Ý nghĩa:
- Tạo batch dạng: `[v1_0, v2_0, v1_1, v2_1, ...]`.

### `make_batch_symmetric(batch)`
- Input: `(view1, view2)`.
- Tạo:
  - view1 = interleave(view1, view2)
  - view2 = interleave(view2, view1)

Kết quả: batch chứa cả (A,B) và (B,A) theo thứ tự xen kẽ để model có thể kích hoạt tối ưu “half forward” ở encoder.

### `loss_of_one_batch(...)`
Đây là wrapper “chuẩn hoá một bước chạy model”.

Các điểm chính:
- `ignore_keys` bỏ qua các key không cần `.to(device)` (vd `true_shape`, `idx`, `instance`, ...).
- Với các key còn lại, chuyển tensor lên device.
- Nếu `symmetrize_batch=True` → gọi `make_batch_symmetric`.
- Chạy model trong AMP nếu `use_amp`.
- Nếu có `criterion` → tính loss trong autocast disabled (float32).

Trả về dict:
- `view1`, `view2`, `pred1`, `pred2`, `loss`.

### `inference(pairs, model, device, batch_size=8, verbose=True)`
- Kiểm tra xem mọi ảnh trong pairs có cùng shape không (`check_if_same_size`).
  - Nếu không → ép `batch_size = 1` (vì collate batch khác size sẽ khó/không hợp lệ).
- Chạy theo từng chunk:
  - `collate_with_cat(pairs[i:i+bs])` để tạo batch view1/view2.
  - `loss_of_one_batch(..., criterion=None)`.
  - `to_cpu(res)` để đẩy result về CPU.
- Cuối cùng `collate_with_cat(result, lists=multiple_shapes)` để gộp list kết quả.

Điểm đáng học:
- “Batching có điều kiện” dựa trên kích thước ảnh là cách thực dụng để tránh crash và tối ưu tốc độ khi có thể.

### `get_pred_pts3d(gt, pred, use_pose=False)`
Chuẩn hoá cách lấy point cloud dự đoán từ nhiều kiểu head/output:

- Nếu `pred` có `depth` và `pseudo_focal` → gọi `depthmap_to_pts3d(**pred, pp=principal_point)`.
  - principal point lấy từ `gt['camera_intrinsics'][...,:2,2]` nếu có.
- Nếu có `pred['pts3d']` → dùng trực tiếp.
- Nếu có `pred['pts3d_in_other_view']`:
  - chỉ cho phép nếu `use_pose=True` (vì pts đã ở hệ khác).

Nếu `use_pose=True` và `pred` có `camera_pose`:
- Transform điểm bằng `geotrf(camera_pose, pts3d)`.

Ý nghĩa:
- Một hàm “adapter” giúp phần đánh giá/hậu xử lý không phải phân nhánh theo từng kiểu head.

### `find_opt_scaling(...)`
Tìm hệ số scale tốt nhất để khớp pointcloud dự đoán với ground truth (khi metric scale chưa xác định).

- Flatten point cloud và thay invalid → NaNs bằng `invalid_to_nans`.
- Tính các tích vô hướng:
  - `dot_gt_pr = (pr * gt).sum(-1)`
  - `dot_gt_gt = (gt^2).sum(-1)`
- Các mode:
  - `avg*`: scale = mean(dot_gt_pr) / mean(dot_gt_gt)
  - `median*`: median(dot_gt_pr / dot_gt_gt)
  - `weiszfeld*`: IRLS (iteratively reweighted least squares) ~ 10 bước, weight = 1/dist.
- Option `stop_grad`: detach scale.
- Clip scale `>= 1e-3`.

Điểm đáng học:
- Đây là cách “fit scale” robust khi có outlier và NaN mask — cực hay khi làm bài toán 3D từ ảnh (thường bị scale ambiguity).

---

## 3 file này đóng vai trò gì trong DUSt3R?

- `patch_embed.py`: **cổng chuyển đổi ảnh → token**, đảm bảo ràng buộc patch-size, và hỗ trợ multi-aspect-ratio/portrait.
- `model.py`: **định nghĩa kiến trúc và forward pass** (encode → decode bất đối xứng → head), cùng trick symmetrized để tối ưu.
- `inference.py`: **hệ thống chạy inference theo batch** và thống nhất cách đọc output (pts3d/pose/scale).

Nếu ví pipeline như một “dây chuyền”:
- `patch_embed` là máy “chia ảnh thành patch và gắn tọa độ”.
- `model` là nhà máy transformer + head sinh 3D.
- `inference` là bộ phận vận hành: xếp hàng, chạy theo lô, gom kết quả.

---

## Những điều hay nhất bạn có thể học từ 3 script này

1) Thiết kế inference pipeline thực dụng
- Batch khi có thể, fallback `bs=1` khi shape khác nhau.
- AMP bao quanh forward để tăng tốc, nhưng tính loss/head quan trọng ở float32.

2) Tối ưu compute nhờ đối xứng dữ liệu
- Tổ chức batch xen kẽ (A,B) và (B,A) để encoder chỉ phải chạy 1 nửa.

3) Tách kiến trúc “core transformer” và “downstream head”
- `CroCoNet` lo backbone/decoder; head lo biểu diễn output (pts3d/depth/conf).

4) Xử lý portrait/landscape có hệ thống
- Dùng wrapper/transpose và `true_shape` để model không bị phụ thuộc orientation.

5) Tư duy robust cho bài toán 3D
- Có sẵn công cụ fit scale (avg/median/IRLS) và xử lý invalid bằng NaN-mask.

6) Backward compatibility khi kiến trúc tiến hoá
- Nếu checkpoint cũ thiếu `dec_blocks2` thì tự nhân bản weights để vẫn chạy.
