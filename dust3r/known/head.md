# Giải thích `dust3r/heads/dpt_head.py` và `dust3r/heads/linear_head.py`

Hai file này implement **downstream head** của DUSt3R: phần “cuối” của mạng, biến biểu diễn token từ decoder thành output pixel-wise (mỗi pixel có vector 3D và có thể kèm confidence).

Trong `dust3r/model.py`, model tạo 2 head độc lập (`head1`, `head2`) và gọi head để sinh `pred1`, `pred2`. Các head này đều tuân theo cùng một ý tưởng:

- Input: danh sách token/feature theo layer (hoặc chỉ layer cuối), kích thước cơ bản là `B x N x C`.
- Output: tensor theo ảnh `B x (3 + conf) x H x W` rồi được **chuẩn hoá/diễn giải** bởi `postprocess()` (depth_mode, conf_mode).

---

## 1) `dust3r/heads/linear_head.py`

### Mục tiêu
`LinearPts3d` là head đơn giản và rất “thẳng”: mỗi token patch trực tiếp dự đoán một block điểm 3D kích thước `patch_size x patch_size`.

Nó phù hợp khi bạn muốn head nhẹ, tốc độ cao, ít phụ thuộc vào kiến trúc phức tạp.

### Class `LinearPts3d`

#### `__init__(self, net, has_conf=False)`
- Lấy `patch_size` từ `net.patch_embed.patch_size[0]`.
- Lưu `depth_mode`, `conf_mode` từ net để `postprocess` hiểu cần chuẩn hoá output theo kiểu gì.
- Tạo linear projection:

  - Input dim: `net.dec_embed_dim`
  - Output dim mỗi token: `(3 + has_conf) * patch_size^2`

Ý nghĩa:
- Mỗi token (tương ứng 1 patch) dự đoán **(3 hoặc 4) giá trị cho mỗi pixel** trong patch đó.
  - 3 kênh: (x, y, z) của điểm 3D
  - nếu `has_conf=True`: thêm 1 kênh confidence

#### `forward(self, decout, img_shape)`
Input:
- `decout`: list các tensor layer; `tokens = decout[-1]` lấy layer cuối.
- `img_shape`: `(H, W)` của ảnh (sau crop/resize) theo pixel.

Luồng xử lý chi tiết:
1) `feat = self.proj(tokens)`
   - `tokens` shape: `B x S x D`
   - `feat` shape: `B x S x ((3+conf)*p^2)`

2) Reshape thành patch-grid:
   - `feat.transpose(-1, -2)` để đưa channel lên trước.
   - `.view(B, -1, H//p, W//p)` → `B x ((3+conf)*p^2) x (H/p) x (W/p)`

3) `pixel_shuffle`:
   - `F.pixel_shuffle(feat, p)`
   - biến kênh `(3+conf)*p^2` thành `3+conf`, đồng thời upsample không gian từ `(H/p, W/p)` lên `(H, W)`.

   Kết quả: `B x (3+conf) x H x W`.

4) `postprocess(feat, depth_mode, conf_mode)`
   - Chuẩn hoá depth/pts3d/confidence theo các mode.

### Vai trò trong DUSt3R
- Head tuyến tính là cách “giải mã” token → pixel rất hiệu quả khi patch embedding/decoder đã học đủ.
- Nó cũng là baseline tốt: ít thành phần, dễ debug, chạy nhanh.

### Điều hay nhất học được
- Pattern `pixel_shuffle` là một cách cực gọn để “giải nén” thông tin patch-level thành pixel-level mà không cần deconv phức tạp.
- Thiết kế output theo `(channels * patch_size^2)` trên token giúp mapping token ↔ patch rất trực tiếp.
- `postprocess()` làm nhiệm vụ thống nhất (một nơi duy nhất) cho mọi head về cách hiểu depth/conf.

---

## 2) `dust3r/heads/dpt_head.py`

### Mục tiêu
DPT head (Dense Prediction Transformer-style decoder) là head “giàu biểu diễn” hơn, dùng nhiều layer features và cơ chế refine multi-scale để cho output mượt và chi tiết.

File này không chỉ dùng DPT có sẵn của CroCo mà còn **chỉnh sửa** để phù hợp với DUSt3R.

### `DPTOutputAdapter_fix`
Đây là subclass của `models.dpt_block.DPTOutputAdapter` với 2 thay đổi chính:

#### `init(self, dim_tokens_enc=768)`
- Gọi `super().init(dim_tokens_enc)`.
- Xoá các module bị coi là duplicated weights:
  - `act_1_postprocess`, `act_2_postprocess`, `act_3_postprocess`, `act_4_postprocess`

Ý nghĩa:
- Giảm trùng lặp tham số hoặc sửa khác biệt so với phiên bản gốc mà DUSt3R không cần.

#### `forward(self, encoder_tokens: List[Tensor], image_size=None)`
Đầu vào:
- `encoder_tokens`: thực tế là list các tensor theo layer (nên head này yêu cầu backbone trả về all layers).
- `image_size = (H,W)`.

Luồng xử lý:
1) Tính số patch theo chiều cao/rộng:
   - `N_H = H // (stride_level * P_H)`
   - `N_W = W // (stride_level * P_W)`

2) Chọn 4 layer theo `self.hooks`:
   - `layers = [encoder_tokens[hook] for hook in self.hooks]`

3) Lọc token “task-relevant” (bỏ global tokens) bằng `self.adapt_tokens`.

4) Reshape tokens về feature map:
   - `rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W)`

5) Áp activation/postprocess theo từng layer: `self.act_postprocess[idx]`.

6) Project về cùng `feature_dim`: `self.scratch.layer_rn[idx]`.

7) Fuse theo refinement network (multi-scale):
   - `path_4 = refinenet4(layers[3])` và crop cho khớp shape
   - `path_3 = refinenet3(path_4, layers[2])`
   - `path_2 = refinenet2(path_3, layers[1])`
   - `path_1 = refinenet1(path_2, layers[0])`

8) Output head: `out = self.head(path_1)`.

Kết quả `out` thường là `B x num_channels x H x W` (tuỳ cấu hình adapter).

### `PixelwiseTaskWithDPT`
Đây là wrapper module DUSt3R dùng để gọi DPT adapter và (tuỳ chọn) chạy `postprocess`.

#### `__init__(...)`
- `self.return_all_layers = True`: báo hiệu cho backbone rằng head cần toàn bộ layer outputs.
- Nhận các tham số như:
  - `hooks_idx`: chọn 4 layer để hook
  - `dim_tokens`: dim của token ở các layer tương ứng (enc/dec)
  - `num_channels`: số kênh output (ở DUSt3R thường là 3 hoặc 4 nếu có conf)
  - `depth_mode`, `conf_mode`: cho `postprocess`
- Tạo `self.dpt = DPTOutputAdapter_fix(...)` rồi `self.dpt.init(...)`.

#### `forward(self, x, img_info)`
- Gọi `out = self.dpt(x, image_size=(img_info[0], img_info[1]))`.
- Nếu có `postprocess`: `out = postprocess(out, depth_mode, conf_mode)`.

### `create_dpt_head(net, has_conf=False)`
Factory tạo head DPT dựa trên net params:
- Assert `net.dec_depth > 9` (cần đủ layer để hook sâu).
- Chọn:
  - `feature_dim=256`, `last_dim=feature_dim//2`
  - `out_nchan=3` (x,y,z)
- Tính vị trí hooks:
  - `[0, l2*2//4, l2*3//4, l2]` với `l2 = net.dec_depth`
- `dim_tokens=[ed, dd, dd, dd]` (ed=enc_embed_dim, dd=dec_embed_dim)
- `num_channels = out_nchan + has_conf`

Ý nghĩa:
- Head DPT dùng token từ nhiều “độ sâu” khác nhau để ghép multi-scale.

### Vai trò trong DUSt3R
- DPT head cung cấp một đường giải mã giàu ngữ cảnh không gian, thường cho output mượt và chi tiết hơn linear head.
- Cơ chế hook nhiều layer phù hợp với dense prediction: coarse→fine.

### Điều hay nhất học được
- Pattern “adapter + refinement multi-scale” là một cách mạnh để biến token-sequence thành map không gian.
- `return_all_layers=True` là một hợp đồng (contract) rất rõ: head yêu cầu backbone trả gì.
- Chỉnh sửa upstream code (CroCo DPT adapter) một cách tối thiểu để phù hợp yêu cầu downstream là kỹ năng engineering thực tế.

---

## 3) Hai head này khác nhau thế nào (và khi nào dùng cái nào)?

- Linear head:
  - Ưu: đơn giản, nhanh, ít tham số, dễ debug.
  - Nhược: chi tiết/mượt phụ thuộc nhiều vào chất lượng token cuối; ít khai thác multi-scale.

- DPT head:
  - Ưu: khai thác nhiều layer, multi-scale refinement, thường cho map tốt hơn.
  - Nhược: phức tạp hơn, nặng hơn, yêu cầu backbone trả all layers.

Trong DUSt3R, việc có cả 2 head cho thấy thiết kế mở: bạn có thể chọn trade-off tốc độ/chất lượng bằng cấu hình `head_type`.

---

## 4) “Bài học thiết kế” quan trọng rút ra

1) Chuẩn hoá hậu xử lý bằng một hàm chung
- Dù head nào, cuối cùng vẫn đi qua `postprocess(depth_mode, conf_mode)` để output có cùng semantics.

2) Tư duy shape rõ ràng
- Cả hai head đều dựa trên quy ước rõ: token là `B x N x C`, output là `B x C_out x H x W`.

3) Định nghĩa contract giữa backbone và head
- Linear head chỉ cần layer cuối.
- DPT head yêu cầu “all layers” và xác định hooks.

4) Dùng kỹ thuật upsample hiệu quả
- `pixel_shuffle` là ví dụ tối ưu: vừa nhanh vừa gọn.

5) Viết factory để gắn head theo kiến trúc backbone
- `create_dpt_head(net, ...)` dùng `net.dec_depth`, `enc_embed_dim`, `dec_embed_dim` để cấu hình head đúng.
