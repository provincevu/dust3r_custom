# Giải thích `dust3r/utils/image.py` và `dust3r/image_pairs.py`

Tài liệu này mô tả chi tiết hai script:

- `dust3r/utils/image.py`: nạp ảnh/độ sâu (đọc file), chuẩn hoá và cắt/căn chỉnh kích thước để đưa vào model.
- `dust3r/image_pairs.py`: tạo danh sách **cặp ảnh (image pairs)** theo một **đồ thị cảnh (scene graph)** để DUSt3R chạy dự đoán theo cạnh của đồ thị.

Hai script này thường xuất hiện ngay đầu pipeline inference/demo:

1) Nạp ảnh → biến thành tensor đã chuẩn hoá, có metadata về kích thước thật
2) Tạo các cặp (view1, view2) → chạy model trên từng cặp để suy ra quan hệ hình học (depth/pose/…)

---

## 1) `dust3r/utils/image.py`

### Mục tiêu
Script này giải quyết ba vấn đề “đầu vào ảnh” rất hay gặp trong các pipeline thị giác máy tính:

1) **Đọc ảnh từ nhiều nguồn/định dạng** (jpg/png, có thể thêm heic/heif nếu cài thư viện; và đọc EXR cho depthmap qua OpenCV).
2) **Chuẩn hoá hình học (resize/crop)** sao cho phù hợp với yêu cầu của backbone/patch embedding (thường ViT yêu cầu kích thước chia hết cho `patch_size`).
3) **Chuẩn hoá giá trị pixel** về dạng tensor phù hợp với mô hình (chuẩn hoá về khoảng xấp xỉ [-1, 1] theo mean/std = 0.5).

### Các thành phần chính

#### 1.1. Cấu hình hỗ trợ EXR và HEIF
- Đặt biến môi trường `OPENCV_IO_ENABLE_OPENEXR=1` trước khi import OpenCV để bật đọc `.exr` nếu OpenCV build có hỗ trợ.
- Thử import `pillow_heif.register_heif_opener()`:
  - Nếu có, `heif_support_enabled=True` và `load_images()` sẽ chấp nhận `.heic`, `.heif`.
  - Nếu không, vẫn chạy bình thường với jpg/png.

Điểm hay: code “tự degrade” (optional dependency) — có thì dùng, không có thì thôi.

#### 1.2. `ImgNorm`
```python
ImgNorm = Compose([ToTensor(), Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
```
- `ToTensor()` chuyển ảnh PIL (uint8 0..255) thành float tensor (0..1) dạng `[C,H,W]`.
- `Normalize(mean=0.5, std=0.5)` đưa về gần `[-1, 1]` theo công thức:

$$x_{norm} = \frac{x - 0.5}{0.5} = 2x - 1$$

Đây là kiểu chuẩn hoá rất phổ biến cho ViT / transformer vision.

#### 1.3. `imread_cv2(path, options=cv2.IMREAD_COLOR)`
- Đọc ảnh bằng OpenCV.
- Nếu file đuôi `.exr` / `EXR` → đổi `options` sang `cv2.IMREAD_ANYDEPTH` để lấy dữ liệu độ sâu (float).
- Nếu đọc được ảnh màu (3 kênh) → chuyển BGR → RGB.
- Nếu `cv2.imread()` trả `None` → raise lỗi rõ ràng.

Điểm hay:
- Tự động xử lý BGR/RGB (một bẫy rất hay gặp khi trộn PIL/OpenCV).
- Tự chuyển chế độ đọc cho depthmap EXR thay vì ép chung một pipeline.

#### 1.4. `img_to_arr(img)`
- Nếu `img` là string path → gọi `imread_cv2()` để đọc.
- Nếu không → trả nguyên.

Đây là utility nhỏ để “chấp nhận đầu vào linh hoạt”: có thể truyền path hoặc truyền mảng đã đọc sẵn.

#### 1.5. `rgb(ftensor, true_shape=None)`
Mục tiêu: biến một tensor/ndarray đã chuẩn hoá về ảnh hiển thị (float 0..1) cho việc visualize.

Hành vi chính:
- Nhận đầu vào có thể là:
  - list → map đệ quy
  - `torch.Tensor` → `.detach().cpu().numpy()`
  - tensor 3D dạng `[3,H,W]` → transpose sang `[H,W,3]`
  - tensor 4D dạng `[B,3,H,W]` → transpose sang `[B,H,W,3]`
- Nếu có `true_shape=(H,W)` → cắt ảnh về đúng kích thước thật (bù cho việc đã crop/pad trong preprocessing).
- Nếu dtype là `uint8` → chia 255.
- Nếu không phải `uint8` → giả sử đang ở không gian chuẩn hoá `[-1,1]` (hoặc gần vậy), nên chuyển ngược về `[0,1]` bằng:

$$x = 0.5 \cdot ftensor + 0.5$$

- Cuối cùng `clip(0,1)`.

Điểm hay:
- Tách riêng “đường đi vào model” (Normalize) và “đường đi ra để xem” (denormalize) rõ ràng.

#### 1.6. `_resize_pil_image(img, long_edge_size)`
- Tính `S = max(img.size)` (cạnh dài).
- Chọn nội suy:
  - Nếu ảnh đang lớn hơn target → `LANCZOS` (downsample chất lượng cao)
  - Nếu ảnh nhỏ hơn/hoặc bằng target → `BICUBIC` (upsample)
- Tính `new_size` để **giữ tỉ lệ**, sao cho cạnh dài = `long_edge_size`.

Điểm hay:
- Chọn nội suy khác nhau cho upsample/downsample là một best practice “nhỏ nhưng đáng giá”.

#### 1.7. `load_images(folder_or_list, size, square_ok=False, verbose=True, patch_size=16)`
Đây là hàm quan trọng nhất.

##### Đầu vào
- `folder_or_list`:
  - string → coi là folder, `os.listdir()` và sort
  - list → coi là list các path
- `size`: hai mode chính
  - `size == 224`: workflow giống các pipeline classification: resize theo short-side rồi crop vuông
  - `size != 224` (thường 512): resize theo long-side rồi crop theo bội số patch
- `square_ok`: nếu không cho phép crop ra hình vuông (khi input W==H) thì sẽ ép crop thành hình chữ nhật tỉ lệ 4:3 theo logic `halfh = 3*halfw/4`.
- `patch_size`: mặc định 16, dùng để đảm bảo kích thước crop phù hợp patch embedding (ViT).

##### Luồng xử lý
1) Xác định danh sách đuôi file được hỗ trợ:
   - mặc định: `.jpg`, `.jpeg`, `.png`
   - nếu có HEIF support: thêm `.heic`, `.heif`

2) Với mỗi file ảnh hợp lệ:
   - Mở bằng PIL, áp `exif_transpose(...)` để sửa xoay ảnh theo metadata EXIF, rồi `.convert('RGB')`.
   - Lưu kích thước gốc `W1,H1`.

3) Resize:
   - Nếu `size==224`:
     - resize dựa trên **cạnh ngắn**, để sau đó crop vuông trung tâm.
     - Code thực hiện bằng cách truyền vào `_resize_pil_image()` một `long_edge_size` được tính từ `size * max(W1/H1, H1/W1)`.
   - Ngược lại:
     - resize cạnh dài về `size` (thường 512).

4) Crop quanh tâm:
   - Tính tâm `cx=W//2`, `cy=H//2`.
   - Nếu `size==224`: crop vuông theo `half = min(cx,cy)`.
   - Nếu `size!=224`:
     - Tính `halfw` và `halfh` sao cho **kích thước crop** là bội số của `patch_size`.
       - `halfw = floor(W/patch_size)*patch_size/2` (viết khác đi nhưng ý là vậy)
       - tương tự cho `halfh`
     - Nếu ảnh sau resize là vuông (`W==H`) và `square_ok` là False:
       - điều chỉnh `halfh` nhỏ hơn để ra crop dạng 4:3.

5) Chuẩn hoá và đóng gói output:
   - `ImgNorm(img)[None]` → tensor shape `[1,3,H,W]`.
   - `true_shape = np.int32([img.size[::-1]])`
     - `img.size` là `(W,H)`, đảo lại để thành `(H,W)`
     - bọc thêm `[...]` để thành shape `(1,2)`.
   - thêm metadata `idx` và `instance` để định danh.

6) Cuối cùng:
   - `assert imgs, 'no images foud at '+root` để đảm bảo không rỗng.

##### Output format
Một list các dict, mỗi dict có ít nhất:
- `img`: tensor `[1,3,H,W]` đã normalize
- `true_shape`: array `(1,2)` chứa `(H,W)`
- `idx`: chỉ số ảnh
- `instance`: string của chỉ số (thường phục vụ logging/ID)

### Vai trò trong DUSt3R
- Đây là “cổng vào” chuẩn hoá ảnh cho inference/demo.
- Thiết kế crop theo `patch_size` giúp input tương thích với kiến trúc transformer (tránh mismatch do kích thước không chia hết patch).
- Lưu `true_shape` giúp những bước visualize/khôi phục output biết vùng nào là “hình thật” sau crop.

---

## 2) `dust3r/image_pairs.py`

### Mục tiêu
DUSt3R (và nhiều hệ multi-view) thường không xử lý tất cả ảnh một lần mà xử lý theo **cặp ảnh** (pairwise) để dự đoán:
- tương quan hai view,
- độ sâu tương đối / matching,
- ràng buộc hình học,
- hoặc các đại lượng trung gian để tối ưu hoá/ghép nối.

Script này tạo ra danh sách cặp `(img_i, img_j)` theo một “đồ thị” mà bạn chọn, giúp:
- điều khiển số lượng cặp (chi phí tính toán),
- kiểm soát tính liên thông và loop closure,
- ưu tiên cặp gần nhau trong chuỗi (video) hoặc chọn một ảnh làm tham chiếu.

### Các thành phần chính

#### 2.1. `make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)`
Hàm chính tạo danh sách cặp.

##### Các kiểu `scene_graph`
1) `complete`
- Tạo đồ thị đầy đủ (complete graph) nhưng chỉ tạo một chiều (i>j), sau đó có thể `symmetrize`.
- Số cặp ~ $\frac{N(N-1)}{2}$ (hoặc gấp đôi nếu symmetrize).
- Tốt cho tập nhỏ, hoặc khi bạn muốn chất lượng cao bằng nhiều ràng buộc.

2) `swin-K` hoặc `swin-K-noncyclic`
- “sliding window” theo chỉ số ảnh: mỗi ảnh nối với K ảnh tiếp theo.
- `noncyclic` nghĩa là **không** nối vòng (không loop closure cuối→đầu).
- Mặc định nếu parse lỗi `K` thì lấy `winsize=3`.
- Dùng `set` để tránh trùng cạnh.

3) `logwin-K` hoặc `logwin-K-noncyclic`
- Window theo offset lũy thừa: `[1,2,4,8,...]` đến `2^(K-1)`.
- Ý tưởng: vừa có cạnh gần (ổn định), vừa có cạnh xa (ràng buộc global) với số cạnh chỉ tăng tuyến tính.

4) `oneref-R`
- Chọn một ảnh tham chiếu `refid=R` (mặc định 0).
- Ghép ref với tất cả ảnh còn lại.
- Hợp khi bạn muốn neo tất cả vào một view “tốt” (ít blur, rõ vật thể) hoặc để giảm pair count.

##### `symmetrize`
- Nếu True: thêm các cặp đảo `(img2,img1)`.
- Nhiều pipeline pairwise muốn cả hai chiều để model/augment xử lý nhất quán.

##### `prefilter`
Sau khi tạo xong, có thể lọc bớt theo chuỗi thời gian:
- `prefilter='seqT'` → chỉ giữ cặp có khoảng cách chỉ số $|i-j| \le T$.
- `prefilter='cycT'` → tương tự nhưng tính khoảng cách theo vòng (cyclic).

Điểm hay:
- Tách rõ “tạo đồ thị” và “lọc đồ thị”. Bạn có thể thử nhiều chiến lược mà không động vào phần model.

#### 2.2. `sel(x, kept)`
Utility chọn subset theo chỉ số `kept` trên nhiều kiểu dữ liệu:
- dict → đệ quy cho từng field
- tensor/ndarray → index
- tuple/list → giữ phần tử theo index

Điểm hay:
- Đây là pattern cực hữu ích khi bạn có nhiều cấu trúc song song (view1/view2/pred1/pred2) và muốn lọc đồng bộ.

#### 2.3. `_filter_edges_seq(edges, seq_dis_thr, cyclic=False)`
- `edges` là list các cặp chỉ số `(i,j)`.
- Tính `n` = số ảnh (từ max index + 1).
- Với mỗi cạnh:
  - tính `dis = abs(i-j)`
  - nếu `cyclic`: lấy min giữa khoảng cách thẳng và khoảng cách vòng (`i+n-j`, `i-n-j`).
  - nếu `dis <= seq_dis_thr`: giữ.

#### 2.4. `filter_pairs_seq(pairs, seq_dis_thr, cyclic=False)`
- Chuyển `pairs` sang `edges` bằng cách đọc `img['idx']`.
- Lọc bằng `_filter_edges_seq`.
- Trả về list pairs đã giữ.

#### 2.5. `filter_edges_seq(view1, view2, pred1, pred2, seq_dis_thr, cyclic=False)`
Trường hợp bạn đã có:
- `view1`, `view2`: batch metadata cho từng cạnh
- `pred1`, `pred2`: output tương ứng

thì hàm này lọc tất cả đồng bộ và in log:
- “kept X/Y edges”.

### Vai trò trong DUSt3R
- Đây là “bộ điều khiển độ phức tạp” của pipeline multi-view.
- Chọn `scene_graph` phù hợp giúp cân bằng:
  - chất lượng tái dựng / độ ổn định định vị
  - thời gian chạy và VRAM
- Các lựa chọn như `swin`/`logwin`/`oneref` là những chiến lược sampling cạnh rất thực tế cho:
  - chuỗi video
  - bộ ảnh nhiều view
  - dữ liệu có tính tuần hoàn (panorama/loop)

---

## Bạn học được gì hay nhất từ hai script này?

### A) Thiết kế pipeline “đầu vào sạch” (image preprocessing)
- **EXIF transpose**: luôn sửa orientation trước khi resize/crop, tránh lỗi “ảnh xoay 90°” làm hỏng geometry.
- **Tách rõ normalize và denormalize**: vào model dùng Normalize, ra visualize dùng hàm `rgb()`.
- **Đảm bảo ràng buộc kiến trúc**: crop kích thước chia hết `patch_size` để tương thích ViT/transformer.
- **Optional dependencies**: hỗ trợ HEIF nếu có, không có vẫn chạy.

### B) Tư duy đồ thị cho bài toán multi-view
- Thay vì “chạy mọi cặp”, bạn định nghĩa **đồ thị cạnh** theo mục tiêu:
  - complete: tối đa ràng buộc
  - sliding window: ưu tiên cặp gần (ổn định, rẻ)
  - log window: thêm ràng buộc xa mà vẫn tiết kiệm
  - one-ref: neo vào một view tốt

### C) Mẫu code lọc batch đồng bộ
- Hàm `sel()` là một pattern gọn gàng để lọc nhiều cấu trúc dữ liệu song song.
- Đây là kỹ năng rất đáng học cho các pipeline DL: bạn hiếm khi lọc “một tensor”, mà lọc cả một batch gồm metadata + predictions.

### D) Thực dụng trong engineering
- `verbose` logging vừa đủ để debug (kích thước trước/sau), không làm code rối.
- `assert imgs` fail-fast: lỗi sớm và rõ hơn thay vì chạy sâu mới crash.

---

## Gợi ý cách dùng trong thực tế (mental model)

- Khi bạn muốn chạy DUSt3R trên một folder ảnh:
  1) `load_images(folder, size=512, patch_size=16)` → list ảnh đã chuẩn hoá
  2) `make_pairs(imgs, scene_graph='swin-3', prefilter='seq10')` (ví dụ) → list cặp
  3) Feed từng cặp vào phần inference để lấy predictions và tối ưu/ghép nối.

Bạn có thể coi hai file này là “data plumbing”: một file lo **chuẩn hoá ảnh**, một file lo **cấu trúc hoá quan hệ giữa ảnh**.
