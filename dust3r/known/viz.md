# Visualization trong DUSt3R: `dust3r/viz.py`

File `dust3r/viz.py` là bộ tiện ích **hiển thị 3D** của DUSt3R dựa trên `trimesh`. Nó giúp bạn:

- Xem **point cloud** (điểm 3D) được dự đoán/tối ưu.
- Xem **mesh** được tạo từ pointmap (mỗi pixel thành 2 tam giác).
- Vẽ **camera frustum** (hình nón/khung camera) theo pose và (xấp xỉ) intrinsics.
- Thực hiện một số hậu xử lý phục vụ viz như **segment_sky** để bỏ bầu trời.

Trong pipeline DUSt3R, file này thường được gọi từ:
- Demo/inference để quan sát kết quả.
- Global alignment (`cloud_opt/*`) để hiển thị scene đang tối ưu (ví dụ `BasePCOptimizer.show()` dùng `SceneViz`).

---

## 0) Phụ thuộc và nguyên tắc “optional dependency”

Ở đầu file:
- Cố import `trimesh` trong `try/except`.
- Nếu thiếu `trimesh`, code in cảnh báo:
  - `"module trimesh is not installed, cannot visualize results"`

Điểm đáng học:
- Visualization thường là optional; không nên làm cả project fail nếu thiếu.

---

## 1) Các hàm utility nhỏ

### `cat_3d(vecs)`
Mục đích: chuẩn hoá một hoặc nhiều mảng/tensor 3D thành một mảng `(N,3)`.

- Nếu input là `np.ndarray` hoặc `torch.Tensor` → bọc thành list.
- `to_numpy(vecs)` chuyển về numpy.
- `reshape(-1,3)` và `np.concatenate`.

Dùng khi tạo point cloud từ nhiều view hoặc nhiều batch.

### `uint8(colors)`
Chuẩn hoá màu về `np.uint8`:
- Nếu là float (0..1) → nhân 255.
- Assert giá trị trong [0,255].

### `cat(a, b)`
Gộp 2 pointcloud (mỗi cái có thể là HxWx3) thành một `(N,3)` bằng `reshape(-1,3)` rồi concatenate.

---

## 2) Hiển thị point cloud

### `show_raw_pointcloud(pts3d, colors, point_size=2)`
- Tạo `trimesh.Scene()`.
- Tạo `trimesh.PointCloud(cat_3d(pts3d), colors=cat_3d(colors))`.
- `scene.show(line_settings={'point_size': point_size})`.

Vai trò:
- Cách nhanh nhất để xem output 3D từ model/optimizer.

### `show_raw_pointcloud_with_cams(imgs, pts3d, mask, focals, cams2world, ...)`
Hiển thị pointcloud + camera frustums:

- Ghép tất cả pts hợp lệ theo `mask`.
- Tạo `PointCloud` với màu từ ảnh.
- Với mỗi camera pose:
  - gọi `add_scene_cam(...)` để vẽ camera frustum và (tuỳ chọn) texture ảnh lên màn hình camera.

Điểm đáng học:
- Khi debug multi-view, nhìn camera + pointcloud cùng lúc giúp phát hiện scale/pose drift rất nhanh.

---

## 3) Từ pointmap → mesh (tam giác hoá ảnh)

### `pts3d_to_trimesh(img, pts3d, valid=None)`
Mục tiêu: tạo dữ liệu mesh từ một pointmap `pts3d` và ảnh `img` cùng shape `(H,W,3)`.

Các bước:
1) `vertices = pts3d.reshape(-1,3)`.

2) Tạo faces (mỗi pixel cell = 2 triangles):
- Dựng grid index `idx` shape (H,W).
- Lấy 4 góc của mỗi ô: `idx1,idx2,idx3,idx4`.
- Tạo 2 tam giác cho mỗi ô.
- Đồng thời, **thêm lại các tam giác đảo chiều** (backward) để “tắt face culling” (một cách rẻ và phổ biến).

3) Màu theo mặt (face color):
- Lấy màu ảnh ở góc tương ứng.

4) Nếu có `valid` mask:
- Bỏ những face có bất kỳ đỉnh nào invalid.

Output trả về dict:
- `vertices`, `faces`, `face_colors`.

Điểm đáng học:
- Cách tạo mesh từ dense depth/pointmap là một kỹ thuật cơ bản để xuất/quan sát reconstruction.

### `cat_meshes(meshes)`
Gộp nhiều mesh dict thành một mesh lớn:
- Cộng offset index cho `faces` dựa trên số vertex tích luỹ.
- Concatenate `vertices`, `faces`, `face_colors`.

---

## 4) Xem cặp ảnh + confidence + pointcloud

### `show_duster_pairs(view1, view2, pred1, pred2)`
Hàm debug interative (Matplotlib):

- Với mỗi edge/pair e:
  - Lấy idx i/j.
  - Hiển thị img1/img2 (qua `rgb(view['img'][e])`).
  - Hiển thị confidence map `pred1['conf'][e]` và `pred2['conf'][e]`.
  - Tính score thô: `conf1.mean()*conf2.mean()`.
  - Nếu người dùng nhập `y` → gọi `show_raw_pointcloud(...)` để xem pointcloud ghép từ 2 phía.

Vai trò:
- Đây là công cụ “sanity check” cho pairwise prediction (đặc biệt hữu ích khi train/finetune hoặc debug dataset).

---

## 5) Camera visualization

### `OPENGL` và quy ước trục
```python
OPENGL = [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]
```
Ma trận này đảo trục Y và Z để chuyển từ quy ước camera/world (thường của vision) sang quy ước hiển thị kiểu OpenGL.

### `auto_cam_size(im_poses)`
- `cam_size = 0.1 * get_med_dist_between_poses(im_poses)`.

Ý nghĩa:
- Tự chọn kích thước camera frustum theo “khoảng cách pose trung vị” để không quá to/nhỏ khi scene scale thay đổi.

### `add_scene_cam(scene, pose_c2w, edge_color, image=None, focal=None, imsize=None, screen_width=0.03, marker=None)`
Đây là hàm lõi để vẽ camera:

1) Xác định kích thước H/W từ:
- `image` nếu có
- hoặc `imsize`
- hoặc xấp xỉ từ `focal`

2) Xác định `focal` nếu chưa có:
- mặc định `min(H,W)*1.1`.

3) Dựng “fake camera” dạng cone:
- Cone có width/height tỉ lệ với screen_width và focal.
- Apply transform: `pose_c2w @ OPENGL @ aspect_ratio @ rot45`.

4) Nếu có `image`:
- Tạo một quad ở vị trí màn hình camera.
- Map texture bằng `trimesh.visual.TextureVisuals`.

5) Tạo mesh camera edges:
- Nhân 3 bộ vertices hơi khác nhau (0.95* và rotated 2°) để tạo cảm giác “viền” dày.
- Thêm faces đảo chiều để tránh culling.

6) Option `marker='o'`:
- Thêm một icosphere tại tâm camera.

Điểm đáng học:
- Dựng camera frustum + texture là cách trực quan hoá cực mạnh để hiểu pose và hướng nhìn.

---

## 6) Class `SceneViz`

`SceneViz` là wrapper hướng đối tượng quanh `trimesh.Scene`.

### `__init__`
- `self.scene = trimesh.Scene()`.

### `add_pointcloud(self, pts3d, color=(0,0,0), mask=None, denoise=False)`
- Chuẩn hoá pts và mask về list.
- Concatenate các điểm theo mask.
- Tạo `PointCloud` và set vertex colors:
  - Nếu `color` là array/list/tensor → dùng per-point colors.
  - Nếu là tuple rgb → broadcast.
- Option `denoise=True`:
  - Lấy median centroid.
  - Bỏ top 1% xa nhất (quantile 0.99).

Vai trò:
- Đây là hàm hay dùng nhất khi hiển thị kết quả global alignment.

### `add_camera(...)` và `add_cameras(...)`
- `add_camera` hỗ trợ truyền:
  - `focal` dạng scalar hoặc `intrinsics` dạng (3,3).
- `add_cameras` lặp qua danh sách poses.

### `show(point_size=2)`
- `self.scene.show(...)`.

### `add_rgbd(...)` (lưu ý)
Trong file hiện có **2 định nghĩa `add_rgbd`** trong `SceneViz`.
- Ở Python, định nghĩa sau sẽ **ghi đè** định nghĩa trước.
- Bản sau sử dụng `depthmap_to_absolute_camera_coordinates(...)`.

Điểm đáng học:
- Khi code grow nhanh, nên tránh trùng tên method; đây là loại lỗi “khó thấy” vì không crash ngay.

---

## 7) Segment sky (lọc bầu trời)

### `segment_sky(image)`
Mục tiêu: tạo mask sky để bỏ bầu trời khỏi pointcloud (bầu trời thường gây nhiễu depth/3D).

Các bước:
1) Chuyển image về numpy uint8 (0..255) nếu cần.
2) `cv2.cvtColor(image, cv2.COLOR_BGR2HSV)`.
3) Tạo mask màu (blue-ish) + các điều kiện “xám sáng/luminous gray”.
4) Morphological opening.
5) Connected components: giữ CC lớn nhất (hoặc vài CC lớn gần bằng lớn nhất).
6) Trả về `torch.Tensor` boolean mask.

Vai trò:
- Dùng trong global alignment (ví dụ `BasePCOptimizer.mask_sky()` gọi `segment_sky`).

Điểm đáng học:
- Dù đơn giản (HSV threshold + CC), đây là một thủ thuật rất thực dụng để cải thiện chất lượng reconstruction.

---

## 8) Những điều hay nhất bạn có thể học từ script này

1) Optional dependency pattern
- Visualization không bắt buộc; thiếu `trimesh` vẫn chạy inference/training.

2) Chuẩn hoá data boundary (torch ↔ numpy)
- Nhiều hàm đầu tiên đều gọi `to_numpy`, giúp viz code nhất quán.

3) Dựng camera frustum đúng cách
- Dùng `pose_c2w`, chuyển hệ trục với `OPENGL`, scale theo `focal`/H để frustum hợp lý.

4) Từ pointmap → mesh
- Tam giác hoá ảnh là kỹ năng nền tảng cho các pipeline depth/3D.

5) Multi-view debug workflow
- `show_duster_pairs` cho bạn một vòng lặp debug: ảnh → conf → pointcloud.

6) Một vài “cảnh báo” engineering để bạn rút kinh nghiệm
- Trùng method `add_rgbd` (định nghĩa sau ghi đè định nghĩa trước).
- Có một `assert col.shape == pts.shape, bb()` trong `add_pointcloud` – nếu assert fail sẽ gọi `bb()` nhưng `bb` không được định nghĩa trong file này (có thể là leftover debug). Đây là thứ nên dọn dẹp khi production.

---

## Script này nằm ở đâu trong DUSt3R?

- Pairwise inference / demo: dùng để xem predicted pts3d/conf.
- Global alignment (`cloud_opt/*`): dùng `SceneViz` để xem tiến trình tối ưu.
- Hậu xử lý: có công cụ lọc sky (và global alignment có clean_pointcloud riêng).

Nói ngắn gọn: `viz.py` là “cửa sổ” để bạn nhìn vào những thứ DUSt3R đang làm trong 3D.
