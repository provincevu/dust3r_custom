# Global Alignment trong DUSt3R: `base_opt.py`, `modular_optimizer.py`, `init_im_poses.py`

Tài liệu này giải thích 3 script thuộc thư mục `dust3r/cloud_opt/` – phần **căn chỉnh toàn cục** (global alignment) của DUSt3R.

Nếu coi DUSt3R gồm 2 tầng:

1) **Pairwise prediction** (model dự đoán cho từng cặp ảnh): sinh ra các pointmap `pts3d` và `conf` cho từng cặp.
2) **Global alignment**: ghép các dự đoán pairwise thành một scene nhất quán: tìm pose/camera/intrinsics/depth sao cho các pointmap “khớp nhau” tốt nhất trên toàn đồ thị.

Ba file này chính là tầng (2).

---

## Tổng quan: bài toán tối ưu đang giải là gì?

### Đồ thị (graph)
- **Node**: ảnh (view) $i=0..N-1$
- **Edge**: quan sát pairwise giữa hai ảnh $(i,j)$, được model trả về dưới dạng:
  - `pred_i[i_j]`: pointmap của ảnh i (trong frame i)
  - `pred_j[i_j]`: pointmap của ảnh j nhưng đã biểu diễn trong frame i (đây là `pts3d_in_other_view` từ model)
  - `conf_i[i_j]`, `conf_j[i_j]`: độ tin cậy cho từng pixel

Trong code:
- `BasePCOptimizer.edges`: list các cạnh `(i,j)` lấy từ `view1['idx']`, `view2['idx']`.
- `BasePCOptimizer.pred_i`, `pred_j`, `conf_i`, `conf_j`: map từ string key `"i_j"` sang tensor.

### Biến tối ưu (parameters)
Có hai cấp biến:

1) **Pairwise pose parameters** theo cạnh $e=(i,j)$:
- `pw_poses[e]`: tham số hoá pose (unit quaternion + translation + log scale) cho cạnh.
- `pw_adaptors[e]`: “adaptor” nhỏ để scale xy/z khác nhau (giảm mismatch do scale/độ sâu).

2) **Image-wise variables** theo ảnh (chỉ có ở `ModularPointCloudOptimizer`):
- `im_depthmaps[i]`: depthmap cho ảnh i (lưu ở log-depth).
- `im_poses[i]`: pose camera-to-world của ảnh i.
- `im_focals[i]`: focal (có thể 1 giá trị hoặc fx/fy) cho ảnh i.
- `im_pp[i]`: principal point offset cho ảnh i (tuỳ chọn optimize).

### Hàm loss (ý nghĩa trực quan)
Mục tiêu: tìm scene sao cho với mỗi cạnh $(i,j)$, pointcloud hiện tại của ảnh i/j **gần** với pointmap dự đoán sau khi được “đặt” vào cùng world.

Trong `BasePCOptimizer.forward()`:
- `proj_pts3d = self.get_pts3d()` là pointmap hiện tại của từng ảnh (đã ở world frame).
- Với mỗi edge $e=(i,j)$:
  - Lấy predicted points cho phía i và j (`pred_i`, `pred_j`) rồi biến đổi bằng pose cạnh `pw_poses[e]` và adaptor `pw_adapt[e]`.
  - Tính khoảng cách (L1/L2/...) giữa:
    - `proj_pts3d[i]` và `aligned_pred_i`
    - `proj_pts3d[j]` và `aligned_pred_j`
  - Weighted theo confidence.

Nói ngắn gọn: **tối ưu để mọi predicted pairwise pointmaps “ăn khớp” với pointmaps hiện tại của từng ảnh trong world**.

---

## 1) `dust3r/cloud_opt/base_opt.py`

File này cung cấp:

- Base class `BasePCOptimizer`: đóng gói dữ liệu edges/pred/conf và định nghĩa loss.
- Vòng lặp tối ưu `global_alignment_loop()`.
- Các tiện ích: mask sky, visualize, clean pointcloud.

### 1.1. `BasePCOptimizer.__init__` và `_init_from_views(...)`

Có 2 đường khởi tạo:

1) `BasePCOptimizer(other_optimizer)`:
- Nếu truyền vào một object (dạng dict-like), nó `deepcopy` và copy một danh sách attribute.
- Đây là cơ chế “clone optimizer state” đơn giản.

2) Khởi tạo từ dữ liệu:
```python
_init_from_views(view1, view2, pred1, pred2, ...)
```
Các bước chính:

- Chuẩn hoá `view1['idx']`/`view2['idx']` thành list.
- Tạo `self.edges = [(i,j), ...]`.
- `self.is_symmetrized` check xem đồ thị có đủ cả (i,j) và (j,i).
- `self.n_imgs = self._check_edges()`:
  - Assert chỉ số ảnh là 0..N-1 không thiếu.

- Nạp prediction:
  - `pred1['pts3d']` → `self.pred_i`
  - `pred2['pts3d_in_other_view']` → `self.pred_j`
  - `pred1['conf']` và `pred2['conf']` → `self.conf_i`, `self.conf_j`

- Tính `self.imshapes` cho từng ảnh: shape H/W để biết kích thước pointmap.

- Xử lý confidence:
  - `self.min_conf_thr`: ngưỡng mask
  - `self.conf_trf = get_conf_trf(conf)` để đưa conf về không gian weight (ví dụ log/exp tuỳ mode).
  - `self.im_conf = _compute_img_conf(...)`:
    - tạo per-image confidence map bằng cách lấy max qua các cạnh liên quan.

- Tạo tham số pairwise:
  - `self.POSE_DIM = 7` (quat 4 + trans 3)
  - `self.pw_poses = nn.Parameter(rand_pose((n_edges, 1+POSE_DIM)))`
    - `+1` là log-scale.
  - `self.pw_adaptors = nn.Parameter(zeros((n_edges,2)))`:
    - adaptor xy và z; có thể cho phép học hoặc không.

- (Optional) lưu ảnh để viz:
  - Nếu `view1`/`view2` có `img` → build danh sách `imgs` theo index ảnh và convert bằng `rgb()`.

Điểm đáng học:
- Biến các predicted tensors thành cấu trúc map theo edge (`NoGradParamDict`) để truy xuất nhanh và tránh gradient.

### 1.2. Adaptor và pose parameterization

#### `get_adaptors()`
- Từ `pw_adaptors` (2 tham số) tạo ra 3 hệ số `(scale_xy, scale_xy, scale_z)`.
- Option `norm_pw_scale` sẽ trừ mean để product gần 1, giúp ổn định.
- Dùng `exp(adapt/pw_break)` để giới hạn bước thay đổi (brake).

#### `_get_poses(poses)`
- `poses[:, :4]` là quaternion Q.
- `poses[:, 4:7]` là translation T nhưng qua `signed_expm1`.
- Dùng `roma.RigidUnitQuat(Q,T).normalize().to_homogeneous()` để ra ma trận 4x4.

#### `get_pw_poses()`
- Lấy `RT` từ `_get_poses()`.
- Scale cả rotation & translation theo `get_pw_scale()`:
  - scale = `exp(log_scale)` và có thể norm về `base_scale`.

Điểm đáng học:
- Parameterization rất “tối ưu hoá-friendly”: quat unit + log/exp cho scale và translation.

### 1.3. `forward()` = loss toàn cục

Các bước:
- `pw_poses = get_pw_poses()`.
- `pw_adapt = get_adaptors()`.
- `proj_pts3d = get_pts3d()` (abstract – lớp con cung cấp).
- Precompute weight per-edge bằng `conf_trf`.

Với mỗi edge:
- `aligned_pred_i = geotrf(pw_poses[e], pw_adapt[e] * pred_i[i_j])`
- `aligned_pred_j = geotrf(pw_poses[e], pw_adapt[e] * pred_j[i_j])`
- `li = dist(proj_pts3d[i], aligned_pred_i, weight=weight_i).mean()`
- `lj = dist(proj_pts3d[j], aligned_pred_j, weight=weight_j).mean()`
- Loss tổng = mean over edges.

Điểm đáng học:
- Loss được viết theo “graph loop” rõ ràng, dễ debug từng cạnh.

### 1.4. `compute_global_alignment(init=...)`
- Nếu `init == 'mst'/'msp'`: gọi `init_fun.init_minimum_spanning_tree(self)`.
- Nếu `init == 'known_poses'`: gọi `init_fun.init_from_known_poses(self)`.
- Sau đó gọi `global_alignment_loop(self, ...)` để optimize.

### 1.5. `global_alignment_loop()` và schedule
- Tạo optimizer Adam trên tất cả param `requires_grad`.
- Lặp `niter`:
  - Tính `lr` theo `cosine_schedule` hoặc `linear_schedule`.
  - `loss = net()` → backprop → step.

Điểm đáng học:
- “Optimizer là một nn.Module” và loss là `__call__()` là pattern rất gọn cho tối ưu hình học.

### 1.6. `clean_pointcloud(...)`
Mục tiêu: giảm confidence của điểm “bị che khuất” (occluded) khi nhìn từ view khác.

Thuật toán:
1) Với mỗi ảnh i, xét từng ảnh j khác:
2) Project pts3d của i sang camera j: `proj = geotrf(cams[j], pts3d)`.
3) Lấy pixel `(u,v)` bằng intrinsics `K[j]`.
4) Với điểm rơi vào vùng ảnh và có depth > 0:
   - Nếu `proj_depth < (1-tol) * depthmaps[j][u,v]` nghĩa là điểm ở *trước* surface mà j tin là đúng.
   - Nếu confidence của điểm i thấp hơn confidence của j tại pixel đó → coi là “bad point” và clip conf.

Điểm đáng học:
- Một bước hậu xử lý dựa trên multi-view consistency rất thực tế, giúp pointcloud sạch hơn.

---

## 2) `dust3r/cloud_opt/modular_optimizer.py`

### Mục tiêu
`ModularPointCloudOptimizer` là phiên bản chậm hơn nhưng linh hoạt hơn của global alignment:

- Cho phép tối ưu **image-wise**: pose/intrinsics/depth per image.
- Cho phép “preset”/freeze một phần: ví dụ pose đã biết, focal đã biết.

Nó kế thừa `BasePCOptimizer` và implement các abstract method cần thiết.

### 2.1. Các tham số thêm trong `__init__`
- `self.has_im_poses = True`.
- `focal_brake`: hệ số “hãm” khi tham số hoá log(focal).
- `optimize_pp`: có optimize principal point hay không.
- `fx_and_fy`: dùng 1 focal hay 2 (fx,fy).

Tạo biến tối ưu:
- `im_depthmaps`: list param shape (H,W) cho mỗi ảnh, init khoảng `rand/10 - 3` (log depth).
- `im_poses`: list param POSE_DIM cho mỗi ảnh.
- `im_focals`: list param:
  - giá trị khởi tạo ~ `focal_brake * log(max(H,W))`.
- `im_pp`: principal point offset (2,).

Điểm đáng học:
- Parameterize focal theo log để đảm bảo focal luôn dương khi exp.

### 2.2. Preset/freeze (kỹ thuật cực hữu ích)

Các hàm:
- `preset_pose(known_poses, pose_msk=None)`:
  - set pose và `.requires_grad_(False)`.
  - nếu số pose known <= 1 → vẫn giữ `norm_pw_scale` để tránh drift scale.

- `preset_intrinsics(known_intrinsics, msk=None)`:
  - set focal và principal point.

- `preset_focal(known_focals, msk=None)`.
- `preset_principal_point(known_pp, msk=None)`.

Hàm `_get_msk_indices(msk)` cho phép mask là:
- None, int, list/tuple, bool mask, list index.

Điểm đáng học:
- Thiết kế API “mask flexible” giúp workflow downstream cực tiện.

### 2.3. Intrinsics, poses, depth → pts3d

- `_set_focal` lưu `focal_brake * log(focal)`.
- `get_focals` trả `exp(log_focal / focal_brake)`.

- `_set_principal_point` lưu offset so với center `(W/2,H/2)` và chia 10.
- `get_principal_points` trả về center + 10*offset.

- `get_intrinsics` build K (n_imgs,3,3).

- `get_im_poses` lấy ma trận cam-to-world từ `_get_poses`.

- Depth:
  - `_set_depthmap` lưu `log(depth)`.
  - `get_depthmaps` trả `exp(log_depth)`.

- `depth_to_pts3d()`:
  1) Lấy depth, focal, pp, im_poses.
  2) Biến focal thành field dạng `(1,2,H,W)`.
  3) `depthmap_to_pts3d(...)` tạo pointmap trong camera frame.
  4) `geotrf(pose, ptmap)` đưa sang world frame.

Điểm đáng học:
- Dùng depthmap + intrinsics + pose để sinh pointcloud là mối nối giữa “dense depth” và “3D geometry optimization”.

---

## 3) `dust3r/cloud_opt/init_im_poses.py`

### Mục tiêu
Global alignment rất nhạy với khởi tạo. File này cung cấp các cách init:

- Init từ **known poses**.
- Init bằng **Minimum Spanning Tree (MST)** dựa trên score cạnh.
- Dùng PnP (RANSAC) để ước lượng pose/focal khi thiếu.

### 3.1. `init_from_known_poses(self, ...)`
Giả định: mọi pose đều known.

Các bước:
1) Lấy known poses mask và known poses.
2) Lấy known focals và principal points.
3) Với mỗi edge (i,j):
   - Dùng `fast_pnp(self.pred_j[i_j], focal_i, pp=pp_i, msk=...)` để tìm pose tương đối của view2 so với view1.
   - Align hai camera dự đoán (P1=I, P2) với two GT cameras bằng `align_multiple_poses` → ra (s,R,T).
   - Set pairwise pose `self._set_pose(self.pw_poses, e, R, T, scale=s)`.
   - Ghi nhớ edge có score conf tốt nhất để init depthmap.
4) Init depthmaps theo edge tốt nhất.

Điểm đáng học:
- Khi có GT poses, pairwise pose được “kéo” về khớp GT bằng rigid registration, giúp tối ưu nhanh hội tụ.

### 3.2. `init_minimum_spanning_tree(self, ...)` và `minimum_spanning_tree(...)`

Ý tưởng:
- Dùng score cạnh (confidence) để tạo một MST trên đồ thị ảnh.
- MST cho bạn một cấu trúc nối tất cả node với số cạnh tối thiểu nhưng “mạnh nhất”.

Trong `minimum_spanning_tree(...)`:
- Build sparse graph từ `compute_edge_scores`.
- Dùng `scipy.sparse.csgraph.minimum_spanning_tree`.

Quy trình dựng pointcloud theo MST:
1) Chọn cạnh mạnh nhất (i*,j*) làm seed:
   - `pts3d[i] = pred_i[i_j]`
   - `pts3d[j] = pred_j[i_j]`
2) Lặp thêm node mới:
   - Nếu i đã done, j chưa done:
     - rigid registration: align `pred_i[i_j]` với `pts3d[i]` → ra transform.
     - Apply transform lên `pred_j[i_j]` để tạo `pts3d[j]`.
   - Tương tự nếu j đã done.
3) Nếu cần im_poses/focals:
   - Estimate focal từ pointmap (Weiszfeld) nếu thiếu.
   - Dùng `fast_pnp` để estimate pose.

Sau đó `init_from_pts3d(self, ...)`:
- Nếu có nhiều known poses, align toàn bộ scene vào GT bằng `align_multiple_poses`.
- Set pairwise pose cho mọi edge bằng `rigid_points_registration(pred_i, pts3d[i])`.
- Apply `get_pw_norm_scale_factor` để ổn định scale.
- Nếu optimizer có image-wise variables:
  - set depthmap bằng cách project pts3d về camera frame.
  - set im_pose và im_focal.

Điểm đáng học:
- MST init là chiến lược cổ điển nhưng cực hiệu quả để bootstrapping multi-view reconstruction.

### 3.3. PnP và focal estimation

- `estimate_focal(pts3d_i, pp=None)`:
  - ước lượng focal dựa trên pointmap và giả định principal point ở center.

- `fast_pnp(pts3d, focal, msk, ...)`:
  - Dùng RANSAC PnP (OpenCV `solvePnPRansac` với `SOLVEPNP_SQPNP`).
  - Nếu focal None → thử nhiều focal theo geomspace.
  - Trả về `cam_to_world` (đã invert) và best focal.

Điểm đáng học:
- Kết hợp “đoán focal” + PnP RANSAC giúp dựng pose ban đầu ngay cả khi intrinsics không known.

---

## 3 script này đóng vai trò như nào trong DUSt3R?

- `base_opt.py`: định nghĩa **bài toán tối ưu toàn cục** và cung cấp loop optimize + tiện ích làm sạch/visualize.
- `modular_optimizer.py`: cung cấp một optimizer cụ thể có thể tối ưu **pose + intrinsics + depth per-image** và cho phép preset/freeze.
- `init_im_poses.py`: cung cấp **khởi tạo tốt** (MST/known poses/PnP), quyết định lớn đến hội tụ và chất lượng scene.

Nói cách khác:
- `init_im_poses.py` = khởi tạo.
- `BasePCOptimizer.forward()` = định nghĩa loss.
- `global_alignment_loop()` = chạy tối ưu.
- `ModularPointCloudOptimizer` = hiện thực biến ẩn (depth/pose/K) để tạo pointcloud nhất quán.

---

## Những điều hay nhất bạn có thể học từ 3 script này

1) Tư duy “graph optimization” cho bài toán multi-view
- Mọi thứ được viết như tối ưu trên đồ thị edges/nodes, rất gần với bundle adjustment nhưng ở dạng dense pointmap.

2) Parameterization để tối ưu ổn định
- Quat chuẩn hoá, log/exp cho scale/focal, adaptor có brake → giảm nguy cơ nổ gradient.

3) Khởi tạo là sống còn
- MST + rigid registration là một công thức mạnh: bắt đầu từ cạnh tốt nhất và lan dần.

4) Thiết kế API preset/freeze
- Cho phép “đóng đinh” pose/intrinsics khi bạn có sensor/metadata, và chỉ tối ưu phần còn thiếu.

5) Hậu xử lý bằng multi-view consistency
- `clean_pointcloud` là ví dụ rõ của việc dùng các view khác để loại occlusion/outlier.

6) Engineering pattern: optimizer là `nn.Module`
- Dùng cơ chế PyTorch autograd/optimizer/schedule để giải bài toán hình học phức tạp mà code vẫn gọn.
