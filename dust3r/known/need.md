# Lộ trình đọc DUSt3R theo phần cốt lõi

Mục tiêu của lộ trình này là giúp bạn hiểu **cốt lõi thật sự** của DUSt3R, không sa vào các chi tiết dễ đoán như CLI boilerplate, logging phụ, hay các utility quá nhỏ.

Nếu chỉ muốn nắm “xương sống” của dự án, hãy đọc theo thứ tự dưới đây.

---

## 1) `dust3r/utils/image.py`

### Nên đọc gì
- `load_images(...)`
- `imread_cv2(...)`
- `rgb(...)`
- `_resize_pil_image(...)`

### Cần hiểu sau khi đọc xong
- Ảnh được đưa vào model ở dạng nào.
- Tại sao phải resize/crop theo `patch_size`.
- `true_shape` dùng để làm gì.
- Cách ảnh được chuẩn hóa trước khi đi vào backbone.

### Vì sao đây là file đầu tiên
Đây là cổng vào dữ liệu. Nếu không hiểu ảnh được chuẩn hóa thế nào, bạn sẽ không hiểu phần model phía sau đang nhận input gì.

---

## 2) `dust3r/image_pairs.py`

### Nên đọc gì
- `make_pairs(...)`
- `filter_pairs_seq(...)`
- `filter_edges_seq(...)`
- `sel(...)`

### Cần hiểu sau khi đọc xong
- DUSt3R không chạy trên toàn bộ ảnh cùng lúc mà chạy theo **cặp ảnh**.
- “Scene graph” là gì.
- Các kiểu ghép cặp: `complete`, `swin`, `logwin`, `oneref`.
- Vì sao batch pairwise cần symmetrize và lọc theo khoảng cách chuỗi.

### Cần nắm được ý chính
Đây là phần quyết định cấu trúc đồ thị của bài toán. Muốn hiểu global alignment, bạn phải hiểu graph cạnh được tạo ra như thế nào.

---

## 3) `dust3r/patch_embed.py`

### Nên đọc gì
- `PatchEmbedDust3R.forward(...)`
- `ManyAR_PatchEmbed.forward(...)`
- `get_patch_embed(...)`

### Cần hiểu sau khi đọc xong
- Ảnh được biến thành token ra sao.
- Vì sao patch size phải chia hết ảnh.
- Cách DUSt3R xử lý ảnh portrait/landscape trong cùng pipeline.
- `true_shape` tác động thế nào đến positional embedding.

### Chỉ cần nhớ
Đây là lớp chuyển từ ảnh sang token. Nếu không hiểu phần này, bạn sẽ khó hiểu encoder/decoder hoạt động trên cái gì.

---

## 4) `dust3r/model.py`

### Nên đọc gì
- `AsymmetricCroCo3DStereo.__init__(...)`
- `_encode_image(...)`
- `_encode_symmetrized(...)`
- `_decoder(...)`
- `forward(...)`
- `set_downstream_head(...)`

### Cần hiểu sau khi đọc xong
- Kiến trúc thật của DUSt3R: 2 encoder stream, 2 decoder branch, 2 head.
- Vì sao model là **bất đối xứng**.
- Vì sao view2 được biểu diễn trong frame của view1.
- Trick symmetrized batch để tiết kiệm compute.
- Tại sao model có thể dùng nhiều head khác nhau.

### Đây là file cực quan trọng
Nếu chỉ đọc một file về model, hãy đọc file này thật kỹ. Nó là lõi kiến trúc của DUSt3R.

---

## 5) `dust3r/heads/linear_head.py`

### Nên đọc gì
- `LinearPts3d.__init__(...)`
- `LinearPts3d.forward(...)`

### Cần hiểu sau khi đọc xong
- Token cuối của decoder được map sang 3D point map như thế nào.
- Vì sao dùng `pixel_shuffle`.
- Tại sao mỗi token phải dự đoán một block điểm cho cả patch.

### Ý chính cần giữ
Đây là head đơn giản nhất để thấy cách DUSt3R đổi token thành output dense.

---

## 6) `dust3r/heads/dpt_head.py`

### Nên đọc gì
- `PixelwiseTaskWithDPT.__init__(...)`
- `DPTOutputAdapter_fix.forward(...)`
- `create_dpt_head(...)`

### Cần hiểu sau khi đọc xong
- Cách DUSt3R khai thác nhiều layer của backbone để tạo output dense.
- Cơ chế hook nhiều layer và refine multi-scale.
- Vì sao head này nặng hơn linear head nhưng mạnh hơn về biểu diễn.

### Điều cần nắm
Đây là biến thể head giàu ngữ cảnh hơn. Đọc file này để hiểu triết lý “dense prediction từ nhiều tầng token”.

---

## 7) `dust3r/heads/postprocess.py`

### Nên đọc gì
- `postprocess(...)`
- `reg_dense_depth(...)`
- `reg_dense_conf(...)`

### Cần hiểu sau khi đọc xong
- Output raw của head được biến thành `pts3d` và `conf` như thế nào.
- `depth_mode` và `conf_mode` nghĩa là gì.
- Confidence là đầu ra đã được biến đổi, không phải giá trị ngẫu nhiên.

### Vì sao cần đọc
Đây là chỗ chốt semantics của output. Nếu không hiểu postprocess, bạn sẽ hiểu sai đầu ra model.

---

## 8) `dust3r/inference.py`

### Nên đọc gì
- `make_batch_symmetric(...)`
- `loss_of_one_batch(...)`
- `inference(...)`
- `get_pred_pts3d(...)`
- `find_opt_scaling(...)`

### Cần hiểu sau khi đọc xong
- Cách batch được tạo ra cho inference.
- Khi nào phải ép batch size = 1.
- Model output được đổi thành point cloud dùng chung như thế nào.
- Scale fitting được xử lý ra sao khi metric chưa xác định.

### Điều quan trọng
Đây là cầu nối giữa data đã chuẩn hóa và model output. Nó cho bạn thấy inference pipeline thực sự chạy thế nào.

---

## 9) `dust3r/losses.py`

### Nên đọc gì
- `Regr3D`
- `ConfLoss`
- `Regr3D_ScaleInv`
- `Regr3D_ShiftInv`
- `Regr3D_ScaleShiftInv`

### Cần hiểu sau khi đọc xong
- Model được train để dự đoán 3D theo frame nào.
- Vì sao loss phải có invariance theo scale hoặc shift.
- Confidence được học bằng công thức nào.
- Vì sao `ConfLoss` là chìa khóa để confidence có ý nghĩa.

### Đây là phần rất cốt lõi
Nếu bạn muốn hiểu model “học cái gì”, đây là file phải đọc kỹ.

---

## 10) `dust3r/training.py`

### Nên đọc gì
- `train(...)`
- `train_one_epoch(...)`
- `test_one_epoch(...)`
- `build_dataset(...)`

### Cần hiểu sau khi đọc xong
- Vòng lặp train/test diễn ra thế nào.
- AMP, gradient accumulation, DDP, checkpoint, TensorBoard được ghép ra sao.
- Criterion từ `losses.py` được cắm vào training loop như thế nào.

### Chỉ cần đọc sâu phần này ở mức workflow
Không cần sa vào từng dòng logging. Cái cần hiểu là cách training orchestration nối model + loss + dataset.

---

## 11) `dust3r/cloud_opt/init_im_poses.py`

### Nên đọc gì
- `init_minimum_spanning_tree(...)`
- `minimum_spanning_tree(...)`
- `fast_pnp(...)`
- `init_from_known_poses(...)`
- `align_multiple_poses(...)`

### Cần hiểu sau khi đọc xong
- Tại sao global alignment cần khởi tạo tốt.
- MST được dùng để tạo initial scene như thế nào.
- PnP và focal estimation giúp khởi tạo pose ra sao.
- Vì sao khởi tạo sai thì tối ưu toàn cục dễ hỏng.

### Ý chính
Đây là một trong những file quyết định kết quả reconstruction toàn cục.

---

## 12) `dust3r/cloud_opt/base_opt.py`

### Nên đọc gì
- `_init_from_views(...)`
- `_compute_img_conf(...)`
- `get_pw_poses(...)`
- `get_pts3d(...)`
- `forward(...)`
- `compute_global_alignment(...)`
- `clean_pointcloud(...)`

### Cần hiểu sau khi đọc xong
- Bài toán global alignment được biểu diễn bằng biến nào.
- Loss toàn cục được tính như thế nào trên graph cạnh.
- Confidence được dùng làm weight ra sao.
- `min_conf_thr` là ngưỡng lọc hậu xử lý chứ không phải confidence mà model tự “đánh giá” bằng rule ngoài.
- Cách pointcloud được làm sạch bằng consistency đa view.

### Đây là lõi của global alignment
File này là trọng tâm nếu bạn muốn hiểu cách DUSt3R ghép nhiều ảnh thành một scene 3D chung.

---

## 13) `dust3r/cloud_opt/optimizer.py` và `modular_optimizer.py`

### Nên đọc gì
- Trong `optimizer.py`: `PointCloudOptimizer.__init__(...)`, `depth_to_pts3d(...)`, `forward(...)`
- Trong `modular_optimizer.py`: các hàm preset/freeze như `preset_pose(...)`, `preset_focal(...)`, `preset_principal_point(...)`

### Cần hiểu sau khi đọc xong
- Khác nhau giữa optimizer “cổ điển” và bản modular.
- Image-wise variables được tối ưu ra sao.
- Khi nào nên freeze một phần pose/intrinsics.

### Chỉ nên đọc sau khi hiểu base_opt
Đây là lớp cụ thể của global alignment, nên đọc sau khi đã hiểu bài toán gốc.

---

## 14) `dust3r/viz.py`

### Nên đọc gì
- `SceneViz`
- `add_pointcloud(...)`
- `add_cameras(...)`
- `add_scene_cam(...)`
- `pts3d_to_trimesh(...)`
- `segment_sky(...)`

### Cần hiểu sau khi đọc xong
- Cách DUSt3R biến scene 3D thành hình để xem.
- Cách camera pose được vẽ trong không gian 3D.
- Hậu xử lý như sky masking hỗ trợ reconstruction thế nào.

### Mục tiêu khi đọc
Không cần đọc để hiểu thuật toán lõi, nhưng rất hữu ích để hiểu output thực tế của pipeline.

---

## Thứ tự đọc đề xuất, ngắn gọn nhất

1. `dust3r/utils/image.py`
2. `dust3r/image_pairs.py`
3. `dust3r/patch_embed.py`
4. `dust3r/model.py`
5. `dust3r/heads/postprocess.py`
6. `dust3r/heads/linear_head.py`
7. `dust3r/heads/dpt_head.py`
8. `dust3r/inference.py`
9. `dust3r/losses.py`
10. `dust3r/training.py`
11. `dust3r/cloud_opt/init_im_poses.py`
12. `dust3r/cloud_opt/base_opt.py`
13. `dust3r/cloud_opt/optimizer.py`
14. `dust3r/cloud_opt/modular_optimizer.py`
15. `dust3r/viz.py`

---

## Nếu chỉ muốn nắm lõi nhất, hãy ưu tiên 6 file này

1. `dust3r/utils/image.py`
2. `dust3r/image_pairs.py`
3. `dust3r/patch_embed.py`
4. `dust3r/model.py`
5. `dust3r/losses.py`
6. `dust3r/cloud_opt/base_opt.py`

Nếu hiểu chắc 6 file này, bạn đã nắm được gần như toàn bộ xương sống của DUSt3R.

---

## Những phần có thể đọc lướt hoặc bỏ qua ban đầu

- CLI argparse chi tiết trong `training.py` và `demo.py`
- Logging TensorBoard chi tiết
- Các helper nhỏ chỉ dùng một lần
- Các biến thể fallback/compatibility cũ nếu bạn chưa cần debug checkpoint
- Các file dataset cụ thể nếu bạn chưa huấn luyện lại từ đầu

---

## Kết luận

Muốn hiểu DUSt3R nhanh và đúng trọng tâm, hãy đi theo chuỗi:

**ảnh → pairs → patch tokens → model → head/postprocess → inference → loss → training → global alignment**

Nếu bạn đọc đúng thứ tự và chỉ tập trung vào các hàm đã liệt kê ở trên, bạn sẽ nắm được phần lõi thật sự của dự án mà không bị chìm trong chi tiết phụ.
