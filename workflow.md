# DUSt3R Project Workflow (Chi tiết, end-to-end)

Tài liệu này mô tả đầy đủ cách dự án DUSt3R vận hành từ đầu vào đến đầu ra, gồm:
- Pipeline suy luận/tái dụng 3D
- Pipeline huấn luyện
- Pipeline visual localization
- Pipeline tiền xử lý dữ liệu
- Điểm nghẽn hiệu năng
- Điểm có giá trị ứng dụng thực tiễn tốt nhất
- Đề xuất tái cấu trúc
- Xếp hạng 1-10 cho từng phần kèm lý do

Lưu ý:
- Dự án theo giấy phép phi thương mại (CC BY-NC-SA 4.0) và phụ thuộc giấy phép từng dataset.
- Có khoảng cách giữa "workflow kỹ thuật hiện tại" và "workflow sản phẩm hóa"; phần đề xuất bên dưới nhằm lấp đầy khoảng cách đó.

---

## 1) Tổng quan kiến trúc

DUSt3R được tổ chức thành các lớp sau:

1. Entry points
- demo.py (root): chạy Gradio reconstruction demo.
- train.py (root): gọi training loop.
- visloc.py (root): đánh giá visual localization bằng PnP.

2. Core model và suy luận
- dust3r/model.py: AsymmetricCroCo3DStereo (2 encoder stream + 2 decoder stream + head).
- dust3r/inference.py: đóng gói chạy batch, symmetrize, collate kết quả.
- dust3r/image_pairs.py: tạo cặp (pairs) theo scene graph.
- dust3r/heads/*: head linear hoặc DPT để sinh pts3d + confidence.

3. Hậu xử lý và căn chỉnh toàn cục
- dust3r/cloud_opt/*:
  - PointCloudOptimizer: tối ưu toàn bộ pose/depth/focal từ pairwise predictions.
  - ModularPointCloudOptimizer: phiên bản linh hoạt để freeze một phần biến.
  - PairViewer: chế độ nhanh cho 1 cặp ảnh (bỏ qua global optimize đầy đủ).

4. Dữ liệu huấn luyện
- dust3r/datasets/*: adapters cho ARKitScenes, Co3d, MegaDepth, Waymo, v.v.
- dust3r/datasets/base/*: base dataset, random batched sampler theo aspect ratio.

5. Visual localization
- dust3r_visloc/*: nạp map/query, tạo 2D-3D correspondences, chạy PnP (cv2/poselib/pycolmap), đánh giá metric.

6. Preprocess
- datasets_preprocess/*.py: chuẩn hóa dữ liệu đầu vào từng dataset về format dùng cho loader.

7. Triển khai
- docker/*: compose CPU/CUDA + script run demo.

---

## 2) Workflow suy luận và tái dụng 3D (chi tiết)

### B2.1. Nạp model
- AsymmetricCroCo3DStereo.from_pretrained(...)
- Có 2 nguồn:
  - Local checkpoint: load_model() đọc ckpt, khởi tạo model từ string args trong ckpt.
  - HuggingFace: dùng PyTorchModelHubMixin.
- Nếu ckpt cũ thiếu dec_blocks2, load_state_dict sẽ nhân bản từ dec_blocks.

### B2.2. Nạp và chuẩn hóa ảnh
- dust3r/utils/image.py: load_images(...)
- Chuẩn hóa:
  - EXIF transpose
  - resize theo image_size (224 hoặc 512)
  - crop sao cho kích thước hợp patch_size
  - ImgNorm: ToTensor + normalize quanh 0
- Mỗi ảnh được lưu thành dict:
  - img (tensor BCHW)
  - true_shape
  - idx
  - instance

### B2.3. Tạo đồ thị cặp ảnh
- dust3r/image_pairs.py: make_pairs(...)
- Hỗ trợ:
  - complete
  - swin-k
  - logwin-k
  - oneref-r
- Có thể symmetrize để có (i,j) và (j,i) => cải thiện tính đối xứng khi suy luận/huấn luyện.

### B2.4. Inference cặp ảnh
- dust3r/inference.py: inference(pairs, model, ...)
- Nếu ảnh khác resolution -> ép batch_size=1.
- Mỗi batch:
  - loss_of_one_batch(..., criterion=None)
  - chuyển tensor sang device
  - model(view1, view2)

### B2.5. Forward trong model
- dust3r/model.py:
  1) _encode_symmetrized:
     - nếu input đối xứng, tính 1 nửa forward pass rồi interleave lại.
  2) _decoder:
     - 2 decoder branch (dec_blocks và dec_blocks2) trao đổi thông tin theo hướng bất đối xứng.
  3) _downstream_head:
     - head1 cho view1, head2 cho view2.
  4) output:
     - pred1['pts3d'] trong hệ quy chiếu view1
     - pred2['pts3d_in_other_view'] cũng trong hệ quy chiếu view1
     - confidence map

### B2.6. Global alignment (nếu >2 ảnh)
- dust3r/cloud_opt/global_aligner(...)
- Chọn mode:
  - PairViewer: cho 1-2 ảnh
  - PointCloudOptimizer: multi-view optimize đầy đủ
  - ModularPointCloudOptimizer: optimize linh hoạt hơn, chậm hơn

Quy trình PointCloudOptimizer:
1. Khởi tạo biến tối ưu:
- Pairwise poses + adaptors
- Per-image depthmap
- Per-image pose
- Per-image focal (+ principal point tùy chọn)

2. Khởi tạo giá trị ban đầu:
- compute_global_alignment(init='mst') -> minimum spanning tree trên graph edges, căn theo confidence.

3. Vòng lặp tối ưu:
- global_alignment_loop(...)
- Adam + lịch cosine/linear
- Loss: căn chỉnh point cloud toàn cục với pairwise dự đoán có trọng số confidence.

4. Trích xuất kết quả:
- get_im_poses, get_focals, get_pts3d, get_masks

### B2.7. Xuất và hiển thị kết quả
- dust3r/demo.py + dust3r/viz.py
- Có thể:
  - xuất mesh hoặc point cloud (.glb)
  - mask sky
  - clean depth theo occlusion consistency
  - điều chỉnh ngưỡng confidence

---

## 3) Workflow huấn luyện (chi tiết)

### B3.1. Cấu hình train
- train.py -> dust3r/training.py
- parser nhận:
  - model string (eval)
  - train/test criterion string (eval)
  - train/test dataset string (eval)
  - lr schedule, amp, distributed, save/eval freq

### B3.2. Build dataset và dataloader
- dust3r/datasets/get_data_loader(...)
- Nếu dataset có make_sampler -> BatchedRandomSampler (giữ cùng aspect ratio trong 1 batch)
- Nếu distributed -> DistributedSampler
- Hệ thống EasyDataset cho phép:
  - dataset1 + dataset2
  - k * dataset
  - N @ dataset (resize ngẫu nhiên theo epoch)

### B3.3. BaseStereoViewDataset __getitem__
Mỗi sample 2 view đi qua:
1. _get_views() của dataset cụ thể
2. transform ảnh
3. tính pts3d + valid_mask từ depth + intrinsics + pose
4. transpose portrait -> landscape để đồng nhất
5. trả về 2 view dict đã chuẩn hóa

### B3.4. Train step
- train_one_epoch:
  - loss_of_one_batch(..., symmetrize_batch=True)
  - model forward
  - criterion forward
  - gradient accumulation
  - AMP tùy chọn
  - adjust LR theo iteration

### B3.5. Loss
- dust3r/losses.py
- Cốt lõi:
  - Regr3D: hồi quy pts3d của 2 view trong hệ quy chiếu view1
  - ConfLoss: căn trọng số theo confidence học được
- Biến thể đánh giá:
  - Regr3D_ShiftInv
  - Regr3D_ScaleInv
  - Regr3D_ScaleShiftInv

### B3.6. Eval và checkpoint
- test_one_epoch trên nhiều test dataset
- Lưu:
  - checkpoint-last
  - checkpoint-best
  - checkpoint-final
- Ghi TensorBoard + log.txt (JSON lines)

---

## 4) Workflow visual localization (visloc)

### B4.1. Nạp dataset visloc
- visloc.py nhận chuỗi dataset qua eval, ví dụ:
  - VislocAachenDayNight(...)
  - VislocInLoc(...)
  - VislocSevenScenes(...)
  - VislocCambridgeLandmarks(...)
- Dataset map/query đọc từ kapture/COLMAP + pairs retrieval.

### B4.2. Tạo matching từ DUSt3R
Với mỗi query-map pair:
1. Chạy inference 2 ảnh.
2. Lấy confidence mask.
3. Rút pts2d (query) và pts3d (map).
4. reciprocal nearest-neighbor trên không gian 3D.
5. Đổi hệ tọa độ pixel giữa cv2 <-> colmap và scale về ảnh gốc.

### B4.3. PnP
- dust3r_visloc/localization.py: run_pnp
- Backend:
  - cv2
  - poselib
  - pycolmap
- Lọc tối đa số điểm (pnp_max_points), chạy RANSAC-PnP.

### B4.4. Đánh giá
- Tính lỗi translation/rotation
- aggregate metrics acc@m,deg
- export file kết quả pose

---

## 5) Workflow preprocess dữ liệu

Mục tiêu preprocess:
- chuẩn hóa camera intrinsics/extrinsics
- resize/crop theo quy tắc của DUSt3R
- xuất depth/rgb/meta format đồng nhất
- tạo danh sách pairs

Kiểu script:
- preprocess_co3d.py, preprocess_megadepth.py, preprocess_scannetpp.py, preprocess_waymo.py, ...
- đầu vào là dataset gốc + precomputed_pairs
- đầu ra là thư mục *_processed với image/depth/npz metadata

Đặc điểm quan trọng:
- Có script cần render/undistort (vd scannetpp cần pyrender/trimesh/OpenGL)
- Có script đa luồng (thread/process) cho I/O nặng
- Chất lượng preprocess ảnh hưởng trực tiếp chất lượng và độ ổn định training

---

## 6) Điểm nghẽn hiệu năng (performance bottlenecks)

### 6.1. Dùng eval trên string cho model/dataset/loss
- Nội dung: training.py dùng eval cho model/dataset/criterion.
- Tác động:
  - overhead nhỏ về runtime
  - rủi ro bảo trì/bảo mật lớn
  - khó static-analysis
- Ảnh hưởng thực tế: Trung bình về tốc độ, Cao về vận hành.

### 6.2. Multi-shape inference bị ép batch_size=1
- Nội dung: inference.py nếu pair khác shape -> batch_size=1.
- Tác động: GPU utilization giảm rõ rệt khi dữ liệu đa dạng tỉ lệ khung hình.
- Ảnh hưởng thực tế: Cao.

### 6.3. CPU bottleneck ở preprocessing và dataloader
- Nội dung: PIL/OpenCV crop-resize + load EXR/PNG/JPG + tính pts3d trong __getitem__.
- Tác động: GPU chờ đợi nếu num_workers/pin_memory không tối ưu.
- Ảnh hưởng: Cao khi train lớn.

### 6.4. Global alignment phi tuyến và tốn kém
- Nội dung: tối ưu đồng thời depth + pose + focal cho nhiều ảnh.
- Tác động: niter cao sẽ chậm; memory tăng theo số ảnh và độ phân giải.
- Ảnh hưởng: Cao cho ứng dụng online real-time.

### 6.5. Reciprocal matching bằng KDTree CPU
- Nội dung: find_reciprocal_matches dùng scipy KDTree trên CPU.
- Tác động: khi điểm nhiều (high-res, nhiều map views), matching là nút thắt.
- Ảnh hưởng: Trung bình-Cao trong visloc.

### 6.6. Trimesh/GLB export
- Nội dung: xuất mesh đầy đủ trong demo.
- Tác động: chậm và tốn RAM khi point cloud lớn.
- Ảnh hưởng: Trung bình.

### 6.7. Chưa cache mạnh cho map-side visloc
- Nội dung: map ảnh + pts3d map có thể được nạp lặp lại nhiều lần.
- Tác động: I/O lặp lại, chậm đánh giá.
- Ảnh hưởng: Trung bình.

### 6.8. Python-level loops trong nhiều khâu
- Nội dung: nhiều vòng for per-pair/per-image.
- Tác động: overhead Python cao so với batched tensor ops.
- Ảnh hưởng: Trung bình.

---

## 7) Điểm có giá trị ứng dụng thực tiễn tốt nhất

### 7.1. Dùng pointmap trực tiếp + confidence map
- Lợi ích:
  - dễ dùng cho matching, reconstruction, localization
  - confidence dùng để filter và căn trọng số
- Giá trị: rất cao cho bài toán 3D thực tế.

### 7.2. Kiến trúc pair graph linh hoạt
- make_pairs cho phép complete/swin/logwin/oneref.
- Giá trị:
  - tùy biến theo bộ nhớ và kịch bản
  - dùng cho sequence ngắn, dài, loop closure

### 7.3. Global alignment có chế độ nhanh/chậm
- PairViewer cho 2 ảnh nhanh.
- PointCloudOptimizer cho đa ảnh chất lượng cao.
- Giá trị: cân bằng chất lượng-và-tốc độ tùy use-case.

### 7.4. Hệ sinh thái dataset rộng
- Nhiều dataset indoor/outdoor/object-level.
- Giá trị:
  - tổng quát hóa tốt hơn
  - phù hợp nhiều ngành (robotics, mapping, AR)

### 7.5. Visloc tích hợp backend PnP đa dạng
- cv2/poselib/pycolmap.
- Giá trị:
  - dễ adaptation theo hệ thống có sẵn
  - dễ benchmark công bằng

### 7.6. Docker script cho demo
- dễ khởi động nhanh trên máy khác nhau.
- Giá trị: tốt cho PoC/trình diễn.

---

## 8) Đề xuất tái cấu trúc (nếu dùng cho công việc thực tế quy mô lớn)

### 8.1. Bỏ eval-string, thay bằng registry type-safe
Vấn đề:
- eval cho model/loss/dataset khó test và dễ lỗi runtime.

Đề xuất:
- Tạo registry:
  - MODEL_REGISTRY
  - LOSS_REGISTRY
  - DATASET_REGISTRY
- Config dùng YAML/JSON + pydantic dataclass validate.

Lợi ích:
- an toàn hơn, dễ IDE autocomplete, dễ CI.

### 8.2. Tách rõ 3 tầng: core model, pipeline, app
Vấn đề:
- demo/training/visloc còn giao nhau ở nhiều utility.

Đề xuất:
- dust3r/core: model, heads, losses
- dust3r/pipelines: reconstruction, visloc, training
- dust3r/apps: gradio, cli entrypoint

Lợi ích:
- dễ maintain, dễ test theo module.

### 8.3. Chuẩn hóa schema dữ liệu
Vấn đề:
- Dict động key linh hoạt, dễ sai key khi mở rộng.

Đề xuất:
- dataclass cho ViewBatch, PredBatch, SceneState.

Lợi ích:
- rõ ràng contract, giảm bug ngầm.

### 8.4. Tối ưu dataloader và cache
Đề xuất:
- cache metadata/decoded image (LMDB/WebDataset/Zarr tùy bộ dữ liệu)
- precompute một số biến đổi lặp lại (to_orig, intrinsic-adjust)
- profile num_workers, prefetch_factor, pin_memory theo máy

Lợi ích:
- giảm data stall, tăng GPU utilization.

### 8.5. Batched/approximate matching cho visloc
Đề xuất:
- thay KDTree CPU bằng FAISS/GPU ANN nếu quy mô lớn
- hoặc downsample confidence-aware trước matching

Lợi ích:
- giảm độ trễ visloc.

### 8.6. Alignment theo cấp độ
Đề xuất:
- coarse-to-fine:
  1) optimize low-res/low-iter
  2) refine trên tập điểm confident
- bật early-stop theo delta loss.

Lợi ích:
- giảm thời gian tối ưu mà vẫn giữ chất lượng.

### 8.7. Instrumentation và observability
Đề xuất:
- thêm profiler hooks (data time, forward time, optimize time)
- JSON logs có trace id cho pair/scene

Lợi ích:
- dễ tìm bottleneck đúng nơi.

### 8.8. Đồng bộ convention tọa độ và pixel-center
Vấn đề:
- có nhiều chỗ convert cv2 <-> colmap (+/-0.5) thủ công.

Đề xuất:
- gom một module coordinate_conventions.py có test unit bắt buộc.

Lợi ích:
- giảm bug localization khó phát hiện.

---

## 9) Xếp hạng 1-10 theo từng phần (kèm lý do)

Thang điểm:
- 1-3: yếu
- 4-6: tạm được
- 7-8: tốt
- 9-10: rất tốt/xuất sắc

### 9.1. Core model architecture (AsymmetricCroCo3DStereo): 9/10
Lý do:
- Kiến trúc mạnh, đã chứng minh hiệu quả.
- Hỗ trợ multi-resolution và head linh hoạt (linear/DPT).
- Trừ 1 điểm vì phức tạp và khó tối ưu nếu không có GPU mạnh.

### 9.2. Inference pipeline: 8/10
Lý do:
- Rõ ràng, dễ gọi API.
- Xử lý được symmetrize và collate.
- Trừ điểm vì ép batch=1 khi multi-shape và chưa tối ưu throughput.

### 9.3. Pair graph construction: 8/10
Lý do:
- Đủ chế độ graph cho nhiều kịch bản.
- Đơn giản, dễ hiểu.
- Trừ điểm vì thiếu heuristic adaptive graph theo confidence/nội dung cạnh.

### 9.4. Global alignment (cloud_opt): 8/10
Lý do:
- Đầy đủ tính năng, có init MST, có pair viewer nhanh.
- Chất lượng tốt cho multi-view.
- Trừ điểm vì chi phí tính toán cao, dễ thành bottleneck online.

### 9.5. Training engine: 7/10
Lý do:
- Có DDP, AMP, grad accumulation, logging, checkpoint.
- Ổn định cho nghiên cứu.
- Trừ điểm vì dùng eval-string và cần bổ sung config hệ thống hóa.

### 9.6. Loss design: 8/10
Lý do:
- Regr3D + ConfLoss hợp lý cho bài toán pointmap.
- Có biến thể scale/shift invariance cho benchmark.
- Trừ điểm vì cần bổ sung robust losses theo outlier scene khắc nghiệt.

### 9.7. Dataset abstraction: 8/10
Lý do:
- Base class rõ, đa dạng adapter dataset.
- Có sampler theo aspect ratio.
- Trừ điểm vì __getitem__ còn làm nhiều việc nặng, dễ data-loader bottleneck.

### 9.8. Dataset preprocessing toolkit: 7/10
Lý do:
- Đầy đủ script cho nhiều dataset lớn.
- Làm rõ đường preprocess từ dữ liệu gốc.
- Trừ điểm vì convention chưa đồng nhất hoàn toàn, script dài và khó bảo trì.

### 9.9. Visual localization pipeline: 8/10
Lý do:
- End-to-end đầy đủ: retrieval pairs -> matching -> PnP -> metric.
- Nhiều backend PnP.
- Trừ điểm vì matching/IO có thể chậm ở quy mô lớn.

### 9.10. Visualization and demo UX: 8/10
Lý do:
- Demo Gradio dễ dùng, tùy chọn practical.
- Xuất GLB hữu ích cho trình diễn/kiểm tra.
- Trừ điểm vì hiện tại thiên về research demo hơn sản phẩm production.

### 9.11. Dockerization/deployment convenience: 7/10
Lý do:
- Có script docker CPU/CUDA, khởi động nhanh demo.
- Trừ điểm vì chưa có profile production (monitoring, healthcheck, scale-out).

### 9.12. Production readiness tổng thể: 6.5/10
Lý do:
- Rất mạnh cho research/prototyping.
- Cần thêm hardening: config typed, caching, profiling, service boundary, tests tọa độ.

---

## 10) Gợi ý tối ưu trực tiếp theo mục tiêu công việc

Nếu mục tiêu là:

1. Demo nhanh và ổn định
- Dùng PairViewer cho 1-2 ảnh.
- Giảm image size nếu cần latency.
- Tắt xuất mesh nếu chỉ cần điểm 3D.

2. Xử lý loạt (batch offline)
- Cố định 1-2 resolution để tránh batch_size=1.
- Chạy graph swin/logwin thay vì complete cho bộ ảnh dài.
- Pre-cache ảnh đã resize + metadata.

3. Localization quy mô lớn
- Cache map views + pts3d_rescaled.
- Hạn chế pnp_max_points với sampling theo confidence.
- Ưu tiên poselib/pycolmap tùy môi trường.

4. Fine-tune cho domain riêng
- Bắt đầu từ checkpoint 512 linear, sau đó 512 dpt.
- Kiểm soát chất preprocess (intrinsics/depth quality) là ưu tiên số 1.
- Theo dõi data-time/GPU-time song song.

---

## 11) Kết luận ngắn

DUSt3R là codebase rất mạnh cho tái dụng 3D và localization theo hướng pointmap, đặc biệt tốt cho nghiên cứu và PoC. Để ứng dụng thực tiễn quy mô lớn, cần ưu tiên 4 nhóm nâng cấp: 
- bỏ eval-string (registry typed),
- tối ưu dataloader/caching,
- giảm chi phí global alignment,
- chuẩn hóa convention tọa độ + bộ test hệ thống.