# Training trong DUSt3R: `dust3r/losses.py` và `dust3r/training.py`

Tài liệu này giải thích 2 script quan trọng cho huấn luyện DUSt3R:

- `dust3r/losses.py`: định nghĩa các **loss** để model học dự đoán pointmap 3D (và confidence) từ cặp ảnh.
- `dust3r/training.py`: “engine” huấn luyện: parse args → build dataset/dataloader → forward+loss → backprop → log/checkpoint → evaluation.

Bạn có thể xem hai file này như:

- `losses.py` = “định nghĩa đúng/sai” (objective).
- `training.py` = “cách chạy” (training loop & orchestration).

---

## 1) `dust3r/losses.py`

### Vai trò trong DUSt3R
DUSt3R không học trực tiếp pose như một scalar; nó học **dense 3D** (pointmap) và **confidence**. Vì thế loss phải:

1) So sánh pointmap dự đoán với ground truth pointmap theo đúng hệ quy chiếu.
2) Hỗ trợ các bất định phổ biến: scale ambiguity, shift depth, v.v.
3) Khuyến khích confidence học được phản ánh “độ tin cậy” của từng pixel.

File này thiết kế loss theo dạng composable (cộng, nhân hệ số), và hỗ trợ vừa trả loss scalar vừa trả loss per-pixel.

---

### 1.1. Các building blocks

#### `BaseCriterion` / `LLoss`
- `BaseCriterion` chỉ giữ `reduction`.
- `LLoss` là khung cho loss dạng Lp:
  - `forward(a,b)` kiểm tra shape, gọi `distance(a,b)`.
  - `reduction` có thể là `none` / `sum` / `mean`.

#### `L21Loss` (Euclidean)
- `distance = ||a-b||_2` theo trục cuối.
- `L21 = L21Loss()` được tạo sẵn.

Điểm đáng học:
- Tách `distance()` khỏi `forward()` giúp tạo nhiều biến thể loss mà vẫn dùng chung reduction.

---

### 1.2. Pattern “composable loss”

#### `Criterion`
- Wrapper quanh một `BaseCriterion`.
- Có `with_reduction(mode)` để ép toàn bộ chuỗi loss trả về per-pixel hoặc scalar.

#### `MultiLoss`
Cho phép viết loss kiểu:

- `loss = MyLoss1() + 0.1 * MyLoss2()`

Cách hoạt động:
- Mỗi `MultiLoss` có `_alpha` và `_loss2` (nối chuỗi).
- `forward()`:
  1) gọi `compute_loss()` của loss hiện tại.
  2) nhân `_alpha`.
  3) nếu có `_loss2` thì cộng thêm loss2.
  4) gom `details` để log.

Điểm đáng học:
- Đây là một cách rất gọn để cấu hình loss từ string `eval(...)` (được dùng trong `training.py`).

---

### 1.3. `Regr3D`: loss hồi quy 3D chính

#### Ý tưởng
`Regr3D` là loss “core”: đảm bảo pointmap 3D dự đoán đúng.

Đặc điểm:
- **Asymmetric**: view1 là anchor (chuẩn hoá/so sánh trong hệ của camera view1).

#### `get_all_pts3d(gt1, gt2, pred1, pred2, dist_clip=None)`
Các bước quan trọng:

1) Đưa GT về hệ camera1:
- `in_camera1 = inv(gt1['camera_pose'])`
- `gt_pts1 = geotrf(in_camera1, gt1['pts3d'])`
- `gt_pts2 = geotrf(in_camera1, gt2['pts3d'])`

2) Lấy valid masks:
- `valid1 = gt1['valid_mask']`, `valid2 = gt2['valid_mask']`
- Nếu có `dist_clip`: loại điểm quá xa.

3) Lấy predicted pts3d theo chuẩn output head:
- `pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False)`
- `pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True)`
  - Điểm quan trọng: `use_pose=True` cho nhánh view2 để đưa điểm về đúng frame so sánh.

4) Normalize pointcloud (tuỳ `norm_mode`):
- `normalize_pointcloud(pr_pts1, pr_pts2, norm_mode, valid1, valid2)`.
- Nếu `not gt_scale`: GT cũng bị normalize để đánh giá invariant theo mode.

#### `compute_loss(...)`
- Tính 2 loss:
  - `l1 = criterion(pred_pts1[mask1], gt_pts1[mask1])`
  - `l2 = criterion(pred_pts2[mask2], gt_pts2[mask2])`
- Trả về `Sum((l1,mask1),(l2,mask2))`.

Lưu ý quan trọng về `Sum(...)`:
- Nếu `l1`/`l2` là per-pixel (ndim > 0) thì `Sum` trả lại tuple để `ConfLoss` xử lý.
- Nếu là scalar thì cộng lại.

Điểm đáng học:
- Thiết kế loss sao cho cùng một code hỗ trợ cả “pixel-level loss” và “global loss”.

---

### 1.4. `ConfLoss`: học confidence để weight loss

#### Ý tưởng
`ConfLoss` giả định `pixel_loss` trả về loss per-pixel. Sau đó dùng confidence dự đoán để:

- Pixel “tự tin” → penalty thấp hơn (nhưng bị phạt log để tránh conf → 0).
- Pixel “không tự tin” → penalty cao hơn.

Công thức trong code:
- `conf_loss = loss * conf - alpha * log(conf)`

Cơ chế:
- `pixel_loss = pixel_loss.with_reduction('none')`.
- Lấy `pred1['conf'][msk1]`, `pred2['conf'][msk2]`.
- Average về scalar.

Điểm đáng học:
- Đây là phiên bản “heteroscedastic regression” phổ biến: model học uncertainty/confidence và dùng nó để weight residual.

---

### 1.5. Các biến thể invariant

Các biến thể này kế thừa `Regr3D` để xử lý các bất định:

- `Regr3D_ShiftInv`:
  - Trừ median depth chung (joint median) khỏi cả GT và pred, để invariant với shift theo trục z.

- `Regr3D_ScaleInv`:
  - Tính scale của scene (center+scale) cho GT và pred.
  - Nếu `gt_scale=True`: ép prediction theo scale GT.
  - Nếu không: normalize cả hai theo scale riêng → invariant với scale.

- `Regr3D_ScaleShiftInv`:
  - Kết hợp shift-inv và scale-inv.

Điểm đáng học:
- Multi-view reconstruction thường gặp scale/shift ambiguity; đưa invariant trực tiếp vào loss giúp ổn định training và metric.

---

## 2) `dust3r/training.py`

### Vai trò trong DUSt3R
`training.py` là entry-point huấn luyện. Nó làm các việc:

1) Parse cấu hình (model, dataset, criterion, lr, epochs...).
2) Khởi tạo môi trường distributed (nếu có).
3) Build dataset/dataloader.
4) Build model + criterion từ string `eval(...)`.
5) Huấn luyện theo epoch/iteration, log TensorBoard, save checkpoint.
6) Test/eval định kỳ và lưu best.

---

### 2.1. `get_args_parser()`
Các args đáng chú ý:

- `--model` default: `AsymmetricCroCo3DStereo(patch_embed_cls='ManyAR_PatchEmbed')`.
- `--train_criterion` default: `ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)`.
- `--amp`: bật Automatic Mixed Precision.
- `--batch_size`, `--accum_iter`: hỗ trợ gradient accumulation.
- `--blr` + rule scale theo effective batch size.
- `--eval_freq`, `--save_freq`, `--keep_freq`.

Điểm đáng học:
- Cấu hình model/loss bằng string giúp thử nghiệm cực nhanh, nhưng đòi hỏi bạn kiểm soát input (vì `eval`).

---

### 2.2. `train(args)` (orchestration)

#### Distributed + seed
- `misc.init_distributed_mode(args)`.
- Seed = `args.seed + rank`.

#### Dataset
- `build_dataset(args.train_dataset, ...)`.
- `data_loader_test` có thể là nhiều dataset (split bằng '+').

#### Model + criterion
- `model = eval(args.model)`.
- `train_criterion = eval(args.train_criterion).to(device)`.
- `test_criterion = eval(args.test_criterion or ...)`.

Lưu ý: file có một chỗ nhìn như typo:
- `test_criterion = eval(args.test_criterion or args.criterion)`
  - `args.criterion` không thấy được định nghĩa trong parser.
  - Ý định có thể là `args.train_criterion`.

(Ở đây mình chỉ chỉ ra để bạn biết khi chạy có thể gặp lỗi, chứ không tự sửa vì bạn đang yêu cầu tài liệu.)

#### Pretrained/resume
- Nếu `--pretrained` và không resume: load state dict `strict=False`.
- Auto-resume từ `output_dir/checkpoint-last.pth`.

#### Learning rate
- Effective batch size = `batch_size * accum_iter * world_size`.
- Nếu `args.lr` None → `args.lr = blr * eff_batch / 256`.

#### DDP
- Bọc `DistributedDataParallel(..., find_unused_parameters=True, static_graph=True)`.

#### Optimizer
- AdamW với param groups (tách weight decay cho bias/norm theo timm).

#### Logging + checkpoint
- TensorBoard `SummaryWriter`.
- Lưu `checkpoint-last`, `checkpoint-best`, `checkpoint-%d` theo keep_freq.

---

### 2.3. `train_one_epoch(...)`

Luồng chính mỗi iteration:
1) Set `model.train(True)`.
2) LR scheduler per-iteration:
   - nếu `data_iter_step % accum_iter == 0`: `misc.adjust_learning_rate(optimizer, epoch_f, args)`.

3) Forward + loss:
```python
loss_tuple = loss_of_one_batch(batch, model, criterion, device,
                              symmetrize_batch=True,
                              use_amp=args.amp, ret='loss')
loss, loss_details = loss_tuple
```
- `symmetrize_batch=True` rất quan trọng:
  - batch sẽ chứa cả (A,B) và (B,A) → giúp model và loss có tính đối xứng tốt hơn, và tận dụng cơ chế symmetrized encode.

4) Gradient accumulation:
- `loss /= accum_iter`.
- `loss_scaler(loss, optimizer, ..., update_grad=...)`.
- `optimizer.zero_grad()` khi đủ accum step.

5) Logging:
- `MetricLogger` update `loss` và các `loss_details` từ criterion.
- TensorBoard log theo `epoch_1000x` (điều chỉnh trục x để so sánh giữa batch size khác nhau).

Điểm đáng học:
- Kết hợp symmetrize batch + AMP + grad accumulation là công thức thực tế để train mô hình lớn trên VRAM giới hạn.

---

### 2.4. `test_one_epoch(...)`
- `model.eval()`.
- Chạy `loss_of_one_batch(... symmetrize_batch=True ...)` nhưng không backprop.
- `metric_logger` dùng window rất lớn để median ổn định.
- Trả về dict chứa cả `avg` và `med`.

---

## 2 script này đóng vai trò như nào trong DUSt3R?

- `losses.py` định nghĩa mục tiêu học: “3D đúng + confidence có ý nghĩa + invariant cần thiết”.
- `training.py` tổ chức toàn bộ quá trình huấn luyện và đánh giá, biến cấu hình string thành model/loss thực thi.

Nếu bạn muốn thay đổi hành vi học của DUSt3R:
- thay loss → chỉnh trong `losses.py` hoặc string `--train_criterion`.
- thay kiến trúc/model backbone/head → chỉnh string `--model`.
- thay schedule/optimizer/amp/accum → chỉnh trong `training.py`.

---

## Điều hay nhất bạn có thể học từ 2 script này

1) Loss design theo đúng bản chất bài toán 3D
- So sánh pointmap trong đúng frame (anchor view1) + hỗ trợ normalize/invariant.

2) Loss composability (cực hợp nghiên cứu)
- Có thể ghép loss bằng `+` và nhân hệ số bằng `*`, log chi tiết từng thành phần.

3) Học confidence đúng cách
- `loss * conf - alpha * log(conf)` là pattern quan trọng để confidence không bị collapse.

4) Training loop “production-grade”
- AMP, grad accumulation, DDP, auto-resume, checkpoint best/last, multi-testset eval.

5) Symmetrize batch như một kỹ thuật chất lượng
- Đưa tính đối xứng vào dữ liệu/loss giúp mô hình pairwise ổn định hơn.
