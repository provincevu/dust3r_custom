![demo](assets/dust3r.jpg)

Phiên bản chính thức của `DUSt3R: Geometric 3D Vision Made Easy`  
[[Project page](https://dust3r.europe.naverlabs.com/)], [[DUSt3R arxiv](https://arxiv.org/abs/2312.14132)]  

> Bạn cũng nên xem các dự án liên quan:  
> [Grounding Image Matching in 3D with MASt3R](https://github.com/naver/mast3r): DUSt3R with a local feature head, metric pointmaps, and a more scalable global alignment!  
> [Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors](https://github.com/naver/pow3r): DUSt3R with known depth / focal length / poses.  
> [MUSt3R: Multi-view Network for Stereo 3D Reconstruction](https://github.com/naver/must3r): Multi-view predictions (RGB SLAM/SfM) without any global alignment.    

![Example of reconstruction from two images](assets/pipeline1.jpg)

![High level overview of DUSt3R capabilities](assets/dust3r_archi.jpg)

```bibtex
@inproceedings{dust3r_cvpr24,
      title={DUSt3R: Geometric 3D Vision Made Easy}, 
      author={Shuzhe Wang and Vincent Leroy and Yohann Cabon and Boris Chidlovskii and Jerome Revaud},
      booktitle = {CVPR},
      year = {2024}
}

@misc{dust3r_arxiv23,
      title={DUSt3R: Geometric 3D Vision Made Easy}, 
      author={Shuzhe Wang and Vincent Leroy and Yohann Cabon and Boris Chidlovskii and Jerome Revaud},
      year={2023},
      eprint={2312.14132},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Mục lục

- [Mục lục](#mục-lục)
- [Giấy phép](#giấy-phép)
- [Bắt đầu](#bắt-đầu)
  - [Cài đặt](#cài-đặt)
  - [Checkpoint](#checkpoint)
  - [Demo tương tác](#demo-tương-tác)
  - [Demo tương tác với Docker](#demo-tương-tác-với-docker)
- [Cách dùng](#cách-dùng)
- [Huấn luyện](#huấn-luyện)
  - [Bộ dữ liệu](#bộ-dữ-liệu)
  - [Demo](#demo)
  - [Siêu tham số của chúng tôi](#siêu-tham-số-của-chúng-tôi)

## Giấy phép

Mã nguồn được phân phối theo giấy phép CC BY-NC-SA 4.0.
Xem [LICENSE](LICENSE) để biết thêm chi tiết.

```python
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
```

## Bắt đầu

### Cài đặt

1. Clone DUSt3R.
```bash
git clone --recursive https://github.com/naver/dust3r
cd dust3r
# if you have already cloned dust3r:
# git submodule update --init --recursive
```

2. Tạo môi trường, ở đây ví dụ dùng conda.
```bash
conda create -n dust3r python=3.11 cmake=3.14.0
conda activate dust3r 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# Tùy chọn: bạn cũng có thể cài thêm các gói để:
# - hỗ trợ ảnh HEIC
# - thêm pyrender, dùng để render depth map trong một số bước tiền xử lý bộ dữ liệu
# - cài các gói cần cho visloc.py
pip install -r requirements_optional.txt
```

3. Tùy chọn: biên dịch các CUDA kernel cho RoPE (như trong CroCo v2).
```bash
# DUSt3R dùng RoPE positional embeddings; bạn có thể biên dịch CUDA kernel để chạy nhanh hơn.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

### Checkpoint

Bạn có thể lấy checkpoint theo 2 cách:

1) Dùng tích hợp `huggingface_hub`: model sẽ được tải tự động.

2) Hoặc dùng các model pre-trained mà chúng tôi cung cấp:

| Modelname   | Training resolutions | Head | Encoder | Decoder |
|-------------|----------------------|------|---------|---------|
| [`DUSt3R_ViTLarge_BaseDecoder_224_linear.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth) | 224x224 | Linear | ViT-L | ViT-B |
| [`DUSt3R_ViTLarge_BaseDecoder_512_linear.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth)   | 512x384, 512x336, 512x288, 512x256, 512x160 | Linear | ViT-L | ViT-B |
| [`DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth) | 512x384, 512x336, 512x288, 512x256, 512x160 | DPT | ViT-L | ViT-B |

Bạn có thể xem siêu tham số đã dùng để train các model này trong phần [Siêu tham số của chúng tôi](#siêu-tham-số-của-chúng-tôi).

Để tải một model cụ thể, ví dụ `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`:
```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
```

Với các checkpoint, hãy chắc chắn bạn đồng ý với giấy phép của tất cả bộ dữ liệu train công khai và các checkpoint nền chúng tôi đã dùng, bên cạnh CC-BY-NC-SA 4.0. Xem lại phần [Siêu tham số của chúng tôi](#siêu-tham-số-của-chúng-tôi) để biết thêm chi tiết.

### Demo tương tác

Trong demo này, bạn có thể chạy DUSt3R trên máy của mình để tái tạo một cảnh.
Trước tiên hãy chọn các ảnh thể hiện cùng một scene.

Bạn có thể chỉnh lịch tối ưu global alignment và số vòng lặp.

> [!NOTE]
> Nếu bạn chỉ chọn 1 hoặc 2 ảnh, bước global alignment sẽ bị bỏ qua (mode=GlobalAlignerMode.PairViewer)

Bấm "Run" và chờ.
Khi global alignment kết thúc, kết quả reconstruction sẽ xuất hiện.
Dùng slider "min_conf_thr" để hiện hoặc ẩn các vùng có độ tin cậy thấp.

```bash
python3 demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt

# Use --weights to load a checkpoint from a local file, eg --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
# Use --image_size to select the correct resolution for the selected checkpoint. 512 (default) or 224
# Use --local_network to make it accessible on the local network, or --server_name to specify the url manually
# Use --server_port to change the port, by default it will search for an available port starting at 7860
# Use --device to use a different device, by default it's "cuda"
```

### Demo tương tác với Docker

Để chạy DUSt3R bằng Docker, bao gồm hỗ trợ NVIDIA CUDA, làm theo các bước sau:

1. **Cài Docker**: nếu chưa có, tải và cài `docker` và `docker compose` từ [trang Docker](https://www.docker.com/get-started).

2. **Cài NVIDIA Docker Toolkit**: để dùng GPU, cài NVIDIA Docker toolkit từ [trang Nvidia](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

3. **Build image Docker và chạy**: `cd` vào thư mục `./docker` và chạy các lệnh sau: 

```bash
cd docker
bash run.sh --with-cuda --model_name="DUSt3R_ViTLarge_BaseDecoder_512_dpt"
```

Hoặc nếu bạn muốn chạy demo không dùng CUDA, chạy lệnh sau:

```bash 
cd docker
bash run.sh --model_name="DUSt3R_ViTLarge_BaseDecoder_512_dpt"
```

Mặc định, `demo.py` được khởi chạy với tùy chọn `--local_network`.  
Truy cập `http://localhost:7860/` để mở giao diện web (hoặc thay `localhost` bằng tên máy để truy cập qua mạng).  

`run.sh` sẽ chạy docker-compose bằng file cấu hình [docker-compose-cuda.yml](docker/docker-compose-cuda.yml) hoặc [docker-compose-cpu.ym](docker/docker-compose-cpu.yml), sau đó khởi động demo bằng [entrypoint.sh](docker/files/entrypoint.sh).


![demo](assets/demo.jpg)

## Cách dùng

```python
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    images = load_images(['croco/assets/Chateau1.png', 'croco/assets/Chateau2.png'], size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # ở bước này, bạn đã có dự đoán thô của DUSt3R
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    # ở đây view1, pred1, view2, pred2 là dict chứa list có độ dài 2
    #  -> vì ta symmetrize nên có cặp (im1, im2) và (im2, im1)
    # trong mỗi view có:
    # id ảnh dạng số: view1['idx'] và view2['idx']
    # ảnh: view1['img'] và view2['img']
    # kích thước ảnh: view1['true_shape'] và view2['true_shape']
    # chuỗi instance do dataloader xuất ra: view1['instance'] và view2['instance']
    # pred1 và pred2 chứa confidence: pred1['conf'] và pred2['conf']
    # pred1 chứa điểm 3D cho view1['img'] trong hệ tọa độ của view1['img']: pred1['pts3d']
    # pred2 chứa điểm 3D cho view2['img'] trong hệ tọa độ của view1['img']: pred2['pts3d_in_other_view']

    # tiếp theo ta dùng global_aligner để căn chỉnh dự đoán
    # tùy bài toán, bạn có thể chỉ cần output thô và không cần bước này
    # nếu chỉ có 2 ảnh đầu vào, bạn có thể dùng GlobalAlignerMode.PairViewer: nó chỉ chuyển đổi output
    # nếu dùng GlobalAlignerMode.PairViewer, không cần chạy compute_global_alignment
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # lấy các giá trị hữu ích từ scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # trực quan hóa reconstruction
    scene.show()

    # tìm match 2D-2D giữa hai ảnh
    from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
    pts2d_list, pts3d_list = [], []
    for i in range(2):
        conf_i = confidence_masks[i].cpu().numpy()
        pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    print(f'found {num_matches} matches')
    matches_im1 = pts2d_list[1][reciprocal_in_P2]
    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    # trực quan hóa vài match
    import numpy as np
    from matplotlib import pyplot as pl
    n_viz = 10
    match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)

```
![matching example on croco pair](assets/matching.jpg)

## Huấn luyện

Phần này trình bày một demo ngắn để bắt đầu huấn luyện DUSt3R.

### Bộ dữ liệu
Hiện tại, chúng tôi đã hỗ trợ các bộ dữ liệu train sau:
  - [CO3Dv2](https://github.com/facebookresearch/co3d) - [Creative Commons Attribution-NonCommercial 4.0 International](https://github.com/facebookresearch/co3d/blob/main/LICENSE)
  - [ARKitScenes](https://github.com/apple/ARKitScenes) - [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://github.com/apple/ARKitScenes/tree/main?tab=readme-ov-file#license)
  - [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) - [non-commercial research and educational purposes](https://kaldir.vc.in.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf)
  - [BlendedMVS](https://github.com/YoYo000/BlendedMVS) - [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
  - [WayMo Open dataset](https://github.com/waymo-research/waymo-open-dataset) - [Non-Commercial Use](https://waymo.com/open/terms/)
  - [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md)
  - [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/)
  - [StaticThings3D](https://github.com/lmb-freiburg/robustmvd/blob/master/rmvd/data/README.md#staticthings3d)
  - [WildRGB-D](https://github.com/wildrgbd/wildrgbd/)

Với mỗi bộ dữ liệu, chúng tôi cung cấp script tiền xử lý trong thư mục `datasets_preprocess` và một gói chứa danh sách cặp ảnh khi cần.
Bạn cần tự tải dữ liệu từ nguồn chính thức, chấp nhận giấy phép của họ, tải danh sách cặp ảnh của chúng tôi và chạy script tiền xử lý.

Links:  
  
[ARKitScenes pairs](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/arkitscenes_pairs.zip)  
[ScanNet++ v1 pairs](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/scannetpp_pairs.zip)  
[ScanNet++ v2 pairs](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/scannetpp_v2_pairs.zip)  
[BlendedMVS pairs](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/blendedmvs_pairs.npy)  
[WayMo Open dataset pairs](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/waymo_pairs.npz)  
[Habitat metadata](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/habitat_5views_v1_512x512_metadata.tar.gz)  
[MegaDepth pairs](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/megadepth_pairs.npz)  
[StaticThings3D pairs](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/staticthings_pairs.npy)  

> [!NOTE]
> Chúng không hoàn toàn giống 100% với dữ liệu đã dùng để train DUSt3R, nhưng đủ gần.

### Demo
Trong demo huấn luyện này, chúng ta sẽ tải và chuẩn bị một phần nhỏ của [CO3Dv2](https://github.com/facebookresearch/co3d) - [Creative Commons Attribution-NonCommercial 4.0 International](https://github.com/facebookresearch/co3d/blob/main/LICENSE) rồi chạy mã train trên đó.
Model demo sẽ chỉ train vài epoch trên một dataset rất nhỏ.
Kết quả sẽ không tốt.

```bash
# download and prepare the co3d subset
mkdir -p data/co3d_subset
cd data/co3d_subset
git clone https://github.com/facebookresearch/co3d
cd co3d
python3 ./co3d/download_dataset.py --download_folder ../ --single_sequence_subset
rm ../*.zip
cd ../../..

python3 datasets_preprocess/preprocess_co3d.py --co3d_dir data/co3d_subset --output_dir data/co3d_subset_processed  --single_sequence_subset

# download the pretrained croco v2 checkpoint
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTLarge_BaseDecoder.pth -P checkpoints/

# quá trình train DUSt3R được thực hiện trong 3 bước.
# trong ví dụ này ta train ít epoch hơn; để xem siêu tham số thật đã dùng trong paper, xem phần tiếp theo: "Siêu tham số của chúng tôi"
# bước 1 - train DUSt3R ở độ phân giải 224
torchrun --nproc_per_node=4 train.py \
    --train_dataset "1000 @ Co3d(split='train', ROOT='data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter)" \
    --test_dataset "100 @ Co3d(split='test', ROOT='data/co3d_subset_processed', resolution=224, seed=777)" \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --pretrained "checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 10 --batch_size 16 --accum_iter 1 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 \
    --output_dir "checkpoints/dust3r_demo_224"	  

# bước 2 - train DUSt3R ở độ phân giải 512
torchrun --nproc_per_node=4 train.py \
    --train_dataset "1000 @ Co3d(split='train', ROOT='data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" \
    --test_dataset "100 @ Co3d(split='test', ROOT='data/co3d_subset_processed', resolution=(512,384), seed=777)" \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --pretrained "checkpoints/dust3r_demo_224/checkpoint-best.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 10 --batch_size 4 --accum_iter 4 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 \
    --output_dir "checkpoints/dust3r_demo_512"

# bước 3 - train DUSt3R ở độ phân giải 512 với dpt
torchrun --nproc_per_node=4 train.py \
    --train_dataset "1000 @ Co3d(split='train', ROOT='data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" \
    --test_dataset "100 @ Co3d(split='test', ROOT='data/co3d_subset_processed', resolution=(512,384), seed=777)" \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --pretrained "checkpoints/dust3r_demo_512/checkpoint-best.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 10 --batch_size 2 --accum_iter 8 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --disable_cudnn_benchmark \
    --output_dir "checkpoints/dust3r_demo_512dpt"

```

### Siêu tham số của chúng tôi

Đây là các lệnh chúng tôi đã dùng để train các model:

```bash
# NOTE: ROOT path omitted for datasets
# 224 linear
torchrun --nproc_per_node 8 train.py \
    --train_dataset=" + 100_000 @ Habitat(1_000_000, split='train', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ BlendedMVS(split='train', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ MegaDepth(split='train', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ ARKitScenes(aug_crop=256, resolution=224, transform=ColorJitter) + 100_000 @ Co3d(split='train', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ StaticThings3D(aug_crop=256, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ ScanNetpp(split='train', aug_crop=256, resolution=224, transform=ColorJitter) + 100_000 @ InternalUnreleasedDataset(aug_crop=128, resolution=224, transform=ColorJitter) " \
    --test_dataset=" Habitat(1_000, split='val', resolution=224, seed=777) + 1_000 @ BlendedMVS(split='val', resolution=224, seed=777) + 1_000 @ MegaDepth(split='val', resolution=224, seed=777) + 1_000 @ Co3d(split='test', mask_bg='rand', resolution=224, seed=777) " \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=10 --epochs=100 --batch_size=16 --accum_iter=1 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 \
    --output_dir="checkpoints/dust3r_224"

# 512 linear
torchrun --nproc_per_node 8 train.py \
    --train_dataset=" + 10_000 @ Habitat(1_000_000, split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ BlendedMVS(split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ MegaDepth(split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ ARKitScenes(aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ Co3d(split='train', aug_crop=16, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ StaticThings3D(aug_crop=256, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ ScanNetpp(split='train', aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ InternalUnreleasedDataset(aug_crop=128, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) " \
    --test_dataset=" Habitat(1_000, split='val', resolution=(512,384), seed=777) + 1_000 @ BlendedMVS(split='val', resolution=(512,384), seed=777) + 1_000 @ MegaDepth(split='val', resolution=(512,336), seed=777) + 1_000 @ Co3d(split='test', resolution=(512,384), seed=777) " \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="checkpoints/dust3r_224/checkpoint-best.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=20 --epochs=100 --batch_size=4 --accum_iter=2 \
    --save_freq=10 --keep_freq=10 --eval_freq=1 --print_freq=10 \
    --output_dir="checkpoints/dust3r_512"

# 512 dpt
torchrun --nproc_per_node 8 train.py \
    --train_dataset=" + 10_000 @ Habitat(1_000_000, split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ BlendedMVS(split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ MegaDepth(split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ ARKitScenes(aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ Co3d(split='train', aug_crop=16, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ StaticThings3D(aug_crop=256, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ ScanNetpp(split='train', aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ InternalUnreleasedDataset(aug_crop=128, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) " \
    --test_dataset=" Habitat(1_000, split='val', resolution=(512,384), seed=777) + 1_000 @ BlendedMVS(split='val', resolution=(512,384), seed=777) + 1_000 @ MegaDepth(split='val', resolution=(512,336), seed=777) + 1_000 @ Co3d(split='test', resolution=(512,384), seed=777) " \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="checkpoints/dust3r_512/checkpoint-best.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=15 --epochs=90 --batch_size=4 --accum_iter=2 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir="checkpoints/dust3r_512dpt"

```
