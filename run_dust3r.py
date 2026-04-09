import torch
import numpy as np
import cv2
from pathlib import Path
import argparse
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.custom import remove_ground
from dust3r.utils.custom import remove_outlier_cc
from dust3r.utils.custom import tsdf_fuse_views
from dust3r.utils.custom import view_consistent_merge
# from dust3r.utils.custom import statistic_plane
import open3d as o3d
import matplotlib.pyplot as plt
from IPython.display import display, Image as IPImage

def main():
    parser = argparse.ArgumentParser(description="Run DUSt3R reconstruction and export point cloud")
    parser.add_argument("--model-path", default="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    parser.add_argument("--image-dir", default="images/tuong")
    parser.add_argument("--output-dir", default="output/tuong")
    parser.add_argument("--image-size", type=int, default=512, choices=[224, 512])
    parser.add_argument("--niter", type=int, default=200)
    parser.add_argument("--schedule", default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--remove-ground", action="store_true", help="Remove dominant ground plane before export")
    parser.add_argument("--ground-y-preference", default="low", choices=["low", "high"],
                        help="Prefer low-Y or high-Y horizontal plane when selecting ground candidate")
    parser.add_argument("--ground-coplanar-angle-tol-deg", type=float, default=6.0,
                        help="Angular tolerance (deg) to also remove near-coplanar ground fragments")
    parser.add_argument("--ground-coplanar-offset-tol", type=float, default=-1.0,
                        help="Offset tolerance in world units for near-coplanar removal (-1 means auto)")
    parser.add_argument("--remove-outlier-cc", action="store_true", help="Remove outlier clusters via connected components (octree=8, min=20, keep largest)")
    parser.add_argument("--tsdf-fusion", action="store_true", help="Fuse depth maps per view with Open3D TSDF before post-processing")
    parser.add_argument("--view-consistent-merge", action="store_true", help="Merge near-duplicate points from multi-view into one layer")
    parser.add_argument("--vcm-voxel-size", type=float, default=0.005, help="Voxel size for view-consistent merge")
    # parser.add_argument("--statistic-plane", action="store_true", help="Flatten planar surfaces using Open3D RSPD patches")
    args = parser.parse_args()

    model_path = args.model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = args.image_dir
    output_dir = args.output_dir
    
    # Tạo thư mục output
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Load model
    print(f"Loading model from {model_path}...")
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
    print(f"Model loaded on {device}")
    
    # 2. Load images
    print(f"Loading images from {image_dir}...")
    image_paths = sorted(Path(image_dir).glob("*.jpg")) + sorted(Path(image_dir).glob("*.png"))
    
    if len(image_paths) < 2:
        print("Error: Need at least 2 images for reconstruction")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Load images với kích thước 512 (có thể điều chỉnh)
    images = load_images([str(p) for p in image_paths], size=args.image_size)
    
    # 3. Tạo image pairs
    # 'complete': tạo pairs giữa tất cả các ảnh (tốn bộ nhớ nhất)
    # 'sequential': chỉ tạo pairs giữa các ảnh liên tiếp
    # 'mst': Minimum Spanning Tree, tối ưu cho nhiều ảnh
    if len(image_paths) > 15:
        scene_graph = 'mst'  # Tiết kiệm bộ nhớ cho nhiều ảnh
    else:
        scene_graph = 'complete'
    
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    print(f"Created {len(pairs)} image pairs")
    
    # 4. Inference
    print("Running inference...")
    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=1)
    
    # 5. Global alignment
    print("Running global alignment...")
    # Chọn mode dựa trên số lượng ảnh
    if len(image_paths) == 2:
        mode = GlobalAlignerMode.PairViewer
    else:
        mode = GlobalAlignerMode.PointCloudOptimizer
    
    scene = global_aligner(output, device=device, mode=mode)
    
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init="mst", niter=args.niter, schedule=args.schedule, lr=args.lr)
        print(f"Alignment loss: {loss:.4f}")
    
    # 6. Lấy kết quả
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    depthmaps = scene.get_depthmaps()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()
    
    print(f"Reconstructed {len(pts3d)} views")
    for i, pts in enumerate(pts3d):
        n_points = pts.shape[0] * pts.shape[1] if len(pts.shape) > 2 else pts.shape[0]
        print(f"View {i}: {n_points} points")
    
    # 7. Merge tất cả point clouds thành một
    all_points = []
    all_colors = []
    
    for i, (pts, conf, img) in enumerate(zip(pts3d, confidence_masks, imgs)):
        # Lấy các điểm có confidence cao
        mask = conf.cpu().numpy() > 0.5
        points = pts.detach().cpu().numpy()[mask]
        
        # Lấy màu tương ứng từ ảnh gốc
        # img shape: (H, W, 3) với giá trị [0, 1]
        colors = img[mask]
        
        all_points.append(points)
        all_colors.append(colors)
    
    # Ghép tất cả
    if len(all_points) > 0:
        if args.tsdf_fusion:
            tsdf_res = tsdf_fuse_views(
                depthmaps,
                poses,
                focals,
                masks=confidence_masks,
                images=imgs,
                enabled=True,
                strict=False,
            )
            if len(tsdf_res.points) > 0:
                merged_points = tsdf_res.points
                merged_colors = tsdf_res.colors if tsdf_res.colors is not None else np.zeros((len(merged_points), 3), dtype=np.float32)
                print(f"TSDF fused views: {tsdf_res.n_integrated_views}")
            else:
                merged_points = np.vstack(all_points)
                merged_colors = np.vstack(all_colors)
                print("TSDF fusion returned empty cloud, fallback to direct merge")
        else:
            merged_points = np.vstack(all_points)
            merged_colors = np.vstack(all_colors)

        if args.view_consistent_merge:
            vcm_res = view_consistent_merge(
                merged_points,
                merged_colors,
                enabled=True,
                voxel_size=float(args.vcm_voxel_size),
            )
            merged_points = vcm_res.points
            if vcm_res.colors is not None:
                merged_colors = vcm_res.colors
            print(f"View-consistent merge: {vcm_res.n_input} -> {vcm_res.n_output} points")

        if args.remove_ground:
            coplanar_offset_tol = None if float(args.ground_coplanar_offset_tol) < 0 else float(args.ground_coplanar_offset_tol)
            res = remove_ground(
                merged_points,
                merged_colors,
                enabled=True,
                y_preference=str(args.ground_y_preference),
                coplanar_angle_tol_deg=float(args.ground_coplanar_angle_tol_deg),
                coplanar_offset_tol=coplanar_offset_tol,
                strict=False,
            )
            merged_points = res.points
            if res.colors is not None:
                merged_colors = res.colors

        if args.remove_outlier_cc:
            res = remove_outlier_cc(
                merged_points,
                merged_colors,
                enabled=True,
                octree_level=8,
                min_points_per_component=20,
                keep_largest_only=True,
            )
            merged_points = res.points
            if res.colors is not None:
                merged_colors = res.colors

        # statistic_plane is temporarily hidden/disabled.
        # To re-enable, uncomment import, CLI argument and block below.
        # if args.statistic_plane:
        #     res = statistic_plane(merged_points, merged_colors, enabled=True, strict=False)
        #     merged_points = res.points
        #     if res.colors is not None:
        #         merged_colors = res.colors
        
        print(f"Total points after merging: {len(merged_points)}")
        
        # 8. Lưu kết quả
        # Lưu dạng PLY (có màu)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(merged_points)
        pc.colors = o3d.utility.Vector3dVector(merged_colors)
        o3d.io.write_point_cloud(f"{output_dir}/reconstruction.ply", pc)
        print(f"Saved point cloud to {output_dir}/reconstruction.ply")
        
        # Lưu dạng NPZ (để dùng sau)
        np.savez(f"{output_dir}/reconstruction.npz", points=merged_points, colors=merged_colors)
        
        # Lưu camera poses
        if len(poses) > 0:
            np.save(f"{output_dir}/camera_poses.npy", poses.detach().cpu().numpy())
            np.save(f"{output_dir}/focals.npy", focals.detach().cpu().numpy())
        
        print("Done!")
    else:
        print("No valid points found!")

if __name__ == "__main__":
    main()