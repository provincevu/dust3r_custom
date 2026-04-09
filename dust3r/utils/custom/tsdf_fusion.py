from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TSDFFusionResult:
    points: np.ndarray
    colors: Optional[np.ndarray]
    n_integrated_views: int


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def _focal_to_fx_fy(focal_i) -> tuple[float, float]:
    f = _to_numpy(focal_i).reshape(-1)
    if len(f) == 0:
        return 1.0, 1.0
    if len(f) == 1:
        fv = float(f[0])
        return fv, fv
    return float(f[0]), float(f[-1])


def tsdf_fuse_views(
    depthmaps,
    im_poses,
    focals,
    *,
    masks=None,
    images=None,
    enabled: bool = True,
    voxel_length: float = 0.0002,
    sdf_trunc: float = 0.0008,
    depth_trunc: float = 1000.0,
    strict: bool = False,
) -> TSDFFusionResult:
    """Fuse multi-view depth into a single point cloud with Open3D TSDF.

    Inputs are meant to come directly from:
    - scene.get_depthmaps()
    - scene.get_im_poses()   (cam-to-world)
    - scene.get_focals()
    - scene.get_masks()      (optional)
    - scene.imgs             (optional colors)
    """

    if not enabled:
        return TSDFFusionResult(
            points=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            n_integrated_views=0,
        )

    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:  # pragma: no cover
        if strict:
            raise ImportError("tsdf_fuse_views requires open3d. Install with: pip install open3d") from exc
        return TSDFFusionResult(
            points=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            n_integrated_views=0,
        )

    depth_list = list(depthmaps)
    pose_list = list(im_poses)
    focal_list = list(focals)
    if not (len(depth_list) == len(pose_list) == len(focal_list)):
        raise ValueError("depthmaps, im_poses, focals must have same length")

    if masks is not None:
        mask_list = list(masks)
        if len(mask_list) != len(depth_list):
            raise ValueError("masks must have same length as depthmaps")
    else:
        mask_list = [None] * len(depth_list)

    if images is not None:
        img_list = list(images)
        if len(img_list) != len(depth_list):
            raise ValueError("images must have same length as depthmaps")
    else:
        img_list = [None] * len(depth_list)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    integrated = 0
    for depth_i, pose_i, focal_i, mask_i, img_i in zip(depth_list, pose_list, focal_list, mask_list, img_list):
        depth = _to_numpy(depth_i).astype(np.float32)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim != 2:
            continue

        h, w = depth.shape
        valid = np.isfinite(depth) & (depth > 0)
        if mask_i is not None:
            m = _to_numpy(mask_i).astype(bool)
            if m.shape == depth.shape:
                valid &= m
        if not np.any(valid):
            continue

        depth_clean = depth.copy()
        depth_clean[~valid] = 0.0
        depth_clean = np.require(depth_clean, dtype=np.float32, requirements=["C"])

        if img_i is None:
            color = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            color = _to_numpy(img_i)
            if color.ndim == 4 and color.shape[0] == 1:
                color = color[0]
            if color.ndim == 3 and color.shape == (3, h, w):
                color = np.transpose(color, (1, 2, 0))
            if color.ndim == 2 and color.shape == (h, w):
                color = np.repeat(color[..., None], 3, axis=2)
            if color.ndim != 3 or color.shape[0] != h or color.shape[1] != w:
                continue
            if color.shape[2] == 1:
                color = np.repeat(color, 3, axis=2)
            elif color.shape[2] >= 3:
                color = color[:, :, :3]
            else:
                continue
            if color.dtype != np.uint8:
                color = np.clip(color, 0.0, 1.0)
                color = (color * 255.0).astype(np.uint8)
            color = np.require(color, dtype=np.uint8, requirements=["C"])

        fx, fy = _focal_to_fx_fy(focal_i)
        cx = (w - 1.0) * 0.5
        cy = (h - 1.0) * 0.5

        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        cam2world = _to_numpy(pose_i).astype(np.float64)
        if cam2world.shape != (4, 4):
            continue
        world2cam = np.linalg.inv(cam2world)

        try:
            depth_img = o3d.geometry.Image(depth_clean)
            color_img = o3d.geometry.Image(color)
        except Exception:
            # Skip malformed view buffers rather than aborting the full fusion pass.
            continue
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img,
            depth_img,
            depth_scale=1.0,
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )

        volume.integrate(rgbd, intrinsic, world2cam)
        integrated += 1

    pcd = volume.extract_point_cloud()
    points = np.asarray(pcd.points, dtype=np.float32)
    if len(points) == 0:
        return TSDFFusionResult(
            points=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            n_integrated_views=integrated,
        )

    colors = np.asarray(pcd.colors, dtype=np.float32)
    if colors.shape[0] != points.shape[0]:
        colors = np.zeros((points.shape[0], 3), dtype=np.float32)

    return TSDFFusionResult(points=points, colors=colors, n_integrated_views=integrated)
