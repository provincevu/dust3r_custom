from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class StatisticPlaneResult:
    points: np.ndarray
    colors: Optional[np.ndarray]
    affected_mask: np.ndarray
    n_patches: int


def _unit(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(vec)
    if n < eps:
        return vec * 0.0
    return vec / n


def statistic_plane(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    *,
    enabled: bool = True,
    # RSPD / planar patch detection params
    normal_variance_threshold_deg: float = 60.0,
    coplanarity_deg: float = 75.0,
    outlier_ratio: float = 0.75,
    min_plane_edge_length: float = 0.0,
    min_num_points: int = 100,
    # Normal estimation params
    normal_radius: Optional[float] = None,
    normal_max_nn: int = 50,
    # Flattening control
    flatten_distance_threshold: Optional[float] = None,
    normal_alignment_threshold: float = 0.92,
    robust_sigma_k: float = 2.5,
    min_flatten_distance: float = 1e-3,
    strict: bool = False,
) -> StatisticPlaneResult:
    """Flatten planar surfaces using Open3D RSPD planar patches.

    This function uses Open3D's robust planar patch detection
    (`PointCloud.detect_planar_patches`) and projects points close to each
    detected plane onto that plane.

    Parameters
    - points: (N, 3) array
    - colors: optional (N, 3) array
    - enabled: switch for no-op behavior
    - normal_variance_threshold_deg, coplanarity_deg, outlier_ratio,
      min_plane_edge_length, min_num_points: passed to detect_planar_patches
    - normal_radius, normal_max_nn: normal estimation params before detection
        - flatten_distance_threshold: max signed distance to plane to be snapped.
            If None, defaults to a conservative local scale.
        - normal_alignment_threshold: keep only points whose estimated normal is
            aligned with the patch normal (or opposite direction).
        - robust_sigma_k: adaptive threshold multiplier from MAD of signed distances
            inside each candidate patch.
        - min_flatten_distance: lower bound of local snapping threshold.
    - strict: if True and open3d unavailable, raise ImportError

    Returns
    - StatisticPlaneResult(points, colors, affected_mask, n_patches)
    """

    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3)")

    if colors is not None:
        cols = np.asarray(colors)
        if cols.shape[0] != pts.shape[0] or cols.shape[1] != 3:
            raise ValueError("colors must be (N, 3) and match points")
    else:
        cols = None

    if not enabled or len(pts) == 0:
        return StatisticPlaneResult(
            points=pts,
            colors=cols,
            affected_mask=np.zeros((len(pts),), dtype=bool),
            n_patches=0,
        )

    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:  # pragma: no cover
        if strict:
            raise ImportError(
                "statistic_plane(enabled=True) requires open3d. Install with: pip install open3d"
            ) from exc
        return StatisticPlaneResult(
            points=pts,
            colors=cols,
            affected_mask=np.zeros((len(pts),), dtype=bool),
            n_patches=0,
        )

    pts64 = np.asarray(pts, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts64)

    # Estimate normals required by planar patch detection.
    bbox = np.ptp(pts64, axis=0)
    diag = float(np.linalg.norm(bbox))
    est_radius = max(diag * 0.01, 1e-6)
    radius = float(normal_radius) if normal_radius is not None else est_radius
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=int(normal_max_nn))
    )

    patches = pcd.detect_planar_patches(
        normal_variance_threshold_deg=float(normal_variance_threshold_deg),
        coplanarity_deg=float(coplanarity_deg),
        outlier_ratio=float(outlier_ratio),
        min_plane_edge_length=float(diag * 0.02), #float(min_plane_edge_length)
        min_num_points=int(min_num_points),
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=int(normal_max_nn)),
    )

    if not patches:
        return StatisticPlaneResult(
            points=pts,
            colors=cols,
            affected_mask=np.zeros((len(pts),), dtype=bool),
            n_patches=0,
        )

    pts_new = pts64.copy()
    affected = np.zeros((len(pts64),), dtype=bool)

    if flatten_distance_threshold is None:
        # Conservative default to avoid flattening small protrusions into a large plane.
        flatten_thr_global = max(radius * 0.35, 1e-6)
    else:
        flatten_thr_global = float(flatten_distance_threshold)

    point_normals = np.asarray(pcd.normals, dtype=np.float64)
    point_normals /= np.maximum(np.linalg.norm(point_normals, axis=1, keepdims=True), 1e-12)

    # Resolve overlaps by taking the smallest displacement proposal per point.
    best_mag = np.full((len(pts64),), np.inf, dtype=np.float64)
    best_disp = np.zeros_like(pts64)

    for obox in patches:
        # Candidate points inside patch box.
        idx = np.asarray(obox.get_point_indices_within_bounding_box(pcd.points), dtype=np.int64)
        if len(idx) == 0:
            continue

        # Open3D patch convention: OBB local z-axis is patch normal.
        n = _unit(np.asarray(obox.R, dtype=np.float64)[:, 2])
        p0 = np.asarray(obox.center, dtype=np.float64)

        signed = (pts64[idx] - p0) @ n

        # Adaptive local threshold from robust distance spread.
        abs_signed = np.abs(signed)
        med = float(np.median(abs_signed))
        mad = float(np.median(np.abs(abs_signed - med)))
        robust_sigma = 1.4826 * mad
        local_thr = max(min_flatten_distance, min(flatten_thr_global, robust_sigma_k * robust_sigma + min_flatten_distance))

        # Keep only points geometrically close to the plane.
        near = abs_signed <= local_thr
        if not np.any(near):
            continue

        # Keep only points whose normals are compatible with this plane.
        idx_near0 = idx[near]
        pn = point_normals[idx_near0]
        ndot = np.abs(pn @ n)
        normal_ok = ndot >= float(normal_alignment_threshold)
        if not np.any(normal_ok):
            continue

        idx_near = idx_near0[normal_ok]
        signed_near = signed[near][normal_ok]
        if not np.any(near):
            continue

        # Project onto plane (proposal only).
        disp = -signed_near[:, None] * n[None, :]
        mag = np.abs(signed_near)
        better = mag < best_mag[idx_near]
        if np.any(better):
            idx_better = idx_near[better]
            best_mag[idx_better] = mag[better]
            best_disp[idx_better] = disp[better]

    valid = np.isfinite(best_mag)
    if np.any(valid):
        pts_new[valid] = pts_new[valid] + best_disp[valid]
        affected[valid] = True

    if cols is not None:
        cols_out = cols.copy()
    else:
        cols_out = None

    return StatisticPlaneResult(
        points=pts_new.astype(pts.dtype, copy=False),
        colors=cols_out,
        affected_mask=affected,
        n_patches=len(patches),
    )
