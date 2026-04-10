from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, List, Dict, Any

import numpy as np


@dataclass(frozen=True)
class RemoveGroundResult:
    points: np.ndarray
    colors: Optional[np.ndarray]
    kept_mask: np.ndarray
    plane_model: Optional[Tuple[float, float, float, float]]
    transform: Optional[np.ndarray] = None


@dataclass(frozen=True)
class AlignGroundResult:
    points: np.ndarray
    colors: Optional[np.ndarray]
    plane_model: Optional[Tuple[float, float, float, float]]
    transform: Optional[np.ndarray]
    aligned: bool


def _unit(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(vec)
    if n < eps:
        return vec * 0.0
    return vec / n


def _estimate_vertical_axis_pca(points: np.ndarray) -> np.ndarray:
    """Estimate an 'up' axis as the smallest-variance PCA component.

    This is a heuristic to decide which planes are likely 'ground' (normal aligned
    with the vertical axis) without requiring gravity/IMU.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if len(pts) < 10:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)

    center = np.nanmedian(pts, axis=0)
    X = pts - center
    # SVD on centered points; last right-singular vector => smallest variance axis
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    vertical = vt[-1]
    return _unit(vertical)


def _plane_point_dist(plane_model: Tuple[float, float, float, float], points: np.ndarray) -> np.ndarray:
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    denom = np.linalg.norm(n)
    if denom < 1e-12:
        return np.full((len(points),), np.inf, dtype=np.float64)
    return np.abs(points @ n + d) / denom


def _plane_signed_dist_unit(plane_model: Tuple[float, float, float, float], points: np.ndarray) -> np.ndarray:
    """Signed distance for a plane, using a unit normal."""
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    denom = np.linalg.norm(n)
    if denom < 1e-12:
        return np.full((len(points),), np.nan, dtype=np.float64)
    n_u = n / denom
    d_u = d / denom
    return points @ n_u + d_u


def _plane_unit_canonical(plane_model: Tuple[float, float, float, float]) -> Tuple[np.ndarray, float]:
    """Return a unit-normal plane with deterministic sign for stable comparison."""
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64), 0.0
    n = n / nn
    d = float(d) / nn
    # canonical orientation to remove sign ambiguity between equivalent planes
    if (n[2] < 0) or (abs(n[2]) < 1e-12 and n[1] < 0) or (abs(n[2]) < 1e-12 and abs(n[1]) < 1e-12 and n[0] < 0):
        n = -n
        d = -d
    return n, d


def _extract_planes(
    o3d,
    points: np.ndarray,
    *,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
    max_planes: int,
    min_inliers: int,
) -> List[Dict[str, Any]]:
    """Iteratively extract dominant planes and return plane models + inlier indices."""
    pts64 = np.asarray(points, dtype=np.float64)
    remaining = o3d.geometry.PointCloud()
    remaining.points = o3d.utility.Vector3dVector(pts64)
    remaining_idx = np.arange(len(pts64))

    planes: List[Dict[str, Any]] = []
    for _ in range(int(max_planes)):
        if len(remaining.points) < ransac_n:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=float(distance_threshold),
            ransac_n=int(ransac_n),
            num_iterations=int(num_iterations),
        )
        inliers = np.asarray(inliers, dtype=np.int64)
        if len(inliers) < int(min_inliers):
            break

        a, b, c, d = plane_model
        n = _unit(np.array([a, b, c], dtype=np.float64))
        global_inliers = remaining_idx[inliers]
        planes.append(
            {
                "plane_model": (float(a), float(b), float(c), float(d)),
                "normal": n,
                "inliers": global_inliers,
                "support": int(len(global_inliers)),
            }
        )

        remaining = remaining.select_by_index(inliers, invert=True)
        remaining_idx = np.delete(remaining_idx, inliers)

    return planes


def _try_segment_plane(
    o3d,
    pts: np.ndarray,
    *,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
) -> Tuple[Optional[Tuple[float, float, float, float]], np.ndarray]:
    if len(pts) < ransac_n:
        return None, np.empty((0,), dtype=np.int64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=float(distance_threshold),
        ransac_n=int(ransac_n),
        num_iterations=int(num_iterations),
    )
    inliers = np.asarray(inliers, dtype=np.int64)
    if len(inliers) == 0:
        return None, inliers
    a, b, c, d = plane_model
    return (float(a), float(b), float(c), float(d)), inliers


def _plane_basis_from_normal(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build a stable orthonormal basis (u, v) on a plane with normal n."""
    n = _unit(n)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = _unit(np.cross(n, ref))
    v = _unit(np.cross(n, u))
    return u, v


def _rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return R such that R @ a ~= b for unit vectors a,b."""
    a = _unit(a)
    b = _unit(b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-10:
        # 180 deg: pick any orthogonal axis.
        axis = _unit(np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float64)))
        if np.linalg.norm(axis) < 1e-8:
            axis = _unit(np.cross(a, np.array([0.0, 0.0, 1.0], dtype=np.float64)))
        x, y, z = axis
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
        # Rodrigues with theta=pi -> R = I + 2*K^2
        return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + K + (K @ K) * ((1.0 - c) / (s * s + 1e-12))
    return R


def _align_points_ground_to_oxz(
    points: np.ndarray,
    plane_model: Optional[Tuple[float, float, float, float]],
) -> Tuple[np.ndarray, Optional[Tuple[float, float, float, float]], Optional[np.ndarray]]:
    """Rotate (and shift Y) so the detected ground plane aligns to Oxz (y=0)."""
    if plane_model is None or len(points) == 0:
        return points, plane_model, None

    n, d = _plane_unit_canonical(plane_model)
    # Prefer +Y as up to keep orientation stable.
    if n[1] > 0:
        n = -n
        d = -d

    target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    R = _rotation_matrix_from_vectors(n, target)

    pts64 = np.asarray(points, dtype=np.float64)
    rot_pts = (R @ pts64.T).T

    # Shift along Y so plane is at y=0.
    p0 = (-d) * n
    p0r = R @ p0
    y_shift = -float(p0r[1])
    rot_pts[:, 1] += y_shift

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([0.0, y_shift, 0.0], dtype=np.float64)

    # After alignment, ground is y=0.
    aligned_plane = (0.0, 1.0, 0.0, 0.0)
    return rot_pts.astype(points.dtype, copy=False), aligned_plane, T


def align_pointcloud_to_ground_oxz(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    *,
    enabled: bool = True,
    distance_threshold: float = 0.005,
    ransac_n: int = 3,
    num_iterations: int = 1200,
    min_plane_inliers: int = 500,
    strict: bool = False,
) -> AlignGroundResult:
    """Align the full cloud to a detected dominant plane so ground becomes Oxz.

    Unlike ``remove_ground``, this utility does not remove points. It estimates one
    dominant plane, rotates the full point cloud to make that plane ``y=0``, and
    returns transformed coordinates for downstream export.
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

    if (not enabled) or len(pts) == 0:
        return AlignGroundResult(points=pts, colors=cols, plane_model=None, transform=None, aligned=False)

    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:  # pragma: no cover
        if strict:
            raise ImportError(
                "align_pointcloud_to_ground_oxz(enabled=True) requires open3d. Install with: pip install open3d"
            ) from exc
        return AlignGroundResult(points=pts, colors=cols, plane_model=None, transform=None, aligned=False)

    pts64 = np.asarray(pts, dtype=np.float64)
    plane_model, inliers = _try_segment_plane(
        o3d,
        pts64,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    if plane_model is None or len(inliers) < int(min_plane_inliers):
        return AlignGroundResult(points=pts, colors=cols, plane_model=None, transform=None, aligned=False)

    out_pts, out_plane, transform = _align_points_ground_to_oxz(pts, plane_model)
    return AlignGroundResult(
        points=out_pts,
        colors=cols,
        plane_model=out_plane,
        transform=transform,
        aligned=transform is not None,
    )


def _plane_2d_stats(points_on_plane: np.ndarray, normal: np.ndarray, grid_size: int = 12) -> Tuple[float, float]:
    """Return (point_density, center_hole_score) from 2D plane projection.

    center_hole_score is higher when the middle region is less occupied.
    """
    pts = np.asarray(points_on_plane, dtype=np.float64)
    if len(pts) < 4:
        return 0.0, 0.0

    u, v = _plane_basis_from_normal(normal)
    center = np.mean(pts, axis=0)
    X = pts - center[None, :]
    uv = np.stack((X @ u, X @ v), axis=1)

    mn = np.min(uv, axis=0)
    mx = np.max(uv, axis=0)
    span = np.maximum(mx - mn, 1e-9)
    area = float(span[0] * span[1])
    if area <= 1e-12:
        return 0.0, 0.0

    density = float(len(pts)) / area

    g = int(max(4, grid_size))
    ij = np.floor((uv - mn[None, :]) / span[None, :] * g).astype(np.int64)
    ij = np.clip(ij, 0, g - 1)
    occ = np.zeros((g, g), dtype=bool)
    occ[ij[:, 0], ij[:, 1]] = True

    c0 = g // 4
    c1 = g - c0
    center_occ = occ[c0:c1, c0:c1]
    center_occ_ratio = float(center_occ.mean()) if center_occ.size else 1.0
    center_hole_score = float(np.clip(1.0 - center_occ_ratio, 0.0, 1.0))
    return density, center_hole_score


def _mark_exterior_free_space(blocked: np.ndarray) -> np.ndarray:
    """Flood-fill free cells from border to mark exterior region."""
    h, w = blocked.shape
    visited = np.zeros((h, w), dtype=bool)
    stack: List[Tuple[int, int]] = []

    for x in range(w):
        stack.append((0, x))
        stack.append((h - 1, x))
    for y in range(h):
        stack.append((y, 0))
        stack.append((y, w - 1))

    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        if visited[y, x] or blocked[y, x]:
            continue
        visited[y, x] = True
        stack.append((y - 1, x))
        stack.append((y + 1, x))
        stack.append((y, x - 1))
        stack.append((y, x + 1))
    return visited


def _rebuild_ground_in_object_footprint(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    kept_mask: np.ndarray,
    plane_model: Optional[Tuple[float, float, float, float]],
    *,
    enabled: bool,
    distance_threshold: float,
    cell_size: Optional[float],
    max_points: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Recreate ground only inside enclosed interior footprint of the object.

    The interior region is estimated by projecting near-ground object points to the
    selected ground plane, rasterizing boundary cells, then flood-filling from map
    borders to remove exterior area. Remaining free cells are treated as interior.
    """
    pts = np.asarray(points)
    cols = None if colors is None else np.asarray(colors)

    if (not enabled) or plane_model is None or len(pts) == 0:
        return pts[kept_mask], (cols[kept_mask] if cols is not None else None)

    obj_pts = pts[kept_mask]
    if len(obj_pts) < 10:
        return pts[kept_mask], (cols[kept_mask] if cols is not None else None)

    n, d = _plane_unit_canonical(plane_model)
    u, v = _plane_basis_from_normal(n)

    signed_obj = obj_pts @ n + d
    # Ensure object is mostly on positive side of the plane for stable thresholding.
    if float(np.nanmedian(signed_obj)) < 0:
        n = -n
        d = -d
        u, v = _plane_basis_from_normal(n)
        signed_obj = obj_pts @ n + d

    # Use only near-ground shell of the object (walls/furniture near floor) as boundary.
    h_band = max(6.0 * float(distance_threshold), float(np.quantile(np.clip(signed_obj, 0.0, None), 0.20)))
    ring_sel = (signed_obj >= 0.0) & (signed_obj <= h_band)
    ring_pts = obj_pts[ring_sel]
    if len(ring_pts) < 20:
        return pts[kept_mask], (cols[kept_mask] if cols is not None else None)

    uv = np.stack((ring_pts @ u, ring_pts @ v), axis=1)
    mn = np.min(uv, axis=0)
    mx = np.max(uv, axis=0)
    span = np.maximum(mx - mn, 1e-9)

    if cell_size is None:
        cs = max(float(distance_threshold) * 2.0, float(np.linalg.norm(span)) / 350.0, 1e-4)
    else:
        cs = max(float(cell_size), 1e-6)

    ij = np.floor((uv - mn[None, :]) / cs).astype(np.int64)
    if len(ij) == 0:
        return pts[kept_mask], (cols[kept_mask] if cols is not None else None)

    ij -= np.min(ij, axis=0, keepdims=True)
    gh = int(np.max(ij[:, 0]) + 1)
    gw = int(np.max(ij[:, 1]) + 1)
    blocked = np.zeros((gh, gw), dtype=bool)
    blocked[ij[:, 0], ij[:, 1]] = True

    exterior = _mark_exterior_free_space(blocked)
    interior = (~blocked) & (~exterior)
    interior_idx = np.argwhere(interior)
    if len(interior_idx) == 0:
        return pts[kept_mask], (cols[kept_mask] if cols is not None else None)

    if len(interior_idx) > int(max_points):
        step = int(np.ceil(len(interior_idx) / float(max_points)))
        interior_idx = interior_idx[::step]

    origin_idx = np.min(np.floor((uv - mn[None, :]) / cs).astype(np.int64), axis=0)
    ij_global = interior_idx.astype(np.int64) + origin_idx[None, :]
    uv_center = mn[None, :] + (ij_global.astype(np.float64) + 0.5) * cs
    p0 = (-d) * n
    reg_pts = p0[None, :] + uv_center[:, 0:1] * u[None, :] + uv_center[:, 1:2] * v[None, :]

    base_pts = pts[kept_mask]
    out_pts = np.concatenate([base_pts, reg_pts.astype(base_pts.dtype, copy=False)], axis=0)

    if cols is None:
        return out_pts, None

    base_cols = cols[kept_mask]
    removed_cols = cols[~kept_mask]
    if len(removed_cols) > 0:
        fill_color = np.median(removed_cols.astype(np.float64), axis=0)
    else:
        fill_color = np.median(base_cols.astype(np.float64), axis=0) if len(base_cols) > 0 else np.array([0, 0, 0], dtype=np.float64)

    if np.issubdtype(base_cols.dtype, np.integer):
        fill_color = np.clip(np.rint(fill_color), 0, 255).astype(base_cols.dtype)
    else:
        fill_color = np.clip(fill_color, 0.0, 1.0).astype(base_cols.dtype)

    reg_cols = np.repeat(fill_color[None, :], len(reg_pts), axis=0)
    out_cols = np.concatenate([base_cols, reg_cols], axis=0)
    return out_pts, out_cols


def _finalize_remove_ground(
    pts: np.ndarray,
    cols: Optional[np.ndarray],
    kept: np.ndarray,
    plane_model: Optional[Tuple[float, float, float, float]],
    *,
    regenerate_ground: bool,
    regenerate_cell_size: Optional[float],
    regenerate_max_points: int,
    distance_threshold: float,
    align_ground_to_oxz: bool,
    pre_transform: Optional[np.ndarray],
) -> RemoveGroundResult:
    new_pts, new_cols = _rebuild_ground_in_object_footprint(
        pts,
        cols,
        kept,
        plane_model,
        enabled=bool(regenerate_ground),
        distance_threshold=float(distance_threshold),
        cell_size=regenerate_cell_size,
        max_points=int(regenerate_max_points),
    )
    transform = None
    plane_out = plane_model
    if align_ground_to_oxz:
        new_pts, plane_out, transform = _align_points_ground_to_oxz(new_pts, plane_model)
    if pre_transform is not None:
        if transform is None:
            transform = pre_transform
        else:
            transform = transform @ pre_transform
    return RemoveGroundResult(points=new_pts, colors=new_cols, kept_mask=kept, plane_model=plane_out, transform=transform)


def remove_ground(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    *,
    enabled: bool = True,
    distance_threshold: float = 0.005,
    ransac_n: int = 3,
    num_iterations: int = 2000,
    max_planes: int = 3,
    top_k_planes: int = 5,
    y_preference: Literal["high", "low"] = "low",
    coplanar_angle_tol_deg: float = 6.0,
    coplanar_offset_tol: Optional[float] = None,
    regenerate_ground: bool = True,
    regenerate_cell_size: Optional[float] = None,
    regenerate_max_points: int = 120000,
    align_ground_to_oxz: bool = True,
    normal_alignment_threshold: float = 0.95,
    bottom_quantile_threshold: float = 0.20,
    bottom_removal_margin: float = 0.15,
    strategy: Literal["ransac_scored", "planar_patch", "orthogonal", "bottom_slice"] = "ransac_scored",
    patch_normal_variance_threshold_deg: float = 35.0,
    patch_coplanarity_deg: float = 5.0,
    patch_outlier_ratio: float = 0.50,
    patch_min_plane_edge_ratio: float = 0.02,
    patch_min_num_points: int = 100000,
    patch_ground_height_quantile: float = 0.45,
    patch_normal_radius: Optional[float] = None,
    patch_normal_max_nn: int = 50,
    orthogonal_tol_deg: float = 15.0,
    extreme_quantile: float = 0.05,
    min_plane_inliers: int = 500,
    strict: bool = False,
) -> RemoveGroundResult:
    """Remove a dominant ground plane from a point cloud.

    Parameters
    - points: (N,3) float array
    - colors: optional (N,3) array (uint8 or float in [0,1])
    - enabled: if False, returns input unchanged
    - distance_threshold: RANSAC inlier distance (in same unit as points)
        - max_planes: number of planes to try/extract.
        - normal_alignment_threshold: required alignment between the plane normal and
            the chosen vertical axis (axis-aligned heuristic).
        - bottom_quantile_threshold / bottom_removal_margin: parameters for
            strategy="bottom_slice".
        - strategy:
                - "ransac_scored": extract multiple planes with iterative RANSAC,
                    keep top-K by support, score each candidate using alignment to
                    Y-axis, mean Y, point density and center-hole signal; remove the
                    highest-score plane inliers and all near-coplanar plane inliers.
                - "planar_patch": detect large planar patches (RSPD via Open3D),
                    select a low-height, wall-orthogonal patch, and remove its inliers.
                - "orthogonal": extract several dominant planes; pick the one whose
                    normal is orthogonal to many other large planes (walls) and which lies
                    on an extreme side of the scene. This tends to work better for
                    buildings where ground is ~90° from walls.
                - "bottom_slice": fit on a bottom slice along an axis and remove nearby
                    points (more heuristic, may fail if global frame is rotated).
        - patch_*: tuning params for strategy="planar_patch".
        - top_k_planes: how many largest RANSAC planes to score (recommended 3-5).
                - y_preference: choose whether higher-Y or lower-Y planes are preferred.
                - coplanar_angle_tol_deg / coplanar_offset_tol: near-coplanar matching
                    thresholds to remove fragmented pieces of the same ground plane.
                - regenerate_ground: after removing detected ground, recreate a ground layer
                    only under the object's footprint (projected on the selected plane).
                - regenerate_cell_size / regenerate_max_points: controls regenerated ground
                    sampling density and cap.
                - align_ground_to_oxz: rotate full output cloud so selected ground aligns
                    with Oxz (y=0).
                - orthogonal_tol_deg: tolerance for considering two plane normals orthogonal
            (in degrees).
        - extreme_quantile: how close to an extreme the ground plane should be
            along its own normal direction.
        - min_plane_inliers: minimum inliers to accept an extracted plane.
    - strict: if True and Open3D not available, raise; else no-op.

    Returns
    - RemoveGroundResult with filtered points/colors and a boolean kept_mask.

    Notes
    - Requires `open3d` only when enabled=True.
    - This is a heuristic; for best object separation (house vs trees), consider
      combining with a 2D semantic mask projected to 3D.
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
        kept = np.ones((len(pts),), dtype=bool)
        return RemoveGroundResult(points=pts, colors=cols, kept_mask=kept, plane_model=None, transform=None)

    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:  # pragma: no cover
        if strict:
            raise ImportError(
                "remove_ground(enabled=True) requires open3d. Install with: pip install open3d"
            ) from exc
        kept = np.ones((len(pts),), dtype=bool)
        return RemoveGroundResult(points=pts, colors=cols, kept_mask=kept, plane_model=None, transform=None)

    pts64 = np.asarray(pts, dtype=np.float64)
    pre_transform: Optional[np.ndarray] = None

    # Optional pre-alignment so removal runs in a frame where ground ~ Oxz.
    if align_ground_to_oxz:
        pre_plane, pre_inliers = _try_segment_plane(
            o3d,
            pts64,
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=max(800, int(num_iterations) // 2),
        )
        if pre_plane is not None and len(pre_inliers) >= int(min_plane_inliers):
            aligned_pts, _, Tpre = _align_points_ground_to_oxz(pts64, pre_plane)
            pts = aligned_pts.astype(pts.dtype, copy=False)
            pts64 = np.asarray(pts, dtype=np.float64)
            pre_transform = Tpre

    # Strategy -1: Multi-RANSAC + scored plane ranking.
    if strategy == "ransac_scored":
        planes = _extract_planes(
            o3d,
            pts64,
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
            max_planes=max(int(max_planes), int(top_k_planes)),
            min_inliers=min_plane_inliers,
        )

        if planes:
            planes = sorted(planes, key=lambda p: int(p.get("support", 0)), reverse=True)
            k = int(np.clip(top_k_planes, 1, max(1, len(planes))))
            candidates = planes[:k]

            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            cand_stats: List[Dict[str, Any]] = []
            for p in candidates:
                n = _unit(np.asarray(p["normal"], dtype=np.float64))
                inliers = np.asarray(p["inliers"], dtype=np.int64)
                pts_in = pts64[inliers]
                if len(pts_in) == 0:
                    continue

                align_y = float(abs(np.dot(n, y_axis)))
                angle_deg = float(np.rad2deg(np.arccos(np.clip(align_y, -1.0, 1.0))))
                mean_y = float(np.mean(pts_in[:, 1]))
                density, hole = _plane_2d_stats(pts_in, n, grid_size=12)

                cand_stats.append(
                    {
                        "plane": p,
                        "align_y": align_y,
                        "angle_deg": angle_deg,
                        "mean_y": mean_y,
                        "density": float(density),
                        "hole": float(hole),
                    }
                )

            if cand_stats:
                heights = np.array([c["mean_y"] for c in cand_stats], dtype=np.float64)
                densities = np.array([c["density"] for c in cand_stats], dtype=np.float64)

                h_min, h_max = float(np.min(heights)), float(np.max(heights))
                d_min, d_max = float(np.min(densities)), float(np.max(densities))

                for c in cand_stats:
                    h_norm_raw = (c["mean_y"] - h_min) / max(h_max - h_min, 1e-9)
                    h_norm = h_norm_raw if y_preference == "high" else (1.0 - h_norm_raw)
                    d_norm = (c["density"] - d_min) / max(d_max - d_min, 1e-9)
                    # Prioritize horizontal plane (+), higher mean Y (+),
                    # denser support (+), and center-hole signal (+, optional).
                    c["score"] = (
                        0.55 * float(c["align_y"]) +
                        0.25 * float(h_norm) +
                        0.15 * float(d_norm) +
                        0.05 * float(c["hole"])
                    )

                best = max(cand_stats, key=lambda c: float(c["score"]))
                best_plane = best["plane"]
                inliers = np.asarray(best_plane["inliers"], dtype=np.int64)

                # Also remove any extracted plane that is near-coplanar with best.
                if coplanar_offset_tol is None:
                    offset_tol = float(distance_threshold) * 3.0
                else:
                    offset_tol = float(max(0.0, coplanar_offset_tol))
                cos_parallel = float(np.cos(np.deg2rad(float(coplanar_angle_tol_deg))))
                n_best, d_best = _plane_unit_canonical(best_plane["plane_model"])
                union_inliers: List[np.ndarray] = [inliers]

                for p in planes:
                    if p is best_plane:
                        continue
                    n_i, d_i = _plane_unit_canonical(p["plane_model"])
                    # align signs so offset comparison is meaningful
                    if float(np.dot(n_best, n_i)) < 0.0:
                        n_i = -n_i
                        d_i = -d_i
                    parallel = abs(float(np.dot(n_best, n_i))) >= cos_parallel
                    close_offset = abs(float(d_i - d_best)) <= offset_tol
                    if parallel and close_offset:
                        union_inliers.append(np.asarray(p["inliers"], dtype=np.int64))

                inliers = np.unique(np.concatenate(union_inliers)) if union_inliers else inliers
                remove = np.zeros((len(pts64),), dtype=bool)
                remove[inliers] = True

                kept = ~remove
                return _finalize_remove_ground(
                    pts,
                    cols,
                    kept,
                    best_plane["plane_model"],
                    regenerate_ground=regenerate_ground,
                    regenerate_cell_size=regenerate_cell_size,
                    regenerate_max_points=regenerate_max_points,
                    distance_threshold=distance_threshold,
                    align_ground_to_oxz=False,
                    pre_transform=pre_transform,
                )

        # fallback
        strategy = "planar_patch"

    # Strategy 0: Detect large planar patches first (more stable for outdoor scenes).
    if strategy == "planar_patch":
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts64)

        bbox = np.ptp(pts64, axis=0)
        diag = float(np.linalg.norm(bbox))
        est_radius = max(diag * 0.01, 1e-6)
        radius = float(patch_normal_radius) if patch_normal_radius is not None else est_radius

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=int(patch_normal_max_nn))
        )

        min_edge_len = float(max(diag * float(patch_min_plane_edge_ratio), float(distance_threshold) * 4.0))
        patches = pcd.detect_planar_patches(
            normal_variance_threshold_deg=float(patch_normal_variance_threshold_deg),
            coplanarity_deg=float(patch_coplanarity_deg),
            outlier_ratio=float(patch_outlier_ratio),
            min_plane_edge_length=min_edge_len,
            min_num_points=int(patch_min_num_points),
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=int(patch_normal_max_nn)),
        )

        if patches:
            up_axis = _estimate_vertical_axis_pca(pts64)
            up_proj = pts64 @ up_axis
            low_height_gate = float(np.quantile(up_proj, float(patch_ground_height_quantile)))
            ortho_dot_thr = float(np.sin(np.deg2rad(float(orthogonal_tol_deg))))

            candidates: List[Dict[str, Any]] = []
            for obox in patches:
                idx = np.asarray(obox.get_point_indices_within_bounding_box(pcd.points), dtype=np.int64)
                if len(idx) < int(patch_min_num_points):
                    continue

                n = _unit(np.asarray(obox.R, dtype=np.float64)[:, 2])
                p0 = np.asarray(obox.center, dtype=np.float64)
                plane_height = float(p0 @ up_axis)
                if plane_height > low_height_gate:
                    continue

                signed = (pts64[idx] - p0) @ n
                near = np.abs(signed) <= float(distance_threshold)
                if not np.any(near):
                    continue

                inliers = idx[near]
                candidates.append(
                    {
                        "normal": n,
                        "center": p0,
                        "height": plane_height,
                        "inliers": inliers,
                        "support": int(len(inliers)),
                        "align_up": float(abs(np.dot(n, up_axis))),
                    }
                )

            if len(candidates) >= 1:
                for i, ci in enumerate(candidates):
                    ni = np.asarray(ci["normal"], dtype=np.float64)
                    ortho_degree = 0
                    for j, cj in enumerate(candidates):
                        if i == j:
                            continue
                        nj = np.asarray(cj["normal"], dtype=np.float64)
                        if abs(float(np.dot(ni, nj))) <= ortho_dot_thr:
                            ortho_degree += 1
                    ci["ortho_degree"] = int(ortho_degree)

                candidates_sorted = sorted(
                    candidates,
                    key=lambda c: (
                        int(c.get("ortho_degree", 0)),
                        float(c.get("align_up", 0.0)),
                        int(c.get("support", 0)),
                        -float(c.get("height", 0.0)),
                    ),
                    reverse=True,
                )
                best = candidates_sorted[0]

                if int(best.get("support", 0)) >= int(min_plane_inliers):
                    remove = np.zeros((len(pts64),), dtype=bool)
                    remove[np.asarray(best["inliers"], dtype=np.int64)] = True
                    kept = ~remove
                    p0 = np.asarray(best["center"], dtype=np.float64)
                    n = _unit(np.asarray(best["normal"], dtype=np.float64))
                    d = -float(np.dot(n, p0))
                    plane_model = (float(n[0]), float(n[1]), float(n[2]), float(d))
                    return _finalize_remove_ground(
                        pts,
                        cols,
                        kept,
                        plane_model,
                        regenerate_ground=regenerate_ground,
                        regenerate_cell_size=regenerate_cell_size,
                        regenerate_max_points=regenerate_max_points,
                        distance_threshold=distance_threshold,
                        align_ground_to_oxz=False,
                        pre_transform=pre_transform,
                    )

        # fallback
        strategy = "orthogonal"

    # Strategy 1: Use orthogonality relations between planes.
    if strategy == "orthogonal":
        planes = _extract_planes(
            o3d,
            pts64,
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
            max_planes=max_planes,
            min_inliers=min_plane_inliers,
        )

        if len(planes) >= 2:
            ortho_dot_thr = float(np.sin(np.deg2rad(float(orthogonal_tol_deg))))
            # Score each plane by how many other plane normals are ~orthogonal.
            for i, pi in enumerate(planes):
                ni = np.asarray(pi["normal"], dtype=np.float64)
                ortho_degree = 0
                for j, pj in enumerate(planes):
                    if i == j:
                        continue
                    nj = np.asarray(pj["normal"], dtype=np.float64)
                    if abs(float(np.dot(ni, nj))) <= ortho_dot_thr:
                        ortho_degree += 1
                pi["ortho_degree"] = int(ortho_degree)

                # Extremeness along its own normal direction: plane position vs projections.
                a, b, c, d = pi["plane_model"]
                n_raw = np.array([a, b, c], dtype=np.float64)
                denom = np.linalg.norm(n_raw)
                if denom < 1e-12:
                    pi["extreme_dist"] = float("inf")
                    pi["extreme_side"] = 0
                else:
                    n_u = n_raw / denom
                    d_u = d / denom
                    proj = pts64 @ n_u
                    plane_pos = -d_u
                    q_low = float(np.quantile(proj, float(extreme_quantile)))
                    q_high = float(np.quantile(proj, 1.0 - float(extreme_quantile)))
                    dl = abs(plane_pos - q_low)
                    dh = abs(plane_pos - q_high)
                    pi["extreme_dist"] = float(min(dl, dh))
                    # which side is outside (toward the closer extreme)
                    pi["extreme_side"] = -1 if dl <= dh else +1

            # Choose candidate:
            # - highest ortho_degree (ground is orthogonal to many walls)
            # - then highest support
            # - then closest to an extreme
            planes_sorted = sorted(
                planes,
                key=lambda p: (
                    int(p.get("ortho_degree", 0)),
                    int(p.get("support", 0)),
                    -float(p.get("extreme_dist", float("inf"))),
                ),
                reverse=True,
            )
            best = planes_sorted[0]

            # Only accept if it has at least one orthogonal partner.
            if int(best.get("ortho_degree", 0)) >= 1:
                plane_model = best["plane_model"]
                signed = _plane_signed_dist_unit(plane_model, pts64)
                dist = np.abs(signed)
                side = int(best.get("extreme_side", 0))
                if side == 0:
                    remove = dist <= float(distance_threshold)
                elif side < 0:
                    remove = (dist <= float(distance_threshold)) & (signed < 0)
                else:
                    remove = (dist <= float(distance_threshold)) & (signed > 0)

                kept = ~remove
                return _finalize_remove_ground(
                    pts,
                    cols,
                    kept,
                    plane_model,
                    regenerate_ground=regenerate_ground,
                    regenerate_cell_size=regenerate_cell_size,
                    regenerate_max_points=regenerate_max_points,
                    distance_threshold=distance_threshold,
                    align_ground_to_oxz=False,
                    pre_transform=pre_transform,
                )

        # fallback
        strategy = "bottom_slice"

    # Strategy 2: bottom slice on axis (fallback)
    # Heuristic for "house occupies >60%": avoid PCA vertical (often biased by house walls).
    best = None
    best_axis = None
    for axis in range(3):
        coord = pts64[:, axis]
        q = float(np.quantile(coord, bottom_quantile_threshold))
        bottom_idx = np.where(coord <= q)[0]
        if len(bottom_idx) < max(ransac_n, 100):
            continue
        plane_model, inliers_local = _try_segment_plane(
            o3d,
            pts64[bottom_idx],
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        if plane_model is None or len(inliers_local) == 0:
            continue

        a, b, c, d = plane_model
        n = _unit(np.array([a, b, c], dtype=np.float64))
        alignment = abs(float(n[axis]))

        global_inliers_bottom = bottom_idx[inliers_local]
        inlier_coord_med = float(np.median(coord[global_inliers_bottom]))
        is_bottom = inlier_coord_med <= q

        candidate = {
            "plane_model": plane_model,
            "axis": axis,
            "score": int(len(global_inliers_bottom)),
            "alignment": float(alignment),
            "is_bottom": bool(is_bottom),
        }

        if candidate["alignment"] >= normal_alignment_threshold and candidate["is_bottom"]:
            if best is None or candidate["score"] > best["score"]:
                best = candidate
                best_axis = axis

    if best is None or best_axis is None:
        kept = np.ones((len(pts64),), dtype=bool)
        return RemoveGroundResult(points=pts, colors=cols, kept_mask=kept, plane_model=None, transform=pre_transform)

    plane_model = best["plane_model"]
    axis = int(best_axis)

    coord = pts64[:, axis]
    q_remove = float(np.quantile(coord, min(0.95, bottom_quantile_threshold + bottom_removal_margin)))
    dist = _plane_point_dist(plane_model, pts64)
    remove = (dist <= float(distance_threshold)) & (coord <= q_remove)

    kept = ~remove
    return _finalize_remove_ground(
        pts,
        cols,
        kept,
        plane_model,
        regenerate_ground=regenerate_ground,
        regenerate_cell_size=regenerate_cell_size,
        regenerate_max_points=regenerate_max_points,
        distance_threshold=distance_threshold,
        align_ground_to_oxz=False,
        pre_transform=pre_transform,
    )
