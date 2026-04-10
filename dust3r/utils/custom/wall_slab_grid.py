from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .remove_ground import align_pointcloud_to_ground_oxz


@dataclass(frozen=True)
class BBoxScaleInfo:
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    diag: float
    y_span: float
    xz_span: float


@dataclass(frozen=True)
class WallSlabInfo:
    y_min: float
    y_max: float
    y_span: float
    y0: float
    y1: float
    scale_duoi: float
    scale_tren: float
    slab_ratio_min: float
    slab_ratio_max: float


@dataclass(frozen=True)
class OccupancyGrid2D:
    counts: np.ndarray
    occupied: np.ndarray
    origin_xz: np.ndarray
    cell_size: float
    width: int
    height: int
    xz_min: np.ndarray
    xz_max: np.ndarray


@dataclass(frozen=True)
class WallSlabGridResult:
    points: np.ndarray
    colors: Optional[np.ndarray]
    aligned: bool
    align_transform: Optional[np.ndarray]
    bbox: BBoxScaleInfo
    slab: WallSlabInfo
    slab_mask: np.ndarray
    slab_points: np.ndarray
    occupancy: OccupancyGrid2D
    cut_lines: np.ndarray


def _validate_points_colors(
    points: np.ndarray,
    colors: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3)")

    if colors is not None:
        cols = np.asarray(colors)
        if cols.shape[0] != pts.shape[0] or cols.shape[1] != 3:
            raise ValueError("colors must be (N, 3) and match points")
    else:
        cols = None
    return pts, cols


def read_point_cloud_with_bbox(
    cloud_path: str,
    *,
    strict: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], BBoxScaleInfo]:
    """Read point cloud file and compute bbox/diag scale statistics."""
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:  # pragma: no cover
        if strict:
            raise ImportError(
                "read_point_cloud_with_bbox requires open3d. Install with: pip install open3d"
            ) from exc
        return np.zeros((0, 3), dtype=np.float32), None, BBoxScaleInfo(
            bbox_min=np.zeros((3,), dtype=np.float64),
            bbox_max=np.zeros((3,), dtype=np.float64),
            diag=0.0,
            y_span=0.0,
            xz_span=0.0,
        )

    pcd = o3d.io.read_point_cloud(str(cloud_path))
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None

    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_min = np.asarray(bbox.get_min_bound(), dtype=np.float64)
    bbox_max = np.asarray(bbox.get_max_bound(), dtype=np.float64)
    diag = float(np.linalg.norm(bbox_max - bbox_min))

    y_span = float(max(0.0, bbox_max[1] - bbox_min[1]))
    xz_span = float(max(bbox_max[0] - bbox_min[0], bbox_max[2] - bbox_min[2], 0.0))

    info = BBoxScaleInfo(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        diag=diag,
        y_span=y_span,
        xz_span=xz_span,
    )
    return pts, cols, info


def compute_bbox_scale(points: np.ndarray) -> BBoxScaleInfo:
    """Compute bbox, diag and spans from raw points."""
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if len(pts) == 0:
        z = np.zeros((3,), dtype=np.float64)
        return BBoxScaleInfo(bbox_min=z, bbox_max=z, diag=0.0, y_span=0.0, xz_span=0.0)

    bbox_min = np.min(pts.astype(np.float64), axis=0)
    bbox_max = np.max(pts.astype(np.float64), axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    y_span = float(max(0.0, bbox_max[1] - bbox_min[1]))
    xz_span = float(max(bbox_max[0] - bbox_min[0], bbox_max[2] - bbox_min[2], 0.0))
    return BBoxScaleInfo(bbox_min=bbox_min, bbox_max=bbox_max, diag=diag, y_span=y_span, xz_span=xz_span)


def compute_wall_slab_by_y_span(
    points: np.ndarray,
    *,
    scale_duoi: float = 0.02,
    scale_tren: float = 0.18,
    scale_limit: float = 1.00,
) -> Tuple[np.ndarray, WallSlabInfo]:
    """Select a low-height wall slab using [y0, y1] as percentages of y-span."""
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3)")

    sd = float(np.clip(scale_duoi, 0.0, scale_limit))
    st = float(np.clip(scale_tren, 0.0, scale_limit))
    if st < sd:
        sd, st = st, sd

    if len(pts) == 0:
        info = WallSlabInfo(
            y_min=0.0,
            y_max=0.0,
            y_span=0.0,
            y0=0.0,
            y1=0.0,
            scale_duoi=sd,
            scale_tren=st,
            slab_ratio_min=0.0,
            slab_ratio_max=0.0,
        )
        return np.zeros((0,), dtype=bool), info

    y = pts[:, 1].astype(np.float64)
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_span = float(max(y_max - y_min, 1e-12))

    y0 = float(y_min + sd * y_span)
    y1 = float(y_min + st * y_span)
    if y1 < y0:
        y0, y1 = y1, y0

    slab_mask = (y >= y0) & (y <= y1)
    slab_ratio_min = float((y0 - y_min) / y_span)
    slab_ratio_max = float((y1 - y_min) / y_span)

    info = WallSlabInfo(
        y_min=y_min,
        y_max=y_max,
        y_span=y_span,
        y0=y0,
        y1=y1,
        scale_duoi=sd,
        scale_tren=st,
        slab_ratio_min=slab_ratio_min,
        slab_ratio_max=slab_ratio_max,
    )
    return slab_mask, info


def _grid_shape_from_points_xz(points_xz: np.ndarray, cell_size: float) -> Tuple[np.ndarray, np.ndarray, int, int]:
    xz_min = np.min(points_xz, axis=0)
    xz_max = np.max(points_xz, axis=0)
    span = np.maximum(xz_max - xz_min, 1e-9)

    width = int(np.ceil(span[0] / cell_size)) + 1
    height = int(np.ceil(span[1] / cell_size)) + 1
    width = max(width, 1)
    height = max(height, 1)
    return xz_min, xz_max, width, height


def build_slab_occupancy_grid_xz(
    slab_points: np.ndarray,
    *,
    xz_span: Optional[float] = None,
    cell_ratio: float = 0.006,
    cell_ratio_min: float = 0.004,
    cell_ratio_max: float = 0.010,
) -> OccupancyGrid2D:
    """Project slab to XZ and build occupancy/count grid with bbox-relative cell size."""
    pts = np.asarray(slab_points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("slab_points must be (N, 3)")

    ratio = float(np.clip(cell_ratio, cell_ratio_min, cell_ratio_max))

    if len(pts) == 0:
        z2 = np.zeros((2,), dtype=np.float64)
        return OccupancyGrid2D(
            counts=np.zeros((1, 1), dtype=np.int32),
            occupied=np.zeros((1, 1), dtype=bool),
            origin_xz=z2,
            cell_size=0.0,
            width=1,
            height=1,
            xz_min=z2,
            xz_max=z2,
        )

    pts_xz = pts[:, [0, 2]].astype(np.float64)
    span_est = float(xz_span) if xz_span is not None else float(
        max(np.ptp(pts_xz[:, 0]), np.ptp(pts_xz[:, 1]), 1e-9)
    )
    cell = float(max(span_est * ratio, 1e-9))

    xz_min, xz_max, width, height = _grid_shape_from_points_xz(pts_xz, cell)

    ij = np.floor((pts_xz - xz_min[None, :]) / cell).astype(np.int64)
    ij[:, 0] = np.clip(ij[:, 0], 0, width - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, height - 1)

    counts = np.zeros((height, width), dtype=np.int32)
    np.add.at(counts, (ij[:, 1], ij[:, 0]), 1)
    occupied = counts > 0

    return OccupancyGrid2D(
        counts=counts,
        occupied=occupied,
        origin_xz=xz_min,
        cell_size=cell,
        width=width,
        height=height,
        xz_min=xz_min,
        xz_max=xz_max,
    )


def build_slab_cut_lines(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    *,
    y0: float,
    y1: float,
) -> np.ndarray:
    """Build 8 rectangle segments (4 at y0 and 4 at y1) for demo visualization."""
    x0, z0 = float(bbox_min[0]), float(bbox_min[2])
    x1, z1 = float(bbox_max[0]), float(bbox_max[2])

    def _rect_segments(yv: float) -> np.ndarray:
        p00 = np.array([x0, yv, z0], dtype=np.float64)
        p10 = np.array([x1, yv, z0], dtype=np.float64)
        p11 = np.array([x1, yv, z1], dtype=np.float64)
        p01 = np.array([x0, yv, z1], dtype=np.float64)
        return np.stack(
            [
                np.stack([p00, p10], axis=0),
                np.stack([p10, p11], axis=0),
                np.stack([p11, p01], axis=0),
                np.stack([p01, p00], axis=0),
            ],
            axis=0,
        )

    return np.concatenate([_rect_segments(float(y0)), _rect_segments(float(y1))], axis=0)


def wall_slab_grid_from_points(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    *,
    align_ground: bool = True,
    align_distance_threshold: float = 0.005,
    align_ransac_n: int = 3,
    align_num_iterations: int = 1200,
    align_min_plane_inliers: int = 500,
    scale_duoi: float = 0.02,
    scale_tren: float = 0.18,
    slab_scale_limit: float = 1.00,
    cell_ratio: float = 0.006,
    cell_ratio_min: float = 0.004,
    cell_ratio_max: float = 0.010,
    strict: bool = False,
) -> WallSlabGridResult:
    """Run wall-slab pipeline: align -> slab cut by y-span -> occupancy grid in XZ."""
    pts, cols = _validate_points_colors(points, colors)

    aligned = False
    transform = None
    out_pts = pts
    out_cols = cols

    if align_ground and len(pts) > 0:
        ares = align_pointcloud_to_ground_oxz(
            pts,
            cols,
            enabled=True,
            distance_threshold=float(align_distance_threshold),
            ransac_n=int(align_ransac_n),
            num_iterations=int(align_num_iterations),
            min_plane_inliers=int(align_min_plane_inliers),
            strict=bool(strict),
        )
        out_pts = ares.points
        out_cols = ares.colors
        aligned = bool(ares.aligned)
        transform = ares.transform

    bbox = compute_bbox_scale(out_pts)
    slab_mask, slab = compute_wall_slab_by_y_span(
        out_pts,
        scale_duoi=float(scale_duoi),
        scale_tren=float(scale_tren),
        scale_limit=float(slab_scale_limit),
    )

    slab_points = out_pts[slab_mask]
    occupancy = build_slab_occupancy_grid_xz(
        slab_points,
        xz_span=float(bbox.xz_span),
        cell_ratio=float(cell_ratio),
        cell_ratio_min=float(cell_ratio_min),
        cell_ratio_max=float(cell_ratio_max),
    )

    cut_lines = build_slab_cut_lines(
        bbox.bbox_min,
        bbox.bbox_max,
        y0=float(slab.y0),
        y1=float(slab.y1),
    )

    return WallSlabGridResult(
        points=out_pts,
        colors=out_cols,
        aligned=aligned,
        align_transform=transform,
        bbox=bbox,
        slab=slab,
        slab_mask=slab_mask,
        slab_points=slab_points,
        occupancy=occupancy,
        cut_lines=cut_lines,
    )
