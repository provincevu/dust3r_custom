from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
from scipy.spatial import ConvexHull

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
    footprint_mask: np.ndarray
    footprint_polygon_xz: np.ndarray
    outer_boundary_xz: np.ndarray
    source_points: np.ndarray
    source_mask: np.ndarray
    dist_to_footprint_boundary: np.ndarray
    facade_band_width: float
    facade_band_mask: np.ndarray
    facade_band_points: np.ndarray


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


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            ys0 = max(0, -dy)
            ys1 = min(h, h - dy)
            xs0 = max(0, -dx)
            xs1 = min(w, w - dx)
            yd0 = max(0, dy)
            yd1 = min(h, h + dy)
            xd0 = max(0, dx)
            xd1 = min(w, w + dx)
            out[yd0:yd1, xd0:xd1] |= mask[ys0:ys1, xs0:xs1]
    return out


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    out = np.ones_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            shifted = np.zeros_like(mask, dtype=bool)
            ys0 = max(0, -dy)
            ys1 = min(h, h - dy)
            xs0 = max(0, -dx)
            xs1 = min(w, w - dx)
            yd0 = max(0, dy)
            yd1 = min(h, h + dy)
            xd0 = max(0, dx)
            xd1 = min(w, w + dx)
            shifted[yd0:yd1, xd0:xd1] = mask[ys0:ys1, xs0:xs1]
            out &= shifted
    return out


def morphological_closing(mask: np.ndarray, kernel_radius: int = 2) -> np.ndarray:
    """Binary closing (dilate then erode) with disk-like kernel in grid cells."""
    r = int(max(0, kernel_radius))
    if r == 0:
        return mask.astype(bool, copy=True)
    dil = _binary_dilate(mask.astype(bool), r)
    return _binary_erode(dil, r)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    best_coords = None
    best_size = 0

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            coords = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(coords) > best_size:
                best_size = len(coords)
                best_coords = coords

    out = np.zeros_like(mask, dtype=bool)
    if best_coords is not None:
        yy = np.array([c[0] for c in best_coords], dtype=np.int64)
        xx = np.array([c[1] for c in best_coords], dtype=np.int64)
        out[yy, xx] = True
    return out


def _mark_exterior_empty(occupied_mask: np.ndarray) -> np.ndarray:
    """Flood-fill empty cells from border to mark true exterior region."""
    h, w = occupied_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    stack = []

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
        if visited[y, x] or occupied_mask[y, x]:
            continue
        visited[y, x] = True
        stack.append((y - 1, x))
        stack.append((y + 1, x))
        stack.append((y, x - 1))
        stack.append((y, x + 1))
    return visited


def _boundary_from_component(component_mask: np.ndarray) -> np.ndarray:
    h, w = component_mask.shape
    boundary = np.zeros_like(component_mask, dtype=bool)
    yy, xx = np.where(component_mask)
    for y, x in zip(yy, xx):
        if y == 0 or y == h - 1 or x == 0 or x == w - 1:
            boundary[y, x] = True
            continue
        if (
            (not component_mask[y - 1, x])
            or (not component_mask[y + 1, x])
            or (not component_mask[y, x - 1])
            or (not component_mask[y, x + 1])
        ):
            boundary[y, x] = True
    return boundary


def _outer_boundary_from_component(component_mask: np.ndarray) -> np.ndarray:
    """Boundary cells that touch the true exterior (not interior courtyards)."""
    exterior_empty = _mark_exterior_empty(component_mask)
    h, w = component_mask.shape
    outer = np.zeros_like(component_mask, dtype=bool)

    yy, xx = np.where(component_mask)
    for y, x in zip(yy, xx):
        if y > 0 and exterior_empty[y - 1, x]:
            outer[y, x] = True
            continue
        if y + 1 < h and exterior_empty[y + 1, x]:
            outer[y, x] = True
            continue
        if x > 0 and exterior_empty[y, x - 1]:
            outer[y, x] = True
            continue
        if x + 1 < w and exterior_empty[y, x + 1]:
            outer[y, x] = True
            continue
    return outer


def _grid_cells_to_xz(ij: np.ndarray, origin_xz: np.ndarray, cell_size: float) -> np.ndarray:
    if len(ij) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    # ij is (row=z, col=x), convert to center coordinates in XZ.
    x = origin_xz[0] + (ij[:, 1].astype(np.float64) + 0.5) * cell_size
    z = origin_xz[1] + (ij[:, 0].astype(np.float64) + 0.5) * cell_size
    return np.stack([x, z], axis=1)


def extract_outer_footprint_polygon(
    occupancy: OccupancyGrid2D,
    *,
    closing_radius_cells: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Step 4: occupancy -> closing -> largest outer boundary -> footprint polygon."""
    occ_mask = occupancy.counts > 0
    closed = morphological_closing(occ_mask, kernel_radius=int(closing_radius_cells))
    largest = _largest_component(closed)
    boundary = _outer_boundary_from_component(largest)
    if not np.any(boundary):
        boundary = _boundary_from_component(largest)

    boundary_ij = np.argwhere(boundary)
    boundary_xz = _grid_cells_to_xz(boundary_ij, occupancy.origin_xz, occupancy.cell_size)

    if len(boundary_xz) < 3:
        return largest, boundary_xz

    try:
        hull = ConvexHull(boundary_xz)
        poly = boundary_xz[hull.vertices]
    except Exception:
        c = np.mean(boundary_xz, axis=0)
        ang = np.arctan2(boundary_xz[:, 1] - c[1], boundary_xz[:, 0] - c[0])
        poly = boundary_xz[np.argsort(ang)]

    return largest, poly


def _point_to_segment_distance_2d(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return np.linalg.norm(points - a[None, :], axis=1)
    t = ((points - a[None, :]) @ ab) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(points - proj, axis=1)


def distance_to_boundary_points(points_xz: np.ndarray, boundary_xz: np.ndarray) -> np.ndarray:
    """Nearest distance from each point to sampled boundary points in XZ."""
    if len(points_xz) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(boundary_xz) == 0:
        return np.full((len(points_xz),), np.inf, dtype=np.float64)

    dmin = np.full((len(points_xz),), np.inf, dtype=np.float64)
    chunk = 4096
    for i in range(0, len(boundary_xz), chunk):
        b = boundary_xz[i:i + chunk]
        diff = points_xz[:, None, :] - b[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        dmin = np.minimum(dmin, np.min(d, axis=1))
    return dmin


def distance_to_polygon_boundary(points_xz: np.ndarray, polygon_xz: np.ndarray) -> np.ndarray:
    if len(points_xz) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(polygon_xz) < 2:
        return np.full((len(points_xz),), np.inf, dtype=np.float64)

    n = len(polygon_xz)
    dmin = np.full((len(points_xz),), np.inf, dtype=np.float64)
    for i in range(n):
        a = polygon_xz[i]
        b = polygon_xz[(i + 1) % n]
        d = _point_to_segment_distance_2d(points_xz, a, b)
        dmin = np.minimum(dmin, d)
    return dmin


def select_outer_facade_band(
    source_points: np.ndarray,
    footprint_polygon_xz: np.ndarray,
    outer_boundary_xz: Optional[np.ndarray] = None,
    *,
    xz_span: float,
    band_ratio: float = 0.02,
    band_ratio_min: float = 0.01,
    band_ratio_max: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Step 5: keep points near footprint outer boundary within facade band width."""
    pts = np.asarray(source_points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("source_points must be (N, 3)")

    ratio = float(np.clip(band_ratio, band_ratio_min, band_ratio_max))
    bw = float(max(float(xz_span) * ratio, 1e-9))

    if len(pts) == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float64), bw

    pxz = pts[:, [0, 2]].astype(np.float64)
    if outer_boundary_xz is not None and len(outer_boundary_xz) > 0:
        dist = distance_to_boundary_points(pxz, np.asarray(outer_boundary_xz, dtype=np.float64))
    else:
        dist = distance_to_polygon_boundary(pxz, np.asarray(footprint_polygon_xz, dtype=np.float64))
    keep = dist <= bw
    return keep, dist, bw


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
    closing_radius_cells: int = 2,
    facade_source: Literal["slab", "all"] = "slab",
    facade_band_ratio: float = 0.02,
    facade_band_ratio_min: float = 0.01,
    facade_band_ratio_max: float = 0.03,
    strict: bool = False,
) -> WallSlabGridResult:
    """Run wall-slab pipeline with outer-footprint + outer-facade band selection."""
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

    footprint_mask, footprint_polygon_xz = extract_outer_footprint_polygon(
        occupancy,
        closing_radius_cells=int(closing_radius_cells),
    )
    outer_boundary_ij = np.argwhere(_outer_boundary_from_component(footprint_mask))
    outer_boundary_xz = _grid_cells_to_xz(outer_boundary_ij, occupancy.origin_xz, occupancy.cell_size)

    source_mode = str(facade_source).lower()
    if source_mode == "all":
        source_points = out_pts
        source_mask = np.ones((len(out_pts),), dtype=bool)
    else:
        source_points = slab_points
        source_mask = slab_mask

    facade_band_mask, dist_to_boundary, facade_band_width = select_outer_facade_band(
        source_points,
        footprint_polygon_xz,
        outer_boundary_xz=outer_boundary_xz,
        xz_span=float(bbox.xz_span),
        band_ratio=float(facade_band_ratio),
        band_ratio_min=float(facade_band_ratio_min),
        band_ratio_max=float(facade_band_ratio_max),
    )
    facade_band_points = source_points[facade_band_mask]

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
        footprint_mask=footprint_mask,
        footprint_polygon_xz=footprint_polygon_xz,
        outer_boundary_xz=outer_boundary_xz,
        source_points=source_points,
        source_mask=source_mask,
        dist_to_footprint_boundary=dist_to_boundary,
        facade_band_width=facade_band_width,
        facade_band_mask=facade_band_mask,
        facade_band_points=facade_band_points,
    )
