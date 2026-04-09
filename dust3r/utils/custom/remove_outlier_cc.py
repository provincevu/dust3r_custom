from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class RemoveOutlierCCResult:
    points: np.ndarray
    colors: Optional[np.ndarray]
    kept_mask: np.ndarray
    n_components: int
    kept_component_sizes: Tuple[int, ...]
    voxel_size: float


def _as_int_tuple(arr: np.ndarray) -> Tuple[int, int, int]:
    return int(arr[0]), int(arr[1]), int(arr[2])


def remove_outlier_cc(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    *,
    enabled: bool = True,
    octree_level: int = 8,
    min_points_per_component: int = 20,
    keep_largest_only: bool = True,
) -> RemoveOutlierCCResult:
    """Remove sparse outlier components using voxel connected components.

    This approximates CloudCompare's "Label Connected Components" on an octree
    grid by:
    1) quantizing points onto a voxel grid (level -> voxel size),
    2) computing 26-neighborhood connected components between occupied voxels,
    3) keeping only components above `min_points_per_component`, and optionally
       keeping only the largest one (CC#0-like behavior).
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

    n = len(pts)
    if not enabled or n == 0:
        kept = np.ones((n,), dtype=bool)
        return RemoveOutlierCCResult(
            points=pts,
            colors=cols,
            kept_mask=kept,
            n_components=1 if n > 0 else 0,
            kept_component_sizes=(int(n),) if n > 0 else tuple(),
            voxel_size=0.0,
        )

    level = int(max(1, min(16, octree_level)))
    min_pts = int(max(1, min_points_per_component))

    pts64 = np.asarray(pts, dtype=np.float64)
    pmin = np.min(pts64, axis=0)
    pmax = np.max(pts64, axis=0)
    extent = pmax - pmin
    max_extent = float(np.max(extent))

    if max_extent <= 1e-12:
        kept = np.ones((n,), dtype=bool)
        return RemoveOutlierCCResult(
            points=pts,
            colors=cols,
            kept_mask=kept,
            n_components=1,
            kept_component_sizes=(int(n),),
            voxel_size=0.0,
        )

    voxel_size = max(max_extent / float(2 ** level), 1e-12)
    grid = np.floor((pts64 - pmin[None, :]) / voxel_size).astype(np.int64)

    voxel_to_points: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(n):
        key = _as_int_tuple(grid[i])
        bucket = voxel_to_points.get(key)
        if bucket is None:
            voxel_to_points[key] = [i]
        else:
            bucket.append(i)

    visited: Dict[Tuple[int, int, int], bool] = {}
    components: List[List[Tuple[int, int, int]]] = []

    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    for start in voxel_to_points.keys():
        if visited.get(start, False):
            continue

        queue = [start]
        visited[start] = True
        comp_voxels: List[Tuple[int, int, int]] = []

        while queue:
            cur = queue.pop()
            comp_voxels.append(cur)
            cx, cy, cz = cur
            for dx, dy, dz in neighbor_offsets:
                nb = (cx + dx, cy + dy, cz + dz)
                if nb in voxel_to_points and not visited.get(nb, False):
                    visited[nb] = True
                    queue.append(nb)

        components.append(comp_voxels)

    comp_point_indices: List[np.ndarray] = []
    comp_sizes: List[int] = []
    for voxels in components:
        idxs: List[int] = []
        for v in voxels:
            idxs.extend(voxel_to_points[v])
        arr = np.asarray(idxs, dtype=np.int64)
        comp_point_indices.append(arr)
        comp_sizes.append(int(len(arr)))

    valid_ids = [i for i, sz in enumerate(comp_sizes) if sz >= min_pts]
    if keep_largest_only and valid_ids:
        best = max(valid_ids, key=lambda i: comp_sizes[i])
        valid_ids = [best]

    kept_mask = np.zeros((n,), dtype=bool)
    kept_sizes: List[int] = []
    for i in valid_ids:
        kept_mask[comp_point_indices[i]] = True
        kept_sizes.append(comp_sizes[i])

    if not np.any(kept_mask):
        # Fail-safe: never return empty cloud from filtering.
        kept_mask[:] = True
        kept_sizes = [int(n)]

    out_points = pts[kept_mask]
    out_colors = cols[kept_mask] if cols is not None else None

    return RemoveOutlierCCResult(
        points=out_points,
        colors=out_colors,
        kept_mask=kept_mask,
        n_components=len(components),
        kept_component_sizes=tuple(sorted(kept_sizes, reverse=True)),
        voxel_size=float(voxel_size),
    )
