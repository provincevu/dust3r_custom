from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ViewConsistentMergeResult:
    points: np.ndarray
    colors: Optional[np.ndarray]
    weights: np.ndarray
    counts: np.ndarray
    n_input: int
    n_output: int
    voxel_size: float


def view_consistent_merge(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    *,
    enabled: bool = True,
    voxel_size: float = 0.005,
) -> ViewConsistentMergeResult:
    """Merge near-duplicate points from multiple views into one surface layer.

    Points are grouped by voxel index in world space. Each voxel outputs one
    representative point using weighted averaging.
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

    if weights is not None:
        w = np.asarray(weights).reshape(-1)
        if w.shape[0] != pts.shape[0]:
            raise ValueError("weights must be (N,) and match points")
        w = np.clip(w.astype(np.float64), 1e-12, None)
    else:
        w = np.ones((len(pts),), dtype=np.float64)

    if not enabled or len(pts) == 0:
        return ViewConsistentMergeResult(
            points=pts,
            colors=cols,
            weights=w,
            counts=np.ones((len(pts),), dtype=np.int32),
            n_input=int(len(pts)),
            n_output=int(len(pts)),
            voxel_size=float(voxel_size),
        )

    vox = float(max(voxel_size, 1e-8))
    base = np.min(pts.astype(np.float64), axis=0)
    key = np.floor((pts.astype(np.float64) - base[None, :]) / vox).astype(np.int64)

    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    m = uniq.shape[0]

    sum_w = np.zeros((m,), dtype=np.float64)
    np.add.at(sum_w, inv, w)

    out_pts = np.zeros((m, 3), dtype=np.float64)
    for c in range(3):
        acc = np.zeros((m,), dtype=np.float64)
        np.add.at(acc, inv, pts[:, c].astype(np.float64) * w)
        out_pts[:, c] = acc / np.maximum(sum_w, 1e-12)

    out_cols = None
    if cols is not None:
        out_cols = np.zeros((m, 3), dtype=np.float64)
        for c in range(3):
            acc = np.zeros((m,), dtype=np.float64)
            np.add.at(acc, inv, cols[:, c].astype(np.float64) * w)
            out_cols[:, c] = acc / np.maximum(sum_w, 1e-12)
        # keep original color range convention
        if np.issubdtype(cols.dtype, np.integer):
            out_cols = np.clip(np.rint(out_cols), 0, 255).astype(cols.dtype)
        else:
            out_cols = out_cols.astype(cols.dtype)

    counts = np.bincount(inv, minlength=m).astype(np.int32)

    return ViewConsistentMergeResult(
        points=out_pts.astype(pts.dtype, copy=False),
        colors=out_cols,
        weights=sum_w,
        counts=counts,
        n_input=int(len(pts)),
        n_output=int(m),
        voxel_size=vox,
    )
