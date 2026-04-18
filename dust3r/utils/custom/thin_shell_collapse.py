from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ThinShellCollapseResult:
    points: np.ndarray
    colors: Optional[np.ndarray]
    n_input: int
    n_output: int
    spacing: float
    normal_radius: float


def _estimate_spacing(points: np.ndarray, max_samples: int = 2048, rng_seed: int = 0) -> float:
    """Estimate local spacing using median 1-NN distance on a random subset."""
    n = int(points.shape[0])
    if n <= 1:
        return 0.0

    rng = np.random.default_rng(rng_seed)
    m = int(min(max_samples, n))
    sample_idx = rng.choice(n, size=m, replace=False)
    sample_pts = points[sample_idx]

    tree = cKDTree(points)
    # k=2 because nearest neighbor of a point is itself
    dists, _ = tree.query(sample_pts, k=2)
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return 0.0

    return float(np.median(nn))


def _cluster_1d(values: np.ndarray, gap_thr: float) -> list[np.ndarray]:
    """Cluster sorted 1D values by split-on-gap."""
    if values.size == 0:
        return []

    order = np.argsort(values)
    v = values[order]
    gaps = np.diff(v)
    split_idx = np.where(gaps > gap_thr)[0]

    clusters: list[np.ndarray] = []
    start = 0
    for s in split_idx:
        clusters.append(order[start : s + 1])
        start = s + 1
    clusters.append(order[start:])
    return clusters


def thin_shell_collapse(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    *,
    enabled: bool = True,
    seed_stride: int = 6,
    normal_radius_mult: float = 4.0,
    cluster_gap_mult: float = 1.5,
    layer_select_mode: str = "propagate",
    seed_neighbor_k: int = 8,
    continuity_weight: float = 1.0,
    roughness_weight: float = 0.35,
    min_neighbors: int = 16,
    max_neighbors: int = 256,
    rng_seed: int = 0,
) -> ThinShellCollapseResult:
    """Collapse overlapping thin layers into roughly one layer on sampled seeds.

    Pipeline:
    1) estimate spacing from 1-NN distances,
    2) estimate seed normal from local PCA,
    3) project neighbors to normal axis,
    4) split 1D values into clusters by gap,
    5) choose cluster either by roughness-only or by graph propagation,
    6) project seed to selected cluster median.

    `layer_select_mode`:
    - "roughness": independent per-seed smallest roughness.
    - "propagate": branch-consistent selection propagated on seed kNN graph.
    """
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3)")

    if colors is not None:
        cols = np.asarray(colors)
        if cols.ndim != 2 or cols.shape[1] != 3 or cols.shape[0] != pts.shape[0]:
            raise ValueError("colors must be (N, 3) and match points")
    else:
        cols = None

    n = int(pts.shape[0])
    if not enabled or n == 0:
        return ThinShellCollapseResult(
            points=pts,
            colors=cols,
            n_input=n,
            n_output=n,
            spacing=0.0,
            normal_radius=0.0,
        )

    stride = int(max(1, seed_stride))
    min_nb = int(max(8, min_neighbors))
    max_nb = int(max(min_nb, max_neighbors))

    spacing = _estimate_spacing(pts, rng_seed=rng_seed)
    if spacing <= 0.0:
        return ThinShellCollapseResult(
            points=pts[::stride].copy(),
            colors=(cols[::stride].copy() if cols is not None else None),
            n_input=n,
            n_output=int(np.ceil(n / stride)),
            spacing=0.0,
            normal_radius=0.0,
        )

    radius = float(max(1e-8, normal_radius_mult * spacing))
    gap_thr = float(max(1e-8, cluster_gap_mult * spacing))

    tree = cKDTree(pts)
    seed_ids = np.arange(0, n, stride, dtype=np.int64)

    out_pts = np.empty((len(seed_ids), 3), dtype=pts.dtype)
    out_cols = np.empty((len(seed_ids), 3), dtype=cols.dtype) if cols is not None else None

    candidate_t: list[np.ndarray] = []
    candidate_rough: list[np.ndarray] = []
    candidate_size: list[np.ndarray] = []
    seed_normals = np.empty((len(seed_ids), 3), dtype=np.float64)
    seed_points = pts[seed_ids].astype(np.float64, copy=False)
    seed_nb_count = np.zeros((len(seed_ids),), dtype=np.int64)

    for oi, si in enumerate(seed_ids):
        p0 = pts[si]

        nb = tree.query_ball_point(p0, r=radius)
        if len(nb) < min_nb:
            # fallback to knn if fixed-radius neighborhood is too sparse
            k = min(max_nb, n)
            _, idx = tree.query(p0, k=k)
            nb_idx = np.atleast_1d(idx)
        else:
            nb_idx = np.asarray(nb, dtype=np.int64)
            if nb_idx.size > max_nb:
                # deterministic truncation by distance
                d2 = np.sum((pts[nb_idx] - p0[None, :]) ** 2, axis=1)
                order = np.argsort(d2)
                nb_idx = nb_idx[order[:max_nb]]

        seed_nb_count[oi] = int(nb_idx.size)

        local = pts[nb_idx]
        centered = local - local.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(1, centered.shape[0] - 1)
        evals, evecs = np.linalg.eigh(cov)
        nrm = evecs[:, int(np.argmin(evals))]

        # orientation consistency: point normal roughly outward from local centroid
        if np.dot(nrm, p0 - local.mean(axis=0)) < 0:
            nrm = -nrm

        seed_normals[oi] = nrm

        t = (local - p0[None, :]) @ nrm
        clusters = _cluster_1d(t, gap_thr=gap_thr)
        if not clusters:
            candidate_t.append(np.asarray([0.0], dtype=np.float64))
            candidate_rough.append(np.asarray([0.0], dtype=np.float64))
            candidate_size.append(np.asarray([1.0], dtype=np.float64))
            continue

        meds = []
        roughs = []
        sizes = []
        for c in clusters:
            tc = t[c]
            med = float(np.median(tc))
            rough = float(np.median(np.abs(tc - med)))
            size = float(c.size)
            meds.append(med)
            roughs.append(rough)
            sizes.append(size)

        candidate_t.append(np.asarray(meds, dtype=np.float64))
        candidate_rough.append(np.asarray(roughs, dtype=np.float64))
        candidate_size.append(np.asarray(sizes, dtype=np.float64))

    mode = str(layer_select_mode).strip().lower()
    chosen_t = np.zeros((len(seed_ids),), dtype=np.float64)

    if mode == "propagate" and len(seed_ids) > 1:
        k = int(max(1, seed_neighbor_k))
        k = int(min(k + 1, len(seed_ids)))
        seed_tree = cKDTree(seed_points)
        _, knn_idx = seed_tree.query(seed_points, k=k)
        knn_idx = np.atleast_2d(knn_idx)

        assigned = np.zeros((len(seed_ids),), dtype=bool)
        # Use most stable seed (largest neighborhood) as root.
        root = int(np.argmax(seed_nb_count))

        root_t = candidate_t[root]
        root_r = candidate_rough[root]
        root_idx = int(np.argmin(np.abs(root_t) + float(max(0.0, roughness_weight)) * root_r))
        chosen_t[root] = float(root_t[root_idx])
        assigned[root] = True

        queue = [root]
        cont_w = float(max(0.0, continuity_weight))
        rough_w = float(max(0.0, roughness_weight))

        while queue:
            cur = queue.pop(0)
            cur_t = chosen_t[cur]
            for nb in np.atleast_1d(knn_idx[cur])[1:]:
                j = int(nb)
                if assigned[j]:
                    continue

                tj = candidate_t[j]
                rj = candidate_rough[j]
                sj = candidate_size[j]
                smax = float(max(1.0, np.max(sj)))

                cost = cont_w * np.abs(tj - cur_t) + rough_w * rj / max(spacing, 1e-8) - 0.05 * (sj / smax)
                pick = int(np.argmin(cost))
                chosen_t[j] = float(tj[pick])
                assigned[j] = True
                queue.append(j)

        # Disconnected islands: fallback to roughness selection.
        for i in range(len(seed_ids)):
            if assigned[i]:
                continue
            ti = candidate_t[i]
            ri = candidate_rough[i]
            si = candidate_size[i]
            pick = int(np.argmin(ri - 0.05 * (si / max(1.0, np.max(si)))))
            chosen_t[i] = float(ti[pick])
    else:
        # Independent per-seed selection (original behavior).
        for i in range(len(seed_ids)):
            ti = candidate_t[i]
            ri = candidate_rough[i]
            si = candidate_size[i]
            pick = int(np.argmin(ri - 0.05 * (si / max(1.0, np.max(si)))))
            chosen_t[i] = float(ti[pick])

    for oi, si in enumerate(seed_ids):
        p0 = pts[si]
        nrm = seed_normals[oi]
        out_pts[oi] = p0 + chosen_t[oi] * nrm
        if out_cols is not None:
            out_cols[oi] = cols[si]

    return ThinShellCollapseResult(
        points=out_pts,
        colors=out_cols,
        n_input=n,
        n_output=int(out_pts.shape[0]),
        spacing=float(spacing),
        normal_radius=float(radius),
    )
