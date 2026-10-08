"""Entropy estimators with explicitly declared random variables (protocol §4).

All reported entropies are base-2 (bits per observation of the named variable).
``X_{F,A} = Q(F(U))`` with ``U`` uniform over valid pixels of region ``A``;
``H_F(A)`` is the plug-in entropy of its empirical distribution.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LN2 = np.log(2.0)


def entropy_bits(counts: np.ndarray) -> float:
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum()
    if n <= 0:
        return float("nan")
    p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum())


def entropy_nats(counts: np.ndarray) -> float:
    return entropy_bits(counts) * LN2


def miller_madow_bits(counts: np.ndarray) -> float:
    """Sensitivity estimate H_ML + (m_occupied - 1) / (2 n ln 2); not clamped."""
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum()
    if n <= 0:
        return float("nan")
    m = int((c > 0).sum())
    return entropy_bits(c) + (m - 1) / (2.0 * n * LN2)


def hist_counts(codes: np.ndarray, nbins: int, weights: np.ndarray | None = None) -> np.ndarray:
    w = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    return np.bincount(np.asarray(codes, dtype=np.int64).ravel(), weights=w, minlength=nbins)[:nbins]


def feature_entropy_from_codes(codes: np.ndarray, nbins: int, weights: np.ndarray | None = None) -> dict[str, Any]:
    cnt = hist_counts(codes, nbins, weights)
    return {
        "H": entropy_bits(cnt),
        "H_mm": miller_madow_bits(cnt) if weights is None else float("nan"),
        "n": int(cnt.sum()) if weights is None else float(cnt.sum()),
        "occupancy": int((cnt > 0).sum()),
        "nbins": nbins,
    }


# ----------------------------------------------------------------------------
# Binary pipeline
# ----------------------------------------------------------------------------
def otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """Otsu threshold on values in [0, 1] (whole-face, one per image)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.5
    hist, edges = np.histogram(v, bins=nbins, range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(np.float64)
    total = w.sum()
    if total == 0:
        return 0.5
    cw = np.cumsum(w)
    cm = np.cumsum(w * centers)
    mu_t = cm[-1] / total
    w0 = cw
    w1 = total - cw
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = cm / w0
        mu1 = (cm[-1] - cm) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = -1
    k = int(np.argmax(between))
    # threshold at the upper edge of bin k: B = 1{I >= t}
    return float(edges[k + 1])


def binarize(i_w: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(i_w) >= t


def block_codes(B: np.ndarray, rotation: int = 0) -> np.ndarray:
    """C4 = B00 + 2 B01 + 4 B10 + 8 B11 at each anchor (shape (H-1, W-1)); rotation rotates the block."""
    b = B.astype(np.int64)
    blocks = [b[:-1, :-1], b[:-1, 1:], b[1:, :-1], b[1:, 1:]]  # B00, B01, B10, B11
    if rotation:
        # rotate block positions: 90 deg rotations of the 2x2 (00,01,10,11) -> (10,00,11,01) etc.
        order = [[0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]][rotation % 4]
        blocks = [blocks[i] for i in order]
    return blocks[0] + 2 * blocks[1] + 4 * blocks[2] + 8 * blocks[3]


def binary_entropy_p(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def marginal_binary_entropy(B: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> dict[str, Any]:
    p = float(B[ys, xs].mean())
    return {"H_marginal": binary_entropy_p(p), "p": p, "n": int(len(ys))}


def block_entropies(C4: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> dict[str, Any]:
    """H4, H3 (from the same anchors) and Hcond = H4 - H3 = H(B11 | B00, B01, B10)."""
    c4 = C4[ys, xs]
    c3 = c4 & 7
    h4 = entropy_bits(np.bincount(c4, minlength=16))
    h3 = entropy_bits(np.bincount(c3, minlength=8))
    return {"H4": h4, "H3": h3, "H_cond": h4 - h3, "n": int(len(ys)), "occupancy4": int((np.bincount(c4, minlength=16) > 0).sum())}


# ----------------------------------------------------------------------------
# Local neighbourhood maps via integral images
# ----------------------------------------------------------------------------
def _box_sums(ind: np.ndarray, side: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Sum of ``ind`` (H, W[, K]) over the side x side window centred at (y, x) for grid points."""
    r = side // 2
    H, W = ind.shape[:2]
    pad = np.zeros((H + 1, W + 1) + ind.shape[2:], dtype=np.float64)
    pad[1:, 1:] = np.cumsum(np.cumsum(ind, axis=0), axis=1)
    y0 = np.clip(ys - r, 0, H)
    y1 = np.clip(ys + r + 1, 0, H)
    x0 = np.clip(xs - r, 0, W)
    x1 = np.clip(xs + r + 1, 0, W)
    return pad[y1[:, None], x1[None, :]] - pad[y0[:, None], x1[None, :]] - pad[y1[:, None], x0[None, :]] + pad[y0[:, None], x0[None, :]]


def grid_points(H: int, W: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    off = stride // 2
    return np.arange(off, H, stride), np.arange(off, W, stride)


def local_entropy_map(
    codes: np.ndarray,
    valid: np.ndarray,
    nbins: int,
    side: int,
    stride: int,
    min_support: float = 0.9,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Neighbourhood entropy e_F(u) = H(Q(F(U)) | U uniform in N_r(u) ∩ valid) on a stride grid.

    Stored at native grid positions; NaN where valid support < min_support * side^2.
    """
    H, W = codes.shape
    ys, xs = grid_points(H, W, stride)
    onehot = np.zeros((H, W, nbins), dtype=np.float64)
    c = np.clip(codes.astype(np.int64), 0, nbins - 1)
    onehot[np.arange(H)[:, None], np.arange(W)[None, :], c] = 1.0
    onehot *= valid[..., None]
    if weights is not None:
        onehot *= weights[..., None]
    counts = _box_sums(onehot, side, ys, xs)  # (ny, nx, nbins)
    support = _box_sums(valid.astype(np.float64), side, ys, xs)
    tot = counts.sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = counts / tot[..., None]
        ent = -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=2)
    ok = support >= min_support * side * side
    ent = np.where(ok, ent, np.nan)
    return {"map": ent, "support": support, "ys": ys, "xs": xs, "side": side, "stride": stride, "nbins": nbins}


def local_binary_cond_map(B: np.ndarray, valid: np.ndarray, side: int, stride: int, min_support: float = 0.9) -> dict[str, Any]:
    """Local H(C4) - H(C3) over 2x2 anchors whose four pixels are valid."""
    C4 = block_codes(B)
    v4 = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    m4 = local_entropy_map(C4, v4, 16, side, stride, min_support)
    m3 = local_entropy_map(C4 & 7, v4, 8, side, stride, min_support)
    out = dict(m4)
    out["map"] = m4["map"] - m3["map"]
    out["map_H4"] = m4["map"]
    return out


def local_marginal_map(B: np.ndarray, valid: np.ndarray, side: int, stride: int, min_support: float = 0.9) -> dict[str, Any]:
    return local_entropy_map(B.astype(np.int64), valid, 2, side, stride, min_support)


def map_to_full(m: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour expansion of a stride map to full resolution (display only)."""
    H, W = shape
    out = np.full((H, W), np.nan)
    ys, xs, stride = m["ys"], m["xs"], m["stride"]
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            out[max(y - stride // 2, 0) : y + stride // 2 + 1, max(x - stride // 2, 0) : x + stride // 2 + 1] = m["map"][i, j]
    return out
