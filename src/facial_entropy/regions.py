"""Prespecified comparison patches, secondary regions, atlas masks and sampling rules."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation

from facial_entropy.alignment import face_axes, polygon_mask
from facial_entropy.util import rng_for

CENTRAL_PATCHES = ("eye_inner_left", "eye_inner_right", "nose_bridge", "nose_tip")
PERIPHERAL_PATCHES = ("cheek_left", "cheek_right", "forehead_left", "forehead_right")
PATCH_NAMES = CENTRAL_PATCHES + PERIPHERAL_PATCHES
PATCH_CLASS = {**{p: "central" for p in CENTRAL_PATCHES}, **{p: "peripheral" for p in PERIPHERAL_PATCHES}}


def patch_centers(lm: np.ndarray) -> dict[str, np.ndarray]:
    """Patch centre rules from protocol §5.4 (iBUG indices, image-side left/right)."""
    lm = np.asarray(lm, dtype=np.float64)
    d, v, _ = face_axes(lm)
    return {
        "eye_inner_left": lm[39],
        "eye_inner_right": lm[42],
        "nose_bridge": 0.5 * (lm[27] + lm[28]),
        "nose_tip": lm[30],
        "cheek_left": 0.25 * (lm[2] + lm[3] + lm[31] + lm[48]),
        "cheek_right": 0.25 * (lm[13] + lm[14] + lm[35] + lm[54]),
        "forehead_left": lm[19] - 0.45 * d * v,
        "forehead_right": lm[24] - 0.45 * d * v,
    }


def round_anchor(c: np.ndarray, size: int) -> tuple[int, int]:
    """Rounding rule: anchor = floor(centre + 0.5) - size // 2, returned as (y0, x0)."""
    cx, cy = float(c[0]), float(c[1])
    return int(np.floor(cy + 0.5)) - size // 2, int(np.floor(cx + 0.5)) - size // 2


def patch_anchors(centers: dict[str, np.ndarray], size: int) -> dict[str, tuple[int, int]]:
    return {k: round_anchor(c, size) for k, c in centers.items()}


def patch_slice(anchor: tuple[int, int], size: int) -> tuple[slice, slice]:
    y0, x0 = anchor
    return slice(y0, y0 + size), slice(x0, x0 + size)


def patch_mask(anchor: tuple[int, int], size: int, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    y0, x0 = anchor
    ys, xs = slice(max(y0, 0), min(y0 + size, shape[0])), slice(max(x0, 0), min(x0 + size, shape[1]))
    m[ys, xs] = True
    return m


def patch_validity(valid: np.ndarray, anchors: dict[str, tuple[int, int]], size: int) -> dict[str, float]:
    out = {}
    for k, a in anchors.items():
        m = patch_mask(a, size, valid.shape)
        out[k] = float((valid & m).sum() / (size * size))
    return out


def patch_overlaps(anchors: dict[str, tuple[int, int]], size: int, shape: tuple[int, int]) -> dict[str, Any]:
    masks = {k: patch_mask(a, size, shape) for k, a in anchors.items()}
    pairs = {}
    cross = False
    for i, a in enumerate(PATCH_NAMES):
        for b in PATCH_NAMES[i + 1 :]:
            frac = float((masks[a] & masks[b]).sum() / (size * size))
            if frac > 0:
                pairs[f"{a}|{b}"] = frac
                if PATCH_CLASS[a] != PATCH_CLASS[b]:
                    cross = True
    return {"central_peripheral_overlap": cross, "pair_overlap_fraction": pairs}


def sample_positions(valid: np.ndarray, sl: tuple[slice, slice], n: int, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Seeded sample of ``n`` valid positions (without replacement) inside a patch; None if too few."""
    sub = valid[sl]
    ys, xs = np.nonzero(sub)
    if len(ys) < n:
        return None
    idx = rng_for("patch-sample", base=seed).choice(len(ys), size=n, replace=False)
    return ys[idx] + sl[0].start, xs[idx] + sl[1].start


def block_anchor_positions(valid: np.ndarray, sl: tuple[slice, slice]) -> tuple[np.ndarray, np.ndarray]:
    """All 2x2 block anchors inside the patch whose four pixels are valid."""
    sub = valid[sl]
    ok = sub[:-1, :-1] & sub[:-1, 1:] & sub[1:, :-1] & sub[1:, 1:]
    ys, xs = np.nonzero(ok)
    return ys + sl[0].start, xs + sl[1].start


def sample_block_anchors(valid: np.ndarray, sl: tuple[slice, slice], n: int, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    ys, xs = block_anchor_positions(valid, sl)
    if len(ys) < n:
        return None
    idx = rng_for("block-sample", base=seed).choice(len(ys), size=n, replace=False)
    return ys[idx], xs[idx]


# ----------------------------------------------------------------------------
# Secondary regions and atlas masks
# ----------------------------------------------------------------------------
def secondary_region_masks(lm: np.ndarray, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    lm = np.asarray(lm, dtype=np.float64)
    d, v, _ = face_axes(lm)
    hull_eye_l = _hull(lm[36:42])
    hull_eye_r = _hull(lm[42:48])
    out = {
        "mouth": polygon_mask(_hull(lm[48:60]), shape),
        "eye_left_whole": binary_dilation(polygon_mask(hull_eye_l, shape), iterations=4),
        "eye_right_whole": binary_dilation(polygon_mask(hull_eye_r, shape), iterations=4),
        "jaw": _band_below(lm, shape, d, v),
    }
    return out


def _hull(pts: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return pts
    h = ConvexHull(pts)
    return pts[h.vertices]


def _band_below(lm: np.ndarray, shape: tuple[int, int], d: float, v: np.ndarray) -> np.ndarray:
    # jaw band: polygon between the jaw line (points 3..13) and the same line shifted
    # 0.15 d toward the eyes.
    jaw = lm[3:14]
    inner = (jaw - 0.15 * d * v)[::-1]
    return polygon_mask(np.vstack([jaw, inner]), shape)


def central_union_mask(anchors: dict[str, tuple[int, int]], size: int, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for k in CENTRAL_PATCHES:
        m |= patch_mask(anchors[k], size, shape)
    return m


def features_union_mask(lm: np.ndarray, shape: tuple[int, int], dilate_px: int = 8) -> np.ndarray:
    """Prespecified union-of-features mask: convex hull of eyes+nose landmarks, dilated."""
    pts = np.vstack([np.asarray(lm)[27:48]])
    m = polygon_mask(_hull(pts), shape)
    return binary_dilation(m, iterations=dilate_px)
