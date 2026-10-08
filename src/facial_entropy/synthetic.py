"""Synthetic pseudo-faces with explicitly supplied 68-point landmarks (smoke profile + controls).

These are NOT face data and never enter the real-data manifest.  They exist to
exercise the full orchestration offline and to run the ROI-translation control.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from facial_entropy.config import ALIGNED_SIZE
from facial_entropy.util import rng_for


def template_landmarks(size: int = ALIGNED_SIZE) -> np.ndarray:
    """A canonical 68-point layout in the aligned frame (eyes at (84,100)/(172,100), mouth (128,184))."""
    s = size / ALIGNED_SIZE
    pts = np.zeros((68, 2))
    # jaw 0..16: lower half-ellipse from image-left temple to image-right temple
    t = np.linspace(np.pi, 2 * np.pi, 17)
    pts[0:17, 0] = 128 + 92 * np.cos(t)
    pts[0:17, 1] = 110 - 118 * np.sin(t)
    # brows 17..21 (left), 22..26 (right)
    for k, cx in ((17, 84), (22, 172)):
        xs = np.linspace(cx - 24, cx + 24, 5)
        pts[k : k + 5, 0] = xs
        pts[k : k + 5, 1] = 78 - 6 * np.cos(np.linspace(-np.pi / 2, np.pi / 2, 5))
    # nose bridge 27..30, nostrils 31..35
    pts[27:31, 0] = 128
    pts[27:31, 1] = np.linspace(100, 148, 4)
    pts[31:36, 0] = np.linspace(112, 144, 5)
    pts[31:36, 1] = [152, 156, 158, 156, 152]
    # eyes 36..41 (left), 42..47 (right)
    for k, cx in ((36, 84), (42, 172)):
        ang = np.array([180, 135, 45, 0, -45, -135]) * np.pi / 180
        pts[k : k + 6, 0] = cx + 15 * np.cos(ang)
        pts[k : k + 6, 1] = 100 - 6 * np.sin(ang)
    # mouth outer 48..59, inner 60..67
    ang = np.linspace(np.pi, -np.pi, 13)[:-1]
    pts[48:60, 0] = 128 + 26 * np.cos(ang)
    pts[48:60, 1] = 184 - 11 * np.sin(ang)
    ang = np.linspace(np.pi, -np.pi, 9)[:-1]
    pts[60:68, 0] = 128 + 17 * np.cos(ang)
    pts[60:68, 1] = 184 - 5 * np.sin(ang)
    return pts * s


def _ellipse(shape: tuple[int, int], cx: float, cy: float, rx: float, ry: float, angle: float = 0.0) -> np.ndarray:
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]
    dx, dy = xs - cx, ys - cy
    ca, sa = np.cos(angle), np.sin(angle)
    u = (dx * ca + dy * sa) / rx
    v = (-dx * sa + dy * ca) / ry
    return (u * u + v * v) <= 1.0


def render_pseudo_face(
    lm: np.ndarray,
    size: int,
    seed: int,
    feature_offset: tuple[float, float] = (0.0, 0.0),
    texture: float = 0.02,
) -> np.ndarray:
    """Smooth skin shading + drawn features at ``lm + offset``; returns float32 RGB in [0, 1]."""
    rng = rng_for("pseudo-face", seed)
    H = W = size
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = W / 2, H * 0.52
    # ellipsoidal shading (bright centre, darker rim) + lateral light
    r2 = ((xs - cx) / (0.42 * W)) ** 2 + ((ys - cy) / (0.55 * H)) ** 2
    base = 0.72 - 0.28 * np.clip(r2, 0, 1.5) + 0.05 * (xs - cx) / W
    skin = np.stack([base * 1.00, base * 0.82, base * 0.70], axis=-1)
    skin += texture * rng.standard_normal((H, W, 1))
    skin = gaussian_filter(skin, sigma=(0.7, 0.7, 0))
    off = np.array(feature_offset)
    p = np.asarray(lm) + off
    img = skin.copy()
    # eyes: dark ellipse + bright highlight
    for k in (36, 42):
        c = p[k : k + 6].mean(axis=0)
        m = _ellipse((H, W), c[0], c[1], 14 * size / 256, 6 * size / 256)
        img[m] *= 0.35
        hl = _ellipse((H, W), c[0] + 3, c[1] - 1, 2.5 * size / 256, 2.0 * size / 256)
        img[hl] = 0.95
    # brows
    for k in (17, 22):
        for q in range(4):
            a, b = p[k + q], p[k + q + 1]
            for tt in np.linspace(0, 1, 12):
                pt = a + tt * (b - a)
                img[_ellipse((H, W), pt[0], pt[1], 3 * size / 256, 2 * size / 256)] *= 0.55
    # nostrils + nose shading
    for k in (31, 35):
        img[_ellipse((H, W), p[k][0], p[k][1], 3 * size / 256, 2.5 * size / 256)] *= 0.45
    img[_ellipse((H, W), p[30][0] - 8 * size / 256, p[30][1] - 10 * size / 256, 6 * size / 256, 22 * size / 256, 0.2)] *= 0.88
    # mouth
    c = p[48:60].mean(axis=0)
    img[_ellipse((H, W), c[0], c[1], 26 * size / 256, 9 * size / 256)] *= 0.6
    img = np.clip(gaussian_filter(img, sigma=(0.5, 0.5, 0)), 0, 1)
    return img.astype(np.float32)


def synthetic_cohort(n: int, seed: int, size: int = ALIGNED_SIZE) -> list[dict]:
    """n pseudo-subjects with small per-subject landmark jitter and feature-level variation."""
    out = []
    base = template_landmarks(size)
    for i in range(n):
        rng = rng_for("synthetic-subject", i, base=seed)
        lm = base + rng.normal(0, 1.5 * size / 256, size=base.shape)
        rgb = render_pseudo_face(lm, size, seed=seed * 1000 + i, texture=float(rng.uniform(0.01, 0.04)))
        out.append({"image_id": f"synth{i:03d}", "subject_id": f"synth{i:03d}", "rgb": rgb, "landmarks": lm})
    return out
