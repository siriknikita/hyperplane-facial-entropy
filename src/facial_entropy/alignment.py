"""Landmarks (68-point iBUG schema via face-alignment FAN), similarity alignment, face masks.

Landmark adapter
----------------
``face_alignment.FaceAlignment(LandmarksType.TWO_D, face_detector='sfd')`` —
2D-FAN (4 stacked hourglasses, Bulat & Tzimiropoulos 2017) with the S3FD
detector.  Both checkpoints are downloaded on first use from adrianbulat.com
into ``torch.hub`` checkpoints.  This project treats them as *opt-in assets*:
without ``--download`` their absence is reported, never fetched silently.

Point schema (zero-indexed): 0-16 jaw, 17-21 image-left brow, 22-26
image-right brow, 27-30 nose bridge, 31-35 nostrils, 36-41 image-left eye,
42-47 image-right eye, 48-67 mouth.  "Left"/"right" are image-side labels in
the iBUG convention (the subject's right eye appears on the image left).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import binary_erosion, map_coordinates

from facial_entropy.config import ALIGNED_SIZE, TARGET_EYE_LEFT, TARGET_EYE_RIGHT, TARGET_MOUTH

LANDMARK_ASSETS = {
    "2DFAN4-11f355bf06.pth.tar": ("https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar", 93_000_000),
    "s3fd-619a316812.pth": ("https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth", 90_000_000),
}


class MissingAssetError(RuntimeError):
    pass


def torch_checkpoint_dir() -> Path:
    import torch  # noqa: PLC0415

    return Path(torch.hub.get_dir()) / "checkpoints"


def landmark_asset_status() -> dict[str, Any]:
    ckpt = torch_checkpoint_dir()
    present, missing, est = {}, {}, 0
    for name, (url, size) in LANDMARK_ASSETS.items():
        p = ckpt / name
        if p.exists():
            present[name] = {"path": str(p), "bytes": p.stat().st_size}
        else:
            missing[name] = {"url": url, "estimated_bytes": size}
            est += size
    try:
        import face_alignment  # noqa: PLC0415

        fa_version = face_alignment.__version__
    except ImportError:
        fa_version = None
    return {
        "face_alignment_version": fa_version,
        "checkpoint_dir": str(ckpt),
        "present": present,
        "missing": missing,
        "estimated_download_bytes": est,
    }


class LandmarkAdapter:
    """68-point landmarks on uint8/float RGB images; largest detected face is used."""

    def __init__(self, device: str = "cpu", allow_download: bool = False) -> None:
        status = landmark_asset_status()
        if status["face_alignment_version"] is None:
            raise MissingAssetError("face-alignment is not installed: `uv sync --extra facial-entropy`")
        if status["missing"] and not allow_download:
            lines = [f"  {k}: {v['url']} (~{v['estimated_bytes']/1e6:.0f} MB)" for k, v in status["missing"].items()]
            raise MissingAssetError(
                "landmark model weights are missing and --download was not given:\n" + "\n".join(lines)
                + f"\nThey would be stored in {status['checkpoint_dir']} (total ~{status['estimated_download_bytes']/1e6:.0f} MB)."
            )
        import face_alignment  # noqa: PLC0415

        self.device = device
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, device=device, face_detector="sfd", flip_input=True, compile=False
        )
        self.version = {
            "library": "face-alignment",
            "version": face_alignment.__version__,
            "network": "2DFAN-4",
            "detector": "s3fd",
            "schema": "iBUG-68 (zero-indexed)",
            "flip_input": True,
            "device": device,
            "assets": landmark_asset_status()["present"],
        }

    def landmarks(self, rgb: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
        img = rgb
        if img.dtype != np.uint8:
            img = np.clip(np.asarray(rgb, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        out = self.fa.get_landmarks_from_image(img, return_bboxes=True, return_landmark_score=True)
        lms, scores, boxes = out
        if lms is None or len(lms) == 0:
            return None, {"n_faces": 0}
        areas = [float((b[2] - b[0]) * (b[3] - b[1])) for b in boxes]
        i = int(np.argmax(areas))
        return np.asarray(lms[i], dtype=np.float64)[:, :2], {
            "n_faces": len(lms),
            "bbox": [float(v) for v in boxes[i][:4]],
            "mean_score": float(np.mean(scores[i])),
            "min_score": float(np.min(scores[i])),
        }


# ----------------------------------------------------------------------------
# Similarity transform
# ----------------------------------------------------------------------------
def key_points_from_68(lm: np.ndarray) -> dict[str, np.ndarray]:
    lm = np.asarray(lm, dtype=np.float64)
    return {
        "eye_left": lm[36:42].mean(axis=0),
        "eye_right": lm[42:48].mean(axis=0),
        "mouth": lm[48:68].mean(axis=0),
    }


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares similarity (rotation + uniform scale + translation), no reflection.

    Parametrised as x' = a x - b y + tx, y' = b x + a y + ty.  Returns the 2x3
    matrix ``A`` with ``dst ~ A @ [x, y, 1]`` and the RMS fit residual in pixels.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    M = np.zeros((2 * n, 4))
    rhs = np.zeros(2 * n)
    M[0::2, 0] = src[:, 0]
    M[0::2, 1] = -src[:, 1]
    M[0::2, 2] = 1.0
    M[1::2, 0] = src[:, 1]
    M[1::2, 1] = src[:, 0]
    M[1::2, 3] = 1.0
    rhs[0::2] = dst[:, 0]
    rhs[1::2] = dst[:, 1]
    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    a, b, tx, ty = sol
    A = np.array([[a, -b, tx], [b, a, ty]])
    res = apply_affine(A, src) - dst
    rms = float(np.sqrt((res**2).sum(axis=1).mean()))
    return A, rms


def apply_affine(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ A[:, :2].T + A[:, 2]


def invert_affine(A: np.ndarray) -> np.ndarray:
    R = A[:, :2]
    t = A[:, 2]
    Ri = np.linalg.inv(R)
    return np.hstack([Ri, (-Ri @ t)[:, None]])


def warp_rgb(rgb: np.ndarray, A: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``rgb`` (H, W, 3) into the aligned frame with bilinear interpolation.

    Pixel index coordinates are used throughout: aligned pixel (u, v) samples the
    source at ``A^{-1} (u, v)``.  Returns the aligned image and an in-bounds mask.
    """
    Ai = invert_affine(A)
    us, vs = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
    src = apply_affine(Ai, np.stack([us.ravel(), vs.ravel()], axis=1))
    xs, ys = src[:, 0].reshape(size, size), src[:, 1].reshape(size, size)
    H, W = rgb.shape[:2]
    inb = (xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)
    out = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        out[..., c] = map_coordinates(np.asarray(rgb[..., c], dtype=np.float64), [ys, xs], order=1, mode="nearest")
    return out, inb


def canonical_targets() -> np.ndarray:
    return np.array([TARGET_EYE_LEFT, TARGET_EYE_RIGHT, TARGET_MOUTH], dtype=np.float64)


# ----------------------------------------------------------------------------
# Masks
# ----------------------------------------------------------------------------
def polygon_mask(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    inside = MplPath(np.asarray(points, dtype=np.float64)).contains_points(np.stack([xs.ravel(), ys.ravel()], axis=1))
    return inside.reshape(h, w)


def face_axes(lm: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Inter-eye distance d, unit vector v from eye midpoint toward mouth centre, eye midpoint."""
    kp = key_points_from_68(lm)
    d = float(np.linalg.norm(kp["eye_right"] - kp["eye_left"]))
    mid = 0.5 * (kp["eye_left"] + kp["eye_right"])
    v = kp["mouth"] - mid
    v = v / max(np.linalg.norm(v), 1e-9)
    return d, v, mid


def face_polygon_mask(lm: np.ndarray, shape: tuple[int, int], forehead_extension: float = 0.80) -> np.ndarray:
    """Landmark-polygon visible-face approximation (NOT a parsed skin mask).

    Polygon = jaw points 0..16 followed by the brow points 26..17 displaced by
    ``-forehead_extension * d * v`` (toward the forehead).  Eyes, brows,
    nostrils and lips stay inside the support.  Ears are outside the jaw line.
    Hair at the forehead edge may leak in; this is reported as a limitation and
    inspected in QA overlays.
    """
    lm = np.asarray(lm, dtype=np.float64)
    d, v, _ = face_axes(lm)
    brows = lm[17:27][::-1] - forehead_extension * d * v
    poly = np.vstack([lm[0:17], brows])
    return polygon_mask(poly, shape)


def erode_support(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode by a (2r+1) square and remove an r-pixel image border (derivative validity)."""
    if radius <= 0:
        return mask.copy()
    st = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    out = binary_erosion(mask, structure=st, border_value=0)
    out[:radius, :] = False
    out[-radius:, :] = False
    out[:, :radius] = False
    out[:, -radius:] = False
    return out


def support_erosion_radius(smooth_sigma: float, stencil_radius: int = 1) -> int:
    return int(np.ceil(3.0 * smooth_sigma)) + stencil_radius + 1


# ----------------------------------------------------------------------------
# Aligned face container
# ----------------------------------------------------------------------------
@dataclass
class AlignedFace:
    image_id: str
    rgb: np.ndarray  # (S, S, 3) float32 in [0, 1]
    landmarks: np.ndarray  # (68, 2) aligned coordinates (x, y)
    transform: np.ndarray  # (2, 3) original -> aligned
    fit_residual_px: float
    face_mask: np.ndarray  # landmark polygon (bool)
    valid: np.ndarray  # eroded analysis support (bool)
    inbounds: np.ndarray
    size: int
    meta: dict[str, Any] = field(default_factory=dict)


def align_face(
    image_id: str,
    rgb_orig: np.ndarray,
    lm_orig: np.ndarray,
    size: int = ALIGNED_SIZE,
    forehead_extension: float = 0.80,
    erosion_radius: int = 5,
    landmark_override_aligned: np.ndarray | None = None,
) -> AlignedFace:
    kp = key_points_from_68(lm_orig)
    src = np.array([kp["eye_left"], kp["eye_right"], kp["mouth"]])
    scale = size / ALIGNED_SIZE
    A, rms = fit_similarity(src, canonical_targets() * scale)
    rgb_al, inb = warp_rgb(rgb_orig, A, size)
    lm_al = apply_affine(A, lm_orig) if landmark_override_aligned is None else landmark_override_aligned
    fmask = face_polygon_mask(lm_al, (size, size), forehead_extension) & inb
    valid = erode_support(fmask, erosion_radius)
    return AlignedFace(
        image_id=image_id,
        rgb=rgb_al,
        landmarks=lm_al,
        transform=A,
        fit_residual_px=rms,
        face_mask=fmask,
        valid=valid,
        inbounds=inb,
        size=size,
        meta={"erosion_radius": erosion_radius, "forehead_extension": forehead_extension},
    )


def yaw_proxy_from_landmarks(lm: np.ndarray) -> float:
    """Descriptive-only yaw proxy: signed asymmetry of nose tip between the eye corners.

    0 = nose tip centred between outer eye corners; +/-1 = at an eye corner.  This
    is not a calibrated pose angle and is never used as an inclusion filter.
    """
    lm = np.asarray(lm, dtype=np.float64)
    left, right, nose = lm[36], lm[45], lm[30]
    axis = right - left
    L = np.linalg.norm(axis)
    if L < 1e-9:
        return float("nan")
    t = np.dot(nose - left, axis) / (L * L)
    return float(2.0 * t - 1.0)
