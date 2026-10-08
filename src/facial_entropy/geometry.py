"""Graph-surface geometry: normalisation, validated derivatives, curvatures, normals.

Coordinate convention (protocol §4.3): at aligned resolution ``S`` the spacing
is ``h = 1 / (S - 1)`` so x, y in [0, 1].  Arrays are indexed ``[y, x]``; x is
axis 1.  The solver's own unit-pixel convention is untouched; only this
downstream analysis uses ``h``.  Derivative smoothing (Gaussian, sigma in
pixels) is an *analysis* choice applied to the whole normalised field before
differentiation; the support mask is applied afterwards.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


def robust_scale(z: np.ndarray, valid: np.ndarray) -> float:
    v = np.asarray(z, dtype=np.float64)[valid]
    if v.size == 0:
        return float("nan")
    q25, q75 = np.percentile(v, [25, 75])
    return float((q75 - q25) / 1.349)


def normalise_height(z: np.ndarray, valid: np.ndarray, s_z: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    med = float(np.median(z[valid])) if valid.any() else 0.0
    return (z - med) / s_z


def derivatives(z: np.ndarray, h: float, sigma: float = 0.0) -> dict[str, np.ndarray]:
    """Central-difference first and second derivatives with spacing ``h`` (after optional smoothing)."""
    z = np.asarray(z, dtype=np.float64)
    if sigma > 0:
        z = gaussian_filter(z, sigma=sigma, mode="reflect")
    zy, zx = np.gradient(z, h, h, edge_order=2)
    zyy, zxy = np.gradient(zy, h, h, edge_order=2)
    _, zxx = np.gradient(zx, h, h, edge_order=2)
    return {"z": z, "zx": zx, "zy": zy, "zxx": zxx, "zxy": zxy, "zyy": zyy}


def curvatures(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    zx, zy, zxx, zxy, zyy = d["zx"], d["zy"], d["zxx"], d["zxy"], d["zyy"]
    g = 1.0 + zx**2 + zy**2
    kg = (zxx * zyy - zxy**2) / g**2
    hm = ((1.0 + zy**2) * zxx - 2.0 * zx * zy * zxy + (1.0 + zx**2) * zyy) / (2.0 * g**1.5)
    return kg, hm


def normals(d: dict[str, np.ndarray]) -> np.ndarray:
    zx, zy = d["zx"], d["zy"]
    n = np.stack([-zx, -zy, np.ones_like(zx)], axis=-1)
    return n / np.sqrt(1.0 + zx**2 + zy**2)[..., None]


def normal_direction_codes(d: dict[str, np.ndarray], tilt_threshold: float) -> np.ndarray:
    """16 states: 8 azimuth sectors of (-zx, -zy) x 2 tilt classes (tan(theta) <= / > threshold)."""
    zx, zy = d["zx"], d["zy"]
    az = np.arctan2(-zy, -zx)  # [-pi, pi]
    sector = np.floor((az + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    tilt = (np.sqrt(zx**2 + zy**2) > tilt_threshold).astype(int)
    return sector + 8 * tilt


def geometry_fields(z_star: np.ndarray, valid: np.ndarray, s_z: float, sigma: float, h: float) -> dict[str, Any]:
    zt = normalise_height(z_star, valid, s_z)
    d = derivatives(zt, h, sigma)
    kg, hm = curvatures(d)
    return {
        "z_tilde": zt,
        "z_smooth": d["z"],
        "zx": d["zx"],
        "zy": d["zy"],
        "gaussian_curvature": kg,
        "mean_curvature": hm,
        "grad_mag": np.sqrt(d["zx"] ** 2 + d["zy"] ** 2),
        "laplacian_analysis": d["zxx"] + d["zyy"],
        "area_weight": np.sqrt(1.0 + d["zx"] ** 2 + d["zy"] ** 2),
        "derivs": d,
    }


# Descriptor registry: name -> (kind, source)
#   kind: "signed" (asinh transform), "magnitude" (log1p), "unit" (identity on [0,1]), "code"
DESCRIPTORS: dict[str, dict[str, str]] = {
    "gaussian_curvature": {"kind": "signed", "source": "surface"},
    "mean_curvature": {"kind": "signed", "source": "surface"},
    "laplacian_analysis": {"kind": "signed", "source": "surface"},
    "grad_mag": {"kind": "magnitude", "source": "surface"},
    "residual": {"kind": "signed", "source": "rust"},
    "trust": {"kind": "unit", "source": "rust"},
    "normal_dir": {"kind": "code", "source": "surface"},
}
