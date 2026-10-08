"""Calibration-frozen quantizers (protocol §4.4-4.5): transforms, bin edges, terciles, codebooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ScalarQuantizer:
    name: str
    kind: str  # signed | magnitude | unit | code
    nbins: int
    scale: float
    lo: float
    hi: float
    degenerate: bool
    n_fit: int

    def transform(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=np.float64)
        if self.kind == "signed":
            return np.arcsinh(f / self.scale)
        if self.kind == "magnitude":
            return np.log1p(np.maximum(f, 0.0) / self.scale)
        return f

    def apply(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (codes int16, clipped_low bool, clipped_high bool)."""
        if self.kind == "code":
            c = np.asarray(f, dtype=np.int64)
            return c.astype(np.int16), np.zeros(c.shape, bool), np.zeros(c.shape, bool)
        t = self.transform(f)
        if self.degenerate:
            z = np.zeros(t.shape, dtype=np.int16)
            return z, np.zeros(t.shape, bool), np.zeros(t.shape, bool)
        width = (self.hi - self.lo) / self.nbins
        idx = np.floor((t - self.lo) / width)
        lo_clip = idx < 0
        hi_clip = idx >= self.nbins
        idx = np.clip(idx, 0, self.nbins - 1)
        idx = np.where(np.isfinite(t), idx, 0)
        return idx.astype(np.int16), lo_clip, hi_clip

    def edges(self) -> list[float]:
        return list(np.linspace(self.lo, self.hi, self.nbins + 1))

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "edges": self.edges()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScalarQuantizer:
        return cls(**{k: d[k] for k in ("name", "kind", "nbins", "scale", "lo", "hi", "degenerate", "n_fit")})


def fit_scalar_quantizer(
    name: str,
    kind: str,
    samples: np.ndarray,
    nbins: int,
    q_lo: float = 0.005,
    q_hi: float = 0.995,
    scale_floor: float = 1e-9,
) -> ScalarQuantizer:
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if kind == "code":
        return ScalarQuantizer(name, kind, nbins, 1.0, 0.0, float(nbins), False, int(s.size))
    if kind == "unit":
        scale, lo, hi = 1.0, 0.0, 1.0
        deg = s.size == 0 or float(np.ptp(s)) <= 0
        return ScalarQuantizer(name, kind, nbins, scale, lo, hi, bool(deg), int(s.size))
    scale = float(max(np.median(np.abs(s)) if s.size else 0.0, scale_floor))
    q = ScalarQuantizer(name, kind, nbins, scale, 0.0, 1.0, False, int(s.size))
    t = q.transform(s)
    lo, hi = (np.quantile(t, [q_lo, q_hi]) if t.size else (0.0, 0.0))
    deg = not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= 1e-12
    q.lo, q.hi, q.degenerate = float(lo), float(hi), bool(deg)
    return q


@dataclass
class Terciles:
    name: str
    scale: float
    cuts: tuple[float, float]

    def apply(self, f: np.ndarray) -> np.ndarray:
        t = np.arcsinh(np.asarray(f, dtype=np.float64) / self.scale)
        return (t > self.cuts[0]).astype(np.int16) + (t > self.cuts[1]).astype(np.int16)


def fit_terciles(name: str, samples: np.ndarray, scale_floor: float = 1e-9) -> Terciles:
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    scale = float(max(np.median(np.abs(s)) if s.size else 0.0, scale_floor))
    t = np.arcsinh(s / scale)
    c1, c2 = np.quantile(t, [1 / 3, 2 / 3]) if t.size else (0.0, 0.0)
    return Terciles(name, scale, (float(c1), float(c2)))


@dataclass
class Codebook:
    names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    centroids: np.ndarray
    n_fit: int
    seed: int

    def standardise(self, X: np.ndarray) -> np.ndarray:
        # robust standardisation followed by asinh tail compression (same as at fit time)
        return np.arcsinh((np.asarray(X, dtype=np.float64) - self.mean) / self.scale)

    def apply(self, X: np.ndarray) -> np.ndarray:
        Z = self.standardise(X)
        d = ((Z[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d, axis=1).astype(np.int16)

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": self.names,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "centroids": self.centroids.tolist(),
            "n_fit": self.n_fit,
            "seed": self.seed,
            "k": int(self.centroids.shape[0]),
            "dim": int(self.centroids.shape[1]),
        }


def fit_codebook(names: list[str], X: np.ndarray, k: int, seed: int) -> Codebook:
    """k-means codebook on robustly standardised calibration vectors (frozen afterwards)."""
    from sklearn.cluster import KMeans  # noqa: PLC0415

    X = np.asarray(X, dtype=np.float64)
    X = X[np.isfinite(X).all(axis=1)]
    # robust standardisation: median / (IQR / 1.349) with floor
    med = np.median(X, axis=0)
    iqr = (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)) / 1.349
    iqr = np.where(iqr > 1e-9, iqr, 1.0)
    Z = (X - med) / iqr
    # asinh-compress heavy tails before clustering
    Z = np.arcsinh(Z)
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(Z)
    return Codebook(names, med, iqr, km.cluster_centers_, int(Z.shape[0]), seed)


@dataclass
class Calibration:
    """Everything frozen after the calibration split is processed."""

    height_scale: float
    height_scale_per_image: dict[str, float]
    tilt_threshold: float
    quantizers: dict[str, dict[int, ScalarQuantizer]]  # descriptor -> nbins -> quantizer
    terciles: dict[str, Terciles]
    codebooks: dict[int, Codebook]
    smoothed_luminance_sigma: float
    fit_population: dict[str, Any] = field(default_factory=dict)
    degenerate_flags: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "height_scale": self.height_scale,
            "height_scale_per_image": self.height_scale_per_image,
            "tilt_threshold": self.tilt_threshold,
            "quantizers": {d: {str(n): q.to_dict() for n, q in qs.items()} for d, qs in self.quantizers.items()},
            "terciles": {k: {"name": t.name, "scale": t.scale, "cuts": list(t.cuts)} for k, t in self.terciles.items()},
            "codebooks": {str(k): cb.to_dict() for k, cb in self.codebooks.items()},
            "smoothed_luminance_sigma": self.smoothed_luminance_sigma,
            "fit_population": self.fit_population,
            "degenerate_flags": self.degenerate_flags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Calibration:
        qs = {desc: {int(n): ScalarQuantizer.from_dict(q) for n, q in qq.items()} for desc, qq in d["quantizers"].items()}
        ter = {k: Terciles(t["name"], t["scale"], tuple(t["cuts"])) for k, t in d["terciles"].items()}
        cbs = {}
        for k, c in d["codebooks"].items():
            cbs[int(k)] = Codebook(c["names"], np.array(c["mean"]), np.array(c["scale"]), np.array(c["centroids"]), c["n_fit"], c["seed"])
        return cls(
            height_scale=d["height_scale"],
            height_scale_per_image=d["height_scale_per_image"],
            tilt_threshold=d["tilt_threshold"],
            quantizers=qs,
            terciles=ter,
            codebooks=cbs,
            smoothed_luminance_sigma=d["smoothed_luminance_sigma"],
            fit_population=d.get("fit_population", {}),
            degenerate_flags=d.get("degenerate_flags", {}),
        )
