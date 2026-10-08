"""Adapter around the existing Hyperplane-Forge Rust core (``forge_detect._core``).

Reuse contract
--------------
* The baseline surface is produced by the *existing* CPU backend
  (:class:`forge_detect.backends.cpu.CpuBackend`) with the *existing* structured
  parameter object (:class:`forge_detect.config.PipelineParams`). No numerics
  are re-implemented for the baseline.
* Trust ``W`` is the repository's deterministic ``heuristic_trust_map`` (the mode
  ``forge_detect.pipeline.detect`` falls back to when no CNN is supplied); the
  constant-trust ablation supplies ``W = 1`` through the same interface.
* ``I_w`` (weighted luminance) and ``K`` (local albedo, 5x5 median) are needed
  by the solver audit and are not exported by the Rust API; they are recomputed
  here with the *same formulas* and validated against the exported energy terms
  in Experiment 0 (``controls.py``).  They are not used to build the baseline.
* If the extension cannot be imported this module raises ``RustUnavailable``
  with a concrete build diagnostic.  There is no silent Python/CUDA fallback.
* The ``corrected_gradient_torch`` variant is a separately labelled *diagnostic*
  solver (float64 torch L-BFGS on a replica of the implemented discrete energy)
  used only to quantify the solver-sign finding; it is never the baseline.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from facial_entropy.config import SolverConfig
from facial_entropy.util import sha256_json


class RustUnavailable(RuntimeError):
    pass


def set_rayon_threads(n: int) -> None:
    """Rayon reads RAYON_NUM_THREADS once at pool creation; set before first import."""
    if "forge_detect._core" in __import__("sys").modules:
        return
    os.environ.setdefault("RAYON_NUM_THREADS", str(int(n)))


def import_core() -> Any:
    try:
        from forge_detect import _core  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RustUnavailable(
            "forge_detect._core (Rust extension) is not importable: "
            f"{exc!r}. Build it with `uv sync --reinstall-package forge-detect` from this project "
            "(the sibling checkout ../deepfake-detection-v2 is built by maturin with features "
            "['pyo3/extension-module', 'python-extension']), or run `make build-release` inside "
            "that checkout."
        ) from exc
    if not hasattr(_core, "solve_and_extract"):
        raise RustUnavailable("forge_detect._core lacks solve_and_extract; unexpected build")
    return _core


def core_info() -> dict[str, Any]:
    import forge_detect  # noqa: PLC0415

    core = import_core()
    return {
        "forge_detect_file": forge_detect.__file__,
        "core_file": getattr(core, "__file__", None),
        "core_symbols": sorted(n for n in dir(core) if not n.startswith("_")),
        "rayon_num_threads_env": os.environ.get("RAYON_NUM_THREADS"),
    }


def pipeline_params(cfg: SolverConfig, **overrides: float):
    """Build the existing structured ``PipelineParams`` from our config (+overrides)."""
    from forge_detect.config import PdeParams, PipelineParams  # noqa: PLC0415

    d = {**cfg.__dict__, **overrides}
    return PipelineParams(
        w_r=d["w_r"],
        w_g=d["w_g"],
        w_b=d["w_b"],
        sigma_base=d["sigma_base"],
        k_ratio=d["k_ratio"],
        n_scales=int(d["n_scales"]),
        tensor_window_radius=int(d["tensor_window_radius"]),
        p_flat=d["p_flat"],
        p_edge=d["p_edge"],
        neigh_radius=int(d["neigh_radius"]),
        albedo_window_radius=int(d["albedo_window_radius"]),
        epsilon=d["epsilon"],
        c_z=d["c_z"],
        sigma_ref=d["sigma_ref"],
        pde=PdeParams(
            lambda_=d["pde_lambda"],
            alpha=d["pde_alpha"],
            beta=d["pde_beta"],
            max_iter=int(d["pde_max_iter"]),
            tol=d["pde_tol"],
            log_every=int(d["pde_log_every"]),
        ),
    )


def weighted_luminance(rgb: np.ndarray, cfg: SolverConfig | None = None) -> np.ndarray:
    """Repository luminance ``I_w = w_R R + w_G G + w_B B`` (same weights, float32)."""
    cfg = cfg or SolverConfig()
    rgb = np.asarray(rgb, dtype=np.float32)
    return (cfg.w_r * rgb[..., 0] + cfg.w_g * rgb[..., 1] + cfg.w_b * rgb[..., 2]).astype(np.float32)


def local_albedo(i_w: np.ndarray, radius: int = 2) -> np.ndarray:
    """Replica of ``hyperplane::local_albedo`` (square median, mirror boundary)."""
    from scipy.ndimage import median_filter  # noqa: PLC0415

    if radius == 0:
        return np.asarray(i_w, dtype=np.float32).copy()
    return median_filter(np.asarray(i_w, dtype=np.float32), size=2 * radius + 1, mode="mirror")


def trust_map(rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "heuristic":
        from forge_detect.trust_map import heuristic_trust_map  # noqa: PLC0415

        return heuristic_trust_map(np.ascontiguousarray(rgb, dtype=np.float32))
    if mode == "constant":
        return np.ones(rgb.shape[:2], dtype=np.float32)
    raise ValueError(f"unknown trust mode {mode!r}")


@dataclass
class SurfaceResult:
    variant: str
    backend: str
    z_forged: np.ndarray
    z_star: np.ndarray
    residual: np.ndarray
    laplacian: np.ndarray
    trust: np.ndarray
    luminance: np.ndarray
    energy_total: float
    energy_data: float
    energy_smoothness: float
    energy_consistency: float
    energy_trace: np.ndarray
    iterations: int
    converged: bool
    finite: bool
    runtime_s: float
    params: dict[str, Any] = field(default_factory=dict)
    threads: str | None = None

    def meta(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "backend": self.backend,
            "energy_total": self.energy_total,
            "energy_data": self.energy_data,
            "energy_smoothness": self.energy_smoothness,
            "energy_consistency": self.energy_consistency,
            "iterations": self.iterations,
            "converged": self.converged,
            "finite": self.finite,
            "runtime_s": self.runtime_s,
            "params": self.params,
            "threads": self.threads,
        }


# Variant registry: name -> (trust_mode, solver overrides, description).
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {"trust": "heuristic", "overrides": {}, "desc": "existing Rust pipeline, heuristic trust"},
    "constant_trust": {"trust": "constant", "overrides": {}, "desc": "existing pipeline with W = 1"},
    "alpha_x0.5": {"trust": "heuristic", "overrides": {"pde_alpha": 0.25}, "desc": "alpha x 0.5"},
    "alpha_x2": {"trust": "heuristic", "overrides": {"pde_alpha": 1.0}, "desc": "alpha x 2"},
    "beta_0": {"trust": "heuristic", "overrides": {"pde_beta": 0.0}, "desc": "no consistency term"},
    "forged_only": {"trust": "heuristic", "overrides": {}, "desc": "z_forged before settlement"},
    "luminance": {"trust": None, "overrides": {}, "desc": "graph of I_w (simple comparator)"},
    "smoothed_luminance": {"trust": None, "overrides": {}, "desc": "graph of G_sigma * I_w (comparator)"},
    "corrected_gradient_torch": {
        "trust": "heuristic",
        "overrides": {},
        "desc": "diagnostic: float64 L-BFGS minimiser of the replica energy (not the baseline)",
    },
}


def variant_key(variant: str, cfg: SolverConfig, extra: dict[str, Any] | None = None) -> str:
    spec = VARIANTS[variant]
    payload = {
        "variant": variant,
        "trust": spec["trust"],
        "solver": {**cfg.__dict__, **spec["overrides"]},
        "extra": extra or {},
    }
    return sha256_json(payload)[:16]


def solve_surface(
    rgb: np.ndarray,
    cfg: SolverConfig,
    variant: str = "baseline",
    smooth_sigma: float | None = None,
) -> SurfaceResult:
    """Run one surface construction variant on an aligned float32 RGB image in [0, 1]."""
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be (H, W, 3), got {rgb.shape}")
    spec = VARIANTS[variant]
    i_w = weighted_luminance(rgb, cfg)
    t0 = time.perf_counter()

    if variant in ("luminance", "smoothed_luminance"):
        from scipy.ndimage import gaussian_filter  # noqa: PLC0415

        z = i_w.copy()
        if variant == "smoothed_luminance":
            if smooth_sigma is None:
                raise ValueError("smoothed_luminance needs smooth_sigma (chosen on calibration)")
            z = gaussian_filter(i_w, sigma=float(smooth_sigma), mode="mirror").astype(np.float32)
        zeros = np.zeros_like(z)
        return SurfaceResult(
            variant=variant,
            backend="numpy-comparator",
            z_forged=z,
            z_star=z,
            residual=zeros,
            laplacian=zeros,
            trust=np.ones_like(z),
            luminance=i_w,
            energy_total=float("nan"),
            energy_data=float("nan"),
            energy_smoothness=float("nan"),
            energy_consistency=float("nan"),
            energy_trace=np.zeros(0, dtype=np.float32),
            iterations=0,
            converged=True,
            finite=bool(np.isfinite(z).all()),
            runtime_s=time.perf_counter() - t0,
            params={"smooth_sigma": smooth_sigma},
            threads=None,
        )

    if variant == "corrected_gradient_torch":
        from facial_entropy.solver_replica import lbfgs_minimise  # noqa: PLC0415

        base = solve_surface(rgb, cfg, "baseline")
        k = local_albedo(i_w, cfg.albedo_window_radius)
        out = lbfgs_minimise(base.z_forged, i_w, base.trust, k, cfg)
        z_star = out["z_star"].astype(np.float32)
        from scipy.ndimage import gaussian_filter  # noqa: PLC0415

        z_ideal = gaussian_filter(base.z_forged, sigma=cfg.sigma_ref, mode="mirror")
        from facial_entropy.solver_replica import laplacian_np  # noqa: PLC0415

        return SurfaceResult(
            variant=variant,
            backend="torch-lbfgs-replica(float64)",
            z_forged=base.z_forged,
            z_star=z_star,
            residual=(z_star - z_ideal).astype(np.float32),
            laplacian=laplacian_np(z_star).astype(np.float32),
            trust=base.trust,
            luminance=i_w,
            energy_total=out["energy_total"],
            energy_data=out["energy_data"],
            energy_smoothness=out["energy_smoothness"],
            energy_consistency=out["energy_consistency"],
            energy_trace=np.asarray(out["trace"], dtype=np.float32),
            iterations=int(out["iterations"]),
            converged=bool(out["converged"]),
            finite=bool(np.isfinite(z_star).all()),
            runtime_s=time.perf_counter() - t0,
            params={"grad_norm_ratio": out["grad_norm_ratio"]},
            threads=None,
        )

    # --- existing Rust pipeline through the existing CPU backend -------------
    set_rayon_threads(cfg.rayon_threads)
    import_core()
    from forge_detect.backends.cpu import CpuBackend  # noqa: PLC0415

    w = trust_map(rgb, spec["trust"])
    params = pipeline_params(cfg, **spec["overrides"])
    res = CpuBackend().solve(rgb, w, params)
    runtime = time.perf_counter() - t0
    z_star = res.z_forged if variant == "forged_only" else res.z_star
    finite = bool(np.isfinite(res.z_star).all() and np.isfinite(res.z_forged).all())
    return SurfaceResult(
        variant=variant,
        backend="rust-cpu(forge_detect._core.solve_and_extract via CpuBackend)",
        z_forged=np.asarray(res.z_forged, dtype=np.float32),
        z_star=np.asarray(z_star, dtype=np.float32),
        residual=np.asarray(res.residual, dtype=np.float32),
        laplacian=np.asarray(res.laplacian, dtype=np.float32),
        trust=np.asarray(w, dtype=np.float32),
        luminance=i_w,
        energy_total=float(res.energy_total),
        energy_data=float(res.energy_data),
        energy_smoothness=float(res.energy_smoothness),
        energy_consistency=float(res.energy_consistency),
        energy_trace=np.asarray(res.energy_trace, dtype=np.float32),
        iterations=int(res.iterations),
        converged=bool(res.converged),
        finite=finite,
        runtime_s=runtime,
        params=params.as_core_kwargs(),
        threads=os.environ.get("RAYON_NUM_THREADS"),
    )


def direct_core_call(rgb: np.ndarray, w: np.ndarray, cfg: SolverConfig) -> dict[str, Any]:
    """Call ``_core.solve_and_extract`` directly (parity check against the backend path)."""
    core = import_core()
    params = pipeline_params(cfg)
    out = core.solve_and_extract(
        np.ascontiguousarray(rgb, dtype=np.float32),
        np.ascontiguousarray(w, dtype=np.float32),
        **params.as_core_kwargs(),
    )
    return {k: (np.asarray(v) if hasattr(v, "shape") else v) for k, v in out.items()}
