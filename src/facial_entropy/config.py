"""Profile configuration (TOML) and the frozen analysis protocol dataclasses."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
CONFIG_DIR = PROJECT_ROOT / "experiments" / "facial_entropy" / "configs"
DEFAULT_FEI_ROOT = Path("/Users/mykytasiryk/Development/datasets/fei-face-data-dataset")

# Canonical similarity-alignment targets (x, y) in the 256 x 256 aligned frame.
ALIGNED_SIZE = 256
TARGET_EYE_LEFT = (84.0, 100.0)  # image-left eye centre (iBUG points 36-41)
TARGET_EYE_RIGHT = (172.0, 100.0)  # image-right eye centre (iBUG points 42-47)
TARGET_MOUTH = (128.0, 184.0)  # mouth centre (iBUG points 48-67)


@dataclass
class SolverConfig:
    """Mirror of forge-detect PipelineParams defaults (source-audited, unchanged)."""

    w_r: float = 0.299
    w_g: float = 0.587
    w_b: float = 0.114
    sigma_base: float = 1.0
    k_ratio: float = 1.4142135623730951
    n_scales: int = 4
    tensor_window_radius: int = 2
    p_flat: float = 0.30
    p_edge: float = 0.70
    neigh_radius: int = 6
    albedo_window_radius: int = 2
    epsilon: float = 1.0e-3
    c_z: float = 1.0
    sigma_ref: float = 4.0
    pde_lambda: float = 1.0
    pde_alpha: float = 0.5
    pde_beta: float = 5.0
    pde_max_iter: int = 500
    pde_tol: float = 1.0e-5
    pde_log_every: int = 10
    trust_mode: str = "heuristic"  # forge_detect.trust_map.heuristic_trust_map
    rayon_threads: int = 4


@dataclass
class AnalysisConfig:
    aligned_size: int = ALIGNED_SIZE
    smooth_sigma: float = 1.0
    smooth_sigmas: tuple[float, ...] = (0.5, 1.0, 2.0)
    derivative_stencil_radius: int = 1
    patch_size: int = 32
    samples_per_patch: int = 900
    block_samples_per_patch: int = 800
    min_support: float = 0.90
    primary_bins: int = 16
    bins: tuple[int, ...] = (8, 16, 32)
    primary_window: int = 21
    windows: tuple[int, ...] = (11, 21, 31)
    stride: int = 4
    binary_thresholds: tuple[float, ...] = (0.35, 0.50, 0.65)
    calib_quantile_lo: float = 0.005
    calib_quantile_hi: float = 0.995
    calib_samples_per_class: int = 900  # central / peripheral samples per calibration image
    scale_floor: float = 1.0e-9
    height_scale_floor: float = 1.0e-6
    forehead_extension: float = 0.80  # fraction of inter-eye distance d, see protocol.md
    codebook_sizes: tuple[int, ...] = (16, 32)
    tile_size: int = 16
    policy_area_fractions: tuple[float, ...] = (0.10, 0.20, 0.40)
    policy_random_seeds: int = 20
    jitter_sds: tuple[float, ...] = (1.0, 2.0)
    jitter_draws: int = 20
    resolutions: tuple[int, ...] = (128, 256, 512)


@dataclass
class StatsConfig:
    bootstrap: int = 10000
    permutations: int = 10000
    alpha: float = 0.05
    verification_bootstrap: int = 1000


@dataclass
class ProfileConfig:
    name: str
    synthetic_only: bool
    n_subjects: int
    calibration_fraction: float = 0.20
    robustness_subset: int = 10
    primary_pose: str = "11"
    verification_poses: tuple[str, ...] = ("05", "06", "11", "12", "13", "14")
    strata_poses: tuple[str, ...] = ("12", "13", "14")
    experiments: tuple[str, ...] = ("exp0", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6")
    run_resolution_sensitivity: bool = True
    landmark_device: str = "cpu"
    solver: SolverConfig = field(default_factory=SolverConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply(obj: Any, d: dict[str, Any]) -> None:
    for k, v in d.items():
        if not hasattr(obj, k):
            raise KeyError(f"unknown config key {k!r} for {type(obj).__name__}")
        cur = getattr(obj, k)
        if isinstance(cur, tuple):
            v = tuple(v)
        setattr(obj, k, v)


def load_profile(name_or_path: str | Path) -> ProfileConfig:
    """Load ``experiments/facial_entropy/configs/<name>.toml`` or an explicit TOML path."""
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIG_DIR / f"{name_or_path}.toml"
    if not path.exists():
        raise FileNotFoundError(f"profile config not found: {name_or_path} (looked in {CONFIG_DIR})")
    with path.open("rb") as f:
        raw = tomllib.load(f)
    prof = raw.get("profile", {})
    cfg = ProfileConfig(
        name=prof.get("name", path.stem),
        synthetic_only=bool(prof.get("synthetic_only", False)),
        n_subjects=int(prof.get("n_subjects", 0)),
    )
    _apply(cfg, {k: v for k, v in prof.items() if k not in {"name", "synthetic_only", "n_subjects"}})
    _apply(cfg.solver, raw.get("solver", {}))
    _apply(cfg.analysis, raw.get("analysis", {}))
    _apply(cfg.stats, raw.get("stats", {}))
    cfg.extra = raw.get("extra", {})
    cfg.extra["config_path"] = str(path)
    return cfg
