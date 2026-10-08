"""Rust output parity, replica energy validation, cache invalidation."""

import numpy as np
import pytest

from facial_entropy import solver_replica as rep
from facial_entropy.config import ProfileConfig, SolverConfig
from facial_entropy.pipeline import SurfaceCache
from facial_entropy.rust_adapter import direct_core_call, local_albedo, solve_surface, trust_map, variant_key, weighted_luminance
from facial_entropy.synthetic import render_pseudo_face, template_landmarks

pytest.importorskip("forge_detect._core")


@pytest.fixture(scope="module")
def rgb():
    return render_pseudo_face(template_landmarks(64), 64, seed=3)


def test_adapter_matches_direct_core_and_is_deterministic(rgb):
    cfg = SolverConfig()
    a = solve_surface(rgb, cfg, "baseline")
    b = solve_surface(rgb, cfg, "baseline")
    d = direct_core_call(rgb, trust_map(rgb, "heuristic"), cfg)
    assert np.abs(a.z_star - b.z_star).max() == 0.0
    assert np.abs(a.z_star - d["z_star"]).max() < 1e-6
    assert np.abs(a.laplacian - d["L"]).max() < 1e-6
    assert a.backend.startswith("rust-cpu")


def test_replica_energy_matches_rust_at_logged_iterate(rgb):
    cfg = SolverConfig(pde_log_every=1)
    r = solve_surface(rgb, cfg, "baseline")
    iw = weighted_luminance(rgb, cfg).astype(np.float64)
    k = local_albedo(iw.astype(np.float32), cfg.albedo_window_radius).astype(np.float64)
    ix, iy = rep.scharr_np(iw)
    e = rep.energy_np(r.z_star, r.z_forged, ix, iy, r.trust, k, cfg.pde_lambda, cfg.pde_alpha, cfg.pde_beta)
    assert e["total"] == pytest.approx(r.energy_total, rel=1e-3)
    assert e["consistency"] == pytest.approx(r.energy_consistency, rel=1e-3)


def test_replica_laplacian_matches_rust_export(rgb):
    r = solve_surface(rgb, SolverConfig(), "baseline")
    assert np.abs(rep.laplacian_np(r.z_star) - r.laplacian).max() < 1e-4


def test_variant_overrides_change_cache_key():
    cfg = ProfileConfig(name="t", synthetic_only=True, n_subjects=0)
    k1 = variant_key("baseline", cfg.solver)
    k2 = variant_key("alpha_x2", cfg.solver)
    k3 = variant_key("baseline", SolverConfig(pde_alpha=0.6))
    assert len({k1, k2, k3}) == 3
    c = SurfaceCache(__import__("pathlib").Path("/tmp/fe-cache-test"), "digestA")
    c2 = SurfaceCache(__import__("pathlib").Path("/tmp/fe-cache-test"), "digestB")
    assert c.key("sha", "v1", "baseline", cfg) != c2.key("sha", "v1", "baseline", cfg)
    assert c.key("sha", "v1", "baseline", cfg) != c.key("sha", "v2", "baseline", cfg)


def test_constant_trust_and_beta_zero_variants_differ_from_baseline(rgb):
    cfg = SolverConfig()
    base = solve_surface(rgb, cfg, "baseline")
    ct = solve_surface(rgb, cfg, "constant_trust")
    b0 = solve_surface(rgb, cfg, "beta_0")
    assert (ct.trust == 1).all()
    assert np.abs(ct.z_star - base.z_star).max() > 1e-6
    assert b0.params["pde_beta"] == 0.0
    assert b0.energy_consistency == 0.0
