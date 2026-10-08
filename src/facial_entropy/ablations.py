"""Experiment 3: what the Forge construction adds (variants under common and within-variant calibration)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from facial_entropy import statistics as st
from facial_entropy.experiments import RunContext
from facial_entropy.pipeline import AnalysisSettings, analyse_image, fit_calibration
from facial_entropy.util import read_json, write_json

ABLATION_VARIANTS = ("baseline", "luminance", "smoothed_luminance", "forged_only", "constant_trust", "alpha_x0.5", "alpha_x2", "beta_0", "corrected_gradient_torch")


def run_ablations(ctx: RunContext, df_primary: pd.DataFrame) -> dict[str, Any]:
    assert ctx.cohort is not None and ctx.calibration is not None
    lock = read_json(ctx.out / "protocol.lock.json")
    excluded = set(lock["excluded_primary"])
    prim = ctx.cohort.primary[~ctx.cohort.primary["image_id"].isin(excluded)]
    cal_rows = prim[prim["split"] == "calibration"]
    test_rows = prim[prim["split"] == "test"]
    s = ctx.settings
    s_nomap = AnalysisSettings.from_cfg(ctx.cfg, maps=False)
    rows: list[dict[str, Any]] = []
    map_corr: list[dict[str, Any]] = []
    runtime: dict[str, list[float]] = {v: [] for v in ABLATION_VARIANTS}
    conv: dict[str, list[bool]] = {v: [] for v in ABLATION_VARIANTS}
    # baseline maps for correlation
    base_maps: dict[str, np.ndarray] = {}
    key = f"gaussian_curvature_b{s.nbins}_w{s.window}"
    within_calibs = {}
    t0 = time.perf_counter()
    for variant in ABLATION_VARIANTS:
        # within-variant calibration (sensitivity)
        faces, surfs = [], []
        for _, row in cal_rows.iterrows():
            f = ctx.face_for(row)
            if f is None:
                continue
            su = ctx.surface_for(f, variant)
            faces.append(f)
            surfs.append(su)
        within = fit_calibration(faces, surfs, ctx.cfg, s_nomap, ctx.seed)
        within_calibs[variant] = {"height_scale": within.height_scale, "degenerate": within.degenerate_flags}
        for _, row in test_rows.iterrows():
            f = ctx.face_for(row)
            if f is None:
                continue
            su = ctx.surface_for(f, variant)
            runtime[variant].append(su.runtime_s)
            conv[variant].append(bool(su.converged and su.finite))
            prod = analyse_image(f, su, ctx.calibration, s, ctx.seed, variant=variant, descriptors=("gaussian_curvature", "mean_curvature", "residual"))
            for r in prod.rows:
                rows.append({**r, "calibration": "common"})
            m = prod.maps[key]["map"]
            if variant == "baseline":
                base_maps[f.image_id] = m
            elif f.image_id in base_maps:
                map_corr.append({"variant": variant, "image_id": f.image_id, **st.map_comparison(base_maps[f.image_id], m)})
            prod_w = analyse_image(f, su, within, s_nomap, ctx.seed, variant=variant, descriptors=("gaussian_curvature",))
            for r in prod_w.rows:
                rows.append({**r, "calibration": "within"})
    ab = pd.DataFrame(rows)
    ab.to_csv(ctx.out / "ablation_metrics.csv", index=False)
    pd.DataFrame(map_corr).to_csv(ctx.out / "ablation_map_correlation.csv", index=False)
    summary: dict[str, Any] = {"variants": {}, "paired_vs_baseline": {}, "within_variant_calibration": within_calibs}
    base_subj = None
    for variant in ABLATION_VARIANTS:
        sub = ab[(ab["variant"] == variant) & (ab["calibration"] == "common") & (ab["metric"] == "gaussian_curvature")]
        subj = st.subject_differences(sub, "gaussian_curvature", "fixed")
        summ = st.paired_summary(subj, ctx.cfg.stats.bootstrap, ctx.cfg.stats.permutations, ctx.seed)
        subw = ab[(ab["variant"] == variant) & (ab["calibration"] == "within") & (ab["metric"] == "gaussian_curvature")]
        summ_w = st.paired_summary(st.subject_differences(subw, "gaussian_curvature", "fixed"), ctx.cfg.stats.bootstrap, ctx.cfg.stats.permutations, ctx.seed)
        mc = [m for m in map_corr if m["variant"] == variant]
        summary["variants"][variant] = {
            "delta_K_common_calibration": summ, "delta_K_within_calibration": summ_w,
            "runtime_mean_s": float(np.mean(runtime[variant])) if runtime[variant] else float("nan"),
            "converged_fraction": float(np.mean(conv[variant])) if conv[variant] else float("nan"),
            "map_spearman_vs_baseline_mean": float(np.nanmean([m["spearman"] for m in mc])) if mc else None,
            "map_top_decile_overlap_mean": float(np.nanmean([m["top_decile_overlap"] for m in mc])) if mc else None,
        }
        if variant == "baseline":
            base_subj = subj.set_index("subject_id")["D"]
        elif base_subj is not None and not subj.empty:
            d = (subj.set_index("subject_id")["D"] - base_subj).dropna()
            summary["paired_vs_baseline"][variant] = {
                "mean_delta_D": float(d.mean()), "ci95": list(st.group_bootstrap_ci(d.to_numpy(), ctx.cfg.stats.bootstrap, ctx.seed)),
                "signflip_p": st.sign_flip_test(d.to_numpy(), ctx.cfg.stats.permutations, ctx.seed)["p"], "n": int(len(d)),
            }
        # rank consistency of patch entropies with baseline (per image Spearman over 8 patches)
        if variant != "baseline":
            b = ab[(ab["variant"] == "baseline") & (ab["calibration"] == "common") & (ab["metric"] == "gaussian_curvature") & (ab["estimator"] == "fixed")]
            v = sub[sub["estimator"] == "fixed"]
            merged = b.merge(v, on=["image_id", "patch"], suffixes=("_b", "_v"))
            rhos = [st.spearman(g["value_b"].to_numpy(), g["value_v"].to_numpy()) for _, g in merged.groupby("image_id")]
            summary["variants"][variant]["patch_rank_spearman_vs_baseline_mean"] = float(np.nanmean(rhos)) if rhos else None
    summary["runtime_total_s"] = time.perf_counter() - t0
    summary["luminance_smoothing_sigma"] = ctx.calibration.smoothed_luminance_sigma
    write_json(ctx.out / "ablation_results.json", summary)
    return summary
