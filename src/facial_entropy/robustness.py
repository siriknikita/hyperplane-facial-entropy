"""Experiment 4: nuisance, discretisation and parameter robustness on the prespecified subset."""

from __future__ import annotations

import io
import time
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

from facial_entropy import regions as reg
from facial_entropy import statistics as st
from facial_entropy.experiments import RunContext
from facial_entropy.pipeline import AnalysisSettings, analyse_image, load_maps, save_maps
from facial_entropy.util import read_json, rng_for, write_json


def jpeg_cycle(rgb: np.ndarray, quality: int) -> np.ndarray:
    im = Image.fromarray(np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0


def photometric_perturbations(rgb: np.ndarray, image_id: str) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    out: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for q in (95, 75, 50):
        out[f"jpeg_q{q}"] = (jpeg_cycle(rgb, q), {})
    for sg in (0.5, 1.0):
        out[f"blur_s{sg}"] = (gaussian_filter(rgb, (sg, sg, 0)).astype(np.float32), {})
    for sd in (0.01, 0.03):
        rng = rng_for("noise", image_id, sd)
        p = np.clip(rgb + sd * rng.standard_normal(rgb.shape), 0, 1).astype(np.float32)
        out[f"noise_sd{sd}"] = (p, {"clip_fraction": float(((rgb + sd * 0) < 0).mean())})
    for g in (0.8, 1.2):
        p = np.clip(rgb * g, 0, 1).astype(np.float32)
        out[f"brightness_x{g}"] = (p, {"clip_fraction": float((rgb * g > 1).mean())})
    for g in (0.8, 1.2):
        out[f"gamma_{g}"] = (np.clip(rgb, 0, 1).astype(np.float32) ** g, {"note": "applied to encoded [0,1] values"})
    for gains in ((1.05, 1.00, 0.95), (0.95, 1.00, 1.05)):
        p = np.clip(rgb * np.array(gains, dtype=np.float32), 0, 1).astype(np.float32)
        out[f"gains_{gains[0]}_{gains[2]}"] = (p, {"clip_fraction": float((rgb * np.array(gains) > 1).mean())})
    return out


def _contrast(rows: list[dict[str, Any]], metric: str) -> float:
    vals = {r["patch"]: r["value"] for r in rows if r["metric"] == metric and r["estimator"] == "fixed"}
    c = np.nanmean([vals.get(p, np.nan) for p in reg.CENTRAL_PATCHES])
    p = np.nanmean([vals.get(p, np.nan) for p in reg.PERIPHERAL_PATCHES])
    return float(c - p)


def run_robustness(ctx: RunContext) -> dict[str, Any]:
    assert ctx.cohort is not None and ctx.calibration is not None
    lock = read_json(ctx.out / "protocol.lock.json")
    ids = lock["robustness_subset"]
    prim = ctx.cohort.primary.set_index("image_id")
    s = ctx.settings
    s_nomap = AnalysisSettings.from_cfg(ctx.cfg, maps=False)
    key = f"gaussian_curvature_b{s.nbins}_w{s.window}"
    bkey = f"binary_cond_otsu_w{s.window}"
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for iid in ids:
        row = prim.loc[iid]
        row = row.copy()
        row["image_id"] = iid
        face = ctx.face_for(row)
        if face is None:
            continue
        base_surf = ctx.surface_for(face, "baseline")
        base = analyse_image(face, base_surf, ctx.calibration, s, ctx.seed, descriptors=("gaussian_curvature", "mean_curvature", "residual"))
        dK0, dB0 = _contrast(base.rows, "gaussian_curvature"), _contrast(base.rows, "binary_cond_otsu")
        bmap, bbmap = base.maps[key]["map"], base.maps[bkey]["map"]

        def record(kind: str, name: str, prod, extra: dict[str, Any] | None = None, surf=None):
            d = {"image_id": iid, "subject_id": row["subject_id"], "kind": kind, "perturbation": name,
                 "dK_base": dK0, "dK": _contrast(prod.rows, "gaussian_curvature"), "dB_base": dB0, "dB": _contrast(prod.rows, "binary_cond_otsu"),
                 "retained": prod.patch_info["retained"]}
            if key in prod.maps:
                d.update({f"KG_{k}": v for k, v in st.map_comparison(bmap, prod.maps[key]["map"]).items()})
                d.update({f"BIN_{k}": v for k, v in st.map_comparison(bbmap, prod.maps[bkey]["map"]).items()})
            if surf is not None:
                d.update({"converged": surf.converged, "iterations": surf.iterations})
            clip = [r.get("clip_lo", 0) + r.get("clip_hi", 0) for r in prod.rows if r["metric"] == "gaussian_curvature" and r["estimator"] == "fixed"]
            d["KG_clip_fraction_mean"] = float(np.mean(clip)) if clip else np.nan
            d.update(extra or {})
            rows.append(d)

        # photometric: regenerate the surface from perturbed RGB, same landmarks
        for name, (prgb, extra) in photometric_perturbations(face.rgb, iid).items():
            surf = ctx.surface_for(face, "baseline", rgb_override=prgb)
            prod = analyse_image(face, surf, ctx.calibration, s, ctx.seed, descriptors=("gaussian_curvature", "mean_curvature", "residual"))
            save_maps(ctx.maps_path(iid, f"pert_{name}"), prod)
            record("photometric", name, prod, {**extra, "rgb_saturation_fraction": float((prgb >= 1.0).mean())}, surf)
        # analysis window / bins / smoothing
        for w in ctx.cfg.analysis.windows:
            prod = analyse_image(face, base_surf, ctx.calibration, AnalysisSettings.from_cfg(ctx.cfg, window=w, extra_windows=()), ctx.seed, descriptors=("gaussian_curvature",))
            record("window", f"w{w}", prod)
        for nb in ctx.cfg.analysis.bins:
            prod = analyse_image(face, base_surf, ctx.calibration, AnalysisSettings.from_cfg(ctx.cfg, nbins=nb, extra_windows=()), ctx.seed, descriptors=("gaussian_curvature",))
            record("bins", f"b{nb}", prod)
        for sg in ctx.cfg.analysis.smooth_sigmas:
            f2 = ctx.face_for(row, sigma=sg)
            prod = analyse_image(f2, base_surf, ctx.calibration, AnalysisSettings.from_cfg(ctx.cfg, sigma=sg, extra_windows=()), ctx.seed, descriptors=("gaussian_curvature",))
            record("smoothing", f"sigma{sg}", prod)
        # landmark jitter (patch placement only)
        for sd in ctx.cfg.analysis.jitter_sds:
            for k in range(ctx.cfg.analysis.jitter_draws):
                rng = rng_for("jitter", iid, sd, k, base=ctx.seed)
                lm = face.landmarks + rng.normal(0, sd, face.landmarks.shape)
                prod = analyse_image(face, base_surf, ctx.calibration, s_nomap, ctx.seed, landmarks=lm, descriptors=("gaussian_curvature",))
                record("landmark_jitter", f"sd{sd}", prod, {"draw": k})
        # resolution sensitivity (unchanged solver pixel parameters; footprints scaled)
        if ctx.cfg.run_resolution_sensitivity:
            for res in ctx.cfg.analysis.resolutions:
                if res == ctx.cfg.analysis.aligned_size:
                    continue
                fr = ctx.face_for(row, size=res)
                if fr is None:
                    continue
                sr = ctx.surface_for(fr, "baseline")
                prod = analyse_image(fr, sr, ctx.calibration, AnalysisSettings.from_cfg(ctx.cfg, size=res, extra_windows=()), ctx.seed, descriptors=("gaussian_curvature",))
                record("resolution", f"res{res}", prod, {"note": "solver pixel params unchanged; patch/window/samples scaled with resolution; calibration frozen at 256"}, sr)
    df = pd.DataFrame(rows)
    df.to_csv(ctx.out / "robustness_metrics.csv", index=False)
    summary: dict[str, Any] = {"n_images": len(ids), "runtime_s": time.perf_counter() - t0, "by_perturbation": {}}
    if not df.empty:
        for (kind, name), g in df.groupby(["kind", "perturbation"]):
            summary["by_perturbation"][f"{kind}:{name}"] = {
                "n": int(len(g)),
                "dK_change_mean": float((g["dK"] - g["dK_base"]).mean()), "dK_change_sd": float((g["dK"] - g["dK_base"]).std(ddof=1)) if len(g) > 1 else None,
                "dK_sign_agreement": float((np.sign(g["dK"]) == np.sign(g["dK_base"])).mean()),
                "dB_change_mean": float((g["dB"] - g["dB_base"]).mean()),
                "KG_map_spearman_mean": float(g["KG_spearman"].mean()) if "KG_spearman" in g else None,
                "KG_map_top_decile_overlap_mean": float(g["KG_top_decile_overlap"].mean()) if "KG_top_decile_overlap" in g else None,
                "BIN_map_spearman_mean": float(g["BIN_spearman"].mean()) if "BIN_spearman" in g else None,
                "KG_clip_fraction_mean": float(g["KG_clip_fraction_mean"].mean()),
                "retained_fraction": float(g["retained"].mean()),
            }
    write_json(ctx.out / "robustness_results.json", summary)
    return summary
