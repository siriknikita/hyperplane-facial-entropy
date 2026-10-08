"""Orchestration: prepare (manifest, alignment, calibration, protocol lock), Experiments 1-2, atlas."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from facial_entropy import regions as reg
from facial_entropy import statistics as st
from facial_entropy.alignment import AlignedFace, LandmarkAdapter, MissingAssetError, align_face, landmark_asset_status, support_erosion_radius, yaw_proxy_from_landmarks
from facial_entropy.config import ProfileConfig
from facial_entropy.datasets import discover_auto, discover_fei, discover_lfw
from facial_entropy.manifest import Cohort, build_manifest, near_duplicate_report, robustness_subset, save_manifest, select_cohort
from facial_entropy.pipeline import AnalysisSettings, FacePreparer, ImageProducts, SurfaceCache, analyse_image, choose_luminance_sigma, fit_calibration, load_maps, save_maps
from facial_entropy.quantization import Calibration
from facial_entropy.rust_adapter import SurfaceResult, core_info
from facial_entropy.synthetic import synthetic_cohort
from facial_entropy.util import Timer, environment_info, git_info, read_json, sha256_json, write_json

FORGE_REPO = Path(__file__).resolve().parents[3] / "deepfake-detection-v2"


def source_digest() -> str:
    """Digest of the forge-detect Rust sources + python backend as installed (cache invalidation)."""
    import forge_detect  # noqa: PLC0415

    parts: dict[str, str] = {}
    pkg = Path(forge_detect.__file__).parent
    for p in sorted(pkg.rglob("*.py")) + sorted(pkg.glob("*.so")):
        parts[str(p.relative_to(pkg))] = sha256_json(p.stat().st_size)
    src = FORGE_REPO / "src"
    if src.exists():
        for p in sorted(src.glob("*.rs")):
            parts["rs/" + p.name] = sha256_json(p.read_text())
    return sha256_json(parts)[:16]


@dataclass
class RunContext:
    cfg: ProfileConfig
    seed: int
    out: Path
    data_root: str | None
    dataset: str
    manifest_path: str | None
    download: bool
    manifest: pd.DataFrame | None = None
    cohort: Cohort | None = None
    preparer: FacePreparer | None = None
    cache: SurfaceCache | None = None
    calibration: Calibration | None = None
    synthetic: dict[str, dict[str, Any]] = field(default_factory=dict)
    run_manifest: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    status: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    @property
    def settings(self) -> AnalysisSettings:
        return AnalysisSettings.from_cfg(self.cfg)

    def log(self, msg: str) -> None:
        print(f"[forge-entropy] {msg}", flush=True)

    def save_run_manifest(self) -> None:
        self.run_manifest.update({"status": self.status, "timings_s": self.timings})
        write_json(self.out / "run_manifest.json", self.run_manifest)

    def face_for(self, row: pd.Series, **kw: Any) -> AlignedFace | None:
        if self.cfg.synthetic_only:
            s = self.synthetic[row["image_id"]]
            size = kw.get("size") or self.cfg.analysis.aligned_size
            sigma = kw.get("sigma", self.cfg.analysis.smooth_sigma)
            r = support_erosion_radius(sigma, self.cfg.analysis.derivative_stencil_radius)
            rgb = kw.get("rgb", s["rgb"])
            lm = kw.get("landmarks", s["landmarks"])
            face = align_face(row["image_id"], rgb, lm, size=size, forehead_extension=self.cfg.analysis.forehead_extension, erosion_radius=r)
            face.meta.update({"sha256": row["sha256"], "subject_id": row["subject_id"], "split": row.get("split", ""), "landmark_meta": {"source": "synthetic"}})
            return face
        assert self.preparer is not None
        return self.preparer.prepare(row, **kw)

    def surface_for(self, face: AlignedFace, variant: str = "baseline", **kw: Any) -> SurfaceResult:
        assert self.cache is not None
        if variant == "smoothed_luminance" and "smooth_sigma" not in kw:
            assert self.calibration is not None
            kw["smooth_sigma"] = self.calibration.smoothed_luminance_sigma
        return self.cache.get_or_solve(face, variant, self.cfg, **kw)

    def maps_path(self, image_id: str, tag: str) -> Path:
        p = self.out / "cache" / "maps"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{image_id}__{tag}.npz"

    def primary_rows(self, split: str | None = None) -> pd.DataFrame:
        assert self.cohort is not None
        df = self.cohort.primary
        return df if split is None else df[df["split"] == split]


# ----------------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------------
def prepare(ctx: RunContext) -> None:
    cfg = ctx.cfg
    out = ctx.out
    out.mkdir(parents=True, exist_ok=True)
    tm = Timer()
    ctx.run_manifest = {
        "profile": cfg.to_dict(), "seed": ctx.seed, "environment": environment_info(), "forge_repo_git": git_info(FORGE_REPO),
        "project_git": git_info(Path(__file__).resolve().parents[2]), "core": core_info(), "source_digest": source_digest(),
        "dataset_arg": ctx.dataset, "data_root": ctx.data_root, "download": ctx.download, "output": str(out),
    }
    ctx.cache = SurfaceCache(out / "cache", ctx.run_manifest["source_digest"])

    if cfg.synthetic_only:
        n = cfg.n_subjects or 12
        cohort = synthetic_cohort(n, ctx.seed)
        rows = []
        for c in cohort:
            ctx.synthetic[c["image_id"]] = c
            rows.append({"image_id": c["image_id"], "path": "(synthetic)", "sha256": sha256_json(c["image_id"]), "subject_id": c["subject_id"],
                         "source_video_id": c["subject_id"], "dataset": "SYNTHETIC", "label": "synthetic", "provenance": "rendered pseudo-face (not a photograph)",
                         "width": c["rgb"].shape[1], "height": c["rgb"].shape[0], "frame_index": None, "pose": "11", "task_label": c["subject_id"],
                         "landmark_asset_version": "synthetic-template", "alignment_transform": "", "alignment_residual_px": 0.0, "quality_flags": "synthetic",
                         "split": "", "role": "", "exclusion_reason": ""})
        ctx.manifest = pd.DataFrame(rows)
        ctx.run_manifest["dataset"] = {"name": "SYNTHETIC", "n_images": n, "note": "synthetic fixtures only; never a real-data result"}
    else:
        if ctx.dataset == "lfw":
            recs = discover_lfw(ctx.data_root, ctx.download)
            name, cands = "LFW-funneled", []
        elif ctx.dataset == "manifest":
            if not ctx.manifest_path:
                raise ValueError("--dataset manifest needs --manifest <csv>")
            ctx.manifest = pd.read_csv(ctx.manifest_path)
            recs, name, cands = [], "manifest", [ctx.manifest_path]
        else:
            name, recs, cands = discover_auto(ctx.data_root) if ctx.dataset == "auto" else ("FEI", discover_fei(ctx.data_root or ""), [])
        if ctx.manifest is None:
            ctx.log(f"building manifest for {len(recs)} images ({name})")
            ctx.manifest = build_manifest(recs)
        ctx.run_manifest["dataset"] = {"name": name, "candidates": cands, "n_images": int(len(ctx.manifest)),
                                       "n_subjects": int(ctx.manifest["subject_id"].nunique()), "dedup_sha256_excluded": int((ctx.manifest["exclusion_reason"] != "").sum())}
    ctx.timings["manifest"] = tm.lap()

    coh = select_cohort(ctx.manifest, cfg.primary_pose, cfg.n_subjects, cfg.calibration_fraction, ctx.seed, cfg.verification_poses)
    ctx.cohort = coh
    ctx.run_manifest["cohort"] = {"requested_subjects": coh.requested_subjects, "available_subjects": coh.available_subjects,
                                  "n_primary": int(len(coh.primary)), "n_secondary": int(len(coh.secondary)),
                                  "n_calibration": int((coh.primary["split"] == "calibration").sum()), "n_test": int((coh.primary["split"] == "test").sum()),
                                  "notes": coh.notes, "primary_pose": cfg.primary_pose}

    # landmarks + alignment for primary (and secondary) images
    if not cfg.synthetic_only:
        status = landmark_asset_status()
        ctx.run_manifest["landmark_assets"] = status
        try:
            lm = LandmarkAdapter(device=cfg.landmark_device, allow_download=ctx.download)
        except MissingAssetError as exc:
            ctx.status["prepare"] = f"blocked: {exc}"
            ctx.save_run_manifest()
            raise
        ctx.preparer = FacePreparer(cfg, out, lm)
        ctx.run_manifest["landmark_assets_after"] = landmark_asset_status()
    exclusions: list[dict[str, Any]] = []
    qa_rows = []
    s = ctx.settings
    for role, df in (("primary", coh.primary), ("secondary", coh.secondary)):
        t_role = time.perf_counter()
        n_done = 0
        for i, (_, row) in enumerate(df.iterrows()):
            face = ctx.face_for(row)
            if face is None:
                exclusions.append({"image_id": row["image_id"], "subject_id": row["subject_id"], "role": role, "reason": "no_face_detected"})
                continue
            info = {"image_id": row["image_id"], "role": role, "fit_residual_px": face.fit_residual_px, "yaw_proxy": yaw_proxy_from_landmarks(face.landmarks),
                    "face_mask_px": int(face.face_mask.sum()), "valid_px": int(face.valid.sum()), "landmark_meta": face.meta.get("landmark_meta", {})}
            if role == "primary":
                from facial_entropy.pipeline import patch_layout  # noqa: PLC0415

                layout = patch_layout(face, s)
                info.update({"retained": layout["retained"], "rejection_reasons": layout["rejection_reasons"], "validity": layout["validity"]})
                if not layout["retained"]:
                    exclusions.append({"image_id": row["image_id"], "subject_id": row["subject_id"], "role": role, "reason": ";".join(layout["rejection_reasons"])})
            qa_rows.append(info)
            n_done += 1
            if role == "primary" and i == 9 and not cfg.synthetic_only:
                dt = time.perf_counter() - t_role
                ctx.log(f"first 10 primary faces aligned in {dt:.1f}s -> projected {dt / 10 * (len(coh.primary) + len(coh.secondary)):.0f}s for all faces")
        ctx.timings[f"align_{role}"] = time.perf_counter() - t_role
        ctx.log(f"{role}: {n_done}/{len(df)} faces aligned")
    write_json(out / "alignment_qa.json", qa_rows)
    excl_df = pd.DataFrame(exclusions, columns=["image_id", "subject_id", "role", "reason"])
    save_manifest(excl_df, out / "exclusions.csv")
    excluded_ids = set(excl_df[excl_df["role"] == "primary"]["image_id"])
    full = pd.concat([coh.primary, coh.secondary])
    full = full.copy()
    full.loc[full["image_id"].isin(set(excl_df["image_id"])), "exclusion_reason"] = full["image_id"].map(dict(zip(excl_df["image_id"], excl_df["reason"], strict=False))).fillna("")
    full["landmark_asset_version"] = "synthetic-template" if cfg.synthetic_only else str(ctx.preparer.landmarker.version if ctx.preparer and ctx.preparer.landmarker else "")
    for q in qa_rows:
        full.loc[full["image_id"] == q["image_id"], "alignment_residual_px"] = q["fit_residual_px"]
    manifest_out = pd.concat([full, ctx.manifest[~ctx.manifest["image_id"].isin(full["image_id"])]]) if not cfg.synthetic_only else full
    save_manifest(manifest_out, out / "dataset_manifest.csv")
    if not cfg.synthetic_only:
        ctx.run_manifest["near_duplicates"] = near_duplicate_report(coh.primary)
    ctx.timings["qa"] = tm.lap()

    # calibration (baseline surfaces on calibration subjects only)
    cal = coh.primary[(coh.primary["split"] == "calibration") & ~coh.primary["image_id"].isin(excluded_ids)]
    faces, surfs = [], []
    t_cal = time.perf_counter()
    for _, row in cal.iterrows():
        f = ctx.face_for(row)
        if f is None:
            continue
        faces.append(f)
        surfs.append(ctx.surface_for(f, "baseline"))
    n_planned = int(len(coh.primary) + len(coh.secondary))
    proj = ctx.cache.projected_bytes(n_planned)
    ctx.run_manifest["disk_projection"] = {"planned_cache_entries": n_planned, "mean_entry_bytes": float(np.mean(ctx.cache.entry_bytes)) if ctx.cache.entry_bytes else None,
                                           "projected_cache_bytes": proj, "budget_bytes": 2 * 1024**3,
                                           "within_budget": bool(proj <= 2 * 1024**3) if np.isfinite(proj) else None}
    ctx.log(f"projected surface cache {proj/1e9:.2f} GB for {n_planned} entries (budget 2 GB)")
    if np.isfinite(proj) and proj > 2 * 1024**3:
        raise RuntimeError(f"projected cache {proj/1e9:.2f} GB exceeds the 2 GB output budget; reduce the cohort or cache policy")
    lum = choose_luminance_sigma(faces, surfs)
    ctx.calibration = fit_calibration(faces, surfs, cfg, s, ctx.seed, luminance_choice=lum)
    ctx.timings["calibration"] = time.perf_counter() - t_cal
    write_json(out / "calibration.json", ctx.calibration.to_dict())
    lock = {
        "protocol_version": "1.0", "seed": ctx.seed, "profile": cfg.to_dict(), "subject_split": coh.subject_split,
        "primary_image_ids": sorted(coh.primary["image_id"]), "excluded_primary": sorted(excluded_ids),
        "robustness_subset": robustness_subset(coh.primary[~coh.primary["image_id"].isin(excluded_ids)], cfg.robustness_subset, ctx.seed),
        "patch_rule": "protocol.md §5.4, 32x32, anchor = floor(c+0.5)-16", "estimator": "plug-in, base 2, 900 fixed samples per patch, 800 block anchors",
        "quantizer": "calibration.json (asinh/log1p, 16 bins, quantiles 0.005/0.995)", "solver": cfg.solver.__dict__,
        "calibration_sha256": sha256_json(ctx.calibration.to_dict()), "source_digest": ctx.run_manifest["source_digest"],
        "endpoints": ["binary_cond_otsu", "gaussian_curvature(b16, projected-area sampling)"],
    }
    lock["hash"] = sha256_json(lock)
    write_json(out / "protocol.lock.json", lock)
    ctx.status["prepare"] = "done"
    ctx.save_run_manifest()
    ctx.log(f"prepare done: {len(faces)} calibration faces, s_z={ctx.calibration.height_scale:.4g}, lum sigma={lum['sigma']}")


def load_prepared(ctx: RunContext) -> None:
    """Reload a prepared output directory (manifest, cohort, calibration) for run/report."""
    out = ctx.out
    ctx.run_manifest = read_json(out / "run_manifest.json")
    lock = read_json(out / "protocol.lock.json")
    ctx.cache = SurfaceCache(out / "cache", ctx.run_manifest["source_digest"])
    ctx.calibration = Calibration.from_dict(read_json(out / "calibration.json"))
    df = pd.read_csv(out / "dataset_manifest.csv", dtype={"pose": str, "subject_id": str, "image_id": str}, keep_default_na=False)
    if ctx.cfg.synthetic_only:
        for c in synthetic_cohort(ctx.cfg.n_subjects or 12, ctx.seed):
            ctx.synthetic[c["image_id"]] = c
    prim = df[df["role"] == "primary"].copy()
    sec = df[df["role"] == "secondary"].copy()
    ctx.cohort = Cohort(primary=prim, secondary=sec, subject_split=lock["subject_split"], requested_subjects=ctx.cfg.n_subjects,
                        available_subjects=ctx.run_manifest.get("cohort", {}).get("available_subjects", 0))
    ctx.manifest = df
    if not ctx.cfg.synthetic_only:
        ctx.preparer = FacePreparer(ctx.cfg, out, None)  # landmarks are cached by prepare()
    ctx.status = ctx.run_manifest.get("status", {})
    ctx.timings = ctx.run_manifest.get("timings_s", {})


# ----------------------------------------------------------------------------
# Experiments 1 + 2: metrics on all primary images, inference, atlas
# ----------------------------------------------------------------------------
SENSITIVITY_SETTINGS = [
    {"nbins": 8}, {"nbins": 32}, {"sigma": 0.5}, {"sigma": 2.0},
]


def run_primary_metrics(ctx: RunContext) -> pd.DataFrame:
    assert ctx.cohort is not None and ctx.calibration is not None
    lock = read_json(ctx.out / "protocol.lock.json")
    excluded = set(lock["excluded_primary"])
    rows_all: list[dict[str, Any]] = []
    prim = ctx.cohort.primary[~ctx.cohort.primary["image_id"].isin(excluded)]
    t0 = time.perf_counter()
    base_s = ctx.settings
    for i, (_, row) in enumerate(prim.iterrows()):
        face = ctx.face_for(row)
        if face is None:
            continue
        surf = ctx.surface_for(face, "baseline")
        prod = analyse_image(face, surf, ctx.calibration, base_s, ctx.seed)
        rows_all.extend(prod.rows)
        save_maps(ctx.maps_path(face.image_id, "baseline"), prod)
        for kw in SENSITIVITY_SETTINGS:
            s2 = AnalysisSettings.from_cfg(ctx.cfg, maps=False, **kw)
            if "sigma" in kw:
                f2 = ctx.face_for(row, sigma=kw["sigma"])
                if f2 is None:
                    continue
                rows_all.extend(analyse_image(f2, surf, ctx.calibration, s2, ctx.seed, descriptors=("gaussian_curvature", "mean_curvature", "residual")).rows)
            else:
                rows_all.extend(analyse_image(face, surf, ctx.calibration, s2, ctx.seed, descriptors=("gaussian_curvature", "mean_curvature", "residual")).rows)
        if i == 9:
            dt = time.perf_counter() - t0
            ctx.log(f"first 10 primary images analysed in {dt:.1f}s -> projected {dt / 10 * len(prim):.0f}s")
    df = pd.DataFrame(rows_all)
    df.to_csv(ctx.out / "image_metrics.csv", index=False)
    ctx.timings["primary_metrics"] = time.perf_counter() - t0
    return df


def primary_inference(ctx: RunContext, df: pd.DataFrame) -> dict[str, Any]:
    cfg = ctx.cfg
    test = df[df["split"] == "test"]
    base = test[(test["nbins"] == cfg.analysis.primary_bins) & (test["sigma"] == cfg.analysis.smooth_sigma)]
    endpoints = {"binary_conditional": "binary_cond_otsu", "gaussian_curvature": "gaussian_curvature"}
    results: dict[str, Any] = {"primary": {}, "secondary": {}, "sensitivity": {}, "family_note": {}}
    subj_frames = []
    pvals = {}
    for ep, metric in endpoints.items():
        subj = st.subject_differences(base, metric, "fixed")
        subj["metric"] = metric
        subj_frames.append(subj)
        summ = st.paired_summary(subj, cfg.stats.bootstrap, cfg.stats.permutations, ctx.seed, cfg.stats.alpha)
        results["primary"][ep] = {"metric": metric, "estimator": "fixed (900 samples / 800 blocks)", **summ}
        pvals[ep] = summ["signflip_p"]
    adj = st.holm(pvals)
    for ep in endpoints:
        results["primary"][ep]["signflip_p_holm"] = adj[ep]
    results["family_note"]["primary"] = "Holm over the two confirmatory endpoints (sign-flip p-values)"
    # secondary metrics (own family)
    secondary = ["binary_marginal_otsu", "binary_block4_otsu", "mean_curvature", "laplacian_analysis", "grad_mag", "residual", "trust",
                 "normal_dir", "joint_KG_HM_R_27", "codebook_16", "codebook_32",
                 "binary_cond_t0.35", "binary_cond_t0.50", "binary_cond_t0.65", "binary_cond_otsu_rot1", "binary_cond_otsu_rot2", "binary_cond_otsu_rot3"]
    sec_p = {}
    for metric in secondary:
        for est in ("fixed",):
            subj = st.subject_differences(base, metric, est)
            if subj.empty:
                continue
            subj["metric"] = metric
            subj_frames.append(subj)
            summ = st.paired_summary(subj, cfg.stats.bootstrap, cfg.stats.permutations, ctx.seed, cfg.stats.alpha)
            results["secondary"][metric] = summ
            sec_p[metric] = summ["signflip_p"]
    adj2 = st.holm(sec_p)
    for m in sec_p:
        results["secondary"][m]["signflip_p_holm_secondary_family"] = adj2[m]
    results["family_note"]["secondary"] = f"Holm over {len(sec_p)} secondary metrics (exploratory)"
    # estimator / sampling sensitivity
    for metric in ("gaussian_curvature", "binary_cond_otsu"):
        for est in ("all", "surface_weighted"):
            subj = st.subject_differences(base, metric, est)
            if not subj.empty:
                results["sensitivity"][f"{metric}__{est}"] = st.paired_summary(subj, cfg.stats.bootstrap, cfg.stats.permutations, ctx.seed, cfg.stats.alpha)
    for nb in cfg.analysis.bins:
        sub = test[(test["nbins"] == nb) & (test["sigma"] == cfg.analysis.smooth_sigma)]
        subj = st.subject_differences(sub, "gaussian_curvature", "fixed")
        if not subj.empty:
            results["sensitivity"][f"gaussian_curvature__bins{nb}"] = st.paired_summary(subj, cfg.stats.bootstrap, cfg.stats.permutations, ctx.seed, cfg.stats.alpha)
    for sg in cfg.analysis.smooth_sigmas:
        sub = test[(test["nbins"] == cfg.analysis.primary_bins) & (np.isclose(test["sigma"], sg))]
        subj = st.subject_differences(sub, "gaussian_curvature", "fixed")
        if not subj.empty:
            results["sensitivity"][f"gaussian_curvature__sigma{sg}"] = st.paired_summary(subj, cfg.stats.bootstrap, cfg.stats.permutations, ctx.seed, cfg.stats.alpha)
    # Miller-Madow sensitivity for the surface endpoint
    mm = base[(base["metric"] == "gaussian_curvature") & (base["estimator"] == "fixed")].copy()
    mm["value"] = mm["value_mm"]
    subj = st.subject_differences(mm, "gaussian_curvature", "fixed")
    if not subj.empty:
        results["sensitivity"]["gaussian_curvature__miller_madow"] = st.paired_summary(subj, cfg.stats.bootstrap, cfg.stats.permutations, ctx.seed, cfg.stats.alpha)
    # calibration-split pilot SD -> power scenarios
    calib = df[(df["split"] == "calibration") & (df["nbins"] == cfg.analysis.primary_bins) & (df["sigma"] == cfg.analysis.smooth_sigma)]
    pw = {}
    for ep, metric in endpoints.items():
        sj = st.subject_differences(calib, metric, "fixed")
        sd = float(sj["D"].std(ddof=1)) if len(sj) > 1 else float("nan")
        pw[ep] = {"pilot_sd_from_calibration": sd, "n_pilot": int(len(sj)), **st.power_scenarios(sd), "ci_width_scenarios": st.ci_width_scenarios(sd) if np.isfinite(sd) else {}}
    results["power"] = pw
    # counts
    n_test_subjects = int(base["subject_id"].nunique())
    n_retained = int(base[base["retained"].astype(bool)]["subject_id"].nunique())
    results["counts"] = {"test_subjects": n_test_subjects, "test_subjects_with_all_patches_retained": n_retained,
                         "calibration_subjects": int(calib["subject_id"].nunique()), "images_per_subject": 1}
    results["seeds"] = {"bootstrap": ctx.seed, "permutation": ctx.seed}
    subj_df = pd.concat(subj_frames) if subj_frames else pd.DataFrame()
    subj_df.to_csv(ctx.out / "subject_metrics.csv", index=False)
    write_json(ctx.out / "primary_results.json", results)
    return results


def atlas(ctx: RunContext, df: pd.DataFrame) -> dict[str, Any]:
    """Dense landmark-relative atlas (aligned grid): mean maps over test subjects; max-location descriptives."""
    assert ctx.cohort is not None and ctx.calibration is not None
    s = ctx.settings
    test_ids = sorted(df[(df["split"] == "test") & df["retained"].astype(bool)]["image_id"].unique())
    keys = {"KG": f"gaussian_curvature_b{s.nbins}_w{s.window}", "BIN": f"binary_cond_otsu_w{s.window}", "HM": f"mean_curvature_b{s.nbins}_w{s.window}", "R": f"residual_b{s.nbins}_w{s.window}"}
    stacks: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    central_masks, feature_masks, valid_masks = [], [], []
    max_in_central: dict[str, list[bool]] = {k: [] for k in keys}
    enrich: dict[str, list[float]] = {k: [] for k in keys}
    kg_fields = []
    ys = xs = None
    for iid in test_ids:
        row = ctx.cohort.primary[ctx.cohort.primary["image_id"] == iid].iloc[0]
        face = ctx.face_for(row)
        if face is None:
            continue
        maps = load_maps(ctx.maps_path(iid, "baseline"))
        anchors = {k: tuple(v) for k, v in maps["_meta"]["patch_info"]["anchors"].items()}
        cm = reg.central_union_mask(anchors, s.patch_size, face.valid.shape)
        fm = reg.features_union_mask(face.landmarks, face.valid.shape)
        ys, xs = maps[keys["KG"]]["ys"], maps[keys["KG"]]["xs"]
        cm_g, fm_g, vm_g = cm[np.ix_(ys, xs)], fm[np.ix_(ys, xs)], face.valid[np.ix_(ys, xs)]
        central_masks.append(cm_g)
        feature_masks.append(fm_g)
        valid_masks.append(vm_g)
        for k, name in keys.items():
            m = maps[name]["map"]
            stacks[k].append(m)
            fin = np.isfinite(m)
            if fin.any():
                idx = np.unravel_index(np.nanargmax(np.where(fin, m, -np.inf)), m.shape)
                max_in_central[k].append(bool(fm_g[idx]))
                q = np.nanquantile(m[fin], 0.9)
                top = fin & (m >= q)
                frac_top_in = (top & fm_g).sum() / max(top.sum(), 1)
                frac_area = (fm_g & fin).sum() / max(fin.sum(), 1)
                enrich[k].append(float(frac_top_in / frac_area) if frac_area > 0 else np.nan)
        # between-subject field sample: K_G codes at grid points
        surf = ctx.surface_for(face, "baseline")
        from facial_entropy.pipeline import compute_fields  # noqa: PLC0415

        f = compute_fields(face, surf, ctx.calibration, s)
        codes, _, _ = ctx.calibration.quantizers["gaussian_curvature"][s.nbins].apply(f["gaussian_curvature"])
        kg_fields.append(np.where(face.valid, codes, -1)[np.ix_(ys, xs)])
    out: dict[str, Any] = {"n_subjects": len(central_masks), "ys": ys, "xs": xs}
    if not central_masks:
        return out
    for k in keys:
        arr = np.stack(stacks[k])
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            out[f"mean_{k}"] = np.nanmean(arr, axis=0)
        out[f"n_support_{k}"] = np.isfinite(arr).sum(axis=0)
        out[f"frac_max_in_features_mask_{k}"] = float(np.mean(max_in_central[k])) if max_in_central[k] else float("nan")
        out[f"top_decile_enrichment_{k}"] = {"mean": float(np.nanmean(enrich[k])), "median": float(np.nanmedian(enrich[k])), "n": len(enrich[k])}
    fm_mean = np.mean(np.stack(feature_masks), axis=0)
    vm_mean = np.mean(np.stack(valid_masks), axis=0)
    out["features_mask_area_fraction_of_valid"] = float(np.stack(feature_masks).sum() / max(np.stack(valid_masks).sum(), 1))
    out["central_patch_union_area_fraction_of_valid"] = float(np.stack(central_masks).sum() / max(np.stack(valid_masks).sum(), 1))
    out["features_mask_mean"] = fm_mean
    out["valid_mean"] = vm_mean
    # between-subject entropy at fixed coordinates (subject-balanced: one image per subject)
    K = np.stack(kg_fields)
    hb = np.full(K.shape[1:], np.nan)
    for i in range(K.shape[1]):
        for j in range(K.shape[2]):
            col = K[:, i, j]
            col = col[col >= 0]
            if len(col) >= max(10, K.shape[0] // 2):
                from facial_entropy.entropy import entropy_bits, hist_counts  # noqa: PLC0415

                hb[i, j] = entropy_bits(hist_counts(col, s.nbins))
    out["between_subject_KG_entropy"] = hb
    np.savez_compressed(ctx.out / "atlas.npz", **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
    write_json(ctx.out / "atlas_summary.json", {k: v for k, v in out.items() if not isinstance(v, np.ndarray)})
    return out
