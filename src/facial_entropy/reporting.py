"""Render results.md from the saved machine-readable outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from facial_entropy.util import atomic_write_text, dir_size_bytes, read_json


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        if isinstance(x, (list, tuple)):
            return "[" + ", ".join(_f(v, nd) for v in x) + "]"
        if isinstance(x, bool):
            return str(x)
        xf = float(x)
        if not np.isfinite(xf):
            return "nan"
        return f"{xf:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _summ_row(name: str, s: dict[str, Any], extra_p: str | None = None) -> str:
    holm = s.get("signflip_p_holm", s.get("signflip_p_holm_secondary_family"))
    return (f"| {name} | {s['n_groups']} | {_f(s['central_mean'])} ± {_f(s['central_sd'])} | {_f(s['peripheral_mean'])} ± {_f(s['peripheral_sd'])} | "
            f"{_f(s['mean_diff'])} | {_f(s['median_diff'])} | {_f(s['ci95_bootstrap'])} | {_f(s['d_z'])} | {_f(s['prop_positive'], 2)} | "
            f"{_f(s['signflip_p'], 4)} | {_f(holm, 4) if holm is not None else 'n/a'} | {_f(s['signtest_p'], 4)} |")


def build_report(out: Path) -> str:
    rm = read_json(out / "run_manifest.json") if (out / "run_manifest.json").exists() else {}
    cfg = rm.get("profile", {})
    synthetic = bool(cfg.get("synthetic_only", False))
    L: list[str] = []
    L.append(f"# Regional facial entropy on the Hyperplane-Forge surface — results ({cfg.get('name', '?')} profile)\n")
    L.append(f"**Run type:** {'SYNTHETIC-ONLY (rendered pseudo-faces; no real-data conclusion)' if synthetic else 'REAL DATA: ' + str(rm.get('dataset', {}).get('name'))}  ")
    L.append(f"**Seed:** {rm.get('seed')}  **Status:** {rm.get('status', {})}  **Output:** `{out}`\n")

    # ---------------- findings first
    pr = read_json(out / "primary_results.json") if (out / "primary_results.json").exists() else None
    ctl = read_json(out / "controls.json") if (out / "controls.json").exists() else None
    L.append("## 1. Findings\n")
    if ctl:
        s = ctl["summary"]
        L.append(f"**Experiment 0 (controls):** {s['n_checks']} checks, {len(s.get('failed_integrity', []))} integrity failures, "
                 f"{len(s.get('audit_findings', []))} audit findings.  ")
        if s.get("failed_integrity"):
            L.append(f"**Integrity failures (block the corresponding claims): {s['failed_integrity']}**  ")
        if s.get("audit_findings"):
            L.append(f"Audit findings (solver discrepancies, reported not corrected in the baseline): {s['audit_findings']}  ")
        sa = ctl.get("solver_audit", {})
        if sa:
            L.append(f"\n**Solver audit.** The implemented update uses the divergence term with sign `{sa.get('implemented_divergence_sign')}`; "
                     f"the true gradient of the implemented discrete energy carries `{sa.get('energy_gradient_divergence_sign')}`. "
                     f"At the Rust `z*` the energy-gradient norm is {_f(sa.get('baseline_stationarity_ratio'))} × its value at `z_forged` "
                     f"(iterations {sa.get('baseline_iterations')}, converged flag {sa.get('baseline_converged_flag')}). "
                     f"The baseline `z*` is therefore a damped partial descent, not a stationary point of E. The baseline is preserved unchanged; "
                     f"`corrected_gradient_torch` (float64 L-BFGS on the validated replica energy) is reported as a separately labelled diagnostic variant "
                     f"(gradient ratio {_f(sa.get('corrected_variant', {}).get('params', {}).get('grad_norm_ratio'), 6)}).\n")
    if pr:
        c = pr["counts"]
        L.append(f"**Co-primary endpoints** (test split only; subjects = independent unit; {c['test_subjects']} test subjects, "
                 f"{c['test_subjects_with_all_patches_retained']} with all eight patches retained; {c['calibration_subjects']} calibration subjects never used for inference).\n")
        L.append("| endpoint | n subj | central mean ± SD (bits) | peripheral mean ± SD | mean D | median D | 95% boot CI | d_z | P(D>0) | sign-flip p | Holm p | sign-test p |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for ep, s in pr["primary"].items():
            L.append(_summ_row(f"{ep} (`{s['metric']}`)", s))
        L.append("")
        for ep, s in pr["primary"].items():
            sign = "positive (central > peripheral)" if s["mean_diff"] > 0 else "non-positive"
            sig = "Holm-adjusted p < 0.05" if (s.get("signflip_p_holm") is not None and s["signflip_p_holm"] < 0.05) else "not significant after Holm"
            L.append(f"- **{ep}:** estimated contrast is {sign}; {sig}. Interpretation: see decision rules (§6).")
        L.append("")
        L.append("**Secondary metrics (exploratory family, own Holm correction):**\n")
        L.append("| metric | n subj | central ± SD | peripheral ± SD | mean D | median D | 95% CI | d_z | P(D>0) | p | Holm p (secondary) | sign-test p |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for m, s in pr["secondary"].items():
            L.append(_summ_row(m, s))
        L.append("")
        L.append("**Estimator / discretisation sensitivity of the surface endpoint:**\n")
        L.append("| variant | n | mean D | 95% CI | p |")
        L.append("|---|---|---|---|---|")
        for k, s in pr["sensitivity"].items():
            L.append(f"| {k} | {s['n_groups']} | {_f(s['mean_diff'])} | {_f(s['ci95_bootstrap'])} | {_f(s['signflip_p'], 4)} |")
        L.append("")
    else:
        L.append("Primary endpoints: **not run** (no `primary_results.json`).\n")

    at = read_json(out / "atlas_summary.json") if (out / "atlas_summary.json").exists() else None
    if at:
        L.append("**Maximum-location claim (descriptive atlas, not a test):** fraction of test subjects whose highest supported local-entropy cell lies inside the "
                 f"prespecified features mask — K_G: {_f(at.get('frac_max_in_features_mask_KG'), 2)}, binary conditional: {_f(at.get('frac_max_in_features_mask_BIN'), 2)}; "
                 f"features-mask area fraction of valid support: {_f(at.get('features_mask_area_fraction_of_valid'), 2)}; "
                 f"top-decile enrichment (mean) K_G {_f(at.get('top_decile_enrichment_KG', {}).get('mean'))}, binary {_f(at.get('top_decile_enrichment_BIN', {}).get('mean'))}. "
                 "Ties, overlapping windows and non-uniform support complicate any uniform-area null; no naive significance test is attached.\n")

    ab = read_json(out / "ablation_results.json") if (out / "ablation_results.json").exists() else None
    if ab:
        L.append("### Experiment 3 — what the Forge construction adds\n")
        L.append("| variant | Δ_K common calib. (mean [CI]) | p | Δ_K within-variant calib. | paired Δ vs baseline [CI], p | map ρ vs baseline | top-decile overlap | patch-rank ρ | converged | runtime s |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for v, r in ab["variants"].items():
            dc, dw = r["delta_K_common_calibration"], r["delta_K_within_calibration"]
            pv = ab["paired_vs_baseline"].get(v)
            pvs = f"{_f(pv['mean_delta_D'])} {_f(pv['ci95'])}, p={_f(pv['signflip_p'], 3)}" if pv else "—"
            L.append(f"| {v} | {_f(dc['mean_diff'])} {_f(dc['ci95_bootstrap'])} | {_f(dc['signflip_p'], 3)} | {_f(dw['mean_diff'])} {_f(dw['ci95_bootstrap'])} | {pvs} | "
                     f"{_f(r.get('map_spearman_vs_baseline_mean'))} | {_f(r.get('map_top_decile_overlap_mean'))} | {_f(r.get('patch_rank_spearman_vs_baseline_mean'))} | {_f(r['converged_fraction'], 2)} | {_f(r['runtime_mean_s'])} |")
        L.append(f"\nMatched-smoothed luminance sigma chosen on calibration: {ab.get('luminance_smoothing_sigma')} px. Common calibration = baseline quantizers applied to every variant; within = quantizers refitted per variant on calibration images (sensitivity).\n")

    rb = read_json(out / "robustness_results.json") if (out / "robustness_results.json").exists() else None
    if rb and rb.get("by_perturbation"):
        L.append(f"### Experiment 4 — robustness (n = {rb['n_images']} prespecified images)\n")
        L.append("| perturbation | n | mean ΔD_K (bits) | SD | sign agreement | mean ΔD_binary | K_G map ρ | top-decile overlap | binary map ρ | K_G clip frac | retained |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for k, r in rb["by_perturbation"].items():
            L.append(f"| {k} | {r['n']} | {_f(r['dK_change_mean'])} | {_f(r['dK_change_sd'])} | {_f(r['dK_sign_agreement'], 2)} | {_f(r['dB_change_mean'])} | {_f(r['KG_map_spearman_mean'])} | "
                     f"{_f(r['KG_map_top_decile_overlap_mean'])} | {_f(r['BIN_map_spearman_mean'])} | {_f(r['KG_clip_fraction_mean'])} | {_f(r['retained_fraction'], 2)} |")
        L.append("\nPose was not numerically filtered: FEI pose index 11 (frontal neutral) is the inclusion rule; a landmark yaw proxy is recorded descriptively in `alignment_qa.json`. Glasses/facial hair are not annotated in FEI; no such strata could be formed.\n")

    po = read_json(out / "policy_results.json") if (out / "policy_results.json").exists() else None
    if po and po.get("by_policy"):
        L.append("### Experiment 5 — sampling policies (full-image-guided; usefulness untested here)\n")
        L.append("| policy@q | n | achieved q | frac in central patches | frac in features mask | Jaccard stability |")
        L.append("|---|---|---|---|---|---|")
        for k, r in po["by_policy"].items():
            L.append(f"| {k} | {r['n']} | {_f(r['q_achieved_mean'], 3)} | {_f(r['frac_in_central_patches_mean'], 2)} | {_f(r['frac_in_features_mask_mean'], 2)} | {_f(r['jaccard_stability_mean'], 2)} |")
        L.append("")

    vf = read_json(out / "verification_results.json") if (out / "verification_results.json").exists() else None
    if vf:
        L.append("### Experiment 6 — task-relevant information (face verification, unseen identities)\n")
        if not vf.get("ran"):
            L.append(f"Not run: {vf.get('reason')}.  Subjects with ≥4 images: {vf.get('n_subjects_with_ge4_images')}.\n")
        else:
            p = vf["pairs"]
            L.append(f"Identity split train/calibration/test = {len(vf['split']['train'])}/{len(vf['split']['calibration'])}/{len(vf['split']['test'])} subjects; "
                     f"test pairs {p['test']['n_pairs']} ({p['test']['n_positive']} positive) from {p['test']['n_subjects']} subjects — pairs are dependent, subjects are the unit.\n")
            res = vf["results"]
            L.append("| representation | dim | C | test AUROC [subject-boot CI] | balanced acc. | log loss | pred. info score (bits) | TPR@cal-FPR0.1 (achieved FPR) |")
            L.append("|---|---|---|---|---|---|---|---|")
            for blk in ("whole_face", "policy_only"):
                for n, r in res[blk].items():
                    t = r["test"]
                    L.append(f"| {blk}:{n} | {r['dim']} | {r['C']} | {_f(t['auroc'])} {_f(t.get('auroc_ci95_subject_bootstrap'))} | {_f(t['balanced_accuracy'])} | {_f(t['log_loss'])} | {_f(t.get('predictive_information_score_bits'))} | {_f(t['tpr_at_calibrated_fpr0.1_threshold'])} ({_f(t['fpr_at_calibrated_fpr0.1_threshold'])}) |")
            L.append("\n**Region-only verifiers** (same protocol, 32×32 gray patch → 16×16):\n")
            L.append("| region | class | AUROC [CI] |")
            L.append("|---|---|---|")
            for n, r in res["region_only"].items():
                L.append(f"| {n} | {r['region_class']} | {_f(r['test']['auroc'])} {_f(r['test'].get('auroc_ci95_subject_bootstrap'))} |")
            L.append("\n**Occlusion of the frozen whole-face verifier** (AUROC drop; two fill rules; surface variant re-runs the Rust construction on occluded RGB):\n")
            L.append("| region | fill | AUROC | drop |")
            L.append("|---|---|---|---|")
            for k, r in res["occlusion"].items():
                reg_, fill = k.split("|")
                L.append(f"| {reg_} | {fill} | {_f(r['auroc'])} | {_f(r['auroc_drop_vs_intact'])} |")
            L.append("")
            for m, r in res.get("entropy_vs_utility_ranking", {}).items():
                L.append(f"- Entropy-vs-utility rank agreement over {r['n_regions']} regions for `{m}`: Spearman ρ = {_f(r['spearman_entropy_vs_auroc_over_regions'])} ({r['note']}).")
            L.append(f"\nDeepfake task: {vf.get('deepfake_task')}.\n")

    # ---------------- how it was measured
    L.append("## 2. How it was measured\n")
    L.append(f"- Surface: existing Rust pipeline `forge_detect._core.solve_and_extract` via `CpuBackend` with unchanged defaults ({cfg.get('solver', {})}); backend recorded per image in the cache metadata (CPU Rust; never MPS).")
    L.append("- Random variable: `X_{F,A} = Q(F(U))`, `U` uniform over valid pixels of a 32×32 patch; plug-in entropy in bits; 900 fixed seeded samples per patch (800 block anchors for binary blocks).")
    L.append("- Quantizer: calibration-frozen asinh/log1p transform, 16 equal-width bins between calibration quantiles 0.005/0.995 (`calibration.json`); 8/32 bins as sensitivity.")
    L.append("- Geometry: `z̃ = (z* − median)/s_z`, Gaussian σ=1 px analysis smoothing, central differences with h = 1/255, `K_G`, `H_M`, normals as in protocol §4.3.")
    L.append("- Patches: protocol §5.4 rules on iBUG-68 landmarks (face-alignment 2DFAN-4 + S3FD), similarity alignment to (84,100)/(172,100)/(128,184).")
    L.append("- Inference: subjects are the unit; 10,000 subject bootstrap resamples; 10,000 seeded sign-flips; Holm over the two confirmatory endpoints; sign test as a different estimand.")
    coh = rm.get("cohort", {})
    L.append(f"- Cohort: requested {coh.get('requested_subjects')} subjects, available {coh.get('available_subjects')}, primary images {coh.get('n_primary')} "
             f"(calibration {coh.get('n_calibration')}, test {coh.get('n_test')}), secondary images {coh.get('n_secondary')}. Notes: {coh.get('notes')}")
    if (out / "exclusions.csv").exists():
        ex = pd.read_csv(out / "exclusions.csv")
        L.append(f"- Exclusions: {len(ex)} rows (`exclusions.csv`); reasons: {ex['reason'].value_counts().to_dict() if len(ex) else {}}")
    if pr:
        L.append("- Power (planning only, pilot SD from calibration subjects): " + "; ".join(
            f"{ep}: pilot SD {_f(v['pilot_sd_from_calibration'])} (n={v['n_pilot']}), n for d=0.3 at 80% power ≈ {v['power_0.8']['d_0.3']}" for ep, v in pr["power"].items()))
    L.append(f"- Resource use: timings {rm.get('timings_s', {})}; output directory size {dir_size_bytes(out)/1e6:.0f} MB; threads RAYON_NUM_THREADS={rm.get('environment', {}).get('rayon_num_threads_env')}.")
    L.append(f"- Source: forge repo commit {rm.get('forge_repo_git', {}).get('commit')} (dirty={rm.get('forge_repo_git', {}).get('dirty')}); source digest {rm.get('source_digest')}.\n")

    # ---------------- controls detail
    if ctl:
        L.append("## 3. Experiment 0 — control details\n")
        L.append("| check | category | passed | blocks |")
        L.append("|---|---|---|---|")
        for c in ctl["summary"]["checks"]:
            L.append(f"| {c['name']} | {c.get('category', 'integrity')} | {'✅' if c['passed'] else '❌'} | {c.get('blocks', '')} |")
        L.append("")
        nl = ctl.get("controls", {}).get("nonlocal", {})
        if nl:
            L.append("Non-local settlement influence (synthetic): " + "; ".join(f"{k}: max |Δz*| in the untouched patch = {_f(v['max_abs_dz_star_in_unperturbed_patch'], 5)}" for k, v in nl.items()) + ".")
        tr = ctl.get("controls", {}).get("translation", {})
        if tr:
            L.append("ROI-translation control (features drawn at offset from the ROI layout): " + "; ".join(f"offset {k}: D_K = {_f(v['D_KG'])}, D_bin = {_f(v['D_binary_cond'])}" for k, v in tr.items()) + ".\n")

    # ---------------- unrun stages + limitations
    L.append("## 4. Stages not run / limitations\n")
    status = rm.get("status", {})
    for stage in ("prepare", "exp0", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "report"):
        L.append(f"- {stage}: {status.get(stage, 'not run')}")
    L.append("")
    L.append("- The settled field is an algorithm-constructed proxy surface; nothing here validates it as a metric face reconstruction.")
    L.append("- Regional surface entropy does not localise the source of information: settlement is global (see non-local control).")
    L.append("- The landmark-polygon mask is an approximation, not a parsed skin mask; forehead extension may include hair for some subjects (see `figures/qa_overlays.png`).")
    L.append("- FEI provides one acquisition session per subject with a white background and controlled lighting; conclusions are population- and acquisition-dependent.")
    L.append("- Entropy is not salience, semantic relevance, identity, physical truth or authenticity; Experiment 6 measures utility for one lightweight decoder only.")
    L.append("- The exported Rust `energy_*` terms belong to the last *logged* iterate (every `pde_log_every` iterations), not necessarily to the returned `z*`.")
    if synthetic:
        L.append("- **This run used synthetic pseudo-faces only. No empirical claim about faces follows from it.**")
    L.append("\n## 5. Figures\n")
    figs = sorted((out / "figures").glob("*.png")) if (out / "figures").exists() else []
    for f in figs:
        L.append(f"- `figures/{f.name}`")
    L.append("\n## 6. Decision rules\n")
    L.append("See `protocol.md` §18 of the implementation prompt: a positive binary endpoint stable to thresholds supports a central advantage for the binary estimator; "
             "a positive surface endpoint alone supports a central advantage in this constructed surface's quantised curvature distribution; luminance/Forge agreement means a simpler explanation suffices; "
             "trust-dependent contrast implicates the trust prior; weak verification utility means entropy did not identify useful identity information for the tested decoder; "
             "non-positive or uncertain contrasts leave the hypothesis unsupported for this protocol.")
    text = "\n".join(L) + "\n"
    atomic_write_text(out / "results.md", text)
    return text
