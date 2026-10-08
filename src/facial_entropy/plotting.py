"""Figures: QA overlays, example panels, paired differences, atlas, ablation/robustness/policy/verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from facial_entropy import regions as reg  # noqa: E402
from facial_entropy.alignment import invert_affine  # noqa: E402
from facial_entropy.entropy import map_to_full  # noqa: E402
from facial_entropy.experiments import RunContext  # noqa: E402
from facial_entropy.pipeline import analyse_image, load_maps, load_rgb  # noqa: E402
from facial_entropy.util import read_json  # noqa: E402


def _draw_rois(ax, anchors: dict[str, Any], size: int) -> None:
    for k, (y0, x0) in anchors.items():
        col = "cyan" if reg.PATCH_CLASS[k] == "central" else "orange"
        ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), size, size, fill=False, edgecolor=col, linewidth=1.2))


def qa_overlays(ctx: RunContext, figdir: Path, n: int = 8) -> list[str]:
    assert ctx.cohort is not None
    ids = sorted(ctx.cohort.primary["image_id"])[:n]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    s = ctx.settings
    for ax, iid in zip(axes.ravel(), ids, strict=False):
        row = ctx.cohort.primary[ctx.cohort.primary["image_id"] == iid].iloc[0]
        face = ctx.face_for(row)
        if face is None:
            ax.set_title(f"{iid}: no face")
            continue
        ax.imshow(face.rgb)
        ax.contour(face.face_mask, levels=[0.5], colors="yellow", linewidths=0.8)
        ax.contour(face.valid, levels=[0.5], colors="lime", linewidths=0.8)
        ax.scatter(face.landmarks[:, 0], face.landmarks[:, 1], s=4, c="red")
        from facial_entropy.pipeline import patch_layout  # noqa: PLC0415

        lay = patch_layout(face, s)
        _draw_rois(ax, {k: tuple(v) for k, v in lay["anchors"].items()}, s.patch_size)
        ax.set_title(f"{iid} retained={lay['retained']} res={face.fit_residual_px:.2f}px", fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(ids):]:
        ax.axis("off")
    fig.suptitle("QA: landmark polygon mask (yellow), eroded analysis support (green), 68 landmarks, ROIs (cyan=central, orange=peripheral)")
    p = figdir / "qa_overlays.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return [str(p)]


def example_panels(ctx: RunContext, figdir: Path, df: pd.DataFrame, calib) -> list[str]:
    """Deterministic examples: test subjects at D_K quantiles 0, .25, .5, .75, 1 (min = counterexample)."""
    assert ctx.cohort is not None
    from facial_entropy import statistics as st  # noqa: PLC0415

    cfg = ctx.cfg
    base = df[(df["split"] == "test") & (df["nbins"] == cfg.analysis.primary_bins) & (df["sigma"] == cfg.analysis.smooth_sigma)]
    subj = st.subject_differences(base, "gaussian_curvature", "fixed").sort_values("D")
    if subj.empty:
        return []
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    picks = [subj.iloc[min(int(round(q * (len(subj) - 1))), len(subj) - 1)]["subject_id"] for q in qs]
    out = []
    s = ctx.settings
    key = f"gaussian_curvature_b{s.nbins}_w{s.window}"
    vmax_kg = None
    for q, sid in zip(qs, picks, strict=True):
        row = ctx.cohort.primary[ctx.cohort.primary["subject_id"] == sid].iloc[0]
        face = ctx.face_for(row)
        if face is None:
            continue
        surf = ctx.surface_for(face, "baseline")
        prod = analyse_image(face, surf, calib, s, ctx.seed)
        anchors = {k: tuple(v) for k, v in prod.patch_info["anchors"].items()}
        fig, axes = plt.subplots(2, 5, figsize=(22, 9))
        ax = axes.ravel()
        ax[0].imshow(face.rgb); ax[0].set_title(f"{row['image_id']} aligned RGB (D_K quantile {q})")
        ax[1].imshow(prod.fields["binary_otsu"], cmap="gray"); ax[1].set_title(f"binary (Otsu t={prod.flags['otsu_threshold']:.3f})")
        ax[2].imshow(face.rgb); ax[2].contour(face.valid, levels=[0.5], colors="lime"); _draw_rois(ax[2], anchors, s.patch_size); ax[2].set_title("support + ROIs")
        zt = np.where(face.valid, prod.fields["z_tilde"], np.nan)
        ax[3].imshow(zt, cmap="viridis"); ax[3].set_title("z* (median-centred, common scale)")
        kg = np.where(face.valid, np.arcsinh(prod.fields["gaussian_curvature"] / calib.quantizers["gaussian_curvature"][16].scale), np.nan)
        ax[4].imshow(kg, cmap="coolwarm", vmin=-4, vmax=4); ax[4].set_title("asinh(K_G / s_K)")
        for i, (name, title, vmin, vmax) in enumerate([
            (key, f"local H(K_G) bits, 16 bins, w={s.window}", 0, 4),
            (f"mean_curvature_b{s.nbins}_w{s.window}", "local H(H_M) bits", 0, 4),
            (f"residual_b{s.nbins}_w{s.window}", "local H(R) bits", 0, 4),
            (f"binary_cond_otsu_w{s.window}", "local H(B11|B00,B01,B10) bits", 0, 1),
        ]):
            m = map_to_full(prod.maps[name], face.valid.shape)
            im = ax[5 + i].imshow(m, cmap="magma", vmin=vmin, vmax=vmax)
            ax[5 + i].set_title(title)
            plt.colorbar(im, ax=ax[5 + i], fraction=0.046)
        ax[9].imshow(surf.trust, cmap="gray", vmin=0, vmax=1); ax[9].set_title(f"trust W; converged={surf.converged} it={surf.iterations}")
        for a in ax:
            a.axis("off")
        fig.suptitle("Each heatmap pixel = neighbourhood entropy estimate (stride 4 grid, shown nearest-neighbour); not entropy of that pixel", fontsize=10)
        p = figdir / f"example_q{int(q*100):03d}_{row['image_id']}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=90)
        plt.close(fig)
        out.append(str(p))
        # projection to the original image (real data only)
        if not cfg.synthetic_only and Path(str(row["path"])).exists():
            orig = load_rgb(row["path"])
            Ai = invert_affine(face.transform)
            m = map_to_full(prod.maps[key], face.valid.shape)
            from scipy.ndimage import map_coordinates  # noqa: PLC0415

            H, W = orig.shape[:2]
            us, vs = np.meshgrid(np.arange(W), np.arange(H))
            A = face.transform
            al = np.stack([us.ravel(), vs.ravel()], 1) @ A[:, :2].T + A[:, 2]
            mm = np.nan_to_num(m, nan=-1)
            proj = map_coordinates(mm, [al[:, 1].reshape(H, W), al[:, 0].reshape(H, W)], order=0, mode="constant", cval=-1)
            proj = np.where(proj < 0, np.nan, proj)
            fig, ax2 = plt.subplots(1, 2, figsize=(14, 6))
            ax2[0].imshow(orig); ax2[0].set_title("original image")
            ax2[1].imshow(orig); ax2[1].imshow(proj, cmap="magma", alpha=0.6, vmin=0, vmax=4); ax2[1].set_title("local H(K_G) projected via inverse similarity (display only)")
            for a in ax2:
                a.axis("off")
            p2 = figdir / f"projected_{row['image_id']}.png"
            fig.tight_layout(); fig.savefig(p2, dpi=90); plt.close(fig)
            out.append(str(p2))
    return out


def paired_difference_plot(ctx: RunContext, figdir: Path, results: dict[str, Any]) -> list[str]:
    subj = pd.read_csv(ctx.out / "subject_metrics.csv") if (ctx.out / "subject_metrics.csv").exists() else pd.DataFrame()
    if subj.empty:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (ep, metric) in zip(axes, (("binary_conditional", "binary_cond_otsu"), ("gaussian_curvature", "gaussian_curvature")), strict=True):
        d = subj[subj["metric"] == metric]["D"].to_numpy()
        r = results["primary"][ep]
        ax.axhline(0, color="k", lw=0.8)
        ax.scatter(np.random.default_rng(0).normal(0, 0.04, len(d)), d, s=10, alpha=0.6)
        ax.errorbar([0.3], [r["mean_diff"]], yerr=[[r["mean_diff"] - r["ci95_bootstrap"][0]], [r["ci95_bootstrap"][1] - r["mean_diff"]]], fmt="o", color="red", capsize=4)
        ax.set_xlim(-0.3, 0.6); ax.set_xticks([])
        ax.set_title(f"{metric}: D_s = C - P per subject (n={r['n_groups']})\nmean {r['mean_diff']:.3f} [{r['ci95_bootstrap'][0]:.3f},{r['ci95_bootstrap'][1]:.3f}] bits, p={r['signflip_p']:.3g}", fontsize=9)
        ax.set_ylabel("bits")
    p = figdir / "paired_differences.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    return [str(p)]


def atlas_figure(ctx: RunContext, figdir: Path) -> list[str]:
    p = ctx.out / "atlas.npz"
    if not p.exists():
        return []
    z = np.load(p)
    keys = [k for k in ("mean_KG", "mean_BIN", "mean_HM", "mean_R", "between_subject_KG_entropy") if k in z]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 4.5))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys, strict=True):
        im = ax.imshow(z[k], cmap="magma")
        if "features_mask_mean" in z:
            ax.contour(z["features_mask_mean"], levels=[0.5], colors="cyan", linewidths=0.8)
        ax.set_title(k if k != "between_subject_KG_entropy" else "between-subject H(Q(K_G)) at fixed coords\n(exploratory, separate variable)", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis("off")
    fig.suptitle("Aligned-grid atlas: mean local entropy maps over test subjects (stride-4 grid); cyan = prespecified features mask", fontsize=9)
    out = figdir / "atlas.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return [str(out)]


def ablation_figure(ctx: RunContext, figdir: Path) -> list[str]:
    p = ctx.out / "ablation_results.json"
    if not p.exists():
        return []
    r = read_json(p)["variants"]
    names = list(r)
    means = [r[n]["delta_K_common_calibration"]["mean_diff"] for n in names]
    cis = np.array([r[n]["delta_K_common_calibration"]["ci95_bootstrap"] for n in names], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=[np.array(means) - cis[:, 0], cis[:, 1] - np.array(means)], capsize=3, color="steelblue")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Delta_K (bits), common calibration")
    ax.set_title("Experiment 3: central-minus-peripheral K_G entropy by construction variant (95% subject bootstrap)")
    out = figdir / "ablations.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return [str(out)]


def robustness_figure(ctx: RunContext, figdir: Path) -> list[str]:
    p = ctx.out / "robustness_results.json"
    if not p.exists():
        return []
    r = read_json(p)["by_perturbation"]
    if not r:
        return []
    names = list(r)
    ch = [r[n]["dK_change_mean"] for n in names]
    rho = [r[n]["KG_map_spearman_mean"] if r[n]["KG_map_spearman_mean"] is not None else np.nan for n in names]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = np.arange(len(names))
    axes[0].bar(x, ch, color="indianred"); axes[0].axhline(0, color="k", lw=0.8); axes[0].set_ylabel("mean change in D_K (bits)")
    axes[1].bar(x, rho, color="seagreen"); axes[1].set_ylabel("map Spearman vs baseline"); axes[1].set_ylim(-1, 1)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    fig.suptitle("Experiment 4: robustness of the surface endpoint on the prespecified subset")
    out = figdir / "robustness.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return [str(out)]


def policy_figure(ctx: RunContext, figdir: Path) -> list[str]:
    p = ctx.out / "policy_results.json"
    if not p.exists():
        return []
    r = read_json(p)["by_policy"]
    if not r:
        return []
    names = list(r)
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r[n]["frac_in_features_mask_mean"] for n in names], width=0.4, label="fraction of selected area in features mask")
    ax.bar(x + 0.2, [r[n]["jaccard_stability_mean"] or 0 for n in names], width=0.4, label="Jaccard stability under perturbations (entropy policies)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7); ax.legend(fontsize=8)
    ax.set_title("Experiment 5: sampling policies (usefulness untested here)")
    out = figdir / "policies.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return [str(out)]


def verification_figure(ctx: RunContext, figdir: Path) -> list[str]:
    p = ctx.out / "verification_results.json"
    if not p.exists():
        return []
    v = read_json(p)
    if not v.get("ran"):
        return []
    res = v["results"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (title, block) in zip(axes, (("whole-face descriptors", res["whole_face"]), ("region-only (gray 32x32 -> 16x16)", res["region_only"]), ("policy-only (q=0.2 tiles)", res["policy_only"])), strict=True):
        names = list(block)
        auc = [block[n]["test"]["auroc"] for n in names]
        ci = [block[n]["test"].get("auroc_ci95_subject_bootstrap") or [np.nan, np.nan] for n in names]
        ci = np.array(ci, dtype=float)
        x = np.arange(len(names))
        ax.bar(x, auc, yerr=[np.array(auc) - ci[:, 0], ci[:, 1] - np.array(auc)], capsize=3, color="slateblue")
        ax.axhline(0.5, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7); ax.set_ylim(0.4, 1.0); ax.set_title(title, fontsize=9); ax.set_ylabel("test AUROC")
    fig.suptitle("Experiment 6: face verification on unseen identities (subject-bootstrap 95% CI)")
    out = figdir / "verification.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return [str(out)]


def make_all_figures(ctx: RunContext, df: pd.DataFrame | None, results: dict[str, Any] | None) -> list[str]:
    figdir = ctx.out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    try:
        paths += qa_overlays(ctx, figdir)
    except Exception as exc:  # noqa: BLE001
        ctx.log(f"qa figure failed: {exc!r}")
    if df is not None and ctx.calibration is not None:
        try:
            paths += example_panels(ctx, figdir, df, ctx.calibration)
        except Exception as exc:  # noqa: BLE001
            ctx.log(f"example panels failed: {exc!r}")
    if results is not None:
        try:
            paths += paired_difference_plot(ctx, figdir, results)
        except Exception as exc:  # noqa: BLE001
            ctx.log(f"paired plot failed: {exc!r}")
    for fn in (atlas_figure, ablation_figure, robustness_figure, policy_figure, verification_figure):
        try:
            paths += fn(ctx, figdir)
        except Exception as exc:  # noqa: BLE001
            ctx.log(f"{fn.__name__} failed: {exc!r}")
    return paths
