"""Experiment 6: does regional entropy predict task-relevant information?  Default task: face verification.

Design (protocol §13): identity-disjoint 60/20/20 split of subjects with >= 4
frontal-ish photographs; fixed pair manifest; lightweight transparent verifiers
(regularised logistic regression on |descriptor difference|); region-only,
policy-only and occlusion analyses; cluster-aware (subject) bootstrap.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve

from facial_entropy import regions as reg
from facial_entropy.experiments import RunContext
from facial_entropy.geometry import geometry_fields
from facial_entropy.policies import POLICIES, central_first_order, select_tiles, tile_grid, tile_scores
from facial_entropy.util import read_json, rng_for, write_json


def _pool16(a: np.ndarray, size: int = 16) -> np.ndarray:
    H, W = a.shape
    return a.reshape(size, H // size, size, W // size).mean(axis=(1, 3))


def descriptors_for(ctx: RunContext, face, surf, calib) -> dict[str, np.ndarray]:
    g = geometry_fields(surf.z_star, face.valid, calib.height_scale, ctx.cfg.analysis.smooth_sigma, 1.0 / (face.size - 1))
    gray = surf.luminance.astype(np.float64)
    kg = np.arcsinh(g["gaussian_curvature"] / calib.quantizers["gaussian_curvature"][16].scale)
    desc = {
        "gray16": _pool16(gray).ravel(),
        "surface_z16": _pool16(g["z_smooth"]).ravel(),
        "surface_KG16": _pool16(kg).ravel(),
    }
    desc["combined"] = np.concatenate([desc["gray16"], desc["surface_z16"], desc["surface_KG16"]])
    try:
        from skimage.feature import hog  # noqa: PLC0415

        desc["hog"] = hog(gray, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2), feature_vector=True)
    except ImportError:
        pass
    return desc


def region_descriptors(face, surf, anchors: dict[str, tuple[int, int]], size: int) -> dict[str, np.ndarray]:
    gray = surf.luminance.astype(np.float64)
    out = {}
    for p in reg.PATCH_NAMES:
        y0, x0 = anchors[p]
        patch = gray[y0 : y0 + size, x0 : x0 + size]
        if patch.shape != (size, size):
            patch = np.zeros((size, size))
        out[p] = patch.reshape(16, size // 16, 16, size // 16).mean(axis=(1, 3)).ravel()
    return out


def make_pairs(ids_by_subject: dict[str, list[str]], seed: int, tag: str) -> pd.DataFrame:
    rows = []
    subs = sorted(ids_by_subject)
    for s in subs:
        ims = ids_by_subject[s]
        for i in range(len(ims)):
            for j in range(i + 1, len(ims)):
                rows.append({"a": ims[i], "b": ims[j], "subject_a": s, "subject_b": s, "label": 1})
    n_pos = len(rows)
    rng = rng_for("pairs", tag, base=seed)
    all_ims = [(s, im) for s in subs for im in ids_by_subject[s]]
    neg = set()
    tries = 0
    while len(neg) < n_pos and tries < 50 * n_pos:
        i, j = rng.integers(0, len(all_ims), size=2)
        tries += 1
        if all_ims[i][0] == all_ims[j][0]:
            continue
        key = tuple(sorted([all_ims[i][1], all_ims[j][1]]))
        neg.add((key, all_ims[i][0], all_ims[j][0]))
    for (a, b), sa, sb in [(k, sa, sb) for k, sa, sb in neg]:
        rows.append({"a": a, "b": b, "subject_a": sa, "subject_b": sb, "label": 0})
    return pd.DataFrame(rows)


def pair_features(pairs: pd.DataFrame, D: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([np.abs(D[a] - D[b]) for a, b in zip(pairs["a"], pairs["b"], strict=True)])


def fit_verifier(Xtr, ytr, Xcal, ycal, seed: int) -> tuple[Any, dict[str, Any]]:
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(sc.transform(Xtr), ytr)
        p = clf.predict_proba(sc.transform(Xcal))[:, 1]
        ll = log_loss(ycal, p, labels=[0, 1])
        if best is None or ll < best[0]:
            best = (ll, C, sc, clf)
    ll, C, sc, clf = best
    pc = clf.predict_proba(sc.transform(Xcal))[:, 1]
    # calibration-selected threshold for balanced accuracy and a fixed-FPR operating point
    ths = np.unique(pc)
    bal = [(0.5 * ((pc >= t)[ycal == 1].mean() + (pc < t)[ycal == 0].mean()), t) for t in ths]
    t_bal = max(bal)[1]
    fpr, tpr, thr = roc_curve(ycal, pc)
    target_fpr = 0.1
    idx = np.where(fpr <= target_fpr)[0]
    t_fpr = float(thr[idx[-1]]) if len(idx) else float("inf")
    return (sc, clf), {"C": C, "calibration_log_loss": float(ll), "threshold_balanced": float(t_bal), "threshold_fpr0.1": t_fpr}


def evaluate(model, info, X, y, pairs: pd.DataFrame, n_boot: int, seed: int) -> dict[str, Any]:
    sc, clf = model
    p = clf.predict_proba(sc.transform(X))[:, 1]
    pred = (p >= info["threshold_balanced"]).astype(int)
    out = {
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "balanced_accuracy": float(0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "predictive_information_score_bits": float(1.0 - log_loss(y, p, labels=[0, 1]) / np.log(2)) if abs(y.mean() - 0.5) < 0.05 else None,
        "tpr_at_calibrated_fpr0.1_threshold": float((p[y == 1] >= info["threshold_fpr0.1"]).mean()),
        "fpr_at_calibrated_fpr0.1_threshold": float((p[y == 0] >= info["threshold_fpr0.1"]).mean()),
        "n_pairs": int(len(y)), "n_positive": int(y.sum()), "n_subjects": int(pd.concat([pairs["subject_a"], pairs["subject_b"]]).nunique()),
    }
    # cluster-aware bootstrap: resample test subjects, keep pairs whose both subjects are in the sample (weighted by multiplicity)
    subs = sorted(set(pairs["subject_a"]) | set(pairs["subject_b"]))
    rng = rng_for("verif-boot", base=seed)
    aucs = []
    for _ in range(n_boot):
        samp = rng.choice(subs, size=len(subs), replace=True)
        mult = pd.Series(samp).value_counts()
        w = pairs["subject_a"].map(mult).fillna(0).to_numpy() * pairs["subject_b"].map(mult).fillna(0).to_numpy()
        w = np.where(pairs["subject_a"] == pairs["subject_b"], pairs["subject_a"].map(mult).fillna(0).to_numpy(), w)
        m = w > 0
        if len(np.unique(y[m])) < 2:
            continue
        aucs.append(roc_auc_score(y[m], p[m], sample_weight=w[m]))
    out["auroc_ci95_subject_bootstrap"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))] if aucs else None
    out["bootstrap_B"] = len(aucs)
    return out


def run_verification(ctx: RunContext) -> dict[str, Any]:
    assert ctx.cohort is not None and ctx.calibration is not None
    cfg = ctx.cfg
    t0 = time.perf_counter()
    lock = read_json(ctx.out / "protocol.lock.json")
    excluded = set(pd.read_csv(ctx.out / "exclusions.csv")["image_id"]) if (ctx.out / "exclusions.csv").exists() else set()
    all_rows = pd.concat([ctx.cohort.primary, ctx.cohort.secondary])
    all_rows = all_rows[~all_rows["image_id"].isin(excluded)]
    by_subj = {s: sorted(g["image_id"]) for s, g in all_rows.groupby("subject_id")}
    by_subj = {s: v for s, v in by_subj.items() if len(v) >= 4}
    status: dict[str, Any] = {"n_subjects_with_ge4_images": len(by_subj), "images_per_subject": {s: len(v) for s, v in by_subj.items()}}
    if len(by_subj) < 15:
        status["ran"] = False
        status["reason"] = "fewer than 15 subjects with >= 4 photographs; verification not run"
        write_json(ctx.out / "verification_results.json", status)
        return status
    subs = sorted(by_subj)
    rng = rng_for("verif-split", base=ctx.seed)
    perm = list(rng.permutation(subs))
    n = len(perm)
    n_tr, n_cal = int(0.6 * n), int(0.2 * n)
    split = {**{s: "train" for s in perm[:n_tr]}, **{s: "calibration" for s in perm[n_tr : n_tr + n_cal]}, **{s: "test" for s in perm[n_tr + n_cal :]}}
    pairs = {sp: make_pairs({s: by_subj[s] for s in subs if split[s] == sp}, ctx.seed, sp) for sp in ("train", "calibration", "test")}
    for sp, pdf in pairs.items():
        pdf["split"] = sp
    pd.concat(pairs.values()).to_csv(ctx.out / "verification_pairs.csv", index=False)
    status["pairs"] = {sp: {"n_pairs": int(len(p)), "n_positive": int(p["label"].sum()), "n_subjects": int(len([s for s in subs if split[s] == sp]))} for sp, p in pairs.items()}
    status["note"] = "pair counts are not independent sample counts; subjects are the independent unit"

    # descriptors for all images (baseline surfaces)
    D: dict[str, dict[str, np.ndarray]] = {}
    R: dict[str, dict[str, np.ndarray]] = {}
    faces: dict[str, Any] = {}
    s = ctx.settings
    rows_by_id = all_rows.set_index("image_id")
    for s_id in subs:
        for iid in by_subj[s_id]:
            row = rows_by_id.loc[iid].copy()
            row["image_id"] = iid
            face = ctx.face_for(row)
            if face is None:
                continue
            surf = ctx.surface_for(face, "baseline")
            D[iid] = descriptors_for(ctx, face, surf, ctx.calibration)
            from facial_entropy.pipeline import patch_layout  # noqa: PLC0415

            anchors = {k: tuple(v) for k, v in patch_layout(face, s)["anchors"].items()}
            R[iid] = region_descriptors(face, surf, anchors, s.patch_size)
            faces[iid] = (face, surf, anchors)
    for sp in pairs:
        pairs[sp] = pairs[sp][pairs[sp]["a"].isin(D) & pairs[sp]["b"].isin(D)].reset_index(drop=True)
    y = {sp: pairs[sp]["label"].to_numpy() for sp in pairs}
    results: dict[str, Any] = {"whole_face": {}, "region_only": {}, "policy_only": {}, "occlusion": {}}
    models = {}
    for name in ("gray16", "hog", "surface_z16", "surface_KG16", "combined"):
        if name not in next(iter(D.values())):
            continue
        X = {sp: pair_features(pairs[sp], {k: v[name] for k, v in D.items()}) for sp in pairs}
        model, info = fit_verifier(X["train"], y["train"], X["calibration"], y["calibration"], ctx.seed)
        models[name] = (model, info)
        results["whole_face"][name] = {"dim": int(X["train"].shape[1]), **info, "test": evaluate(model, info, X["test"], y["test"], pairs["test"], cfg.stats.verification_bootstrap, ctx.seed)}
    # region-only verifiers
    for p in reg.PATCH_NAMES:
        X = {sp: pair_features(pairs[sp], {k: v[p] for k, v in R.items()}) for sp in pairs}
        model, info = fit_verifier(X["train"], y["train"], X["calibration"], y["calibration"], ctx.seed)
        results["region_only"][p] = {"region_class": reg.PATCH_CLASS[p], "dim": 256, **info, "test": evaluate(model, info, X["test"], y["test"], pairs["test"], cfg.stats.verification_bootstrap, ctx.seed)}
    # occlusion of the frozen gray16 whole-face verifier (two fill rules)
    model, info = models["gray16"]
    for p in reg.PATCH_NAMES:
        for fill in ("mean_gray_0.5", "face_median"):
            Dp = {}
            for iid, (face, surf, anchors) in faces.items():
                g = surf.luminance.astype(np.float64).copy()
                y0, x0 = anchors[p]
                val = 0.5 if fill == "mean_gray_0.5" else float(np.median(g[face.valid]))
                g[y0 : y0 + s.patch_size, x0 : x0 + s.patch_size] = val
                Dp[iid] = _pool16(g).ravel()
            X = pair_features(pairs["test"], Dp)
            ev = evaluate(model, info, X, y["test"], pairs["test"], 0, ctx.seed)
            results["occlusion"][f"{p}|{fill}"] = {"auroc": ev["auroc"], "auroc_drop_vs_intact": results["whole_face"]["gray16"]["test"]["auroc"] - ev["auroc"]}
    # surface-descriptor occlusion applied to RGB before the Rust call (subset of regions for cost: all 8, mean fill)
    if "surface_z16" in models:
        model_s, info_s = models["surface_z16"]
        for p in reg.PATCH_NAMES:
            Dp = {}
            for iid, (face, surf, anchors) in faces.items():
                if rows_by_id.loc[iid].get("split", "") != "test" and split.get(rows_by_id.loc[iid]["subject_id"]) != "test":
                    continue
                rgb = face.rgb.copy()
                y0, x0 = anchors[p]
                rgb[y0 : y0 + s.patch_size, x0 : x0 + s.patch_size] = 0.5
                su = ctx.surface_for(face, "baseline", rgb_override=rgb)
                Dp[iid] = descriptors_for(ctx, face, su, ctx.calibration)["surface_z16"]
            pt = pairs["test"][pairs["test"]["a"].isin(Dp) & pairs["test"]["b"].isin(Dp)]
            if len(pt) and pt["label"].nunique() > 1:
                X = pair_features(pt, Dp)
                ev = evaluate(model_s, info_s, X, pt["label"].to_numpy(), pt, 0, ctx.seed)
                results["occlusion"][f"{p}|surface_rgb_fill_0.5"] = {"auroc": ev["auroc"], "auroc_drop_vs_intact": results["whole_face"]["surface_z16"]["test"]["auroc"] - ev["auroc"]}
    # policy-only verifiers (tile mask x gray + mask channel), q = 0.2, from each image's own maps
    from facial_entropy.entropy import map_to_full  # noqa: PLC0415
    from facial_entropy.pipeline import AnalysisSettings, analyse_image  # noqa: PLC0415

    tile = cfg.analysis.tile_size
    s_map = AnalysisSettings.from_cfg(cfg, extra_windows=())
    key = f"gaussian_curvature_b{s.nbins}_w{s.window}"
    bkey = f"binary_cond_otsu_w{s.window}"
    pol_desc: dict[str, dict[str, np.ndarray]] = {p: {} for p in POLICIES}
    for iid, (face, surf, anchors) in faces.items():
        prod = analyse_image(face, surf, ctx.calibration, s_map, ctx.seed, descriptors=("gaussian_curvature",))
        _, tiles = tile_grid(face.valid, tile)
        if not tiles:
            continue
        budget = max(1, int(np.floor(0.2 * len(tiles))))
        gray = surf.luminance.astype(np.float64)
        scores = {
            "kg_entropy": tile_scores(map_to_full(prod.maps[key], face.valid.shape), tiles, tile),
            "binary_cond_entropy": tile_scores(map_to_full(prod.maps[bkey], face.valid.shape), tiles, tile),
            "gradient_magnitude": tile_scores(np.sqrt(np.sum(np.stack(np.gradient(gray)) ** 2, axis=0)), tiles, tile),
            "central_first": central_first_order(tiles, face.landmarks, tile),
            "random": rng_for("policy-random", iid, 0, base=ctx.seed).random(len(tiles)),
        }
        for pol in POLICIES:
            sel = select_tiles(scores[pol], budget, tiles)
            m = np.zeros(face.valid.shape, bool)
            for i in sel:
                yy, xx = tiles[i]
                m[yy : yy + tile, xx : xx + tile] = True
            pol_desc[pol][iid] = np.concatenate([_pool16(gray * m).ravel(), _pool16(m.astype(float)).ravel()])
    for pol in POLICIES:
        X = {sp: pair_features(pairs[sp], pol_desc[pol]) for sp in pairs}
        model, info = fit_verifier(X["train"], y["train"], X["calibration"], y["calibration"], ctx.seed)
        results["policy_only"][pol] = {"q": 0.2, "dim": 512, **info, "test": evaluate(model, info, X["test"], y["test"], pairs["test"], cfg.stats.verification_bootstrap, ctx.seed)}
    # entropy ranking vs utility ranking over the 8 regions (test subjects' primary images)
    im = pd.read_csv(ctx.out / "image_metrics.csv") if (ctx.out / "image_metrics.csv").exists() else pd.DataFrame()
    rank = {}
    if not im.empty:
        test_subj = [s_ for s_ in subs if split[s_] == "test"]
        base = im[(im["subject_id"].isin(test_subj)) & (im["nbins"] == cfg.analysis.primary_bins) & (im["sigma"] == cfg.analysis.smooth_sigma) & (im["estimator"] == "fixed")]
        for metric in ("gaussian_curvature", "binary_cond_otsu"):
            ent_by_region = base[base["metric"] == metric].groupby("patch")["value"].mean()
            util_by_region = pd.Series({p: results["region_only"][p]["test"]["auroc"] for p in reg.PATCH_NAMES})
            from facial_entropy.statistics import spearman  # noqa: PLC0415

            common = [p for p in reg.PATCH_NAMES if p in ent_by_region]
            rank[metric] = {"spearman_entropy_vs_auroc_over_regions": spearman(ent_by_region[common].to_numpy(), util_by_region[common].to_numpy()),
                            "n_regions": len(common), "entropy_by_region": ent_by_region[common].to_dict(), "auroc_by_region": util_by_region[common].to_dict(),
                            "note": "eight dependent anatomical regions give very few ranking points; descriptive only"}
    results["entropy_vs_utility_ranking"] = rank
    status.update({"ran": True, "split": {sp: [s_ for s_ in subs if split[s_] == sp] for sp in ("train", "calibration", "test")}, "results": results,
                   "runtime_s": time.perf_counter() - t0, "deepfake_task": "not applicable: FEI has no manipulated images / original-manipulated links"})
    write_json(ctx.out / "verification_results.json", status)
    return status
