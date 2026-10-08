"""Experiment 5: entropy-guided tile sampling policies — stability and what they select."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from facial_entropy import regions as reg
from facial_entropy.experiments import RunContext
from facial_entropy.pipeline import load_maps
from facial_entropy.util import read_json, rng_for, write_json

POLICIES = ("kg_entropy", "binary_cond_entropy", "gradient_magnitude", "central_first", "random")


def tile_grid(valid: np.ndarray, tile: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Complete tiles (fully inside valid support) as (row, col) anchors; returns tile-id map and list."""
    H, W = valid.shape
    tiles = []
    for y in range(0, H - tile + 1, tile):
        for x in range(0, W - tile + 1, tile):
            if valid[y : y + tile, x : x + tile].all():
                tiles.append((y, x))
    return np.array(tiles, dtype=int).reshape(-1, 2), tiles


def tile_scores(full_map: np.ndarray, tiles: list[tuple[int, int]], tile: int) -> np.ndarray:
    return np.array([np.nanmean(full_map[y : y + tile, x : x + tile]) for y, x in tiles])


def select_tiles(scores: np.ndarray, budget: int, tiles: list[tuple[int, int]]) -> list[int]:
    """Top-``budget`` tiles by score; ties broken by (row, col) ascending; NaN scores last."""
    order = sorted(range(len(tiles)), key=lambda i: (-(scores[i] if np.isfinite(scores[i]) else -np.inf), tiles[i][0], tiles[i][1]))
    return order[:budget]


def central_first_order(tiles: list[tuple[int, int]], lm: np.ndarray, tile: int) -> np.ndarray:
    """Fixed anatomical order: distance of tile centre to the nearest prespecified central patch centre,
    then to the nose tip, then row/col."""
    centers = reg.patch_centers(lm)
    cen = np.array([centers[k] for k in reg.CENTRAL_PATCHES])
    tc = np.array([[x + tile / 2, y + tile / 2] for y, x in tiles], dtype=float)
    d = np.min(np.linalg.norm(tc[:, None, :] - cen[None, :, :], axis=2), axis=1)
    return -d  # higher = closer


def run_policies(ctx: RunContext) -> dict[str, Any]:
    assert ctx.cohort is not None
    lock = read_json(ctx.out / "protocol.lock.json")
    ids = lock["robustness_subset"]
    s = ctx.settings
    tile = ctx.cfg.analysis.tile_size
    key = f"gaussian_curvature_b{s.nbins}_w{s.window}"
    bkey = f"binary_cond_otsu_w{s.window}"
    prim = ctx.cohort.primary.set_index("image_id")
    from facial_entropy.entropy import map_to_full  # noqa: PLC0415
    from facial_entropy.pipeline import compute_fields  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    pert_names = [p.name.split("__pert_")[1][:-4] for p in (ctx.out / "cache" / "maps").glob(f"{ids[0]}__pert_*.npz")] if ids else []
    for iid in ids:
        row = prim.loc[iid].copy()
        row["image_id"] = iid
        face = ctx.face_for(row)
        if face is None:
            continue
        surf = ctx.surface_for(face, "baseline")
        fields = compute_fields(face, surf, ctx.calibration, s)
        maps = load_maps(ctx.maps_path(iid, "baseline"))
        anchors = {k: tuple(v) for k, v in maps["_meta"]["patch_info"]["anchors"].items()}
        cmask = reg.central_union_mask(anchors, s.patch_size, face.valid.shape)
        fmask = reg.features_union_mask(face.landmarks, face.valid.shape)
        _, tiles = tile_grid(face.valid, tile)
        if not tiles:
            continue
        score_maps = {
            "kg_entropy": tile_scores(map_to_full(maps[key], face.valid.shape), tiles, tile),
            "binary_cond_entropy": tile_scores(map_to_full(maps[bkey], face.valid.shape), tiles, tile),
            "gradient_magnitude": tile_scores(np.sqrt(np.sum(np.stack(np.gradient(surf.luminance.astype(float))) ** 2, axis=0)), tiles, tile),
            "central_first": central_first_order(tiles, face.landmarks, tile),
        }
        pert_scores: dict[str, dict[str, np.ndarray]] = {}
        for pn in pert_names:
            p = ctx.maps_path(iid, f"pert_{pn}")
            if p.exists():
                pm = load_maps(p)
                pert_scores[pn] = {"kg_entropy": tile_scores(map_to_full(pm[key], face.valid.shape), tiles, tile),
                                   "binary_cond_entropy": tile_scores(map_to_full(pm[bkey], face.valid.shape), tiles, tile)}
        for q in ctx.cfg.analysis.policy_area_fractions:
            budget = int(np.floor(q * len(tiles)))
            if budget == 0:
                continue
            for pol in POLICIES:
                seeds = range(ctx.cfg.analysis.policy_random_seeds) if pol == "random" else [0]
                for sd in seeds:
                    if pol == "random":
                        sc = rng_for("policy-random", iid, sd, base=ctx.seed).random(len(tiles))
                    else:
                        sc = score_maps[pol]
                    sel = select_tiles(sc, budget, tiles)
                    selm = np.zeros(face.valid.shape, bool)
                    for i in sel:
                        y, x = tiles[i]
                        selm[y : y + tile, x : x + tile] = True
                    d = {"image_id": iid, "subject_id": row["subject_id"], "policy": pol, "seed": sd, "q_requested": q,
                         "q_achieved": float(selm.sum() / face.valid.sum()), "n_tiles": budget, "n_tiles_available": len(tiles),
                         "frac_selected_in_central_patches": float((selm & cmask).sum() / max(selm.sum(), 1)),
                         "frac_selected_in_features_mask": float((selm & fmask).sum() / max(selm.sum(), 1)),
                         "mean_KG_entropy_selected": float(np.nanmean(map_to_full(maps[key], face.valid.shape)[selm])),
                         "mean_binary_cond_selected": float(np.nanmean(map_to_full(maps[bkey], face.valid.shape)[selm]))}
                    # stability under perturbation (same budget): Jaccard of selected tile sets
                    if pol in ("kg_entropy", "binary_cond_entropy"):
                        jac = []
                        for pn, psc in pert_scores.items():
                            sel2 = set(select_tiles(psc[pol], budget, tiles))
                            jac.append(len(set(sel) & sel2) / max(len(set(sel) | sel2), 1))
                            d[f"jaccard_{pn}"] = jac[-1]
                        d["jaccard_mean_over_perturbations"] = float(np.mean(jac)) if jac else np.nan
                    rows.append(d)
    df = pd.DataFrame(rows)
    df.to_csv(ctx.out / "policy_metrics.csv", index=False)
    summary: dict[str, Any] = {"n_images": len(ids), "usefulness": "untested here (see Experiment 6)", "by_policy": {}}
    if not df.empty:
        for (pol, q), g in df.groupby(["policy", "q_requested"]):
            summary["by_policy"][f"{pol}@q{q}"] = {
                "n": int(len(g)), "q_achieved_mean": float(g["q_achieved"].mean()),
                "frac_in_central_patches_mean": float(g["frac_selected_in_central_patches"].mean()),
                "frac_in_features_mask_mean": float(g["frac_selected_in_features_mask"].mean()),
                "jaccard_stability_mean": float(g["jaccard_mean_over_perturbations"].mean()) if "jaccard_mean_over_perturbations" in g and g["jaccard_mean_over_perturbations"].notna().any() else None,
            }
    write_json(ctx.out / "policy_results.json", summary)
    return summary
