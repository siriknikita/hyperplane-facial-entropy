"""Per-image processing: faces, cached surfaces, calibration fitting, patch metrics, local maps."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from facial_entropy import entropy as ent
from facial_entropy import regions as reg
from facial_entropy.alignment import AlignedFace, LandmarkAdapter, align_face, support_erosion_radius
from facial_entropy.config import ProfileConfig
from facial_entropy.geometry import DESCRIPTORS, geometry_fields, normal_direction_codes, robust_scale
from facial_entropy.quantization import Calibration, fit_codebook, fit_scalar_quantizer, fit_terciles
from facial_entropy.rust_adapter import SurfaceResult, solve_surface, variant_key
from facial_entropy.util import atomic_savez, read_json, rng_for, sha256_json, stable_seed, write_json


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


# ----------------------------------------------------------------------------
# Landmarks + alignment (cached per image SHA)
# ----------------------------------------------------------------------------
class FacePreparer:
    def __init__(self, cfg: ProfileConfig, out: Path, landmarker: LandmarkAdapter | None) -> None:
        self.cfg = cfg
        self.out = out
        self.lm_dir = out / "landmarks"
        self.lm_dir.mkdir(parents=True, exist_ok=True)
        self.landmarker = landmarker

    def landmarks_for(self, sha: str, rgb: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
        p = self.lm_dir / f"{sha}.json"
        if p.exists():
            d = read_json(p)
            return (np.asarray(d["landmarks"]) if d["landmarks"] is not None else None), d["meta"]
        if self.landmarker is None:
            raise RuntimeError("landmarks not cached and no landmark adapter available")
        lm, meta = self.landmarker.landmarks(rgb)
        meta = {**meta, "version": self.landmarker.version}
        write_json(p, {"landmarks": None if lm is None else lm.tolist(), "meta": meta})
        return lm, meta

    def prepare(self, row: pd.Series, size: int | None = None, rgb: np.ndarray | None = None,
                landmarks: np.ndarray | None = None, sigma: float | None = None) -> AlignedFace | None:
        a = self.cfg.analysis
        size = size or a.aligned_size
        sigma = a.smooth_sigma if sigma is None else sigma
        rgb = load_rgb(row["path"]) if rgb is None else rgb
        if landmarks is None:
            lm, meta = self.landmarks_for(row["sha256"], rgb)
        else:
            lm, meta = landmarks, {"n_faces": 1, "source": "supplied"}
        if lm is None:
            return None
        r = support_erosion_radius(sigma, a.derivative_stencil_radius)
        face = align_face(row["image_id"], rgb, lm, size=size, forehead_extension=a.forehead_extension, erosion_radius=r)
        face.meta.update({"landmark_meta": meta, "sha256": row["sha256"], "subject_id": row["subject_id"], "split": row.get("split", "")})
        return face


# ----------------------------------------------------------------------------
# Surface cache
# ----------------------------------------------------------------------------
class SurfaceCache:
    """Bounded raw-surface cache.

    Only the *baseline* variant on the unperturbed aligned RGB is persisted (one
    entry per image, ~1 MB: z_forged, z_star, residual, trust as float32).
    Luminance is recomputed exactly from the RGB; the Laplacian is recomputed
    with the validated replica stencil (``laplacian_source = "replica"``).
    Variant/perturbed solves cost ~0.1 s each and are recomputed on demand so
    the cache stays within the 2 GB output budget.
    """

    PERSISTED = ("z_forged", "z_star", "residual", "trust")

    def __init__(self, root: Path, source_digest: str) -> None:
        self.root = root / "surfaces"
        self.root.mkdir(parents=True, exist_ok=True)
        self.source_digest = source_digest
        self.entry_bytes: list[int] = []

    def key(self, sha: str, align_version: str, variant: str, cfg: ProfileConfig, extra: dict[str, Any] | None = None) -> str:
        return sha256_json(
            {"sha": sha, "align": align_version, "variant": variant_key(variant, cfg.solver, extra), "src": self.source_digest}
        )[:24]

    def get_or_solve(self, face: AlignedFace, variant: str, cfg: ProfileConfig, extra: dict[str, Any] | None = None,
                     rgb_override: np.ndarray | None = None, align_version: str = "v1",
                     smooth_sigma: float | None = None) -> SurfaceResult:
        rgb = face.rgb if rgb_override is None else rgb_override
        persist = variant == "baseline" and rgb_override is None
        if not persist:
            return solve_surface(rgb, cfg.solver, variant, smooth_sigma=smooth_sigma)
        ext = dict(extra or {})
        ext["size"] = int(rgb.shape[0])
        k = self.key(face.meta["sha256"], align_version, variant, cfg, ext)
        p = self.root / f"{k}.npz"
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                arrays = {n: z[n] for n in self.PERSISTED}
            from facial_entropy.rust_adapter import weighted_luminance  # noqa: PLC0415
            from facial_entropy.solver_replica import laplacian_np  # noqa: PLC0415

            return SurfaceResult(
                variant=meta["variant"], backend=meta["backend"], z_forged=arrays["z_forged"], z_star=arrays["z_star"],
                residual=arrays["residual"], laplacian=laplacian_np(arrays["z_star"]).astype(np.float32), trust=arrays["trust"],
                luminance=weighted_luminance(rgb, cfg.solver),
                energy_total=meta["energy_total"], energy_data=meta["energy_data"],
                energy_smoothness=meta["energy_smoothness"], energy_consistency=meta["energy_consistency"],
                energy_trace=np.asarray(meta.get("energy_trace", []), dtype=np.float32), iterations=meta["iterations"], converged=meta["converged"],
                finite=meta["finite"], runtime_s=meta["runtime_s"], params={**meta["params"], "laplacian_source": "replica"}, threads=meta["threads"],
            )
        res = solve_surface(rgb, cfg.solver, variant, smooth_sigma=smooth_sigma)
        meta = {**res.meta(), "cache_key": k, "image_id": face.image_id, "energy_trace": res.energy_trace.tolist()}
        atomic_savez(p, **{n: getattr(res, n) for n in self.PERSISTED}, meta=np.array(json.dumps(meta, default=float)))
        self.entry_bytes.append(p.stat().st_size)
        return res

    def projected_bytes(self, n_entries: int) -> float:
        return float(np.mean(self.entry_bytes)) * n_entries if self.entry_bytes else float("nan")


def _arr_digest(a: np.ndarray) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Analysis settings and per-image products
# ----------------------------------------------------------------------------
@dataclass
class AnalysisSettings:
    nbins: int = 16
    window: int = 21
    sigma: float = 1.0
    stride: int = 4
    h: float = 1.0 / 255.0
    patch_size: int = 32
    samples_per_patch: int = 900
    block_samples_per_patch: int = 800
    min_support: float = 0.9
    binary_thresholds: tuple[float, ...] = (0.35, 0.5, 0.65)
    extra_windows: tuple[int, ...] = ()
    maps: bool = True

    @classmethod
    def from_cfg(cls, cfg: ProfileConfig, size: int | None = None, **kw: Any) -> AnalysisSettings:
        a = cfg.analysis
        size = size or a.aligned_size
        scale = size / a.aligned_size
        s = cls(
            nbins=a.primary_bins, window=max(3, int(round(a.primary_window * scale)) | 1), sigma=a.smooth_sigma,
            stride=max(1, int(round(a.stride * scale))), h=1.0 / (size - 1),
            patch_size=int(round(a.patch_size * scale)), samples_per_patch=int(round(a.samples_per_patch * scale * scale)),
            block_samples_per_patch=int(round(a.block_samples_per_patch * scale * scale)), min_support=a.min_support,
            binary_thresholds=a.binary_thresholds, extra_windows=tuple(w for w in a.windows if w != a.primary_window),
        )
        for k, v in kw.items():
            setattr(s, k, v)
        return s


@dataclass
class ImageProducts:
    image_id: str
    rows: list[dict[str, Any]]
    maps: dict[str, dict[str, Any]]
    patch_info: dict[str, Any]
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)


def compute_fields(face: AlignedFace, surf: SurfaceResult, calib: Calibration, s: AnalysisSettings) -> dict[str, np.ndarray]:
    g = geometry_fields(surf.z_star, face.valid, calib.height_scale, s.sigma, s.h)
    fields = {
        "gaussian_curvature": g["gaussian_curvature"],
        "mean_curvature": g["mean_curvature"],
        "laplacian_analysis": g["laplacian_analysis"],
        "grad_mag": g["grad_mag"],
        "residual": surf.residual.astype(np.float64),
        "trust": surf.trust.astype(np.float64),
        "normal_dir": normal_direction_codes(g["derivs"], calib.tilt_threshold),
        "area_weight": g["area_weight"],
        "z_tilde": g["z_tilde"],
        "zx": g["zx"],
        "zy": g["zy"],
    }
    return fields


def patch_layout(face: AlignedFace, s: AnalysisSettings, landmarks: np.ndarray | None = None) -> dict[str, Any]:
    lm = face.landmarks if landmarks is None else landmarks
    centers = reg.patch_centers(lm)
    anchors = reg.patch_anchors(centers, s.patch_size)
    validity = reg.patch_validity(face.valid, anchors, s.patch_size)
    overlaps = reg.patch_overlaps(anchors, s.patch_size, face.valid.shape)
    retained = all(v >= s.min_support for v in validity.values()) and not overlaps["central_peripheral_overlap"]
    reasons = []
    for k, v in validity.items():
        if v < s.min_support:
            reasons.append(f"{k}_support_{v:.2f}")
    if overlaps["central_peripheral_overlap"]:
        reasons.append("central_peripheral_overlap")
    return {
        "centers": {k: [float(v[0]), float(v[1])] for k, v in centers.items()},
        "anchors": {k: [int(a[0]), int(a[1])] for k, a in anchors.items()},
        "validity": validity,
        "overlaps": overlaps,
        "retained": bool(retained),
        "rejection_reasons": reasons,
    }


def analyse_image(
    face: AlignedFace,
    surf: SurfaceResult,
    calib: Calibration,
    s: AnalysisSettings,
    seed: int,
    variant: str = "baseline",
    landmarks: np.ndarray | None = None,
    descriptors: tuple[str, ...] | None = None,
) -> ImageProducts:
    """All patch metrics and local maps for one image under frozen calibration."""
    layout = patch_layout(face, s, landmarks)
    anchors = {k: tuple(v) for k, v in layout["anchors"].items()}
    valid = face.valid
    fields = compute_fields(face, surf, calib, s)
    rows: list[dict[str, Any]] = []
    base_row = {
        "image_id": face.image_id, "subject_id": face.meta.get("subject_id", ""), "split": face.meta.get("split", ""),
        "variant": variant, "nbins": s.nbins, "window": s.window, "sigma": s.sigma, "converged": surf.converged,
        "finite": surf.finite, "iterations": surf.iterations, "retained": layout["retained"],
    }
    descriptors = descriptors or tuple(DESCRIPTORS)

    # --- binary pipeline ------------------------------------------------------
    i_w = surf.luminance.astype(np.float64)
    t_otsu = ent.otsu_threshold(i_w[valid])
    thresholds = {"otsu": t_otsu, **{f"t{t:.2f}": t for t in s.binary_thresholds}}
    binaries = {name: ent.binarize(i_w, t) for name, t in thresholds.items()}

    # --- quantised descriptors --------------------------------------------------
    quant = {}
    for d in descriptors:
        kind = DESCRIPTORS[d]["kind"]
        if kind == "code":
            quant[d] = (fields[d].astype(np.int16), np.zeros(valid.shape, bool), np.zeros(valid.shape, bool), 16)
        else:
            q = calib.quantizers[d][s.nbins]
            codes, lo, hi = q.apply(fields[d])
            quant[d] = (codes, lo, hi, s.nbins)
    joint = (
        calib.terciles["gaussian_curvature"].apply(fields["gaussian_curvature"])
        + 3 * calib.terciles["mean_curvature"].apply(fields["mean_curvature"])
        + 9 * calib.terciles["residual"].apply(fields["residual"])
    )
    cb_codes = {}
    for k, cb in calib.codebooks.items():
        X = np.stack([fields[n].ravel() for n in cb.names], axis=1)
        cb_codes[k] = cb.apply(X).reshape(valid.shape)

    # --- per-patch statistics ---------------------------------------------------
    for pname in reg.PATCH_NAMES:
        sl = reg.patch_slice(anchors[pname], s.patch_size)
        pseed = stable_seed(face.image_id, pname, base=seed)
        pos = reg.sample_positions(valid, sl, s.samples_per_patch, pseed)
        support = layout["validity"][pname]
        common = {**base_row, "patch": pname, "region_class": reg.PATCH_CLASS[pname], "support_fraction": support}
        if pos is None:
            rows.append({**common, "metric": "_patch_rejected", "estimator": "", "value": np.nan})
            continue
        ys, xs = pos
        sub_valid = valid[sl]
        ally, allx = np.nonzero(sub_valid)
        ally, allx = ally + sl[0].start, allx + sl[1].start

        for d, (codes, lo, hi, nb) in quant.items():
            for est, (yy, xx) in (("fixed", (ys, xs)), ("all", (ally, allx))):
                r = ent.feature_entropy_from_codes(codes[yy, xx], nb)
                rows.append({**common, "metric": d, "estimator": est, "value": r["H"], "value_mm": r["H_mm"], "n": r["n"],
                             "occupancy": r["occupancy"], "clip_lo": float(lo[yy, xx].mean()), "clip_hi": float(hi[yy, xx].mean())})
            rw = ent.feature_entropy_from_codes(codes[ys, xs], nb, weights=fields["area_weight"][ys, xs])
            rows.append({**common, "metric": d, "estimator": "surface_weighted", "value": rw["H"], "value_mm": np.nan,
                         "n": rw["n"], "occupancy": rw["occupancy"], "clip_lo": float(lo[ys, xs].mean()), "clip_hi": float(hi[ys, xs].mean())})
        rj = ent.feature_entropy_from_codes(joint[ys, xs], 27)
        rows.append({**common, "metric": "joint_KG_HM_R_27", "estimator": "fixed", "value": rj["H"], "value_mm": rj["H_mm"], "n": rj["n"], "occupancy": rj["occupancy"]})
        for k, cc in cb_codes.items():
            rc = ent.feature_entropy_from_codes(cc[ys, xs], k)
            rows.append({**common, "metric": f"codebook_{k}", "estimator": "fixed", "value": rc["H"], "value_mm": rc["H_mm"], "n": rc["n"], "occupancy": rc["occupancy"]})

        # binary statistics
        bseed = stable_seed(face.image_id, pname, "blocks", base=seed)
        bpos = reg.sample_block_anchors(valid, sl, s.block_samples_per_patch, bseed)
        for tname, B in binaries.items():
            m = ent.marginal_binary_entropy(B, ys, xs)
            rows.append({**common, "metric": f"binary_marginal_{tname}", "estimator": "fixed", "value": m["H_marginal"], "n": m["n"], "p_one": m["p"], "threshold": thresholds[tname]})
            if bpos is None:
                rows.append({**common, "metric": f"binary_cond_{tname}", "estimator": "fixed", "value": np.nan, "n": 0, "flag": "insufficient_block_anchors"})
                continue
            by, bx = bpos
            for rot in range(4):
                C4 = ent.block_codes(B, rotation=rot)
                be = ent.block_entropies(C4, by, bx)
                suffix = "" if rot == 0 else f"_rot{rot}"
                rows.append({**common, "metric": f"binary_block4_{tname}{suffix}", "estimator": "fixed", "value": be["H4"], "n": be["n"], "occupancy": be["occupancy4"], "threshold": thresholds[tname]})
                rows.append({**common, "metric": f"binary_cond_{tname}{suffix}", "estimator": "fixed", "value": be["H_cond"], "n": be["n"], "occupancy": be["occupancy4"], "threshold": thresholds[tname]})
            # all-anchor sensitivity
            ay, ax = reg.block_anchor_positions(valid, sl)
            be = ent.block_entropies(ent.block_codes(B), ay, ax)
            rows.append({**common, "metric": f"binary_cond_{tname}", "estimator": "all", "value": be["H_cond"], "n": be["n"], "threshold": thresholds[tname]})

    # --- local maps -------------------------------------------------------------
    maps: dict[str, dict[str, Any]] = {}
    if s.maps:
        windows = (s.window, *s.extra_windows)
        for w in windows:
            tag = f"w{w}"
            for d in ("gaussian_curvature", "mean_curvature", "residual", "trust"):
                if d in quant:
                    codes, lo, hi, nb = quant[d]
                    maps[f"{d}_b{nb}_{tag}"] = ent.local_entropy_map(codes, valid, nb, w, s.stride, s.min_support)
            maps[f"binary_cond_otsu_{tag}"] = ent.local_binary_cond_map(binaries["otsu"], valid, w, s.stride, s.min_support)
            maps[f"binary_marginal_otsu_{tag}"] = ent.local_marginal_map(binaries["otsu"], valid, w, s.stride, s.min_support)
        codes, lo, hi, nb = quant["gaussian_curvature"]
        maps[f"gaussian_curvature_b{nb}_w{s.window}_surfacew"] = ent.local_entropy_map(
            codes, valid, nb, s.window, s.stride, s.min_support, weights=fields["area_weight"]
        )
    return ImageProducts(
        image_id=face.image_id, rows=rows, maps=maps, patch_info=layout,
        fields={"gaussian_curvature": fields["gaussian_curvature"], "mean_curvature": fields["mean_curvature"],
                "z_tilde": fields["z_tilde"], "binary_otsu": binaries["otsu"], "grad_mag": fields["grad_mag"]},
        flags={"otsu_threshold": t_otsu, "converged": surf.converged, "finite": surf.finite, "iterations": surf.iterations},
    )


def save_maps(path: Path, prod: ImageProducts) -> None:
    arrays: dict[str, np.ndarray] = {}
    for name, m in prod.maps.items():
        arrays[f"{name}__map"] = m["map"].astype(np.float32)
        arrays[f"{name}__support"] = m["support"].astype(np.int32)
        arrays[f"{name}__ys"] = m["ys"]
        arrays[f"{name}__xs"] = m["xs"]
    arrays["meta"] = np.array(json.dumps({"image_id": prod.image_id, "patch_info": prod.patch_info, "flags": prod.flags}, default=float))
    atomic_savez(path, **arrays)


def load_maps(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with np.load(path, allow_pickle=False) as z:
        names = {k.split("__")[0] for k in z.files if "__" in k}
        for n in names:
            out[n] = {"map": z[f"{n}__map"], "support": z[f"{n}__support"], "ys": z[f"{n}__ys"], "xs": z[f"{n}__xs"], "stride": int(z[f"{n}__ys"][1] - z[f"{n}__ys"][0]) if len(z[f"{n}__ys"]) > 1 else 1}
        out["_meta"] = json.loads(str(z["meta"]))
    return out


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------
def balanced_samples(face: AlignedFace, s: AnalysisSettings, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Equal numbers of valid positions from the central-patch union and the peripheral union."""
    layout = patch_layout(face, s)
    anchors = {k: tuple(v) for k, v in layout["anchors"].items()}
    shape = face.valid.shape
    cen = np.zeros(shape, bool)
    per = np.zeros(shape, bool)
    for k in reg.CENTRAL_PATCHES:
        cen |= reg.patch_mask(anchors[k], s.patch_size, shape)
    for k in reg.PERIPHERAL_PATCHES:
        per |= reg.patch_mask(anchors[k], s.patch_size, shape)
    rng = rng_for("calibration-sample", face.image_id, base=seed)
    out_y, out_x, cls = [], [], []
    for lab, m in (("central", cen & face.valid), ("peripheral", per & face.valid)):
        ys, xs = np.nonzero(m)
        n = min(s.samples_per_patch, len(ys))
        if n == 0:
            return None
        idx = rng.choice(len(ys), size=n, replace=False)
        out_y.append(ys[idx])
        out_x.append(xs[idx])
        cls.append(np.full(n, 0 if lab == "central" else 1))
    return np.concatenate(out_y), np.concatenate(out_x), np.concatenate(cls)


def fit_calibration(
    faces: list[AlignedFace],
    surfaces: list[SurfaceResult],
    cfg: ProfileConfig,
    s: AnalysisSettings,
    seed: int,
    luminance_choice: dict[str, Any] | None = None,
) -> Calibration:
    a = cfg.analysis
    per_img_scale = {}
    for f, su in zip(faces, surfaces, strict=True):
        per_img_scale[f.image_id] = robust_scale(su.z_star, f.valid)
    scales = np.array([v for v in per_img_scale.values() if np.isfinite(v)])
    degenerate = {}
    s_z = float(np.median(scales)) if scales.size else float("nan")
    if not np.isfinite(s_z) or s_z < a.height_scale_floor:
        degenerate["height_scale"] = True
        s_z = max(s_z if np.isfinite(s_z) else 0.0, a.height_scale_floor)
    else:
        degenerate["height_scale"] = False
    tmp = Calibration(s_z, per_img_scale, 0.0, {}, {}, {}, 0.0)

    pooled: dict[str, list[np.ndarray]] = {d: [] for d in DESCRIPTORS if DESCRIPTORS[d]["kind"] != "code"}
    pooled["grad_mag_for_tilt"] = []
    vec_names = ["gaussian_curvature", "mean_curvature", "grad_mag", "residual", "trust", "laplacian_analysis"]
    vecs: list[np.ndarray] = []
    n_used = 0
    for f, su in zip(faces, surfaces, strict=True):
        bs = balanced_samples(f, s, seed)
        if bs is None:
            continue
        ys, xs, _ = bs
        fields = compute_fields(f, su, tmp, s)
        for d in pooled:
            key = "grad_mag" if d == "grad_mag_for_tilt" else d
            pooled[d].append(fields[key][ys, xs])
        vecs.append(np.stack([fields[n][ys, xs] for n in vec_names], axis=1))
        n_used += 1
    quantizers: dict[str, dict[int, Any]] = {}
    for d, spec in DESCRIPTORS.items():
        quantizers[d] = {}
        samples = np.concatenate(pooled[d]) if spec["kind"] != "code" and pooled[d] else np.zeros(0)
        for nb in a.bins:
            q = fit_scalar_quantizer(d, spec["kind"], samples, nb, a.calib_quantile_lo, a.calib_quantile_hi, a.scale_floor)
            quantizers[d][nb] = q
            degenerate[f"{d}_b{nb}"] = q.degenerate
    tilt = float(np.median(np.concatenate(pooled["grad_mag_for_tilt"]))) if pooled["grad_mag_for_tilt"] else 1.0
    terciles = {d: fit_terciles(d, np.concatenate(pooled[d]), a.scale_floor) for d in ("gaussian_curvature", "mean_curvature", "residual")}
    X = np.concatenate(vecs) if vecs else np.zeros((0, len(vec_names)))
    codebooks = {k: fit_codebook(vec_names, X, k, stable_seed("codebook", k, base=seed)) for k in a.codebook_sizes} if len(X) >= max(a.codebook_sizes) * 10 else {}
    return Calibration(
        height_scale=s_z,
        height_scale_per_image=per_img_scale,
        tilt_threshold=tilt,
        quantizers=quantizers,
        terciles=terciles,
        codebooks=codebooks,
        smoothed_luminance_sigma=float(luminance_choice["sigma"]) if luminance_choice else 2.0,
        fit_population={"n_images": n_used, "image_ids": [f.image_id for f in faces], "samples_per_class_per_image": s.samples_per_patch,
                        "balanced": "equal central/peripheral samples per image, one image per subject",
                        "luminance_choice": luminance_choice},
        degenerate_flags=degenerate,
    )


def choose_luminance_sigma(faces: list[AlignedFace], surfaces: list[SurfaceResult], candidates=(1.0, 2.0, 3.0, 4.0)) -> dict[str, Any]:
    """Calibration-only choice of the matched-smoothing sigma for the luminance comparator.

    Rule: the sigma whose smoothed-luminance graph (median-centred, robust-scaled)
    has the smallest RMS difference to the baseline settled surface (same
    normalisation) over the valid support of calibration faces.
    """
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    scores = {}
    for sg in candidates:
        errs = []
        for f, su in zip(faces, surfaces, strict=True):
            v = f.valid
            zs = su.z_star.astype(np.float64)
            zs = (zs - np.median(zs[v])) / max(robust_scale(zs, v), 1e-9)
            lw = gaussian_filter(su.luminance.astype(np.float64), sg, mode="reflect")
            lw = (lw - np.median(lw[v])) / max(robust_scale(lw, v), 1e-9)
            errs.append(float(np.sqrt(np.mean((zs[v] - lw[v]) ** 2))))
        scores[sg] = float(np.mean(errs))
    best = min(scores, key=scores.get)
    return {"sigma": best, "rms_by_sigma": scores, "rule": "min RMS to normalised baseline z* on calibration"}
