"""Provenance manifest, deduplication, subject grouping, seeded selection and splits."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from facial_entropy.datasets import ImageRecord
from facial_entropy.util import rng_for, sha256_file

MANIFEST_COLUMNS = [
    "image_id",
    "path",
    "sha256",
    "subject_id",
    "source_video_id",
    "dataset",
    "label",
    "provenance",
    "width",
    "height",
    "frame_index",
    "pose",
    "task_label",
    "landmark_asset_version",
    "alignment_transform",
    "alignment_residual_px",
    "quality_flags",
    "split",
    "role",
    "exclusion_reason",
]


def build_manifest(recs: list[ImageRecord]) -> pd.DataFrame:
    rows = []
    for r in recs:
        with Image.open(r.path) as im:
            w, h = im.size
        rows.append(
            {
                "image_id": r.image_id,
                "path": str(r.path),
                "sha256": sha256_file(r.path),
                "subject_id": r.subject_id,
                "source_video_id": r.source_video_id,
                "dataset": r.dataset,
                "label": r.label,
                "provenance": r.provenance,
                "width": w,
                "height": h,
                "frame_index": r.frame_index,
                "pose": r.pose,
                "task_label": r.subject_id,  # identity label for verification
                "landmark_asset_version": "",
                "alignment_transform": "",
                "alignment_residual_px": np.nan,
                "quality_flags": "",
                "split": "",
                "role": "",
                "exclusion_reason": "",
            }
        )
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    # exact duplicate removal by SHA-256 (keep first in sorted image_id order)
    df = df.sort_values("image_id").reset_index(drop=True)
    dup = df.duplicated("sha256", keep="first")
    df.loc[dup, "exclusion_reason"] = "exact_duplicate_sha256"
    return df


@dataclass
class Cohort:
    primary: pd.DataFrame  # one primary image per subject (calibration + test)
    secondary: pd.DataFrame  # additional poses used for strata / verification
    subject_split: dict[str, str]  # subject -> calibration | test
    requested_subjects: int
    available_subjects: int
    notes: list[str] = field(default_factory=list)


def select_cohort(
    manifest: pd.DataFrame,
    primary_pose: str,
    n_subjects: int,
    calibration_fraction: float,
    seed: int,
    verification_poses: tuple[str, ...],
) -> Cohort:
    """Deterministic subject selection + grouped calibration/test split.

    Subjects are the grouping unit; the primary image per subject is the
    prespecified pose (FEI pose 11, frontal neutral).  Selection is seeded and
    never looks at any entropy value.
    """
    notes: list[str] = []
    ok = manifest[manifest["exclusion_reason"] == ""]
    prim = ok[ok["pose"] == primary_pose]
    subjects = sorted(prim["subject_id"].unique())
    available = len(subjects)
    if n_subjects <= 0 or n_subjects >= available:
        chosen = subjects
        if n_subjects > available:
            notes.append(f"requested {n_subjects} subjects but only {available} available; using all")
    else:
        rng = rng_for("subject-selection", base=seed)
        chosen = sorted(rng.choice(np.array(subjects), size=n_subjects, replace=False).tolist())
    rng2 = rng_for("calibration-split", base=seed)
    perm = list(rng2.permutation(np.array(chosen)))
    n_cal = max(1, int(round(calibration_fraction * len(chosen))))
    cal = set(perm[:n_cal])
    split = {s: ("calibration" if s in cal else "test") for s in chosen}
    primary = prim[prim["subject_id"].isin(chosen)].copy()
    primary["split"] = primary["subject_id"].map(split)
    primary["role"] = "primary"
    sec = ok[ok["subject_id"].isin(chosen) & ok["pose"].isin(verification_poses) & (ok["pose"] != primary_pose)].copy()
    sec["split"] = sec["subject_id"].map(split)
    sec["role"] = "secondary"
    return Cohort(primary=primary, secondary=sec, subject_split=split, requested_subjects=n_subjects,
                  available_subjects=available, notes=notes)


def robustness_subset(primary: pd.DataFrame, n: int, seed: int) -> list[str]:
    """Deterministic subject-balanced subset of *test* primary images for the nuisance matrix."""
    test = primary[primary["split"] == "test"].sort_values("image_id")
    ids = test["image_id"].tolist()
    if n >= len(ids):
        return ids
    rng = rng_for("robustness-subset", base=seed)
    return sorted(rng.choice(np.array(ids), size=n, replace=False).tolist())


def near_duplicate_report(manifest: pd.DataFrame, max_pairs: int = 50) -> dict[str, Any]:
    """Cheap near-duplicate screen: 8x8 grayscale thumbnails, Hamming distance of mean-threshold hashes."""
    rows = manifest[manifest["exclusion_reason"] == ""]
    hashes = {}
    for _, r in rows.iterrows():
        with Image.open(r["path"]) as im:
            g = np.asarray(im.convert("L").resize((8, 8), Image.BILINEAR), dtype=np.float32)
        hashes[r["image_id"]] = (g > g.mean()).ravel()
    ids = list(hashes)
    H = np.stack([hashes[i] for i in ids]).astype(np.uint8)
    pairs = []
    for i in range(len(ids)):
        d = (H[i] != H[i + 1 :]).sum(axis=1)
        for j in np.where(d <= 2)[0]:
            pairs.append((ids[i], ids[i + 1 + j], int(d[j])))
    return {"n_screened": len(ids), "n_near_duplicate_pairs_hamming_le_2": len(pairs), "examples": pairs[:max_pairs]}


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
