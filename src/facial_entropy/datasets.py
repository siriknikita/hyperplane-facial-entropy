"""Dataset discovery: FEI face database (local), opt-in LFW loader, download budget."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from facial_entropy.config import DEFAULT_FEI_ROOT

DOWNLOAD_CAP_BYTES = 1 << 30  # 1 GB hard cap for any opt-in download

# FEI Face Database (Centro Universitario da FEI, Sao Bernardo do Campo, Brazil).
# 200 subjects x 14 colour images, 640 x 480, white homogeneous background.
# Poses 01-10: profile rotation (01 and 10 near-profile, 05/06 near-frontal);
# 11: frontal neutral; 12: frontal smiling; 13 and 14: frontal with illumination
# changes.  Source: https://fei.edu.br/~cet/facedatabase.html (research use).
FEI_POSE_NOTES = {
    "01": "rotation step 1 (near profile)",
    "02": "rotation step 2",
    "03": "rotation step 3",
    "04": "rotation step 4",
    "05": "rotation step 5 (near frontal)",
    "06": "rotation step 6 (near frontal)",
    "07": "rotation step 7",
    "08": "rotation step 8",
    "09": "rotation step 9",
    "10": "rotation step 10 (near profile)",
    "11": "frontal, neutral expression",
    "12": "frontal, smiling",
    "13": "frontal, illumination variant A",
    "14": "frontal, illumination variant B (darker)",
}
_FEI_RE = re.compile(r"^(?P<subject>\d+)-(?P<pose>\d{2})\.jpg$", re.IGNORECASE)


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: Path
    subject_id: str
    source_video_id: str
    dataset: str
    pose: str
    label: str  # "original" for curated real photographs
    provenance: str
    frame_index: int | None = None


def discover_fei(root: str | Path = DEFAULT_FEI_ROOT) -> list[ImageRecord]:
    root = Path(root)
    if not root.exists():
        return []
    recs: list[ImageRecord] = []
    for p in sorted(root.rglob("*.jpg")):
        m = _FEI_RE.match(p.name)
        if not m:
            continue
        subj = f"fei{int(m.group('subject')):03d}"
        pose = m.group("pose")
        recs.append(
            ImageRecord(
                image_id=f"{subj}-{pose}",
                path=p.resolve(),
                subject_id=subj,
                source_video_id=subj,  # still photographs: one acquisition session per subject
                dataset="FEI",
                pose=pose,
                label="original",
                provenance="FEI Face Database, original colour photographs (not manipulated)",
                frame_index=None,
            )
        )
    return recs


def discover_auto(data_root: str | Path | None) -> tuple[str, list[ImageRecord], list[str]]:
    """Search documented local locations and pick the only eligible real-face source.

    Returns (dataset_name, records, candidate_descriptions).  Candidates that
    resolve to the same image files (a root and its parent) are merged.
    """
    roots = [Path(data_root)] if data_root else []
    roots += [DEFAULT_FEI_ROOT, Path.home() / "Development" / "datasets"]
    found: dict[frozenset, tuple[str, Path, list[ImageRecord]]] = {}
    seen: set[Path] = set()

    def consider(path: Path) -> None:
        if path in seen or not path.exists():
            return
        seen.add(path)
        recs = discover_fei(path)
        if recs:
            key = frozenset(str(r.path) for r in recs)
            if key not in found:
                found[key] = ("FEI", path, recs)

    for r in roots:
        if not r.exists():
            continue
        if any(d.name.startswith("originalimages_part") for d in r.iterdir() if d.is_dir()):
            consider(r)
        else:
            for sub in sorted(p for p in r.iterdir() if p.is_dir()):
                consider(sub)
    descs = [f"{name}: {path} ({len(recs)} images)" for name, path, recs in found.values()]
    if not found:
        raise FileNotFoundError(
            "no eligible real-face dataset found; pass --data-root <FEI root> (expected files "
            "like originalimages_part1/1-11.jpg) or use --dataset lfw --download"
        )
    if len(found) > 1:
        raise RuntimeError("ambiguous dataset candidates, pass --data-root explicitly:\n  " + "\n  ".join(descs))
    name, _path, recs = next(iter(found.values()))
    return name, recs, descs


def lfw_estimate_bytes() -> int:
    # lfw-funneled.tgz as served by the scikit-learn mirror (~233 MB) + metadata (~0.4 MB).
    return 233_000_000 + 400_000


def discover_lfw(data_home: str | Path | None, download: bool) -> list[ImageRecord]:
    """Opt-in LFW loader via scikit-learn (colour, original 250x250 files, no slicing/resizing).

    ``download`` must be True for a first fetch; the estimated size is checked
    against the 1 GB cap.  Identities are the folder names of the extracted archive.
    """
    from sklearn.datasets import get_data_home  # noqa: PLC0415

    home = Path(get_data_home(data_home))
    extracted = home / "lfw_home" / "lfw_funneled"
    if not extracted.exists():
        est = lfw_estimate_bytes()
        if not download:
            raise FileNotFoundError(
                f"LFW not present at {extracted}; rerun with --download (estimated {est/1e6:.0f} MB, "
                f"cap {DOWNLOAD_CAP_BYTES/1e9:.0f} GB). Source: scikit-learn fetch_lfw_people mirror."
            )
        if est > DOWNLOAD_CAP_BYTES:
            raise RuntimeError("LFW estimate exceeds the download cap")
        from sklearn.datasets import fetch_lfw_people  # noqa: PLC0415

        # color=True is set explicitly; resize/slice are irrelevant because we read the
        # original files from disk afterwards.
        fetch_lfw_people(data_home=str(home), color=True, resize=1.0, slice_=None, download_if_missing=True)
    recs: list[ImageRecord] = []
    for person_dir in sorted(p for p in extracted.iterdir() if p.is_dir()):
        subj = person_dir.name
        for p in sorted(person_dir.glob("*.jpg")):
            recs.append(
                ImageRecord(
                    image_id=p.stem,
                    path=p.resolve(),
                    subject_id=subj,
                    source_video_id=subj,
                    dataset="LFW-funneled",
                    pose="unknown",
                    label="original",
                    provenance="LFW (funneled) via scikit-learn mirror; licence: see http://vis-www.cs.umass.edu/lfw/",
                )
            )
    return recs


def dataset_summary(recs: list[ImageRecord]) -> dict[str, Any]:
    subjects = sorted({r.subject_id for r in recs})
    poses = sorted({r.pose for r in recs})
    return {"n_images": len(recs), "n_subjects": len(subjects), "poses": poses}
