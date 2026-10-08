"""Coordinate projection round trip, patch rules, grouping/leakage invariants."""

import numpy as np
import pandas as pd

from facial_entropy import regions as reg
from facial_entropy.alignment import align_face, apply_affine, fit_similarity, invert_affine, key_points_from_68
from facial_entropy.manifest import select_cohort
from facial_entropy.synthetic import template_landmarks


def test_similarity_round_trip_subpixel():
    lm = template_landmarks(256)
    R = 1.7 * np.array([[np.cos(0.2), -np.sin(0.2)], [np.sin(0.2), np.cos(0.2)]])
    lm_orig = lm @ R.T + np.array([50.0, 30.0])
    face = align_face("t", np.zeros((600, 600, 3), np.float32), lm_orig, size=256)
    kp = key_points_from_68(face.landmarks)
    assert np.linalg.norm(kp["eye_left"] - [84, 100]) < 1e-6
    assert np.linalg.norm(kp["mouth"] - [128, 184]) < 1e-6
    pts = np.random.default_rng(0).random((20, 2)) * 256
    back = apply_affine(face.transform, apply_affine(invert_affine(face.transform), pts))
    assert np.abs(back - pts).max() < 1e-9


def test_fit_similarity_has_no_reflection():
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    dst = np.array([[0.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])  # reflected target
    A, rms = fit_similarity(src, dst)
    assert np.linalg.det(A[:, :2]) > 0  # similarity (rotation+scale) cannot reflect
    assert rms > 0.1


def test_patch_rules_and_rounding():
    lm = template_landmarks(256)
    c = reg.patch_centers(lm)
    a = reg.patch_anchors(c, 32)
    y0, x0 = a["eye_inner_left"]
    assert (y0, x0) == (int(np.floor(lm[39][1] + 0.5)) - 16, int(np.floor(lm[39][0] + 0.5)) - 16)
    # forehead centres lie above the brows (smaller y)
    assert c["forehead_left"][1] < lm[19][1]
    valid = np.ones((256, 256), bool)
    v = reg.patch_validity(valid, a, 32)
    assert all(abs(x - 1.0) < 1e-12 for x in v.values())
    ov = reg.patch_overlaps(a, 32, (256, 256))
    assert not ov["central_peripheral_overlap"]


def test_sampling_is_seeded_and_exact_count():
    valid = np.ones((64, 64), bool)
    sl = (slice(10, 42), slice(10, 42))
    p1 = reg.sample_positions(valid, sl, 900, 123)
    p2 = reg.sample_positions(valid, sl, 900, 123)
    p3 = reg.sample_positions(valid, sl, 900, 124)
    assert p1 is not None and len(p1[0]) == 900
    assert np.array_equal(p1[0], p2[0]) and np.array_equal(p1[1], p2[1])
    assert not np.array_equal(p1[0], p3[0])
    assert reg.sample_positions(np.zeros((64, 64), bool), sl, 900, 1) is None


def test_cohort_split_is_subject_disjoint_and_deterministic():
    rows = []
    for s in range(30):
        for pose in ("05", "11", "12"):
            rows.append({"image_id": f"s{s:03d}-{pose}", "path": "", "sha256": f"h{s}{pose}", "subject_id": f"s{s:03d}",
                         "pose": pose, "exclusion_reason": ""})
    df = pd.DataFrame(rows)
    c1 = select_cohort(df, "11", 20, 0.2, 7, ("05", "11", "12"))
    c2 = select_cohort(df, "11", 20, 0.2, 7, ("05", "11", "12"))
    assert c1.subject_split == c2.subject_split
    cal = {s for s, v in c1.subject_split.items() if v == "calibration"}
    test = {s for s, v in c1.subject_split.items() if v == "test"}
    assert cal.isdisjoint(test) and len(cal) == 4 and len(test) == 16
    assert (c1.primary["pose"] == "11").all()
    assert set(c1.secondary["subject_id"]) <= cal | test
    assert c1.secondary["subject_id"].map(c1.subject_split).equals(c1.secondary["split"])
