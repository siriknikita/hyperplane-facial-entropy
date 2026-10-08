import numpy as np
import pandas as pd
import pytest

from facial_entropy import statistics as st


def test_sign_flip_exact_matches_known_value():
    d = np.array([1.0, 2.0, 3.0])
    r = st.sign_flip_test(d, 10000, 1)
    assert r["exact"] and r["B"] == 8
    # only the all-positive and all-negative sign patterns reach |mean| = 2
    assert r["p"] == pytest.approx(2 / 8)


def test_sign_flip_symmetric_null():
    rng = np.random.default_rng(0)
    ps = [st.sign_flip_test(rng.standard_normal(25), 2000, i)["p"] for i in range(40)]
    assert 0.2 < np.mean(np.array(ps) < 0.5) < 0.8


def test_holm_monotone_and_bounded():
    p = {"a": 0.01, "b": 0.04, "c": 0.5}
    h = st.holm(p)
    assert h["a"] == pytest.approx(0.03) and h["b"] == pytest.approx(0.08) and h["c"] == 0.5


def test_subject_differences_average_within_subject_first():
    rows = []
    for subj, imgs in (("s1", ["a", "b"]), ("s2", ["c"])):
        for im in imgs:
            for p, v in (("eye_inner_left", 2.0), ("eye_inner_right", 2.0), ("nose_bridge", 2.0), ("nose_tip", 2.0),
                         ("cheek_left", 1.0), ("cheek_right", 1.0), ("forehead_left", 1.0), ("forehead_right", 1.0 if im != "b" else 5.0)):
                rows.append({"subject_id": subj, "image_id": im, "patch": p, "metric": "m", "estimator": "fixed", "value": v, "retained": True})
    df = pd.DataFrame(rows)
    s = st.subject_differences(df, "m")
    assert set(s["subject_id"]) == {"s1", "s2"}
    s1 = s[s["subject_id"] == "s1"].iloc[0]
    assert s1["n_images"] == 2
    assert s1["D"] == pytest.approx(((2 - 1) + (2 - 2)) / 2)


def test_paired_summary_dz_undefined_when_sd_zero():
    s = pd.DataFrame({"subject_id": list("abcd"), "n_images": 1, "C": 1.0, "P": 0.5, "D": 0.5})
    out = st.paired_summary(s, 100, 100, 1)
    assert out["d_z"] is None and "undefined" in out["d_z_note"]
