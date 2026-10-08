"""Probabilistic definitions: plug-in/Miller-Madow entropy, binary blocks, local maps."""

import numpy as np
import pytest

from facial_entropy import entropy as ent
from facial_entropy import regions as reg


def test_entropy_bits_uniform_and_degenerate():
    assert ent.entropy_bits(np.ones(16)) == pytest.approx(4.0)
    assert ent.entropy_bits(np.array([0, 5, 0])) == 0.0
    assert np.isnan(ent.entropy_bits(np.zeros(4)))


def test_miller_madow_adds_positive_bias_term():
    c = np.array([10, 10, 10, 0])
    assert ent.miller_madow_bits(c) == pytest.approx(ent.entropy_bits(c) + (3 - 1) / (2 * 30 * np.log(2)))


def test_block_codes_bit_order():
    B = np.array([[1, 0], [1, 1]], bool)
    # B00=1, B01=0, B10=1, B11=1 -> 1 + 0 + 4 + 8 = 13
    assert ent.block_codes(B)[0, 0] == 13
    # rotation permutes positions but keeps 4 bits
    for r in range(4):
        assert 0 <= ent.block_codes(B, r)[0, 0] <= 15


def test_conditional_entropy_bounds_and_identities():
    rng = np.random.default_rng(0)
    B = rng.random((80, 80)) < 0.5
    valid = np.ones_like(B)
    ys, xs = reg.block_anchor_positions(valid, (slice(0, 80), slice(0, 80)))
    be = ent.block_entropies(ent.block_codes(B), ys, xs)
    assert 0 <= be["H_cond"] <= 1.0 + 1e-12
    assert be["H_cond"] == pytest.approx(be["H4"] - be["H3"])
    # checkerboard: marginal 1 bit, conditional 0
    cb = (np.add.outer(np.arange(80), np.arange(80)) % 2) == 0
    be2 = ent.block_entropies(ent.block_codes(cb), ys, xs)
    assert be2["H_cond"] == pytest.approx(0.0, abs=1e-12)
    py, px = np.nonzero(valid)
    assert ent.marginal_binary_entropy(cb, py, px)["H_marginal"] == pytest.approx(1.0)


def test_otsu_separates_bimodal():
    v = np.concatenate([np.full(500, 0.2), np.full(500, 0.8)])
    t = ent.otsu_threshold(v)
    assert 0.2 < t <= 0.8
    try:
        from skimage.filters import threshold_otsu
    except ImportError:
        return
    rng = np.random.default_rng(1)
    v2 = np.clip(np.concatenate([rng.normal(0.3, 0.05, 2000), rng.normal(0.7, 0.05, 2000)]), 0, 1)
    assert abs(ent.otsu_threshold(v2) - threshold_otsu(v2, nbins=256)) < 0.01


def test_local_map_matches_brute_force_and_support_rule():
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 8, size=(40, 40))
    valid = np.ones((40, 40), bool)
    valid[:, :12] = False
    m = ent.local_entropy_map(codes, valid, 8, 7, 4, 0.9)
    for i, y in enumerate(m["ys"]):
        for j, x in enumerate(m["xs"]):
            y0, y1, x0, x1 = max(y - 3, 0), min(y + 4, 40), max(x - 3, 0), min(x + 4, 40)
            v = valid[y0:y1, x0:x1]
            c = codes[y0:y1, x0:x1][v]
            if v.sum() >= 0.9 * 49:
                assert m["map"][i, j] == pytest.approx(ent.entropy_bits(np.bincount(c, minlength=8)))
            else:
                assert np.isnan(m["map"][i, j])


def test_surface_weighted_counts_change_distribution():
    codes = np.array([[0, 1], [0, 1]])
    valid = np.ones((2, 2), bool)
    w = np.array([[1.0, 3.0], [1.0, 3.0]])
    unweighted = ent.feature_entropy_from_codes(codes, 2)["H"]
    weighted = ent.feature_entropy_from_codes(codes, 2, weights=w)["H"]
    assert unweighted == pytest.approx(1.0)
    assert weighted < unweighted
