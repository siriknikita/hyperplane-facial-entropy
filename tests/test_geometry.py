"""Derivative signs/spacing, analytic curvature fixtures, quantizer invariants."""

import numpy as np
import pytest

from facial_entropy.geometry import curvatures, derivatives, normals
from facial_entropy.quantization import fit_scalar_quantizer


def _grid(n):
    h = 1.0 / (n - 1)
    x = np.arange(n) * h
    X, Y = np.meshgrid(x, x)
    return X, Y, h


def test_derivative_spacing_and_axis_order():
    X, Y, h = _grid(65)
    z = 2.0 * X + 3.0 * Y
    d = derivatives(z, h)
    assert np.allclose(d["zx"][5:-5, 5:-5], 2.0)
    assert np.allclose(d["zy"][5:-5, 5:-5], 3.0)


def test_paraboloid_saddle_cylinder():
    X, Y, h = _grid(129)
    a = 2.5
    c = 64
    kg, hm = curvatures(derivatives(a * ((X - 0.5) ** 2 + (Y - 0.5) ** 2), h))
    assert kg[c, c] == pytest.approx(4 * a * a, rel=1e-3)
    assert hm[c, c] == pytest.approx(2 * a, rel=1e-3)
    kg, hm = curvatures(derivatives(a * ((X - 0.5) ** 2 - (Y - 0.5) ** 2), h))
    assert kg[c, c] == pytest.approx(-4 * a * a, rel=1e-3)
    assert abs(hm[c, c]) < 1e-6
    kg, hm = curvatures(derivatives(a * (X - 0.5) ** 2, h))
    assert abs(kg[c, c]) < 1e-9
    assert hm[c, c] == pytest.approx(a, rel=1e-3)


def test_normal_is_unit_and_upward():
    X, Y, h = _grid(33)
    n = normals(derivatives(0.3 * X - 0.1 * Y, h))
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0)
    assert (n[..., 2] > 0).all()
    assert np.allclose(n[5, 5, 0], -0.3 / np.sqrt(1 + 0.09 + 0.01))


def test_quantizer_clipping_and_bin_mass():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(10000) * 3
    q = fit_scalar_quantizer("x", "signed", s, 16)
    codes, lo, hi = q.apply(np.array([-1e9, 0.0, 1e9]))
    assert codes.tolist() == [0, codes[1], 15]
    assert lo[0] and hi[2] and not lo[1] and not hi[1]
    c2, _, _ = q.apply(s)
    assert np.bincount(c2, minlength=16).sum() == len(s)
    # degenerate input -> single state
    qd = fit_scalar_quantizer("c", "signed", np.zeros(100), 16)
    assert qd.degenerate
    assert (qd.apply(rng.standard_normal(10))[0] == 0).all()
