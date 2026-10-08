"""Float64 replica of the implemented discrete energy and solver operators.

Used ONLY for the mandatory solver audit (Experiment 0) and for the clearly
labelled diagnostic variant ``corrected_gradient_torch``.  Every operator here
mirrors the Rust source line-for-line (``stencils.rs``, ``scharr.rs``,
``energy.rs``, ``pde.rs``) with the same mirror boundary convention
(``util::mirror``: z(-1) = z(1), z(n) = z(n-2), i.e. numpy/torch "reflect").
The replica is validated against the Rust-exported energy terms before any
conclusion is drawn from it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from facial_entropy.config import SolverConfig


def _pad(z: np.ndarray, r: int = 1) -> np.ndarray:
    return np.pad(z, r, mode="reflect")


def laplacian_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    p = _pad(z)
    return p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4.0 * z


def scharr_np(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=np.float64)
    p = _pad(z)
    a, b = 3.0, 10.0
    mm, mc, mp = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    cm, cp = p[1:-1, :-2], p[1:-1, 2:]
    pm, pc, pp = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    gx = a * (mp - mm) + b * (cp - cm) + a * (pp - pm)
    gy = a * (pm - mm) + b * (pc - mc) + a * (pp - mp)
    return gx / 32.0, gy / 32.0


def divergence_np(fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    px, py = _pad(np.asarray(fx, dtype=np.float64)), _pad(np.asarray(fy, dtype=np.float64))
    return (px[1:-1, 2:] - px[1:-1, :-2]) * 0.5 + (py[2:, 1:-1] - py[:-2, 1:-1]) * 0.5


def energy_np(z, zf, ix, iy, w, k, lam, alpha, beta) -> dict[str, float]:
    z = np.asarray(z, dtype=np.float64)
    data = lam * float(((z - zf) ** 2).sum())
    smooth = alpha * float((laplacian_np(z) ** 2).sum())
    zx, zy = scharr_np(z)
    cons = beta * float(((w * (ix - k * zx)) ** 2 + (w * (iy - k * zy)) ** 2).sum())
    return {"data": data, "smoothness": smooth, "consistency": cons, "total": data + smooth + cons}


def implemented_residual_np(z, zf, ix, iy, w, k, lam, alpha, beta, sign: float = -1.0) -> np.ndarray:
    """The operator ``R(z)`` evaluated in ``pde.rs`` (sign=-1) or with the opposite divergence sign."""
    z = np.asarray(z, dtype=np.float64)
    zx, zy = scharr_np(z)
    fx = w * w * k * (ix - k * zx)
    fy = w * w * k * (iy - k * zy)
    return lam * (z - zf) + alpha * laplacian_np(laplacian_np(z)) + sign * beta * divergence_np(fx, fy)


def initial_tau(w: np.ndarray, k: np.ndarray, cfg: SolverConfig) -> float:
    max_wk2 = float(((w * k) ** 2).max())
    diag = cfg.pde_lambda + 20.0 * cfg.pde_alpha + 16.0 * cfg.pde_beta * max_wk2
    return 0.5 / diag if diag > 0 else 0.5


# ----------------------------------------------------------------------------
# torch autograd versions (true gradient of the replica energy)
# ----------------------------------------------------------------------------
def _torch():
    import torch  # noqa: PLC0415

    return torch


def energy_torch(z, zf, ix, iy, w, k, lam, alpha, beta):
    torch = _torch()
    F = torch.nn.functional

    def pad(a):
        return F.pad(a[None, None], (1, 1, 1, 1), mode="reflect")[0, 0]

    def lap(a):
        p = pad(a)
        return p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4 * a

    def scharr(a):
        p = pad(a)
        A, B = 3.0, 10.0
        mm, mc, mp = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
        cm, cp = p[1:-1, :-2], p[1:-1, 2:]
        pm, pc, pp = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
        gx = A * (mp - mm) + B * (cp - cm) + A * (pp - pm)
        gy = A * (pm - mm) + B * (pc - mc) + A * (pp - mp)
        return gx / 32, gy / 32

    data = lam * ((z - zf) ** 2).sum()
    smooth = alpha * (lap(z) ** 2).sum()
    zx, zy = scharr(z)
    cons = beta * ((w * (ix - k * zx)) ** 2 + (w * (iy - k * zy)) ** 2).sum()
    return data, smooth, cons, data + smooth + cons


def true_gradient(z, zf, ix, iy, w, k, lam, alpha, beta) -> np.ndarray:
    """dE/dz of the replica discrete energy by autograd (float64)."""
    torch = _torch()
    T = lambda a: torch.tensor(np.asarray(a, dtype=np.float64))  # noqa: E731
    zt = T(z).requires_grad_(True)
    e = energy_torch(zt, T(zf), T(ix), T(iy), T(w), T(k), lam, alpha, beta)[3]
    (g,) = torch.autograd.grad(e, zt)
    return g.numpy()


def lbfgs_minimise(
    z0: np.ndarray,
    i_w: np.ndarray,
    w: np.ndarray,
    k: np.ndarray,
    cfg: SolverConfig,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> dict[str, Any]:
    """Minimise the replica energy from z_forged with L-BFGS (diagnostic variant only)."""
    torch = _torch()
    ix, iy = scharr_np(i_w)
    T = lambda a: torch.tensor(np.asarray(a, dtype=np.float64))  # noqa: E731
    zf, ixt, iyt, wt, kt = T(z0), T(ix), T(iy), T(w), T(k)
    z = zf.clone().requires_grad_(True)
    lam, al, be = cfg.pde_lambda, cfg.pde_alpha, cfg.pde_beta
    opt = torch.optim.LBFGS(
        [z], lr=1.0, max_iter=max_iter, tolerance_grad=tol, tolerance_change=tol, history_size=20,
        line_search_fn="strong_wolfe",
    )
    trace: list[float] = []

    def closure():
        opt.zero_grad()
        e = energy_torch(z, zf, ixt, iyt, wt, kt, lam, al, be)[3]
        e.backward()
        trace.append(float(e.detach()))
        return e

    g0 = true_gradient(z0, z0, ix, iy, w, k, lam, al, be)
    opt.step(closure)
    zs = z.detach().numpy()
    g1 = true_gradient(zs, z0, ix, iy, w, k, lam, al, be)
    terms = energy_np(zs, z0, ix, iy, w, k, lam, al, be)
    ratio = float(np.linalg.norm(g1) / max(np.linalg.norm(g0), 1e-300))
    return {
        "z_star": zs,
        "trace": trace,
        "iterations": len(trace),
        "converged": ratio < 1e-3,
        "grad_norm_ratio": ratio,
        "energy_total": terms["total"],
        "energy_data": terms["data"],
        "energy_smoothness": terms["smoothness"],
        "energy_consistency": terms["consistency"],
    }
