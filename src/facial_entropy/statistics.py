"""Subject-level inference: paired summaries, group bootstrap, sign-flip permutation, Holm, power."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from facial_entropy import regions as reg
from facial_entropy.util import rng_for


def subject_differences(rows: pd.DataFrame, metric: str, estimator: str = "fixed") -> pd.DataFrame:
    """C, P, D per image (equal patch weights) then averaged within subject."""
    r = rows[(rows["metric"] == metric) & (rows["estimator"] == estimator) & rows["retained"].astype(bool)]
    if r.empty:
        return pd.DataFrame(columns=["subject_id", "n_images", "C", "P", "D"])
    piv = r.pivot_table(index=["subject_id", "image_id"], columns="patch", values="value", aggfunc="first")
    cen = [p for p in reg.CENTRAL_PATCHES if p in piv]
    per = [p for p in reg.PERIPHERAL_PATCHES if p in piv]
    piv = piv.dropna(subset=cen + per)
    img = pd.DataFrame({"C": piv[cen].mean(axis=1), "P": piv[per].mean(axis=1)})
    img["D"] = img["C"] - img["P"]
    subj = img.groupby(level="subject_id").agg(n_images=("D", "size"), C=("C", "mean"), P=("P", "mean"), D=("D", "mean")).reset_index()
    return subj


def group_bootstrap_ci(d: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    rng = rng_for("bootstrap", base=seed)
    n = len(d)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def sign_flip_test(d: np.ndarray, n_perm: int, seed: int) -> dict[str, Any]:
    """Two-sided sign-flip permutation test of the mean paired difference.

    p = (1 + #{|T_b| >= |T_obs|}) / (B + 1).  Exact enumeration when 2^n <= n_perm.
    Interpreting this as a test of the population mean assumes sign symmetry /
    exchangeability of subject-level differences under H0.
    """
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    if n == 0:
        return {"p": float("nan"), "T_obs": float("nan"), "exact": False, "B": 0}
    t_obs = d.mean()
    if 2**n <= n_perm:
        signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        ts = (signs * d).mean(axis=1)
        p = float(((np.abs(ts) >= abs(t_obs) - 1e-15).sum()) / len(ts))
        return {"p": p, "T_obs": float(t_obs), "exact": True, "B": int(2**n)}
    rng = rng_for("signflip", base=seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    ts = (signs * d).mean(axis=1)
    p = float((1 + (np.abs(ts) >= abs(t_obs) - 1e-15).sum()) / (n_perm + 1))
    return {"p": p, "T_obs": float(t_obs), "exact": False, "B": int(n_perm)}


def sign_test(d: np.ndarray) -> dict[str, Any]:
    d = np.asarray(d, dtype=np.float64)
    nz = d[d != 0]
    k = int((nz > 0).sum())
    n = int(len(nz))
    if n == 0:
        return {"p": float("nan"), "n_nonzero": 0, "n_positive": 0}
    return {"p": float(sps.binomtest(k, n, 0.5).pvalue), "n_nonzero": n, "n_positive": k}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = [(k, v) for k, v in pvals.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {k: float("nan") for k in pvals}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def paired_summary(subj: pd.DataFrame, n_boot: int, n_perm: int, seed: int, alpha: float = 0.05) -> dict[str, Any]:
    d = subj["D"].to_numpy(dtype=np.float64)
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    lo, hi = group_bootstrap_ci(d, n_boot, seed, alpha)
    perm = sign_flip_test(d, n_perm, seed)
    st = sign_test(d)
    return {
        "n_groups": int(n),
        "n_images": int(subj["n_images"].sum()) if n else 0,
        "central_mean": float(subj["C"].mean()) if n else float("nan"),
        "central_sd": float(subj["C"].std(ddof=1)) if n > 1 else float("nan"),
        "peripheral_mean": float(subj["P"].mean()) if n else float("nan"),
        "peripheral_sd": float(subj["P"].std(ddof=1)) if n > 1 else float("nan"),
        "mean_diff": float(d.mean()) if n else float("nan"),
        "median_diff": float(np.median(d)) if n else float("nan"),
        "sd_diff": sd,
        "ci95_bootstrap": [lo, hi],
        "d_z": (float(d.mean() / sd) if (n > 1 and sd > 0) else None),
        "d_z_note": "undefined (sd = 0)" if (n > 1 and sd == 0) else "",
        "prop_positive": float((d > 0).mean()) if n else float("nan"),
        "signflip_p": perm["p"],
        "signflip_exact": perm["exact"],
        "signflip_B": perm["B"],
        "signtest_p": st["p"],
        "signtest_n_nonzero": st["n_nonzero"],
        "bootstrap_B": n_boot,
        "seed": seed,
    }


def power_scenarios(sd_diff: float, alpha_family: float = 0.025, powers=(0.8, 0.9), effects=(0.2, 0.3, 0.5)) -> dict[str, Any]:
    """Normal-approximation planning: n ~ ((z_{1-a/2} + z_{1-b}) / d)^2 (guidance only)."""
    out = {}
    za = sps.norm.ppf(1 - alpha_family / 2)
    for pw in powers:
        zb = sps.norm.ppf(pw)
        out[f"power_{pw}"] = {f"d_{e}": int(np.ceil(((za + zb) / e) ** 2)) for e in effects}
    out["note"] = "two-sided alpha per endpoint 0.025 (two-endpoint family), standardised paired effect d; planning guidance only"
    out["sd_diff_pilot"] = sd_diff
    return out


def ci_width_scenarios(sd_diff: float, ns=(40, 100, 160, 400)) -> dict[str, float]:
    return {str(n): float(2 * 1.96 * sd_diff / np.sqrt(n)) for n in ns}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(sps.spearmanr(a[m], b[m]).statistic)


def map_comparison(m1: np.ndarray, m2: np.ndarray) -> dict[str, float]:
    """Rank correlation on shared valid support, normalised difference, top-decile overlap."""
    m = np.isfinite(m1) & np.isfinite(m2)
    if m.sum() < 10:
        return {"spearman": float("nan"), "norm_diff": float("nan"), "top_decile_overlap": float("nan"), "n_shared": int(m.sum())}
    a, b = m1[m], m2[m]
    rho = spearman(a, b)
    nd = float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(a**2)), 1e-12))
    qa, qb = np.quantile(a, 0.9), np.quantile(b, 0.9)
    ta, tb = a >= qa, b >= qb
    ov = float((ta & tb).sum() / max((ta | tb).sum(), 1))
    return {"spearman": rho, "norm_diff": nd, "top_decile_overlap": ov, "n_shared": int(m.sum())}
