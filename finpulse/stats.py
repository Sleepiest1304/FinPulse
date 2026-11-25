"""Small, dependency-free statistics used by the v3 event study.

* ``ols_twoway``: OLS with standard errors clustered two ways
  (Cameron, Gelbach and Miller 2011): V = V_a + V_b - V_ab, each with the
  small-sample factor G/(G-1). If the combined variance of a coefficient is
  not positive (possible with few clusters), the larger one-way variance is
  used instead and ``fallback`` is set.
* ``t_sf2``: two-sided p-value of a t statistic (via the regularised
  incomplete beta function), so scipy is not needed.
* ``holm`` and ``benjamini_hochberg``: multiple-comparison adjustments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- t distribution
def _betacf(a: float, b: float, x: float, iters: int = 300, eps: float = 1e-14) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, iters + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf2(t: float, df: float) -> float:
    """Two-sided p-value P(|T| >= |t|) for a t distribution with ``df`` degrees."""
    if not np.isfinite(t) or df <= 0:
        return float("nan")
    return betainc(df / 2.0, 0.5, df / (df + t * t))


# ---------------------------------------------------------------- clustered OLS
@dataclass
class OLSResult:
    coef: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    df: int
    n: int
    n_clusters: tuple[int, int]
    fallback: bool

    def ci(self, k: int, level: float = 0.95) -> tuple[float, float]:
        crit = t_crit(level, self.df)
        return self.coef[k] - crit * self.se[k], self.coef[k] + crit * self.se[k]


def t_crit(level: float, df: int) -> float:
    """Critical value c with P(|T| <= c) = level, found by bisection."""
    lo, hi = 0.0, 1e3
    target = 1.0 - level
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_sf2(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _meat(X: np.ndarray, u: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, int]:
    _, inv = np.unique(groups, return_inverse=True)
    G = int(inv.max()) + 1
    scores = np.zeros((G, X.shape[1]))
    np.add.at(scores, inv, X * u[:, None])
    factor = G / (G - 1) if G > 1 else 1.0
    return factor * scores.T @ scores, G


def ols_twoway(y, X, cluster_a, cluster_b) -> OLSResult:
    """OLS of y on X (include a constant column yourself), two-way clustered."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    a = np.asarray(cluster_a).astype(str)
    b = np.asarray(cluster_b).astype(str)
    ab = np.char.add(np.char.add(a, "|"), b)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    Ma, Ga = _meat(X, u, a)
    Mb, Gb = _meat(X, u, b)
    Mab, _ = _meat(X, u, ab)
    Va = XtX_inv @ Ma @ XtX_inv
    Vb = XtX_inv @ Mb @ XtX_inv
    V = Va + Vb - XtX_inv @ Mab @ XtX_inv
    var = np.diag(V).copy()
    bad = ~(var > 0)
    var[bad] = np.maximum(np.diag(Va), np.diag(Vb))[bad]
    se = np.sqrt(var)
    df = max(min(Ga, Gb) - 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = np.array([t_sf2(float(v), df) for v in t])
    return OLSResult(beta, se, t, p, df, n, (Ga, Gb), bool(bad.any()))


# ---------------------------------------------------------------- adjustments
def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (NaN entries stay NaN)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        out[idx] = running
    return out


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values); NaN entries stay NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 1.0
    for i in range(m - 1, -1, -1):
        idx = order[i]
        running = min(running, p[idx] * m / (i + 1))
        out[idx] = min(running, 1.0)
    return out
