import numpy as np
import pytest

from finpulse.stats import benjamini_hochberg, betainc, holm, ols_twoway, t_crit, t_sf2


@pytest.mark.parametrize("t, df, expected", [
    (2.0, 10, 0.073388),     # reference values from standard t tables
    (1.96, 1000, 0.050278),
    (2.776445, 4, 0.05),
    (0.0, 7, 1.0),
])
def test_t_sf2_matches_reference(t, df, expected):
    assert t_sf2(t, df) == pytest.approx(expected, abs=1e-5)


def test_t_sf2_symmetric_and_invalid():
    assert t_sf2(-2.0, 10) == pytest.approx(t_sf2(2.0, 10))
    assert np.isnan(t_sf2(np.nan, 10))
    assert np.isnan(t_sf2(1.0, 0))


def test_betainc_edges_and_symmetry():
    assert betainc(2, 3, 0) == 0 and betainc(2, 3, 1) == 1
    assert betainc(2, 3, 0.4) == pytest.approx(1 - betainc(3, 2, 0.6))
    assert betainc(1, 1, 0.3) == pytest.approx(0.3)


def test_t_crit():
    assert t_crit(0.95, 10) == pytest.approx(2.228139, abs=1e-4)


def test_ols_recovers_coefficients():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = 1.0 + 2.0 * x + rng.normal(scale=0.1, size=400)
    X = np.column_stack([np.ones(400), x])
    res = ols_twoway(y, X, rng.integers(0, 40, 400), rng.integers(0, 30, 400))
    assert res.coef == pytest.approx([1.0, 2.0], abs=0.03)
    assert res.n == 400 and res.df == min(res.n_clusters) - 1
    lo, hi = res.ci(1)
    assert lo < 2.0 < hi


def test_unique_clusters_reduce_to_robust_se():
    # With every observation its own cluster in both dimensions, V_a + V_b - V_ab
    # equals the heteroskedasticity-robust sandwich with an n/(n-1) factor.
    rng = np.random.default_rng(1)
    n = 50
    x = rng.normal(size=n)
    y = x + rng.normal(size=n) * (1 + np.abs(x))
    X = np.column_stack([np.ones(n), x])
    ids = np.arange(n)
    res = ols_twoway(y, X, ids, ids)
    inv = np.linalg.inv(X.T @ X)
    u = y - X @ res.coef
    hc = inv @ (X.T * u ** 2) @ X @ inv * n / (n - 1)
    assert res.se == pytest.approx(np.sqrt(np.diag(hc)))


def test_clustering_widens_se_for_correlated_errors():
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(20), 20)
    y = rng.normal(size=20)[g] + rng.normal(scale=0.2, size=400)
    X = np.ones((400, 1))
    clustered = ols_twoway(y, X, g, g).se[0]
    naive = ols_twoway(y, X, np.arange(400), np.arange(400)).se[0]
    assert clustered > 2 * naive


def test_holm():
    adj = holm([0.01, 0.04, 0.03, np.nan])
    assert adj[:3] == pytest.approx([0.03, 0.06, 0.06])
    assert np.isnan(adj[3])


def test_benjamini_hochberg():
    adj = benjamini_hochberg([0.01, 0.02, 0.03, 0.5])
    assert adj == pytest.approx([0.04, 0.04, 0.04, 0.5])
    assert benjamini_hochberg([1.0, 1.0]) == pytest.approx([1.0, 1.0])
