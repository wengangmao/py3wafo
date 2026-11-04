import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.integrate import trapezoid

from wafo import trgauss as tr
from wafo.spec import createspec, jonswap, spec2mom
from wafo.trgauss._quadratic_saddlepoint import _eww


def _narrow():
    w = np.linspace(.97, 1.03, 13)
    s = np.exp(-((w-1)/.01)**2)
    return w, s/(s.sum()*(w[1]-w[0]))


def _skewness(x, f):
    m = trapezoid(x*f, x)
    v = trapezoid((x-m)**2*f, x)
    return trapezoid((x-m)**3*f, x)/v**1.5


def test_ewwdir_reduces_to_long_crested_sum_transfer_and_deep_stokes_term():
    w = np.linspace(.3, 2, 7); th = np.zeros(7)
    assert_allclose(tr.ewwdir(w, th, w, th, 40), _eww(w, w, 40), rtol=1e-12, atol=1e-14)
    # Deep-water self interaction is the second-order Stokes coefficient k=w^2/g.
    E = tr.ewwdir(w, th, w, th, 1e9)
    assert_allclose(np.diag(E), w**2/9.80629, rtol=1e-5)
    # Rotating every direction leaves the transfer function unchanged.
    turned = tr.ewwdir(w, th+.7, w, th+.7, 40)
    assert_allclose(turned, tr.ewwdir(w, th, w, th, 40), rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError):
        tr.ewwdir(w, th, w[:-1], th[:-1])


def test_dirsp2chitwo_preserves_linear_variance_and_stokes_skewness():
    w, s = _narrow()
    g, b, S12, S22 = tr.dirsp2chitwo(s, w, h=5000, eps=0)
    assert len(g) == len(b) == 2*len(w)
    assert_allclose(np.sum(b**2), 1, rtol=1e-12)
    assert_allclose(S12, -S12.T, atol=1e-12)
    assert np.linalg.eigvalsh(S22).min() > -1e-12
    skew = (6*np.sum(g*b**2)+8*np.sum(g**3))/(np.sum(b**2)+2*np.sum(g**2))**1.5
    assert_allclose(skew, 3/9.80629, rtol=.05)
    # The literal source scaling doubles every quadratic coefficient.
    g2 = tr.dirsp2chitwo(s, w, h=5000, eps=0, source_scaling=True)[0]
    assert_allclose(np.sort(g2), 2*np.sort(g), rtol=1e-10, atol=1e-14)
    # Reduction merges negligible quadratic components without losing variance.
    gr, br, S12r, S22r = tr.dirsp2chitwo(s, w, h=5000)
    assert len(gr) < len(g) and gr[0] == 0
    assert_allclose(np.sum(br**2), 1, rtol=1e-10)
    assert_allclose(S12r, -S12r.T, atol=1e-12)


def test_chitwo2lc_sp_density_and_rice_term_are_exact_for_scaled_chi_square():
    # X=g(Z1^2+Z2^2) is exponential; saddlepoint densities of gamma laws are exact.
    gamma = .5; lam = 2.
    mu, mu2, mu1, muG, beta, cc, fu, Fu = tr.chitwo2lc_sp([gamma, gamma], [0, 0], np.zeros((2, 2)), lam*np.eye(2), .05)
    x = fu.x[0]
    inner = (x > .2) & (x < 4.5)
    law = stats.expon(scale=2*gamma)
    # As in the source, the density is normalized over the marched level range.
    pdf = law.pdf(x)/(law.cdf(x[-1])-law.cdf(x[0]))
    assert_allclose(fu.f[inner], pdf[inner], rtol=2e-3)
    # Conditionally on X=u, X'~N(0,4*gamma*lam*u): one-term Rice intensity is exact.
    exact = pdf*np.sqrt(4*gamma*lam*x/(2*np.pi))
    assert_allclose(muG[inner, 1], exact[inner], rtol=2e-3)
    # Higher-order crossing corrections vanish for this process.
    assert_allclose(mu1[inner, 1], exact[inner], rtol=2e-3)
    assert_allclose(mu2[inner, 1], exact[inner], rtol=2e-3)
    # The source multiplies the normalized density by Daniels' 1+R0 (=11/12 here).
    assert_allclose(mu[inner, 1], exact[inner]*11/12, rtol=2e-3)
    assert np.all(np.diff(Fu[:, 1]) > -1e-8)
    assert_allclose(Fu[inner, 1], law.cdf(x[inner]), rtol=2e-2, atol=1e-3)
    assert np.all(np.isfinite(beta)) and np.all(cc > 0)


def test_specq2lc_and_chitwo_route_agree_and_match_stokes_skewness():
    S = jonswap(np.linspace(.2, 2.6, 80)); S.h = 5000
    mu, fu, Fu, muG, mu1, mu2 = tr.specq2lc(S, .05)
    m0, m2 = spec2mom(S, 2)
    assert_allclose(trapezoid(fu.f[:, 0], fu.x[0]), 1, rtol=1e-12)
    assert_allclose(mu[:, 2], np.exp(-mu[:, 0]**2/(2*m0))*np.sqrt(m2/m0)/(2*np.pi), rtol=1e-12)
    assert np.all(np.diff(Fu[:, 1]) > -1e-8)
    g, b, S12, S22 = tr.dirsp2chitwo(S.S, S.w, h=5000, eps=0)
    out = tr.chitwo2lc_sp(g, b, S12, S22, .05)
    for level in (-3, 0, 3, 6):
        assert_allclose(np.interp(level, fu.x[0], fu.f[:, 0]), np.interp(level, out[6].x[0], out[6].f), rtol=1e-2)
        assert_allclose(np.interp(level, mu[:, 0], mu[:, 1]), np.interp(level, out[0][:, 0], out[0][:, 1]), rtol=1e-2)
    # High crests are more frequent than in the linear Gaussian sea.
    high = mu[:, 0] > 6
    assert np.all(mu[high, 1] > mu[high, 2])
    w, s = _narrow()
    narrow = tr.specq2lc(createspec(w=w, S=s, h=5000), .02)[1]
    assert_allclose(_skewness(narrow.x[0], narrow.f[:, 0]), 3/9.80629, rtol=.05)
    assert tr.specq2lc(S, .1, nb=1)[0].shape[1] == 3


def test_breitung_expansion_is_rice_for_linear_and_close_to_sorm_for_weak_quadratic():
    b = np.array([1., .5]); S12 = np.array([[0, .3], [-.3, 0]]); S22 = np.eye(2)
    levels = np.array([1., 2., 3.])
    linear = tr.chitwo2lc_sorm(levels, [0, 0], b, S22, S12, ass=1)
    variance = b@b
    rice = np.sqrt(b@S22@b/variance)/(2*np.pi)*np.exp(-levels**2/(2*variance))
    assert_allclose(linear[:, 1], rice, rtol=1e-10)
    weak = tr.chitwo2lc_sorm(levels, [0, .02], b, S22, S12, ass=1)
    sorm = tr.chitwo2lc_sorm(levels, [0, .02], b, S22, S12, ass=0)
    assert_allclose(weak[:, 1], sorm[:, 1], rtol=.03)
    Cb0, Cb1, a0, a1 = tr.rqlf_asympt(2., [0, .02], b, S22, S12)
    assert Cb0 > 0 and abs(Cb1) < Cb0 and a0 > 0


def test_writecov_writes_legacy_spline_input_files(tmp_path):
    t = np.linspace(0, 1, 4)
    cov = np.column_stack((t, np.cos(t), -np.sin(t), -np.cos(t)))
    paths = tr.writecov(cov, directory=tmp_path)
    assert [p.name for p in paths] == ['Cd0.in', 'Cd1.in', 'Cd2.in']
    for k, path in enumerate(paths):
        data = np.loadtxt(path)
        assert data.shape == (4, 5)
        assert_allclose(data[:, :2], cov[:, [0, k+1]], rtol=1e-10, atol=1e-12)
        assert not data[:, 2:].any()
    assert open(paths[0]).readline() == '0.0000000000 1.0000000000E+00 0.00 0.00 0.00\n'
    with pytest.raises(ValueError):
        tr.writecov(cov, nr=3, directory=tmp_path)


def _chain(n, skew, N, seed):
    rng = np.random.default_rng(seed)
    G = np.zeros((n, n)); c = (n-1)/2
    for i in range(n):
        for j in range(i+1, n):
            G[i, j] = np.exp(-((i+j)/2-c-skew)**2/4)*(j-i)*np.exp(-(j-i)/2)
    up = np.cumsum(G/G.sum(axis=1, keepdims=True).clip(1e-300), axis=1)
    down = np.cumsum((G/G.sum(axis=0, keepdims=True).clip(1e-300)).T, axis=1)
    x = np.empty(N, int); x[0] = 0; r = rng.random(N)
    for k in range(1, N):
        x[k] = min(np.searchsorted((up if k % 2 else down)[x[k-1]], r[k]), n-1)
    return G, x


def _trough_crest(x, utc, n):
    T = np.zeros((n, n))
    for k in range(0, len(x)-1, 2):
        if not x[k] < utc < x[k+1]:
            continue
        a, b = k, k+1
        while a >= 2 and x[a-1] <= utc:
            a -= 2
        while b+2 < len(x) and x[b+1] >= utc:
            b += 2
        if a >= 2 and b+2 < len(x):
            T[x[a:k+1:2].min(), x[k+1:b+1:2].max()] += 1
    return T


def test_mctp2tc_matches_simulated_turning_point_chains():
    n, utc = 9, 4.
    # Chains whose cycles all cross utc are their own trough-crest matrix.
    G = np.zeros((n, n)); G[:4, 5:] = np.arange(1., 17.).reshape(4, 4)
    assert_allclose(np.fliplr(tr.mctp2tc(np.fliplr(G), utc, (0, n-1, n))), G, atol=1e-12)
    # Centre, off-centre and between-level reference levels, symmetric and skewed chains.
    for m, level, skew, seed in ((9, 4., 0., 1), (9, 4., 1., 3), (9, 3., 0., 5), (10, 4.5, 0., 7)):
        G, x = _chain(m, skew, 120000, seed)
        F = np.fliplr(tr.mctp2tc(np.fliplr(G), level, (0, m-1, m)))
        assert F.min() >= 0
        # Total frequency equals the expected number of upcrossings.
        u = np.arange(m)
        assert_allclose(F.sum(), G[np.ix_(u < level, u > level)].sum(), rtol=1e-10)
        T = _trough_crest(x, level, m)
        assert abs(F/F.sum()-T/T.sum()).max() < 6e-3
    with pytest.raises(ValueError):
        tr.mctp2tc(np.fliplr(G), 100., (0, n-1, n))
