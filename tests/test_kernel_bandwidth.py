"""Independent formulas and numerical invariants for legacy KDE selectors."""
import numpy as np
from scipy.integrate import trapezoid
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import integrate, stats

from wafo import kdetools as kd
from wafo.kdetools._bandwidth import _bcv2_objective, _normal_mixed


@pytest.fixture
def normal_sample():
    return stats.norm.ppf((np.arange(140)+.5)/140)


@pytest.fixture
def bivariate_sample():
    rng = np.random.default_rng(19)
    return rng.multivariate_normal([0, 0], [[1, .3], [.3, 1]], 180)


def test_bootstrap_cv_exact_three_gaussian_objective_and_kernel_rescaling():
    x = np.array([-1.2, -.7, -.2, .1, .5, 1.4])
    candidates = np.array([.2, .4, .7])
    h, hs, scores = kd.hboot(x, hvec=candidates, full_output=True)
    expected = []
    n = len(x)
    for bw in candidates:
        total = n*np.sqrt(2)
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = (x[i]-x[j])**2/bw**2
                    total += np.exp(-d/8)-4/np.sqrt(3)*np.exp(-d/6)+np.sqrt(2)*np.exp(-d/4)
        expected.append(total/(2*n*n*bw*np.sqrt(2*np.pi)))
    assert_allclose(scores, expected, rtol=2e-14)
    assert_array_equal(hs, candidates)
    assert h == hs[np.argmin(expected)]
    ratio = kd.hns(x, 'epan')[0]/kd.hns(x, 'gauss')[0]
    eh, ehs, escore = kd.hboot(x, 'epan', candidates*ratio, full_output=True)
    assert_allclose(escore, scores)
    assert_allclose(eh, h*ratio)


def test_scv_objective_has_its_own_pilots_and_scale_equivariance(normal_sample):
    x = normal_sample
    grid = np.linspace(.12, .8, 20)
    h, candidates, scores = kd.hscv(x, hvec=grid, full_output=True)
    assert h == candidates[np.argmin(scores)]
    assert np.all(np.isfinite(scores))
    other, _, scaled_scores = kd.hscv(x*3+7, hvec=grid*3, full_output=True)
    assert_allclose(other, 3*h, rtol=1e-13)
    assert_allclose(scaled_scores, scores/3, rtol=5e-12)
    assert not np.allclose(scores, kd.hboot(x, hvec=grid, full_output=True)[2])


def test_solve_equation_selectors_converge_and_preserve_units(normal_sample):
    x = normal_sample
    for selector in (kd.hste, kd.hstt):
        h, info = selector(x, inc=256, releps=1e-6, maxit=200, full_output=True)
        assert h.shape == (1,) and h[0] > 0
        assert info['converged'].all() and info['iterations'][0] > 1
        assert_allclose(selector(2.5*x-4, inc=256, releps=1e-6, maxit=200), 2.5*h, rtol=1e-10)
        multi = selector(np.column_stack((x, 2*x)), inc=256, releps=1e-6, maxit=200)
        assert_allclose(multi, [h[0], 2*h[0]], rtol=1e-10)
        with pytest.warns(RuntimeWarning, match='maxit'):
            last, incomplete = selector(x, inc=64, maxit=1, releps=0, full_output=True)
        assert not incomplete['converged'].any() and np.all(last > 0)


def test_stt_fixed_point_matches_independent_finite_difference_curvature(normal_sample):
    x = normal_sample; inc = 256
    h = kd.hstt(x, inc=inc, releps=1e-9, maxit=300)[0]
    span = np.ptp(x)
    grid = np.linspace(x.min()-span/4, x.max()+span/4, inc)
    density = stats.norm.pdf((grid[:, None]-x[None, :])/h).mean(axis=1)/h
    dx = grid[1]-grid[0]
    curvature = dx*np.sum((np.diff(density, 2)/dx**2)**2)
    expected = (1/(2*np.sqrt(np.pi)*len(x)*curvature))**.2
    assert_allclose(h, expected, rtol=2e-9)


def test_bivariate_normal_reference_derivatives_equal_gaussian_integrals():
    cov = np.diag([1.5**2, .7**2])
    # Independence separates these density-derivative inner products.
    integral_x4 = integrate.quad(lambda x: (x**4-6*x*x+3)*stats.norm.pdf(x)**2, -9, 9)[0]
    integral_x2 = integrate.quad(lambda x: (x*x-1)*stats.norm.pdf(x)**2, -9, 9)[0]
    assert_allclose(_normal_mixed((4, 0), cov), integral_x4/(1.5**5)*1/(2*np.sqrt(np.pi)*.7), rtol=1e-12)
    assert_allclose(_normal_mixed((2, 2), cov), integral_x2**2/(1.5**3*.7**3), rtol=1e-12)
    assert _normal_mixed((6, 0), cov) < 0
    assert _normal_mixed((10, 0), cov) < 0


@pytest.mark.parametrize('L', [0, 1, 2, 3])
def test_bivariate_dpi_stages_exact_and_fft_convergence(bivariate_sample, L):
    x = bivariate_sample
    exact, info = kd.hldpi2(x, L=L, full_output=True)
    coarse = kd.hldpi2fft(x, L=L, inc=64)
    fine = kd.hldpi2fft(x, L=L, inc=256)
    assert np.all(exact > 0) and np.all(np.isfinite(exact))
    assert info['psi40'] > 0 and info['psi04'] > 0
    assert_allclose(fine, exact, rtol=.006)
    assert np.linalg.norm(fine-exact) <= np.linalg.norm(coarse-exact)+1e-13
    assert_allclose(kd.hldpi2(x*[2, .4]+[8, -2], L=L), exact*[2, .4], rtol=2e-13)
    assert_allclose(kd.hldpi2fft(x*[2, .4]+[8, -2], L=L, inc=256), fine*[2, .4], rtol=2e-12)
    assert_allclose(kd.hldpi2(x[:, ::-1], L=L), exact[::-1], rtol=2e-13)


def test_bivariate_bcv_source_objective_and_optimized_descent(bivariate_sample):
    x = bivariate_sample[:30]
    h = np.array([.4, .7]); n = len(x)
    z = (x[:, None, :]-x[None, :, :])/h
    t = (z*z).sum(axis=-1)
    terms = (t*t-8*t+8)*stats.norm.pdf(z[:, :, 0])*stats.norm.pdf(z[:, :, 1])
    np.fill_diagonal(terms, 0)
    expected = 1/(4*np.pi*n*np.prod(h))+terms.sum()/(4*n*(n-1)*np.prod(h))
    assert_allclose(_bcv2_objective(x, h), expected, rtol=1e-14)
    bw, info = kd.hbcv2(x, full_output=True)
    standardized = (x-x.mean(axis=0))/x.std(axis=0, ddof=1)
    assert info['converged'] and np.all(bw > 0)
    assert info['objective'] <= _bcv2_objective(standardized, kd.hns(standardized))+1e-12
    assert_allclose(kd.hbcv2(x*[2, .5]+[3, -4]), bw*[2, .5], rtol=2e-5)


def test_density_height_chi_square_tests_against_equal_probability_groups():
    # For a symmetric normal, |X| quantiles give exact equal-probability
    # density-height groups.  A deterministic probability grid spreads counts.
    probabilities = (np.arange(10000)+.5)/10000
    reference = stats.norm.pdf(stats.norm.ppf(probabilities))
    observed = stats.norm.pdf(stats.norm.ppf((np.arange(400)+.5)/400))
    p, statistic, dof = kd.chi2gof2(observed, reference, full_output=True)
    assert p > .99 and statistic < 1e-10 and dof == 19
    grid = np.linspace(-8, 8, 10001)
    p, statistic, dof = kd.chi2gof({'f': observed}, {'f': stats.norm.pdf(grid), 'x': [grid]}, full_output=True)
    assert p > .99 and statistic < 1 and dof == 19
    # Samples concentrated in the low-density tails strongly reject the null.
    assert kd.chi2gof2(np.full(400, stats.norm.pdf(4)), reference) < 1e-40
    with pytest.raises(ValueError, match='120'):
        kd.chi2gof2(observed[:120], reference)
    with pytest.raises(ValueError, match='Repeated'):
        kd.chi2gof(np.ones(400), np.ones(100))


def test_bandwidth_failures_are_explicit(normal_sample):
    with pytest.raises(ValueError):
        kd.hldpi2(np.column_stack((normal_sample, normal_sample)))
    with pytest.raises(ValueError):
        kd.hldpi2(np.column_stack((normal_sample, normal_sample**2)), L=4)
    with pytest.raises(ValueError):
        kd.hste(normal_sample, h=-1)
    with pytest.raises(ValueError):
        kd.hscv(normal_sample, hvec=[0])


@pytest.mark.parametrize('method', ['hboot', 'hscv', 'hste', 'hstt'])
def test_new_univariate_selectors_drive_real_kde_workflows(normal_sample, method):
    x = normal_sample
    points = np.linspace(-8, 8, 1001)
    f, H, factors = kd.kdefun(x, {'kernel': 'gaussian', 'hsMethod': method}, points, full_output=True)
    expected_h = float(np.asarray(getattr(kd, method)(x)).ravel()[0])
    assert_allclose(H, [[expected_h]])
    assert_allclose(f, stats.norm.pdf((points[:, None]-x[None, :])/expected_h).mean(axis=1)/expected_h)
    assert_allclose(trapezoid(f, points), 1, atol=1e-10)


@pytest.mark.parametrize('method', ['hbcv2', 'hldpi2', 'hldpi2fft', 'hste', 'hstt'])
def test_new_bivariate_selectors_drive_real_kde_workflows(bivariate_sample, method):
    x = bivariate_sample
    points = np.array([-.5, 0., 1.])
    f, H, _ = kd.kdefun(x, {'kernel': 'gaussian', 'hsMethod': method}, points, points/2, full_output=True)
    widths = np.asarray(getattr(kd, method)(x))
    assert_allclose(H, np.diag(widths))
    expected = (stats.norm.pdf((points[:, None]-x[None, :, 0])/widths[0])*
                stats.norm.pdf((points[:, None]/2-x[None, :, 1])/widths[1])).mean(axis=1)/np.prod(widths)
    assert_allclose(f, expected, rtol=1e-12)
    with pytest.raises(ValueError):
        kd.kdefun(x[:, 0], {'hsMethod': 'hldpi2'}, points)


def test_univariate_cross_validation_rejects_multidimensional_dispatch(bivariate_sample):
    for method in ('hboot', 'hscv'):
        with pytest.raises(ValueError, match='one data column'):
            kd.kdefun(bivariate_sample, {'hsMethod': method}, 0., 0.)
