"""Original coefficients, probability identities and independent wave moments."""
import json
from importlib.resources import files

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import integrate, special, stats

from wafo import wavemodels as w
from wafo.wavemodels._tables import _data


def test_tayfun_infinite_limit_broadcasting_and_validated_domain():
    x = np.array([-1., 0., .5, 1.5, np.inf])
    a = 2.
    assert_allclose(w.tay81pdf(x, a, np.inf), stats.rayleigh.pdf(x, scale=a/np.sqrt(2)))
    assert_allclose(w.tay81cdf(x, a, np.inf), stats.rayleigh.cdf(x, scale=a/np.sqrt(2)))
    with pytest.raises(NotImplementedError, match='integer'):
        w.tay81pdf(1, 1, 2.3)
    with pytest.raises(NotImplementedError):
        w.tay81cdf(1, 1, 2)
    with pytest.raises(ValueError):
        w.tay81pdf(1, -1, 4)


@pytest.mark.parametrize('N', [9, 16])
def test_finite_tayfun_normalization_and_independent_phasor_moments(N):
    B = np.sqrt(N)
    x = np.linspace(0, B, 601)
    density, error = w.tay81pdf(x, 1, B, 2e-5, full_output=True)
    # For N independent unit phasors, divided by sqrt(N):
    # E|sum|²/N=1 and E|sum|⁴/N²=2-1/N.
    assert_allclose(integrate.simpson(density, x=x), 1, atol=3e-6)
    assert_allclose(integrate.simpson(x*x*density, x=x), 1, atol=2e-5)
    assert_allclose(integrate.simpson(x**4*density, x=x), 2-1/N, atol=1e-4)
    assert np.all(density >= 0) and np.max(error) <= 2e-5
    assert_allclose(w.tay81pdf(x*3, 3, B, 2e-5), density/3, atol=2e-14)
    assert_array_equal(w.tay81cdf([-1, 0, B, np.inf], 1, B), [0, 0, 1, 1])


def test_tayfun_cdf_differentiates_to_its_hankel_density():
    x = np.array([.2, .5, 1., 1.8, 2.8])
    eps = .0002
    derivative = (w.tay81cdf(x+eps, 1, 4, 1e-6)-w.tay81cdf(x-eps, 1, 4, 1e-6))/(2*eps)
    assert_allclose(derivative, w.tay81pdf(x, 1, 4, 1e-6), atol=6e-7)
    # Linear and cubic scale parameters broadcast rather than flatten inputs.
    assert w.tay81pdf(x[:, None], [1., 2.], 4).shape == (len(x), 2)


def test_climate_source_default_formula_and_splice_mass_are_explicit():
    t, h = 10., 4.
    mean = 1.59+.42*np.log(h+2)
    variance = .005+.09*np.exp(-.13*h**1.34)
    conditional = stats.lognorm.pdf(t, np.sqrt(variance), scale=np.exp(mean))
    marginal = stats.weibull_min.pdf(h, 1.547, scale=2.82)
    actual, mass = w.ltwcpdf(t, h, full_output=True)
    assert_allclose(actual, conditional*marginal)
    assert_allclose(mass, .9996540576505768)
    assert_allclose(w.ltwcpdf(t, h, normalize=True), actual/mass)
    assert_allclose(w.ltwcpdf(t, h, condon=1), conditional)
    assert_allclose(w.ltwcpdf(t, h, condon=5), marginal*stats.lognorm.cdf(t, np.sqrt(variance), scale=np.exp(mean)))
    # The marginal selector is independent of t, including nonpositive t.
    assert_allclose(w.ltwcpdf([-1, 0, 10], h, condon=10), marginal)
    raw = integrate.quad(lambda hh: w.ltwcpdf(0, hh, condon=10), 0, 3.27)[0]
    raw += integrate.quad(lambda hh: w.ltwcpdf(0, hh, condon=10), 3.27, np.inf)[0]
    assert_allclose(raw, mass, atol=1e-10)
    normalized = integrate.quad(lambda hh: w.ltwcpdf(0, hh, condon=10, normalize=True), 0, 3.27)[0]
    normalized += integrate.quad(lambda hh: w.ltwcpdf(0, hh, condon=10, normalize=True), 3.27, np.inf)[0]
    assert_allclose(normalized, 1, atol=1e-10)


@pytest.mark.parametrize('definition', [0, 1, 2, 3])
def test_climate_parametrizations_conditional_moments_and_joint_consistency(definition):
    pt = [1., .2, 1.3, .02, .1, .3, 1.2]
    ph = [2.2, 2.5, 1.3, 2.7, .4, .7]
    h = 3.5
    mean = pt[0]+pt[1]*(np.log(h+pt[2]) if definition % 2 == 0 else h**pt[2])
    variance = pt[3]+pt[4]*np.exp(-pt[5]*h**pt[6])
    options = dict(Pt=pt, Ph=ph, definition=definition, normalize=True)
    marginal = w.ltwcpdf(1., h, condon=10, **options)
    assert_allclose(integrate.quad(lambda t: w.ltwcpdf(t, h, condon=1, **options), 0, np.inf)[0], 1, atol=1e-9)
    assert_allclose(integrate.quad(lambda t: w.ltwcpdf(t, h, **options), 0, np.inf)[0], marginal, atol=1e-9)
    for mode, power in ((3, 1), (4, 2)):
        actual = integrate.quad(lambda t: w.ltwcpdf(t, h, condon=mode, **options), 0, np.inf)[0]
        assert_allclose(actual, np.exp(power*mean+power**2*variance/2), rtol=1e-9)
    assert_allclose(w.ltwcpdf(3, 2, Pt=[np.nan]*7, Ph=[np.nan]*6), w.ltwcpdf(3, 2))


def test_nonlinear_velocity_asset_provenance_and_exact_table_node_formula():
    data = _data('jhvnlpar.mat')
    provenance = json.loads(files('wafo.wavemodels').joinpath('data', 'jhvnlpar.PROVENANCE.json').read_text())
    assert provenance['sha256'] == '9d3b370656ad3ad51fc8f939ac397e05c670b0857fd76d296844065bce156098'
    assert data['A11s'].shape == data['B11s'].shape == (25, 13)
    assert_array_equal(data['gam'].ravel(), np.arange(1, 7.1, .5))
    i, j = 12, 4
    h, g, v = data['h2'].ravel()[i], data['gam'].ravel()[j], 1.2
    a, b = data['A11s'][i, j], data['B11s'][i, j]
    expected_conditional = stats.weibull_min.pdf(v, b, scale=a)
    actual = w.jhvnlpdf(h, v, 6, 10, g, True, 1)
    assert_allclose(actual, expected_conditional, rtol=2e-13)
    marginal = w._remaining_models.wtweibpdf(h, *w.jhnlwparfun(6, 10, g))
    assert_allclose(w.jhvnlpdf(h, v, 6, 10, g, True), marginal*expected_conditional)
    _, _, _, scale_table, shape_table = w.jhvnlpdf(h, v, 6, 10, g, True, full_output=True)
    assert_array_equal(scale_table.f, data['A11s'])
    assert_array_equal(shape_table.f, data['B11s'])


def test_nonlinear_velocity_jacobians_conditional_moments_and_normalization():
    Hm0, Tp, gam = 6., 10., 3.
    h, v = 3., 1.5
    pdf, hs, vs, _, _ = w.jhvnlpdf(h, v, Hm0, Tp, gam, full_output=True)
    assert_allclose(w.jhvnlpdf(h/hs, v/vs, Hm0, Tp, gam, True), pdf*hs*vs)
    assert_allclose(w.jhvnlpdf(h/hs, v/vs, Hm0, Tp, gam, True, 1), w.jhvnlpdf(h, v, Hm0, Tp, gam, condon=1)*vs)
    assert_allclose(integrate.quad(lambda velocity: w.jhvnlpdf(h, velocity, Hm0, Tp, gam, condon=1), 0, np.inf)[0], 1, atol=1e-9)
    eps = 1e-5
    derivative = (w.jhvnlpdf(h, v+eps, Hm0, Tp, gam, condon=6)-w.jhvnlpdf(h, v-eps, Hm0, Tp, gam, condon=6))/(2*eps)
    assert_allclose(derivative, w.jhvnlpdf(h, v, Hm0, Tp, gam, condon=1), rtol=1e-9)
    for condon, power in ((3, 1), (4, 2)):
        moment = integrate.quad(lambda velocity: w.jhvnlpdf(h, velocity, Hm0, Tp, gam, condon=condon), 0, np.inf)[0]
        # Physical moment scales as velocity_scale**power.
        dimensionless = integrate.quad(lambda velocity: w.jhvnlpdf(h/hs, velocity, Hm0, Tp, gam, True, condon), 0, np.inf)[0]
        assert_allclose(moment, dimensionless*vs**power, rtol=1e-9)
    heights, velocities = np.linspace(0, 5, 401), np.linspace(0, 8, 501)
    grid = w.jhvnlpdf2(heights, velocities, Hm0, Tp, gam, True)
    assert grid.f.shape == (401, 501) and grid.norm == 1
    assert_allclose(integrate.simpson(integrate.simpson(grid.f, x=velocities, axis=1), x=heights), 1, atol=1e-4)


def test_nonlinear_joint_cdf_derivative_survival_and_height_marginal():
    Hm0, Tp, gam = 6., 10., 3.
    h, v = 3., 2.
    lower, error = w.jhvnlcdf(h, v, Hm0, Tp, gam, full_output=True)
    assert error < 1e-7 and 0 < lower < 1
    upper = w.jhvnlcdf(h, v, Hm0, Tp, gam, tail=1)
    marginal_h = w.jhvnlcdf(h, np.inf, Hm0, Tp, gam)
    marginal_v = w.jhvnlcdf(np.inf, v, Hm0, Tp, gam)
    assert_allclose(upper, 1-marginal_h-marginal_v+lower, atol=2e-8)
    eps = .001
    mixed = sum(sign*w.jhvnlcdf(h+dh*eps, v+dv*eps, Hm0, Tp, gam, tol=1e-10)
                for dh, dv, sign in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)))/(4*eps**2)
    assert_allclose(mixed, w.jhvnlpdf(h, v, Hm0, Tp, gam), rtol=5e-6)
    assert_allclose(w.jhvnlcdf(h, -1, Hm0, Tp, gam, tail=1), 1-marginal_h)
    assert_allclose(w.jhvnlcdf(np.inf, np.inf, Hm0, Tp, gam), 1)
    assert_allclose(w.jhvnlpdf([1, np.inf], [np.inf, 1], Hm0, Tp, gam), [0, 0])
    assert np.isnan(w.jhvnlpdf(np.nan, 1, Hm0, Tp, gam))
    assert w.jhvnlpdf(1, -np.inf, Hm0, Tp, gam, condon=3) == 0
    with pytest.raises(ValueError, match='sqrt'):
        w.jhvnlpdf(1, 1, 6, 8, 3)
    with pytest.raises(ValueError, match='Peakedness'):
        w.jhvnlpdf(1, 1, 6, 10, 8)
    with pytest.raises(ValueError, match='extrapolation'):
        w.jhvnlpdf(10, 1, 6, 10, 1, True)
