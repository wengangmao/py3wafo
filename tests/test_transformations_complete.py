"""Transformation formulas, normalization equations and source-estimator cases."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.interpolate import make_interp_spline
from scipy.stats import norm
from wafo import trgauss as tr


TESTED_FUNCTIONS = ['createpdf','trmak','trunmak','trangood','tranproc',
    'hermitetr','ochitr','troptset','cdf2tr','lc2tr','dat2tr','dat2gaus','gaus2dat']


def test_transformation_and_pdf_constructors():
    pdf = tr.createpdf(3)
    assert len(pdf.x) == len(pdf.labx) == 3 and pdf.f.size == 0
    assert tr.trmak().tr.shape == (0, 2)
    x = np.linspace(-2, 2, 11)
    table = np.column_stack((x, 2*x+1))
    obj = tr.trmak(table, 1, 2, 'test')
    value, mean, sd, form, note, stamp = tr.trunmak(obj)
    assert_array_equal(value, table)
    assert (mean, sd, form, note) == (1, 2, 'table', 'test')
    assert stamp and tr.trmak(obj).date == stamp
    spline = make_interp_spline(x, 2*x+1)
    assert tr.trmak(spline, 0, 1).form == 'pp'
    assert_allclose(tr.tranproc(x, obj), 2*x+1)
    with pytest.raises(ValueError):
        tr.createpdf(0)
    with pytest.raises(ValueError):
        tr.trunmak({'tr': table})


def test_uniform_resampling_and_linear_extrapolation():
    f = [[2, 5], [0, 1], [1, 3]]
    good = tr.trangood(f, 5, -1, 3, 9)
    assert_allclose(np.diff(good[:, 0]), .5)
    assert_allclose(good[:, 1], 2*good[:, 0]+1)
    assert good[0, 0] <= -1 and good[-1, 0] >= 3
    assert_allclose(tr.tranproc([-2, 4], f), [-3, 9])
    with pytest.raises(ValueError):
        tr.tranproc([-2], f, extrapolate=False)
    with pytest.raises(ValueError):
        tr.trangood([[0, 0], [0, 1]])
    with pytest.raises(ValueError):
        tr.trangood(f, nmax=np.nan)


def test_chain_rule_through_fourth_order_matches_t_power_eight():
    # x(t)=t², g(x)=x⁴, so y(t)=t⁸. This checks all chain-rule cross terms.
    u = np.linspace(-2, 2, 51)
    f = np.column_stack((u, u**4))
    t = np.linspace(.1, 1, 21)
    data = np.column_stack((t*t, 2*t, np.full(len(t), 2), np.zeros((len(t), 2))))
    transformed = tr.tranproc(data, f)
    expected = np.column_stack((t**8, 8*t**7, 56*t**6, 336*t**5, 1680*t**4))
    assert_allclose(transformed, expected, atol=1e-8, rtol=1e-10)
    linear = np.column_stack((u, 2*u+1))
    outside = tr.tranproc([[3, 1, 2, 3, 4]], linear)
    assert_allclose(outside, [[7, 2, 4, 6, 8]])
    spline = make_interp_spline(u, u**4, k=5)
    assert_allclose(tr.tranproc(data, spline), expected, atol=1e-8)
    with pytest.raises(ValueError):
        tr.tranproc(np.ones((2, 6)), f)


def test_gaussian_transform_models_and_data_round_trip():
    x = np.linspace(-6, 10, 501)
    for model, data in [(tr.hermitetr, [2, 0, 3, 2]), (tr.ochitr, [2, 0, 2])]:
        g, departure = model(x, data, full_output=True)
        assert_allclose(g[:, 1], (x-2)/2, atol=1e-14)
        assert_allclose(departure, 0, atol=1e-25)
        signal = np.column_stack((np.arange(5), [-2, 0, 2, 4, 6]))
        assert_allclose(tr.gaus2dat(tr.dat2gaus(signal, g), g), signal, atol=1e-13)


@pytest.mark.parametrize('definition', [1, 2])
def test_hermite_inverse_matches_source_polynomial(definition):
    sigma, skew, kurt, mean = 2, .2, 3.5, 1
    excess = kurt-3
    if definition == 2:
        c4 = (np.sqrt(1+1.5*excess)-1)/18
        c3 = skew/(6*(1+6*c4))
    else:
        c3 = skew/6*(1-.015*abs(skew)+.3*skew**2)/(1+.2*excess)
        c4 = .1*(np.cbrt(1+1.25*excess)-1)*(1-1.43*skew**2/excess)**(1-.1*kurt**.8)
    K = 1/np.sqrt(1+2*c3*c3+6*c4*c4)
    y = np.linspace(-3, 3, 101)
    x = mean+sigma*K*(y+c3*(y*y-1)+c4*(y**3-3*y))
    g, departure = tr.hermitetr(x, [sigma, skew, kurt, mean], definition, full_output=True)
    assert_allclose(g[:, 1], y, atol=1e-12)
    assert departure > 0
    hard = tr.hermitetr(y, [1, .1, 2.5, 0])
    assert_allclose(hard[:, 1], y-.1/6*(y*y-1)-(-.5/24)*(y**3-3*y))
    with pytest.raises(ValueError):
        tr.hermitetr(data=[1, 2, 3.1, 0])


def test_ochi_transform_uses_stated_normalization_equations():
    sigma, skew, mean = 2, .4, 1
    x = np.linspace(-5, 7, 301)
    g, departure = tr.ochitr(x, [sigma, skew, mean], full_output=True)
    # Independently solve the source's v relation by a dense bracket estimate.
    from scipy.optimize import brentq
    v = brentq(lambda v: np.sqrt(2*(v-1))*(4-v)-skew, 1, 2)
    a = np.sqrt((v-1)/(2*v*v))
    ma2 = (-1+np.sqrt(1-4*a*a*v))/(2*a)
    assert_allclose(a*(v+ma2**2)+ma2, 0, atol=1e-13)
    assert_allclose(v-2*a*a*v*v, 1, atol=1e-13)
    assert_allclose(2*a*v*v*(3-8*a*a*v), skew, atol=1e-12)
    z = (x-mean)/sigma
    gamma = np.where(z >= 0, 1.28*a, 3*a)
    expected = (-np.expm1(-gamma*z)/gamma-ma2)/np.sqrt(v)
    assert_allclose(g[:, 1], expected, atol=1e-13)
    assert np.all(np.diff(g[:, 1]) > 0) and departure > 0
    with pytest.raises(ValueError):
        tr.ochitr(data=[1, 3, 0])


def test_option_defaults_cdf_and_crossing_gaussian_identity():
    options = tr.troptset('cdf2tr', 'gs', 1, {'csm': 1}, ne=0, param=[-3, 3, 301])
    assert options['gsm'] == options['csm'] == 1 and options['plotflag'] == 'off'
    x = np.linspace(-6, 6, 1001)
    g, test, empirical = tr.cdf2tr(np.column_stack((2+3*x, norm.cdf(x))), 2, 3, options, full_output=True)
    assert_allclose(g[:, 1], (g[:, 0]-2)/3, atol=1e-8)
    assert test < 1e-8 and empirical.shape[1] == 2
    g2, test2, raw = tr.lc2tr(np.column_stack((2+3*x, 5*np.exp(-x*x/2))), 2, 3, options, full_output=True)
    assert_allclose(g2[:, 1], (g2[:, 0]-2)/3, atol=1e-8)
    assert test2 < 1e-8 and raw.shape[1] == 2
    assert tr.cdf2tr('defaults')['ne'] == 7
    assert tr.lc2tr('defaults')['csm'] == .95
    with pytest.raises(ValueError):
        tr.troptset(gsm=2)
    with pytest.raises(ValueError):
        tr.troptset(multip=2)
    with pytest.raises(ValueError):
        tr.cdf2tr(np.empty((0, 2)))
    with pytest.raises(ValueError):
        tr.cdf2tr([[0, 0], [1, 1]], options={'ne': 0})


def test_data_estimators_linear_marginal_and_separate_columns():
    scores = norm.ppf((np.arange(2000)+.5)/2000)
    values = np.exp(.3*scores)
    options = dict(ne=0, gsm=1, param=[-2, 2, 101])
    data = np.column_stack((np.arange(len(scores)), scores*2+3, scores*4-1))
    g, test, cmax, irr, raw = tr.dat2tr(data, 'linear', dict(options, multip=1), full_output=True)
    assert g.shape == (101, 4) and test.shape == cmax.shape == irr.shape == (2,)
    assert_allclose(g[:, 1], np.linspace(-2, 2, 101))
    assert_allclose(g[:, 3], g[:, 1])
    marginal = tr.dat2tr(values, 'mnonlinear', dict(options, param=[-.5, .5, 101]))
    assert_allclose(marginal[:, 1], np.log(marginal[:, 0])/.3, atol=2e-5)
    empirical = tr.dat2tr(values, 'mempirical', dict(options, param=[-.5, .5, 101]))
    assert_allclose(empirical, marginal)
    symmetric = np.sin(np.arange(10000)*.1)
    hermite = tr.dat2tr(symmetric, 'hermite', dict(options, param=[-1, 1, 101]))
    ochi = tr.dat2tr(symmetric, 'ochitr', dict(options, param=[-1, 1, 101]))
    assert np.all(np.diff(hermite[:, 1]) > 0) and np.all(np.diff(ochi[:, 1]) > 0)
    assert tr.dat2tr('defaults')['multip'] == 0


def test_crossing_data_estimator_and_explicit_rejections():
    # Many smooth stationary Gaussian samples give sufficient empirical levels.
    rng = np.random.default_rng(22)
    noise = rng.normal(size=4000)
    values = np.convolve(noise, np.ones(9)/3, mode='same')
    options = dict(ne=0, csm=.05, gsm=.01, param=[-1, 1, 51], chkder='off')
    for definition in ('nonlinear', 'empirical'):
        g, test, cmax, irr, empirical = tr.dat2tr(values, definition, options, full_output=True)
        assert g.shape == (51, 2) and np.all(np.isfinite(g))
        assert cmax[0] > 0 and irr[0] > 0 and test[0] >= 0
        assert empirical.shape[1] == 2
    with pytest.raises(NotImplementedError):
        tr.cdf2tr([[0, .2], [1, .8]], options={'ne': 0, 'plotflag': 1})
    with pytest.raises(ValueError):
        tr.dat2tr(np.ones(30), 'linear')
