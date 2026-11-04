"""Source formulas, analytical signals and conservation checks for onedim."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from wafo import onedim as o
from wafo.misc import gravity


def test_original_positive_endpoint_windows_and_bandwidth():
    assert_array_equal(o.bingham(7, 0), np.ones(7))
    assert_allclose(o.bingham(10, .4), [.5*(1-np.cos(2*np.pi/4.4)),
        .5*(1-np.cos(4*np.pi/4.4)), 1, 1, 1, 1, 1, 1,
        .5*(1-np.cos(4*np.pi/4.4)), .5*(1-np.cos(2*np.pi/4.4))])
    for n in (1, 2, 5, 10):
        assert_allclose(o.hanning(n), np.sin(np.pi*np.arange(1, n+1)/(n+1))**2)
        p, bw = o.parzen(n, 20, full_output=True)
        assert len(p) == n
        assert_allclose(p, p[::-1], atol=1e-15)
        assert np.all((p > 0) & (p <= 1))
        assert bw == 1.33*20/(n+1)
    assert_allclose(o.parzen(3), [.25, 1, .25])
    assert len(o.hanning(0)) == 0
    with pytest.raises(ValueError):
        o.bingham(8, 2)
    with pytest.raises(ValueError):
        o.parzen(2.5)


def test_covariance_spectrum_matches_analytical_cosine_series_and_variance():
    # For x=[1,-1,1,-1], the biased linear covariances are [1,-.75,.5,-.25].
    # With L=3 the positive half of the five-point Parzen window is [1,5/9,2/27].
    x = np.array([1., -1., 1., -1.])
    s = o.dat2spec2(x, 3, dT=.2)
    lag0, lag1, lag2 = (1+1e-12), (-.75+1e-12)*5/9, (.5+1e-12)*2/27
    expected = (lag0+2*lag1*np.cos(s.w*.2)+2*lag2*np.cos(2*s.w*.2))*.2/np.pi
    assert_allclose(s.S, expected, rtol=2e-14, atol=1e-15)
    assert_allclose(np.trapezoid(s.S, s.w), np.var(x)+1e-12, rtol=2e-14)
    assert s.Bw == 2*np.pi*1.33/(3*.2)
    assert s.dof == int(np.floor(3.71*4/3))
    assert s.type == 'freq' and not s.negative_estimates


def test_covariance_spectrum_multicolumn_transform_missing_and_defaults():
    rng = np.random.default_rng(40)
    y = rng.normal(size=400)
    t = np.arange(len(y))*.25
    x = np.column_stack((t, y, 2*y))
    s = o.dat2spec2(x, 30)
    base = o.dat2spec2(y, 30, dT=.25)
    # Independent columns are averaged, and covariance scales quadratically.
    assert_allclose(s.S, 2.5*base.S, atol=2e-12)
    transform = np.column_stack(([-20, 20], [-10, 10]))
    transformed = o.dat2spec2(x, 30, transform)
    assert_allclose(transformed.S, .25*s.S, atol=2e-12)
    from wafo.trgauss import tranproc
    grid = np.linspace(-10, 10, 101)
    nonlinear = np.column_stack((grid, grid+.1*grid**3))
    transformed = o.dat2spec2(x, 30, nonlinear)
    separate = [o.dat2spec2(tranproc(v, nonlinear), 30, dT=.25).S for v in (y, 2*y)]
    assert_allclose(transformed.S, np.mean(separate, axis=0), rtol=2e-14)
    missing = y.copy(); missing[::7] = np.nan
    censored = o.dat2spec2(missing, 20)
    assert_allclose(np.trapezoid(censored.S, censored.w), np.nanvar(missing)+1e-12, rtol=2e-14)
    # Source default-indexing bugs for n<300 and variance=0 are repaired.
    for n in (2, 3, 10, 400):
        short = o.dat2spec2(np.ones(n))
        assert short.L >= 1 and np.all(np.isfinite(short.S))
    assert o.dat2spec2(y).L <= 398
    with pytest.raises(ValueError, match='uniform'):
        o.dat2spec2([[0, 1], [1, 2], [3, 3]])
    with pytest.raises(NotImplementedError):
        o.dat2spec2(y, chopOffHighFreq=True)


def test_hanning_spectrum_source_window_and_explicit_options():
    y = np.sin(np.arange(120)*.5)
    # This highly coherent signal can make the truncated Hann estimate negative.
    with pytest.warns(RuntimeWarning, match='Negative'):
        s = o.dat2spec2(y, 20, wdef='hanning')
    assert s.negative_estimates and np.all(s.S >= 0)
    assert s.Bw == 2*np.pi/20
    with pytest.warns(RuntimeWarning):
        assert_allclose(o.dat2spec2(y, 20, wdef=2).S, s.S)
    with pytest.raises(ValueError):
        o.dat2spec2(y, 121)
    with pytest.raises(ValueError):
        o.dat2spec2(y, wdef='blackman')


@pytest.mark.parametrize('definition,expected', [
    (1, [0, 1, 1, 0]), (2, [0, 1, 3, 1]),
    (3, [1, 3, 3, 1]), (4, [1, 3, 1, 1]),
])
def test_minmax_crossing_endpoint_definitions_and_alpha(definition, expected):
    pairs = [[0, 2], [1, 2], [1, 3], [8, 7]]
    lc, alpha = o.mm2lc(pairs, definition, full_output=True)
    assert_array_equal(lc[:, 0], [0, 1, 2, 3])
    assert_array_equal(lc[:, 1], expected)
    assert alpha == max(expected)/3
    assert_allclose(lc, o.mm2lc(pairs, definition, sa=12))
    with pytest.raises(ValueError):
        o.mm2lc([[2, 1]])


def test_minmax_ties_and_zero_span_cycles_obey_endpoint_rules():
    rng = np.random.default_rng(3)
    pairs = np.sort(rng.integers(-3, 4, size=(50, 2)), axis=1)
    for definition in (1, 2, 3, 4):
        lc = o.mm2lc(pairs, definition)
        expected = []
        for level in lc[:, 0]:
            left = pairs[:, 0] <= level if definition in (3, 4) else pairs[:, 0] < level
            right = pairs[:, 1] >= level if definition in (2, 3) else pairs[:, 1] > level
            expected.append(np.count_nonzero(left & right))
        if definition == 4:
            expected[-1] = expected[-2]
        assert_array_equal(lc[:, 1], expected)
    assert_array_equal(o.mm2lc([[2, 2]], 1), [[2, 0]])
    assert_array_equal(o.mm2lc([[2, 2]], 3), [[2, 1]])


@pytest.mark.parametrize('method', [0, 1, -1, 2, -2, 3, -3])
def test_steepness_methods_on_analytical_sinusoid(method):
    # Phase prevents endpoint zero ambiguity; 1000 samples per period include
    # crest samples exactly and make crossing interpolation error negligible.
    t = (np.arange(8001)+.25)*.004
    amplitude, period = 2., 4.
    x = np.column_stack((t, amplitude*np.sin(2*np.pi*t/period)))
    s, h, ac, at, front, back, cross, yn = o.dat2steep(x, method=method, full_output=True)
    assert len(s) >= 5 and len(at) == len(ac)+1
    assert_allclose(h, 2*amplitude, atol=1e-5)
    assert_allclose(ac, amplitude, atol=4e-6)
    assert_allclose(at, amplitude, atol=4e-6)
    assert_allclose(front, period/4, atol=.0011)
    assert_allclose(back, period/4, atol=.0011)
    if abs(method) <= 1:
        expected = 4*amplitude/period
        assert_allclose(s, expected, rtol=.0011)
    elif abs(method) == 2:
        expected = 8*np.pi*amplitude/period**2/gravity()
        assert_allclose(s, expected, rtol=.0011)
    else:
        expected = 4*np.pi*amplitude/period**2/gravity()
        assert_allclose(s, expected, rtol=3e-6)
    assert_array_equal(cross, o.dat2crossind(x, 0, 'tw'))
    assert_array_equal(yn, x)


def test_asymmetric_front_back_speed_selection_and_multicolumn_interpolation():
    # Piecewise linear waves: trough at0, upcross1.5, crest2,
    # downcross3, following trough4.  Repeated copies avoid partial edges.
    t = np.arange(25.)
    y = np.resize([-1., -1., 1., 0.], len(t))
    x = np.column_stack((t, y, 2*y))
    front, hd = o.dat2steep(x, method=1)
    back, hu = o.dat2steep(x, method=-1)
    selected, height = o.dat2steep(x, method=0)
    assert_allclose(front, np.r_[np.full(4, 2.), np.full(4, 4.)])
    assert_allclose(back, np.r_[np.full(4, 1.), np.full(4, 2.)])
    assert_allclose(selected, front)
    assert_allclose(height, hd)
    assert_allclose(hu, hd)
    result = o.dat2steep(x, rate=3, full_output=True)
    assert result[-1].shape == (73, 3)
    assert_allclose(result[-1][::3], x, atol=1e-14)
    packed = result[-2]
    first_count = packed[0]
    assert first_count > 0 and len(packed) == first_count+packed[first_count+1]+2
    assert len(o.dat2steep(np.ones(10))[0]) == 0
    with pytest.raises(ValueError):
        o.dat2steep(x, method=4)
