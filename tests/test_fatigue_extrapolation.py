"""Analytic distributions and source-bin identities for fatigue extrapolation."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from wafo import cycles as c


@pytest.fixture
def plots():
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    yield plt
    plt.close('all')


def survival(x, k, s):
    x = np.asarray(x, dtype=float)
    if k == 0:
        return np.exp(-x/s)
    base = np.maximum(1-k*x/s, 0)
    return base**(1/k)


def geometric_matrix(n=121, scale=4., count=1e7):
    """Independent endpoints with exact geometric strict-crossing tails."""
    middle = n//2
    q = np.exp(-1/scale)
    tail = np.r_[(1-q)*q**np.arange(middle-1), q**(middle-1)]
    maximum = np.r_[np.zeros(middle+1), tail]
    minimum = maximum[::-1]
    return count*np.outer(minimum, maximum)


def render(fig, tmp_path, name):
    path = tmp_path/(name+'.png')
    fig.savefig(path)
    assert path.stat().st_size > 1000


def test_extralc_exponential_parameters_threshold_offset_and_reflection(plots, tmp_path):
    levels = np.linspace(-12, 12, 12001)
    scale, count = .7, 100000
    lc = np.column_stack((levels, count*np.exp(-abs(levels)/scale)))
    tails, estimates = c.extralc(lc, [-1, 1], 'exp,ml', 1, full_output=True)
    assert_allclose(estimates['High']['s'], scale, atol=.0011)
    assert_allclose(estimates['Low']['s'], estimates['High']['s'], rtol=1e-11)
    expected_threshold = count*np.exp(-1/scale)+1
    assert_allclose(tails['High'][0], [1, expected_threshold])
    x = tails['High'][:, 0]-1
    assert_allclose(tails['High'][:, 1], expected_threshold*np.exp(-x/estimates['High']['s']))
    assert_allclose(tails['Low'][:, 0], -tails['High'][::-1, 0])
    assert_allclose(tails['Low'][:, 1], tails['High'][::-1, 1], rtol=1e-11)
    assert_allclose(estimates['High']['cov'], estimates['High']['s']**2/estimates['High']['n'])
    render(estimates['figures'][0], tmp_path, 'continuous-exponential')


@pytest.mark.parametrize('shape', [.2, -.1])
def test_extralc_gpd_parameter_recovery_and_finite_endpoint(shape):
    levels = np.linspace(-30, 30, 24001)
    distance = np.maximum(abs(levels)-1, 0)
    counts = np.where(abs(levels) <= 1, 1e5*(1-.1*abs(levels)), 9e4*survival(distance, shape, .5))
    tails, estimates = c.extralc(np.column_stack((levels, counts)), [-1, 1],
                                'gpd,ml', tail_grid=np.linspace(0, 10, 501), full_output=True)
    for name in ('High', 'Low'):
        assert_allclose([estimates[name]['k'], estimates[name]['s']], [shape, .5], atol=.012)
        assert np.all(np.isfinite(tails[name]))
        assert np.all(tails[name][:, 2] <= tails[name][:, 1])
        assert np.all(tails[name][:, 1] <= tails[name][:, 3])
    if shape > 0:
        high = estimates['High']['UpperLimit']
        assert_allclose(high, 3.5, atol=.1)
        assert_allclose(estimates['Low']['LowerLimit'], -high, atol=1e-8)
        assert np.all(tails['High'][tails['High'][:, 0] >= high, 1] == 0)


@pytest.mark.parametrize('method', ['gpd,mom', 'gpd,pwm'])
def test_extralc_gpd_moment_estimators_recover_bounded_tail(method):
    levels = np.linspace(-6, 6, 6001)
    distance = np.maximum(abs(levels)-1, 0)
    counts = np.where(abs(levels) <= 1, 1e6*(1-.1*abs(levels)), 9e5*survival(distance, .2, .8))
    _, estimate = c.extralc(np.column_stack((levels, counts)), [-1, 1], method, full_output=True)
    assert_allclose([estimate['High']['k'], estimate['High']['s']], [.2, .8], atol=.007)


def test_extralc_rayleigh_conditional_scale_and_pickands():
    levels = np.linspace(-8, 8, 8001)
    scale = 1.5
    lc = np.column_stack((levels, 1e5*np.exp(-levels**2/scale**2)))
    result, estimates = c.extralc(lc, [-1, 1], 'ray,ml', full_output=True)
    assert_allclose(estimates['High']['s'], scale, atol=.002)
    x = result['High'][:, 0]-1
    assert_allclose(result['High'][:, 1]/estimates['High']['lcu'],
                    np.exp(-((x+1)**2-1)/estimates['High']['s']**2))
    integer_lc = lc.copy()
    integer_lc[:, 1] = np.floor(integer_lc[:, 1])
    result, estimates = c.extralc(integer_lc, [-1, 1], 'gpd,pkd', full_output=True)
    assert estimates['High']['s'] > 0 and np.all(np.isfinite(result['High']))


@pytest.mark.parametrize('method', ['exp,ml', 'exp,mld', 'exp,ls1', 'exp,ls2',
                                    'exp,ls3', 'exp,ls4', 'exp,wls1', 'exp,wls2',
                                    'exp,wls3', 'exp,wls4', 'ray,ml', 'gpd,ml',
                                    'gpd,mom', 'gpd,pwm'])
def test_cmat2extralc_source_bin_estimates_scales_and_reflection(method):
    F = geometric_matrix()
    result, estimate, internal, mse = c.cmat2extralc([-3, 3, 121], F, [-1, 1], method, full_output=True)
    assert_array_equal(internal['threshold_indices'], [39, 81])
    assert_allclose(internal['F1'], F)
    assert np.all(np.isfinite(result['lc']))
    assert_allclose(result['Low'][:, 1], result['High'][::-1, 1], rtol=1e-11, atol=1e-9)
    assert result['lc'][0, 1] == result['lc'][-1, 1] == 0
    assert np.all(np.diff(result['High'][:, 1]) <= 1e-8)
    assert_allclose(estimate['High']['s'], .05*estimate['D']['High']['s'])
    assert_allclose(estimate['D']['Low']['s'], estimate['D']['High']['s'], rtol=1e-11)
    assert mse['High'] >= 0 and mse['Low'] >= 0
    if method == 'exp,mld':
        assert_allclose(estimate['D']['High']['s'], 4, atol=.002)
    if method.startswith(('exp,ls', 'exp,wls')):
        assert_allclose(estimate['D']['High']['s'], 4, atol=1e-8)
        assert_allclose(mse['High'], 0, atol=1e-20)
    if method == 'exp,ml':
        observations = estimate['D']['High']['observations']
        assert_allclose(estimate['D']['High']['s'], np.average(observations[:, 0], weights=observations[:, 1]))


def test_cmat2extralc_conditioning_removes_isolated_tail_cycles_and_diagnostics(plots, tmp_path):
    F = geometric_matrix()
    F[100, 110] = 200
    F[10, 20] = 100
    result, estimate, internal, mse = c.cmat2extralc([0, 120, 121], F, [40, 80],
                                                  'exp,mld', 1, full_output=True)
    assert internal['F1'][100, 110] == internal['F1'][10, 20] == 0
    assert internal['F1'].sum() == F.sum()-300
    assert result['D']['lc'].shape == (121, 2)
    render(estimate['figures'][0], tmp_path, 'matrix-exponential')


def test_rfmextrapolate_manual_combination_kernel_mass_limits_and_reflection(plots, tmp_path):
    F = geometric_matrix(81, 5, 1e6)
    config = dict(method='exp,mld', u_lev=[25, 55], h=.8,
                  Lim=dict(range=30, min=30, max=50), param=(-2, 2, 81))
    final, parameters, extreme, smooth, before_limits = c.rfmextrapolate(F, config, 2, full_output=True)
    assert_allclose(smooth.sum(), F.sum(), rtol=1e-12)
    I, J = np.indices(F.shape)
    use = (J-I >= 30) & (I <= 30) & (J >= 50)
    assert_allclose(before_limits, np.where(use, extreme, smooth))
    expected = before_limits.copy()
    expected[:parameters['imin']] = 0
    expected[:, parameters['jmax']+1:] = 0
    assert_allclose(final, expected)
    assert_allclose(final, final[::-1, ::-1].T, rtol=1e-10, atol=1e-6)
    assert final.sum() > 0 and np.all(np.isfinite(final)) and np.all(final >= 0)
    assert len(parameters['figures']) == 3
    for i, fig in enumerate(parameters['figures']):
        render(fig, tmp_path, f'rainflow-extrapolation-{i}')


def test_rfmextrapolate_auto_thresholds_follow_crossing_fraction_and_no_smoothing():
    F = geometric_matrix(81, 5, 1e6)
    result, parameters, extreme, smooth, initial = c.rfmextrapolate(F, dict(method='exp,mld', h=0), full_output=True)
    counts = c.cmat2lc((0, 80, 81), F)[:, 1]
    active = np.flatnonzero(counts > .05*counts.max())
    assert_array_equal(parameters['u_lev'], active[[0, -1]])
    assert_allclose(smooth, F)
    assert np.all(np.isfinite(result)) and parameters['h'] == 0
    assert parameters['Lim']['range'] >= 0
    default, defaults, _, automatic_smooth, _ = c.rfmextrapolate(F, full_output=True)
    assert np.all(np.isfinite(default)) and default.sum() > 0
    assert defaults['h'] > 0 and defaults['method'] == 'gpd,ml'
    assert_allclose(automatic_smooth.sum(), F.sum(), rtol=1e-12)


def test_tpextrapolate_conservative_rank_blocks_limits_times_and_zero_tails(plots, tmp_path):
    rng = np.random.default_rng(34)
    maxima, minima = 1+rng.exponential(1.2, 60), -1-rng.exponential(.8, 60)
    values = np.column_stack((minima, maxima)).ravel()
    timed = np.column_stack((np.arange(len(values))*.5, values))
    result, parameters, indices, raw = c.tpextrapolate(timed, 3, dict(u_lev=[-1, 1], lim=[-5, 5]),
                                                      2, seed=123, full_output=True)
    assert_allclose(parameters['a_max'], np.mean(maxima-1))
    assert_allclose(parameters['a_min'], np.mean(-1-minima))
    assert parameters['n_min'] == parameters['n_max'] == 60
    assert raw.shape == (360, 2) and np.all(np.diff(raw[:, 0]) > 0)
    for block in raw[:, 1].reshape(3, -1):
        assert np.all(block[1::2] >= maxima)
        assert np.all(block[::2] <= minima)
        assert_array_equal(np.argsort(block[1::2]), np.argsort(maxima))
        assert_array_equal(np.argsort(-block[::2]), np.argsort(-minima))
    assert np.all(result[:, 1] >= -5) and np.all(result[:, 1] <= 5)
    assert np.all(result[indices['max'], -1] > 1)
    assert np.all(result[indices['min'], -1] < -1)
    assert np.all(np.diff(result[:, 0]) > 0)
    for i, fig in enumerate(parameters['figures']):
        render(fig, tmp_path, f'turning-point-extrapolation-{i}')
    empty, estimates, index, raw = c.tpextrapolate(values, 0, dict(u_lev=[-100, 100], method='gpd'),
                                                 full_output=True)
    assert empty.size == raw.size == 0 and estimates['gpd_max'] is None
    assert estimates['a_max'] == estimates['a_min'] == 0
    repeated = c.tpextrapolate(values, 2, dict(u_lev=[-100, 100]), seed=1)
    assert_allclose(repeated, np.tile(values, 2))


def test_tpextrapolate_gpd_inverse_formula_parameter_recovery_and_join_filtering():
    n, shape, scale = 400, .2, .8
    p = (np.arange(n)+.5)/n
    excess = scale/shape*(1-(1-p)**shape)
    values = np.column_stack((-1-excess, 1+excess[::-1])).ravel()
    result, parameters, indices, raw = c.tpextrapolate(values, 1, dict(method='gpd', u_lev=[-1, 1]),
                                                      seed=15, conservative=False, full_output=True)
    assert_allclose(parameters['gpd_max'], [shape, scale], atol=.012)
    rng = np.random.default_rng(15)
    k, s = parameters['gpd_max']
    simulated = s/k*(1-(1-rng.random(n))**k)
    assert_allclose(raw[1::2][np.argsort(excess[::-1])], 1+np.sort(simulated), rtol=1e-12)
    assert_allclose(result, raw)
    # Odd-length blocks share two minima at their join; the source's final
    # zero-threshold filter removes the non-turning point there.
    joined = c.tpextrapolate([0, 3, -1, 2, 0], 3, dict(u_lev=[-10, 10]), seed=2)
    differences = np.diff(joined)
    assert np.all(differences[:-1]*differences[1:] < 0)


def test_tpextrapolate_automatic_thresholds_follow_source_crossing_rule():
    rng = np.random.default_rng(72)
    values = np.column_stack((-1-rng.exponential(1, 100), 1+rng.exponential(1, 100))).ravel()
    empty, parameters, indices, raw = c.tpextrapolate(values, 0, full_output=True)
    crossings = c.tp2lc(values)
    maximum = crossings[:, 1].max()
    selected = crossings[crossings[:, 1] > np.sqrt(maximum), 0]
    assert_allclose(parameters['LCfrac'], 1/np.sqrt(maximum))
    assert_allclose(parameters['u_lev'], selected[[0, -1]])
    assert parameters['n_max'] > 0 and parameters['n_min'] > 0
    assert empty.shape == raw.shape == (0,)


def test_extrapolation_rejects_unidentifiable_or_invalid_inputs():
    with pytest.raises(ValueError, match='threshold'):
        c.extralc([[0, 0], [1, 2], [2, 0]], [0, 2])
    with pytest.raises(ValueError, match='eight'):
        c.rfmextrapolate(np.triu(np.ones((5, 5)), 1))
    with pytest.raises(ValueError, match='alternating'):
        c.tpextrapolate([0, 1, 2], 1, dict(u_lev=[-.5, .5]))
