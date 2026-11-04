"""Numerical identities for the converted course and RFC demo helpers."""
from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import integrate, signal, special

from wafo import wdemos as demos


@pytest.fixture
def plots():
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    yield plt
    plt.close('all')


def render(fig, tmp_path, name):
    path = tmp_path/(name+'.png')
    fig.savefig(path)
    assert path.stat().st_size > 1000


def test_virkler_slices_source_midpoints_and_interpolation_boundaries():
    v = np.array([[1, 0, 10], [2, 20, 30], [5, 80, 90]], dtype=float)
    assert_allclose(demos.alevel(v, 3), [50, 60])
    assert_allclose(demos.alevel(v, 1), [10, 20])
    assert demos.alevel(v, 5).size == 0
    assert_allclose(demos.nlevel(v, 50), [3.5, 3])
    assert_allclose(demos.nlevel(v, 0), [1])
    assert_allclose(demos.nlevel(v, 10), [1.5, 1])
    assert demos.nlevel(v, 100).size == 0
    with pytest.raises(ValueError):
        demos.nlevel(v[[1, 0, 2]], 50)


def test_load2ud_preserves_strict_touching_and_plateau_rules():
    y = np.array([-2, 2, 0, 2, -2, 0, -2, 2])
    up, down, levels = demos.load2ud([-1, 1, 3], np.column_stack((np.arange(len(y)), y)))
    assert_array_equal(up, [3, 2, 3])
    assert_array_equal(down, [2, 1, 2])
    assert_allclose(levels, [-1, 0, 1])
    assert_array_equal(demos.load2ud([0, 10, 1], y)[0], [2])
    assert_array_equal(demos.load2ud([-1, 1, 3], [])[0], [0, 0, 0])


def test_splitload_aligns_transitions_and_allows_missing_regimes():
    x0 = np.column_stack((np.arange(10), np.arange(10)*10))
    tz = np.array([[1.5, 0], [4, 3], [4.1, 9], [8, 100]])
    groups, x, z = demos.splitload(x0, tz)
    assert_array_equal(x[:, 0], np.arange(2, 8))
    assert_array_equal(z, [0, 0, 3, 9, 9, 9])
    assert_array_equal(groups[3], [[4, 40]])
    assert_array_equal(groups[0], x[:2])
    empty, values, regimes = demos.splitload(x0, [[10, 0], [11, 5]])
    assert empty[0].shape == (0, 2) and values.shape == (0, 2) and regimes.size == 0
    with pytest.raises(ValueError):
        demos.splitload(x0, [[1, 0], [1, 1]])


def test_armacov_ar1_ma2_and_source_ar2_equations():
    rho, variance = .7, 2
    expected = variance/(1-rho*rho)*rho**np.arange(21)
    assert_allclose(demos.armacov(1, [1, -rho], variance, 20), expected)
    c = np.array([1, .3, -.2])
    expected = 3*np.r_[np.correlate(c, c, 'full')[2:], np.zeros(8)]
    assert_allclose(demos.armacov(c, 1, 3, 10), expected)
    r = demos.armacov(1, [1, 1, .9], 1, 30)
    assert_allclose(r[0]+r[1]+.9*r[2], 1)
    assert_allclose(r[1]+r[0]+.9*r[1], 0, atol=1e-12)
    assert_allclose(r[2:]+r[1:-1]+.9*r[:-2], 0, atol=1e-12)
    assert_allclose(demos.armacov(2, 4, 3, 0), [.75])
    with pytest.raises(ValueError, match='stationary'):
        demos.armacov(1, [1, -1.01])


def test_armacov_arma_matches_long_impulse_response_and_spectrum():
    A = [1, -2.06, 1.64, -.98, .41]
    C = [1, .05, -.88]
    impulse = signal.lfilter(C, A, np.r_[1., np.zeros(20000)])
    expected = [np.dot(impulse[:len(impulse)-k], impulse[k:]) for k in range(31)]
    assert_allclose(demos.armacov(C, A, 1, 30), expected, rtol=1e-11, atol=1e-12)
    spectrum = demos.armaspec(C, A, 1, 16384)
    assert spectrum.shape == (16384, 2)
    assert_allclose(spectrum[0], [0, (sum(C)/sum(A))**2])
    assert_allclose(spectrum[-1, 0], .5-.5/16384)
    # Complete the half-open grid at Nyquist for trapezoidal integration;
    # twice the positive half density integrates to covariance at lag zero.
    nyquist = (np.dot(C, (-1.)**np.arange(len(C))) /
               np.dot(A, (-1.)**np.arange(len(A))))**2
    integrated = (spectrum[:, 1].sum()-.5*spectrum[0, 1]+.5*nyquist)/len(spectrum)
    assert_allclose(integrated, expected[0], rtol=1e-11)


def test_f_ar_independent_limit_and_nonzero_stationary_mean():
    y = np.array([-1., 0., 2.])
    m, s = 2, .5
    expected = np.exp(-.5*((y-m)/s)**2)/(np.sqrt(2*np.pi)*s) * (special.ndtr(2)-special.ndtr(-2))
    assert_allclose(demos.f_ar(y, 1, 3, 0, m, s), expected)
    mass = integrate.quad(demos.f_ar, -np.inf, np.inf, args=(1.5, 2.5, -.7, 2, .5))[0]
    sx = .5/np.sqrt(1-.7**2)
    assert_allclose(mass, special.ndtr(.5/sx)-special.ndtr(-.5/sx), rtol=1e-10)


def test_discar_joint_mass_conditional_rows_and_stationary_reversibility():
    single, center = demos.discar([-np.inf, np.inf], .2, 3, 1, full_output=True)
    assert_allclose(single, [[1]])
    assert_allclose(center, [3])
    edges = [-np.inf, -1, 0, 1, np.inf]
    mass = np.diff(special.ndtr(edges))
    joint, centers = demos.discar(edges, 0, full_output=True)
    assert_allclose(joint, mass[:, None]*mass, atol=1e-12)
    assert_allclose(centers, [-np.inf, -.5, .5, np.inf])
    conditional = demos.discar(edges, 0, conditional=True)
    assert_allclose(conditional, np.broadcast_to(mass, (4, 4)), atol=1e-12)
    m, rho, s = 2, .6, .5
    edges = np.r_[-np.inf, np.linspace(1, 3, 5), np.inf]
    joint = demos.discar(edges, -rho, m, s)
    pi = np.diff(special.ndtr((edges-m)/(s/np.sqrt(1-rho*rho))))
    assert_allclose(joint.sum(axis=1), pi, atol=1e-11)
    assert_allclose(joint, joint.T, atol=1e-11)
    P = demos.discar(edges, -rho, m, s, conditional=True)
    assert_allclose(P.sum(axis=1), 1, atol=1e-10)
    assert_allclose(pi@P, pi, atol=1e-11)


def test_democc_point_selection_minmax_rainflow_and_plateau_rules(plots, tmp_path):
    values = [0, 5, 1, 3, 0, 4, -1, 2, 0]
    # Arbitrary sample times exercise the MATLAB integer-time assumption fix.
    proc = np.column_stack((10+np.arange(len(values))*.2, values))
    tp = proc.copy()
    k, ax = demos.democc_markmax(proc, tp, 2, full_output=True)
    assert k == 3
    assert demos.democc_markmax(proc, tp, 100, ax=ax) == 7
    render(ax.figure, tmp_path, 'maximum')
    cycles, ax = demos.democc_mmdef(proc, tp, 2, [[-1, 5]], full_output=True)
    assert_allclose(cycles, [[-1, 5], [1, 3]])
    render(ax.figure, tmp_path, 'minmax')
    cycles, ax, details = demos.democc_rfcdef(proc, tp, 3, full_output=True)
    assert_allclose(cycles, [[1, 3]])
    assert details == dict(peak_index=3, left_boundary=1, right_boundary=5,
                           left_trough=2, right_trough=4, trough_index=2)
    render(ax.figure, tmp_path, 'rainflow')
    terminal, ax, info = demos.democc_rfcdef(proc, tp, 7, full_output=True)
    assert_allclose(terminal, [[-1, 2]])
    assert info['right_trough'] is None
    levels = np.array([[0, 0], [1, 2], [2, 2], [3, -1], [4, -1], [5, 0]])
    ax, maxima, minima = demos.democc_tpdef(levels, full_output=True)
    assert_array_equal(maxima, [2])
    assert_array_equal(minima, [4])
    render(ax.figure, tmp_path, 'turning-points')
    axes = demos.democc_plotmat(None, cycles, terminal)
    assert axes[0].get_title() == 'Rainflow cycles'
    assert axes[1].get_title() == 'Min-max cycles'
    render(axes[0].figure, tmp_path, 'cycle-clouds')


def test_rfcdemo1_calculation_matches_custom_preset_and_renders(plots, tmp_path):
    matrix, info = demos.rfcdemo1(1, param=[-8, 8, 9], samples=200, burnin=20,
                                 seed=123, full_output=True)
    assert matrix.shape == (9, 9) and np.all(np.isfinite(matrix)) and np.all(matrix >= 0)
    assert matrix.sum() > 0
    assert info['samples'].shape == (200,) and info['regimes'].shape == (200,)
    assert info['observed'].shape == matrix.shape
    assert len(info['figures']) == 4
    duplicate, custom = demos.rfcdemo1(0, [[.98, .02], [.01, .99]], [[1, -.5], [1, .5]],
                                      [-1, 3], [1, 1], [-8, 8, 9], samples=200,
                                      burnin=20, seed=123, plot=False, full_output=True)
    assert_allclose(duplicate, matrix)
    assert_allclose(custom['samples'], info['samples'])
    assert custom['figures'] == ()
    for i, fig in enumerate(info['figures']):
        render(fig, tmp_path, f'switching-ar1-{i}')
    with pytest.raises(ValueError, match='custom'):
        demos.rfcdemo1(0)


def test_rfcdemo2_rainflow_damage_crossings_and_smoothed_counts(plots, tmp_path):
    from wafo import cycles, damage, markov
    param = [-1, 1, 9]
    matrix, info = demos.rfcdemo2(1, param=param, samples=300, seed=321, full_output=True)
    assert matrix.shape == (9, 9) and np.all(np.isfinite(matrix)) and np.all(matrix >= 0)
    assert matrix.sum() > 0
    assert_allclose(matrix, markov.smctp2rfm(info['regime_transition'], info['transitions']))
    assert_allclose(info['damage'], damage.cmat2dmat(param, matrix, 3))
    assert_allclose(info['smoothed'].sum(), info['observed'].sum())
    assert_allclose(info['crossing'], cycles.cmat2lc(param, info['observed']))
    repeat = demos.rfcdemo2(1, param=param, samples=100, seed=21, plot=False)
    assert_allclose(repeat, matrix)
    shorthand = demos.rfcdemo2(info['regime_transition'], param=param, samples=100,
                               seed=21, plot=False)
    assert_allclose(shorthand, matrix)
    assert len(info['figures']) == 4
    for i, fig in enumerate(info['figures']):
        render(fig, tmp_path, f'switching-turning-points-{i}')
