"""Inverse reconstruction and independently generated switching-model fits."""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from wafo import markov as m


def models(n=7):
    param = [-1, 1, n]
    return [m.mktestmat(param, [-.35, -.1], .3, .8, both=True),
            m.mktestmat(param, [.1, .35], .23, 1.3, both=True)]


def test_arfm2mctp_exact_two_state_and_scale_independence():
    inverse, info = m.arfm2mctp([[0, 17], [0, 0]], full_output=True)
    assert_allclose(inverse, [[0, 1], [1, 0]])
    assert info['input_mass'] == 17 and info['forward_error'] == 0
    assert info['elapsed_seconds'] >= 0


@pytest.mark.parametrize('reversible', [True, False])
@pytest.mark.parametrize('n', [3, 5, 7, 9])
def test_arfm2mctp_recovers_full_from_to_probabilities(reversible, n):
    rng = np.random.default_rng(623)
    up = np.triu(rng.uniform(.2, 2, (n, n)), 1)
    down = up.T if reversible else np.tril(rng.uniform(.2, 2, (n, n)), -1)
    cycles, forward, _ = m.mctp2arfm((up, down), full_output=True)
    reconstructed, info = m.arfm2mctp(31*cycles, full_output=True)
    assert_allclose(reconstructed, forward[0]+forward[1], atol=2e-8, rtol=2e-7)
    assert_allclose(reconstructed.sum(), 2)
    Q, Qh = info['transitions']
    assert_allclose(Q[:-1].sum(axis=1), 1)
    assert_allclose(Qh[1:].sum(axis=1), 1)
    assert info['forward_error'] < 1e-8


def test_arfm2mctp_rejects_incompatible_and_unidentifiable_matrices():
    with pytest.raises(ValueError, match='zero diagonal'):
        m.arfm2mctp(np.eye(3))
    with pytest.raises(ValueError, match='unidentifiable'):
        m.arfm2mctp([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    cycles = m.mctp2arfm(models()[0])
    cycles[1, 0] = .1
    with pytest.raises(ValueError, match='outer-cycle'):
        m.arfm2mctp(cycles)


@pytest.mark.parametrize('method', ['ML', 'chi2', 'HD', 'KL'])
def test_f_smctp_matches_each_original_matrix_distance(method):
    F = models()
    trueP = np.array([[.83, .17], [.29, .71]])
    otherP = np.array([[.75, .25], [.35, .65]])
    observed = 20000*m.smctp2rfm(trueP, F)
    value, predicted, P, pairs = m.f_smctp(m.tr_p2x(otherP, 1), observed, 'P', method,
                                          dict(F=F), init=dict(P=otherP), full_output=True)
    funcs = dict(ML=lambda obs, fit: -m.loglcmat(obs, fit), chi2=m.chi2cmat, HD=m.hdcmat, KL=m.klcmat)
    assert_allclose(value, funcs[method](observed, m.smctp2rfm(otherP, F)))
    assert_allclose(P, otherP)
    assert_allclose(predicted, m.smctp2rfm(P, pairs))


def test_estsmctp_transition_parameter_recovery_from_exact_intensities():
    F = models()
    trueP = np.array([[.83, .17], [.29, .71]])
    initial = np.array([[.7, .3], [.15, .85]])
    observed = 20000*m.smctp2rfm(trueP, F)
    fitted, estimate, options = m.estsmctp(observed, 'P', 'ML', dict(F=F), init=dict(P=initial),
                                          OPTIONS=dict(TolX=1e-7, TolFun=1e-8), full_output=True)
    assert_allclose(fitted['P'], trueP, atol=2e-5)
    assert_allclose(estimate['P'], trueP, atol=2e-5)
    assert_allclose(fitted['rfm'], observed/observed.sum(), atol=2e-7)
    assert estimate['objective'] < estimate['initial_objective']
    assert estimate['success'] and estimate['nfev'] > 1
    assert options['xatol'] == 1e-7


def test_estsmctp_location_scale_recovery_and_pj_wat_conversion():
    param = [-1, 1, 9]
    base = m.mktestmat(param, [-.2, .2], .3, 1.2)
    location, scale = .1, .85
    wat = np.flipud(base.T)
    transformed = np.flipud(m.scalemat(param, wat, location, scale, param)).T
    observed = 50000*m.mctp2rfm((transformed, None))
    known = dict(P=[[1]], F=[(base, None)], param=param)
    fitted, estimate = m.estsmctp(observed, 'MeanStd', 'chi2', known,
                                  init=dict(MeanStd=[[0., 1.]]),
                                  OPTIONS=dict(xatol=1e-8, fatol=1e-9))
    assert_allclose(estimate['MeanStd'], [[location, scale]], atol=1e-5)
    assert estimate['objective'] < 1e-8
    assert_allclose(fitted['F'][0][0], transformed, rtol=1e-4, atol=1e-6)
    # The combined family must decode the same Fortran-order mean/log-scale
    # vector while preserving a separately supplied downward kernel.
    templates = models()
    P = [[.8, .2], [.3, .7]]
    meanstd = np.array([[0., 1.], [0., 1.]])
    X = np.r_[m.tr_p2x(P, 1), meanstd[:, 0], np.log(meanstd[:, 1])]
    expected = m.smctp2rfm(P, templates)
    distance, _, _, pairs = m.f_smctp(X, 1000*expected, 'P,MeanStd', 'chi2',
                                      dict(F=templates, param=[-1, 1, 7]),
                                      init=dict(P=P), full_output=True)
    assert_allclose(distance, 0, atol=1e-12)
    assert_allclose(pairs[0][1], templates[0][1])


def test_estsmctp_callable_model_recovers_location_and_builtin_calc_dispatch():
    param = [-1, 1, 9]
    def calculate(param, model):
        center = model['center']
        return m.mktestmat(param, [center-.2, center+.2], .3, 1, both=True)
    known = dict(P=[[1]], param=param, Ffun=calculate,
                 trModel2X=lambda model: [model['center']],
                 trX2Model=lambda X, known: dict(center=float(X[0])))
    observed = 10000*m.mctp2rfm(calculate(param, dict(center=.12)))
    fitted, estimate = m.estsmctp(observed, 'CalcF', 'chi2', known,
                                  init=dict(M=[dict(center=-.05)]))
    assert_allclose(estimate['M'][0]['center'], .12, atol=2e-6)
    assert estimate['objective'] < 1e-7
    builtin_models = [dict(x0=[-.35, -.1], s=.3, lam=.8), dict(x0=[.1, .35], s=.23, lam=1.3)]
    P = [[.8, .2], [.3, .7]]
    X = np.r_[m.tr_p2x(P, 1), *(m.tr_m2x(model) for model in builtin_models)]
    expected = m.smctp2rfm(P, models())
    value, calculated, _, _ = m.f_smctp(X, 1000*expected, 'P,CalcF', 'chi2',
                                       dict(param=[-1, 1, 7], nM=[4, 4], Ffun='f_funm', trX2Model='tr_x2m'),
                                       init=dict(P=P), full_output=True)
    assert_allclose(value, 0, atol=1e-12)
    assert_allclose(calculated, expected)


def test_f_smctp_side_regime_marks_and_zero_subdiagonal_band():
    P = [[.8, .2], [.3, .7]]
    F = models()
    _, marks, _ = m.smctp2arfm(P, F, SideInfo=1, full_output=True)
    X = m.tr_p2x(P, 1)
    value, predicted, _, _ = m.f_smctp(X, 1000*marks, 'P', 'chi2', dict(F=F, SideInfo=11),
                                      init=dict(P=P), full_output=True)
    assert predicted.shape == (14, 14)
    assert_allclose(predicted.sum(), marks.sum())
    assert_allclose(value, 0, atol=1e-12)
    banded = [(np.triu(up, 2), np.tril(down, -2)) for up, down in F]
    expected = m.smctp2rfm(P, banded)
    value = m.f_smctp(X, 1000*expected, 'P', 'chi2', dict(F=F, NOsubzero=1), init=dict(P=P))
    assert_allclose(value, 0, atol=1e-12)


def test_estsmctp_reports_nonconvergence_and_source_undefined_families():
    F, P = models(), [[.8, .2], [.3, .7]]
    observed = 1000*m.smctp2rfm(P, F)
    with pytest.raises(RuntimeError, match='did not converge'):
        m.estsmctp(observed, 'P', 'chi2', dict(F=F), init=dict(P=[[.6, .4], [.1, .9]]),
                    OPTIONS=dict(MaxIter=1))
    with pytest.raises(NotImplementedError, match='simulation'):
        m.estsmctp(observed, 'SimF', 'ML', dict(P=P), init={})
    with pytest.raises(NotImplementedError, match='SideInfo'):
        m.f_smctp(m.tr_p2x(P, 1), observed, 'P', 'ML', dict(F=F, SideInfo=21), init=dict(P=P))
    with pytest.raises(ValueError, match='nonnegative'):
        m.f_smctp(np.full(6, 10), np.ones((7, 7)), 'P', 'ML', dict(F=F),
                    init=dict(P=np.eye(3)))
    with pytest.raises(NotImplementedError, match='constraints'):
        m.estsmctp(observed, 'P', 'ML', dict(F=F), whatKnown=dict(P=True), init=dict(P=P))


def test_estsmctp_single_regime_has_no_free_transition_parameters():
    F = [models()[0]]
    observed = 1000*m.smctp2rfm([[1]], F)
    fitted, estimate = m.estsmctp(observed, 'P', 'chi2', dict(F=F),
                                  whatKnown=np.array([]), init=dict(P=[[1]]))
    assert_allclose(fitted['P'], [[1]])
    assert estimate['nit'] == 0 and estimate['nfev'] == 1
    assert estimate['objective'] < 1e-12
