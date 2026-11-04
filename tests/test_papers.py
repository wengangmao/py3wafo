import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import trapezoid

from wafo import papers, data, trgauss


def _wafo_session():
    return papers.wafoinit(Np=17, NNp=9, Nh=np.linspace(0, 8, 7), kde_inc=64, paramu=(-6, 6, 13))


def test_wafo_paper_figures_render_with_disclosed_source_gaps():
    session = _wafo_session()
    results = papers.wafofig(-2, session)
    assert set(results) == set(range(1, 11))
    assert results[8]['status'] == 'unavailable' and 'never implemented' in results[8]['reason']
    assert results[6].status == 'partial' and 'session.theory[6]' in results[6].limitations[0]
    for number in (1, 2, 3, 4, 5, 7, 9, 10):
        assert results[number].status == 'rendered', number
        assert results[number].figure.axes
    # Figure 3: theoretical Tt density from the Torsethaugen spectrum is a density.
    fTt = results[3].data['fTt']
    assert 0.8 < trapezoid(fTt.f, fTt.x[0]) < 1.05
    assert len(results[3].data['Tt']) > 20
    # Figure 10: trough-crest frequencies keep the min-max mass crossing utc=0.
    fmm, ftc = np.asarray(results[10].data['fmm'].f), np.asarray(results[10].data['ftc'].f)
    u = np.linspace(-6, 6, 13)
    assert ftc.min() >= 0
    assert_allclose(ftc.sum(), fmm[np.ix_(u < 0, u > 0)].sum(), rtol=1e-10)
    assert ftc[np.ix_(u >= 0, u >= 0)].sum() == 0
    # Figure 9 rainflow matrix lives on max>=min.
    frfc = np.asarray(results[9].data['frfc'].f)
    assert frfc.min() >= -1e-12 and np.tril(frfc, -1).sum() <= 1e-12*frfc.sum()
    # Figure 5 moves the period/amplitude density to velocity and height=2*Ac.
    f5 = results[5].data['fVcfHd']
    assert_allclose(np.asarray(f5.x[1]), 2*np.asarray(results[5].data['fTcfAc'].x[1]))
    # Figure 7 is the spatial Lc/Ac density in metres.
    fLcAc = results[7].data['fLcAc']
    assert np.asarray(fLcAc.x[0]).max() == pytest.approx(250)
    assert np.all(np.isfinite(fLcAc.f)) and np.asarray(fLcAc.f).max() > 0
    with pytest.raises(NotImplementedError):
        papers.wafofig8(session, strict=True)
    # Numbered wrappers reuse the session's cached calculations.
    plt.close('all')
    for number in (1, 2, 3, 4, 5, 6, 7, 9, 10):
        assert getattr(papers, f'wafofig{number}')(session).number == number
    assert papers.wafocleanup(session) >= 17
    assert not session.figures and not session.cache


def test_supplied_theory_completes_figure_six_and_numbered_wrappers():
    session = _wafo_session()
    theory = trgauss.createpdf(2)
    theory.x = [np.linspace(0, 5, 11), np.linspace(0, 4, 9)]
    theory.f = np.ones((9, 11))/20
    session.theory[6] = theory
    result = papers.wafofig6(session)
    assert result.status == 'rendered' and result.data['theory'] is theory
    assert papers.wafofig2.__name__ == 'wafofig2'
    assert papers.wafofig2(session).number == 2
    papers.wafocleanup(session)


def test_paper_menus_and_descriptions():
    intro = papers.wafointro()
    assert intro['year'] == 2000 and len(intro['topics']) == 4
    assert 'Gullfaks' in papers.recintro()['dataset']
    session = papers.wafodemo(_wafo_session())
    assert session.menus
    controls = papers.wafofig(-1, session)
    assert controls in session.menus
    menu = papers.papermenu()
    assert menu is not None
    papers.wafocleanup(session)
    plt.close('all')
    with pytest.raises(ValueError):
        papers.wafofig(11, session)
    with pytest.raises(TypeError):
        papers.recinit(session)


def test_reconstruction_paper_figures_render_on_a_short_record():
    record = data.load('gfaks89.dat')[:4000].copy()
    record[2000:2150, 1] = np.nan
    session = papers.recinit(assets={'gfaks89.dat': record}, mc_samples=3, mc_length=2000,
                             kde_inc=32, Nsim=2)
    results = papers.recfig(-2, session)
    assert set(results) == set(range(1, 14))
    assert all(result.status == 'rendered' for result in results.values())
    cache = session.cache
    assert not np.isnan(cache['xr'][:, 1]).any()
    assert cache['Hm0'] > 0 and cache['Vrms'] > 0
    assert_allclose(cache['Hrms'], cache['Hm0']/np.sqrt(2))
    # Figures 11 and 12 compare model CDFs with the empirical distributions.
    for number in (11, 12):
        F = results[number].data['F']
        assert np.all(np.diff(F) >= -1e-8) and F[0] < .05 and F[-1] > .9
    assert len(results[6].data['test']) == 3
    plt.close('all')
    for number in range(1, 14):
        assert getattr(papers, f'recfig{number}')(session).number == number
    assert papers.recdemo(session) is session and session.menus
    assert papers.recleanup(session) >= 26


def test_rind_handles_subnormal_conditioning_density_and_tiny_schur_complements():
    S = np.array([[1., .6], [.6, 1.]])
    inf = np.inf
    value = trgauss.rind(S, np.zeros(2), np.array([[-inf]]), np.array([[0.]]), xc=np.array([[0., 38.]]), Nt=1)
    assert_allclose(value[0], .5/np.sqrt(2*np.pi), rtol=1e-6)
    assert value[1] == 0
    # A nearly deterministic integration variable leaves a tiny conditional variance.
    S = np.array([[1., 1-1e-13], [1-1e-13, 1.]])
    value = trgauss.rind(S, np.zeros(2), np.array([[-inf]]), np.array([[1.]]), xc=np.array([[0.]]), Nt=1)
    assert_allclose(value[0], 1/np.sqrt(2*np.pi), rtol=1e-6)
