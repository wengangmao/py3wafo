"""Figures of the WAFO (ISOPE 2000) and reconstruction (ISOPE 1999) papers.

Each renderer reproduces one wafofigN/recfigN script on a new figure, caches
expensive calculations in the session, and returns a PaperFigure holding the
plotted numerical results. Theory branches that cannot be computed are left
out and named in ``limitations`` (or supplied through ``session.theory``);
no substitute curve is drawn.
"""
import numpy as np
from scipy.integrate import trapezoid
from ._sessions import PaperFigure

_WAFO_TITLES = {
    1: 'Linear simulation from a Torsethaugen spectrum',
    2: 'Spectral densities: Torsethaugen (solid) and estimated (dash)',
    3: 'Probability density of wave period Tt: histogram, KDE and theory',
    4: 'Directional spectra using Torsethaugen and cos2s spreading',
    5: 'Joint density of crest front velocity and wave height',
    6: 'Joint density of crest front period Tcf and crest amplitude Ac given Tc=5 s',
    7: 'Joint density of crest wavelength Lc and crest amplitude Ac',
    8: 'Joint density of crest wavelength and amplitude for extremal waves',
    9: 'Intensity of rainflow cycles computed from St',
    10: 'Intensity of trough-crest cycles computed from St',
}
_REC_TITLES = {
    1: 'Location of Gullfaks C and Statfjord A platforms in the North Sea',
    2: '10 minutes mean values of wind (dash) and direction (solid)',
    3: 'Example of the reconstructed (solid) and original data set (pluses)',
    4: 'Estimated spectral density',
    5: 'Transfer function g versus the crossing level u',
    6: 'Variability of simulated e(g(u)-u) and observed value',
    7: 'Estimated Weibull scale parameter versus h=H/Hrms',
    8: 'Estimated Weibull shape parameter versus h=H/Hrms',
    9: 'Conditional mean and standard deviation of V given H',
    10: 'Joint distribution of V and H: model and kernel estimate',
    11: 'Probability of exceeding H: model (dash) and data (dots)',
    12: 'Probability of exceeding V: model (dash) and data (dots)',
    13: 'Conditional probability of exceeding V given H',
}


def _axes(figsize=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    return fig, fig.add_subplot()


def _finish(session, paper, number, fig, data, flag, limitations=()):
    from ..misc import wafostamp
    title = (_WAFO_TITLES if paper == 'wafo' else _REC_TITLES)[number]
    wafostamp(f'Figure {number}', flag, fig=fig)
    result = PaperFigure(paper, number, title, fig, data, tuple(limitations))
    session.figures.append(result)
    return result


def _cached(session, key, compute):
    if key not in session.cache:
        session.cache[key] = compute()
    return session.cache[key]


def _kde(data, kernel, hs, L2, inc=None):
    """kdebin estimate with the source contour levels enclosing pl percent."""
    from ..kdetools import kdebin, kdeoptset, evalpdf, qlevels2
    options = dict(kernel=kernel, hs=hs, L2=L2)
    if inc is not None:
        options['inc'] = inc
    f = kdebin(data, kdeoptset(**options))
    if np.ndim(data) == 2 and np.shape(data)[1] == 2 and 'pl' in f:
        r = evalpdf(f, data[:, 0], data[:, 1], method='linear')
        f.cl = qlevels2(r, f.pl)
    return f


def _theory_options(session):
    from ..trgauss import rindoptset
    options = session.settings['theory_options']
    return rindoptset(**options) if isinstance(options, dict) else options


def _supplied(session, number):
    """An externally calculated theoretical density for figure ``number``."""
    return session.theory.get(number)


# --------------------------------------------------------------------- WAFO
def _wafo1(session):
    from ..onedim import waveplot
    p, xt = session.settings, session.cache['xt']
    fig, ax = _axes()
    waveplot(xt, Nsub=1, Nf=1, sym1='k-', ax=ax)
    ax.set_title(f"Hm0 = {p['Hm0']:g}   Tp = {p['Tp']:g}")
    return _finish(session, 'wafo', 1, fig, dict(xt=xt), '(ER)')


def _wafo2(session):
    from ..spec import wspecplot
    St, Ste = session.cache['St'], session.cache['Ste']
    fig, ax = _axes()
    wspecplot(St, 1, 'k-', ax=ax)
    wspecplot(Ste, 1, '--', ax=ax)
    return _finish(session, 'wafo', 2, fig, dict(St=St, Ste=Ste), '(ER)')


def _wafo3(session):
    from ..onedim import dat2wa
    from ..trgauss import spec2thpdf, pdfplot
    from ..wstats import whisto
    p, c = session.settings, session.cache
    grid = [0, p['period_horizon'], p['Np']]
    Tt = _cached(session, 'Tt', lambda: np.asarray(dat2wa(c['xt'], p['u'], 'd2u', 'dw', p['rate'])).ravel())
    fTt = _cached(session, 'fTt', lambda: spec2thpdf(c['St'], p['u'], 'Tt', grid, options=_theory_options(session)))
    fTte = _cached(session, 'fTte', lambda: spec2thpdf(c['Ste'], p['u'], 'Tt', grid, options=_theory_options(session)))
    kdeTt = _cached(session, 'kdeTt', lambda: _kde(Tt, p['kernel'], p['hs'], p['L2']))
    fig, ax = _axes()
    whisto(Tt, max(22, int(2*np.sqrt(len(Tt)))), 0, 1, ax=ax)
    pdfplot(kdeTt, sym='r--', ax=ax)
    pdfplot(fTt, sym='-', ax=ax)
    pdfplot(fTte, sym='g-.', ax=ax)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    return _finish(session, 'wafo', 3, fig, dict(Tt=Tt, fTt=fTt, fTte=fTte, kdeTt=kdeTt), '(ER)')


def _wafo4(session):
    from ..spec import spreading, mkdspec, wspecplot
    p, St = session.settings, session.cache['St']
    peak = 2*np.pi/p['Tp']
    D = spreading(51, 'cos2s', 0, [p['sp'], p['sp'], peak, 0, 0], St.w, 0)
    Std = mkdspec(St, D)
    Dw = spreading(np.linspace(-np.pi, np.pi, 51), 'cos2s', 0, [p['sp'], p['sp'], peak, p['ma'], p['mb']], St.w)
    Stdw = mkdspec(St, Dw)
    fig, ax = _axes()
    wspecplot(Std, 1, 'k-', ax=ax)
    wspecplot(Stdw, 1, 'k--', ax=ax)
    return _finish(session, 'wafo', 4, fig, dict(Std=Std, Stdw=Stdw), '(ER)')


def _velocity_height(f):
    """wafofig5's th2vhpdf: V=H/T transform, then the amplitude-to-height rate 2."""
    from ..trgauss import th2vhpdf
    from ..kdetools import qlevels
    f2 = th2vhpdf(f)
    f2.f = np.asarray(f2.f)/2
    f2.x[1] = np.asarray(f2.x[1])*2
    f2.pl = list(f.get('pl', [10, 30, 50, 70, 90, 95, 99, 99.9]))
    f2.cl = qlevels(f2.f, f2.pl)
    return f2


def _wafo5(session):
    from ..onedim import dat2steep, dat2spec
    from ..trgauss import spec2thpdf, pdfplot
    p = session.settings
    Nxr = _cached(session, 'Nxr', lambda: session.asset('gfaksr89.dat'))
    NVcf, NHd = _cached(session, 'NVcfHd', lambda: dat2steep(Nxr, p['Nrate'], 0)[:2])
    fTcfAc = _cached(session, 'fTcfAc', lambda: spec2thpdf(dat2spec(Nxr), 0, 'TcfAc',
        [0, p['period_horizon'], p['NNp']], p['Nh'], options=_theory_options(session)))
    fVcfHd = _cached(session, 'fVcfHd', lambda: _velocity_height(fTcfAc))
    kde = _cached(session, 'kdeVcfHd', lambda: _kde(np.column_stack((NVcf, NHd)), 'epan', p['Nhs'], p['NL2']))
    fig, ax = _axes()
    ax.plot(NVcf, NHd, '.')
    pdfplot(kde, sym='r--', ax=ax)
    pdfplot(fVcfHd, sym='k-', ax=ax)
    return _finish(session, 'wafo', 5, fig, dict(Vcf=NVcf, Hd=NHd, fTcfAc=fTcfAc, fVcfHd=fVcfHd, kde=kde), '(NR)')


def _japan_sea(session):
    from ..onedim import detrendma
    x = session.asset('yura87.dat')[4999:55000][:, [0, 2]]
    x[:, 1] = detrendma(x[:, 1], 1500)
    return x


def _wafo6(session):
    from ..onedim import dat2steep
    from ..trgauss import pdfplot
    p = session.settings
    Jxn = _cached(session, 'Jxn', lambda: _japan_sea(session))

    def steep():
        out = dat2steep(Jxn, p['Jrate'], 0, full_output=True)
        Ac, Tcf, Tcr = (np.asarray(out[k]).ravel() for k in (2, 4, 5))
        n = min(len(Ac), len(Tcf), len(Tcr))
        Ac, Tcf, Tcr = Ac[:n], Tcf[:n], Tcr[:n]
        keep = (4.8 < Tcf+Tcr) & (Tcf+Tcr < 5.2)
        return Tcf[keep], Ac[keep]
    Tcf, Ac = _cached(session, 'JTcfAc', steep)
    kde = _cached(session, 'kdeTcfAcTc', lambda: _kde(np.column_stack((Tcf, Ac)), 'epan', p['Jhs'], p['JL2']))
    fig, ax = _axes()
    ax.plot(Tcf, Ac, '.')
    pdfplot(kde, sym='r--', ax=ax)
    theory, limitations = _supplied(session, 6), ()
    if theory is not None:
        pdfplot(theory, sym='k-', ax=ax)
    else:
        limitations = ('The joint Tcf/Ac density conditional on Tc=5 s (spec2thpdf with '
                       'paramt [5 5 JNp]) is not available; supply it as session.theory[6].',)
    return _finish(session, 'wafo', 6, fig, dict(Tcf=Tcf, Ac=Ac, kde=kde, theory=theory), '(NR)', limitations)


def _crest_length(session):
    """Lc/Ac density: the spatial record is a Gaussian process with spectrum S(k),
    so the temporal TcAc driver applies exactly to the wavenumber spectrum.
    On the source's coarse length grid, short crest lengths show
    discretization artifacts."""
    from ..spec import jonswap, spec2spec, createspec
    from ..misc import findpeaks
    from ..trgauss import spec2thpdf
    p = session.settings
    Sj = jonswap()
    S = np.asarray(Sj.S, dtype=float).copy()
    peak = int(np.ravel(findpeaks(S))[0])
    S[:int(np.floor(peak*.5))] = 0
    S[int(np.floor(peak*1.5)):] = 0
    Sj.S = S
    Sk = spec2spec(Sj, 'k1d')
    k, Sk_values = np.asarray(Sk.k, dtype=float), np.asarray(Sk.S, dtype=float)
    keep = k >= 0
    k, Sk_values = k[keep], Sk_values[keep]
    # The Rice drivers need an equidistant grid; resample and keep the variance.
    uniform = np.linspace(0, k[np.flatnonzero(Sk_values)[-1]]*1.05, 257)
    resampled = np.interp(uniform, k, Sk_values, right=0)
    resampled *= trapezoid(Sk_values, k)/trapezoid(resampled, uniform)
    # Measure length in units c=sqrt(m0/m2) so the driver's absolute tolerances
    # see order-one frequencies; Lc=c*T and f_Lc=f_T/c map the result back.
    c = np.sqrt(trapezoid(resampled, uniform)/trapezoid(resampled*uniform**2, uniform))
    as_time = createspec(w=uniform*c, S=resampled/c)
    f = spec2thpdf(as_time, 0, 'TcAc', [0, 250/c, p['NNp']], p['Nh'], options=_theory_options(session))
    f.x[0] = np.asarray(f.x[0])*c
    f.f = np.asarray(f.f)/c
    if 'err' in f:
        f.err = np.asarray(f.err)/c
    f.labx = ['Crest wavelength Lc [m]', 'Crest amplitude Ac [m]']
    f.title = 'Joint density of Lc and Ac'
    return f


def _wafo7(session):
    from ..trgauss import pdfplot
    fLcAc = _cached(session, 'fLcAc', lambda: _crest_length(session))
    fig, ax = _axes()
    pdfplot(fLcAc, sym='k-', ax=ax)
    return _finish(session, 'wafo', 7, fig, dict(fLcAc=fLcAc), '(NR)')


def _wafo8(session):
    raise NotImplementedError('Figure 8 was never implemented in the supplied MATLAB source '
                              '(wafofig8 prints "Not implemented yet").')


def _maxmin(session):
    from ..trgauss import spec2mmtpdf
    p = session.settings
    return spec2mmtpdf(session.cache['St'], 0, 'Mm', [0, 7, p['NNp']], p['paramu'], _theory_options(session))


def _cycle_figure(session, number, matrix, cycles, label):
    from ..trgauss import pdfplot
    from ..kdetools import qlevels
    fmm = _cached(session, 'fmm', lambda: _maxmin(session))
    f = type(fmm)(fmm)
    f.f = matrix(np.asarray(fmm.f))
    f.cl = qlevels(f.f)
    f.title = _WAFO_TITLES[number]
    fig, ax = _axes()
    ax.plot(cycles[:, 0], cycles[:, 1], '.', markersize=3)
    pdfplot(f, sym='k-', ax=ax)
    ax.set_xlabel('min' if number == 9 else 'trough')
    ax.set_ylabel('max' if number == 9 else 'crest')
    ax.set_title(f.title)
    return _finish(session, 'wafo', number, fig, {label: f, 'fmm': fmm, 'cycles': cycles}, '(NR)')


def _wafo9(session):
    from ..markov import mctp2rfm
    from ..onedim import dat2tp
    from ..cycles import tp2rfc
    rfc = np.asarray(tp2rfc(dat2tp(session.cache['xt'])), dtype=float)
    cycles = np.column_stack((rfc.min(axis=1), rfc.max(axis=1)))
    return _cycle_figure(session, 9, lambda F: mctp2rfm((F, None)), cycles, 'frfc')


def _trough_crest_pairs(xt):
    from ..onedim import dat2tc
    tc = np.asarray(dat2tc(xt), dtype=float)[:, 1]
    start = int(tc[0] > 0)   # begin with a trough
    n = (len(tc)-start)//2
    return np.column_stack((tc[start:start+2*n:2], tc[start+1:start+2*n:2]))


def _wafo10(session):
    from ..trgauss import mctp2tc
    p = session.settings
    utc = p['u']
    return _cycle_figure(session, 10, lambda F: np.fliplr(mctp2tc(np.fliplr(F), utc, p['paramu'])),
                         _trough_crest_pairs(session.cache['xt']), 'ftc')


_WAFO = {1: _wafo1, 2: _wafo2, 3: _wafo3, 4: _wafo4, 5: _wafo5, 6: _wafo6,
         7: _wafo7, 8: _wafo8, 9: _wafo9, 10: _wafo10}


def _wafo_figure(number, session):
    return _WAFO[number](session)


# ----------------------------------------------------------- reconstruction
_PLACES = [((1, 62), (1.8, 61.2), 'Statfjord A', 'x'), ((1, 59.5), (2.3, 61.2), 'Gullfaks C', 'x'),
           ((10.4, 60.1), (10.8, 59.85), 'Oslo', 'h'), ((10, 63.05), (10.8, 63.4), 'Trondheim', 'h'),
           ((4, 58.8), (5.52, 58.9), 'Stavanger', 'h'), ((3.5, 60.3), (5.2, 60.3), 'Bergen', 'h')]


def _rec1(session):
    m = session.cache['map']
    fig, ax = _axes()
    ax.plot(m[:, 0], m[:, 1])
    ax.axis([-2, 12, 58, 64]); ax.grid(True)
    for (tx, ty), (px, py), name, marker in _PLACES:
        ax.text(tx, ty, name)
        ax.plot(px, py, marker)
    ax.plot([1.8, 1.8], [62, 61.2], 'k-'); ax.plot([1.8, 2.3], [59.5, 61.2], 'k-')
    return _finish(session, 'rec', 1, fig, dict(map=m, places=_PLACES), '(ER)')


def _rec2(session):
    wind = session.cache['wind']
    fig, ax = _axes()
    for span in ([17, 20], [20.333333, 21.33333333]):
        ax.plot(span, [34, 34], '-', linewidth=2.5)
    ax.plot(wind[:, 0], wind[:, 1], '--')
    ax.axis([0, 24, 0, 40])
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Mean wind speed (m/s)')
    right = ax.twinx()
    right.plot(wind[:, 0], wind[:, 2], '-')
    right.set_ylabel('Wind direction (degrees)')
    ax.grid(True)
    ax.set_title('Wind conditions at Statfjord A   24 Dec. 1989')
    return _finish(session, 'rec', 2, fig, dict(wind=wind), '(ER)')


def _rec3(session):
    from ..onedim import waveplot
    c = session.cache
    fig, ax = _axes()
    waveplot(c['xr'], c['xn'][c['inds']], 1, 1, np.sqrt(c['m0']), 4, sym1='k-', ax=ax)
    return _finish(session, 'rec', 3, fig, dict(xr=c['xr'], outliers=c['xn'][c['inds']]), '(NR)')


def _rec4(session):
    from ..spec import ttspec, wspecplot
    Sf = ttspec(session.cache['Sr'], 'f')
    fig, ax = _axes()
    wspecplot(Sf, ax=ax)
    return _finish(session, 'rec', 4, fig, dict(Sf=Sf), '(NR)')


def _rec5(session):
    from ..trgauss import trplot
    c = session.cache
    fig, ax = _axes()
    trplot(c['g'], c['g2'], float(np.mean(c['xr'][:, 1])), float(np.sqrt(c['m0'])), ax=ax)
    return _finish(session, 'rec', 5, fig, dict(g=c['g'], g2=c['g2']), '(NR)')


def _rec6(session):
    from ..trgauss import mctrtest, troptset
    import matplotlib.pyplot as plt
    p, c = session.settings, session.cache
    options = troptset(csm=p['csm1'], gsm=p['csm2'], param=p['param'])
    before = set(plt.get_fignums())
    test = _cached(session, 'mctrtest', lambda: mctrtest(c['Sr'], [p['mc_length'], p['mc_samples']],
        c['test0'], options=options, seed=p['seed']))
    new = sorted(set(plt.get_fignums())-before)
    if new:
        fig = plt.figure(new[-1])
    else:
        fig, ax = _axes()
        values = np.ravel(test)
        ax.plot(values, 'o', mfc='none')
        ax.axhline(c['test0'], color='k')
        ax.set_ylabel('e(g(u)-u)'); ax.set_xlabel('Simulation')
    return _finish(session, 'rec', 6, fig, dict(test=test, test0=c['test0']), '(NR)')


def _rec_param(session, number, column, label):
    from ..wstats import dist2dparamplot
    import matplotlib.pyplot as plt
    c = session.cache
    fig, ax = _axes()
    dist2dparamplot(c['phat'], c['sphat'], column, axes=ax)
    ax.set_ylabel(label); ax.set_xlabel('h')
    return _finish(session, 'rec', number, fig, dict(phat=c['phat'], sphat=c['sphat']), '(NR)')


def _rec7(session):
    return _rec_param(session, 7, 1, 'A(h)')


def _rec8(session):
    return _rec_param(session, 8, 2, 'B(h)')


def _rec9(session):
    from ..wstats import dist2dstatplot
    p, c = session.settings, session.cache
    fig, ax = _axes()
    stats = dist2dstatplot(c['v'], c['h'], c['sphat'], 2, p['res'], ax=ax, full_output=True)
    return _finish(session, 'rec', 9, fig, dict(statistics=stats), '(NR)')


def _rec10(session):
    from ..trgauss import pdfplot
    c = session.cache
    fig, ax = _axes()
    ax.plot(c['v'], c['h'], 'k.')
    ax.axis([0, 3.5, 0, 3.5]); ax.set_aspect('equal')
    pdfplot(c['ft2'], 1, sym='k', ax=ax)
    pdfplot(c['fkde'], 1, sym='k--', ax=ax)
    ax.set_xlabel('v'); ax.set_ylabel('h')
    ax.set_title('Joint distribution of v and h')
    return _finish(session, 'rec', 10, fig, dict(ft2=c['ft2'], fkde=c['fkde']), '(NR)')


def _exceedance(session, number, sample, model, label):
    from ..wstats import empdistr
    x = np.linspace(0, 3.5, 100)
    F = np.asarray(model(x), dtype=float)
    empirical = empdistr(sample)
    fig, ax = _axes()
    ax.semilogy(empirical[:, 0], 1-empirical[:, 1], 'k.')
    below = F < 1
    ax.semilogy(x[below], 1-F[below], 'r--')
    ax.axis([0, 3.5, 1e-4, 1])
    ax.set_xlabel(label); ax.set_ylabel(f'1-F({label})')
    return _finish(session, 'rec', number, fig, dict(x=x, F=F, empirical=empirical), '(NR)')


def _rec11(session):
    from ..wstats import dist2dcdf
    c = session.cache
    return _exceedance(session, 11, c['h'], lambda x: dist2dcdf(20, x, c['sphat'], 0), 'h')


def _rec12(session):
    from ..wstats import dist2dcdf
    c = session.cache
    return _exceedance(session, 12, c['v'], lambda x: dist2dcdf(x, 20, c['sphat'], 0), 'v')


def _rec13(session):
    from ..wstats import dist2dcdfplot
    import matplotlib.pyplot as plt
    p, c = session.settings, session.cache
    curves = dist2dcdfplot(c['v'], c['h'], c['sphat'], 2, p['res'], 0)
    rows = int(np.ceil(len(curves)/3))
    fig = plt.figure(figsize=(9, 2.6*rows))
    axes = [fig.add_subplot(rows, 3, k+1) for k in range(len(curves))]
    dist2dcdfplot(c['v'], c['h'], c['sphat'], 2, p['res'], 2, axes=axes)
    fig.tight_layout()
    return _finish(session, 'rec', 13, fig, dict(sphat=c['sphat'], curves=curves), '(NR)')


_REC = {1: (_rec1, 2), 2: (_rec2, 2), 3: (_rec3, 3), 4: (_rec4, 3), 5: (_rec5, 3), 6: (_rec6, 3),
        7: (_rec7, 7), 8: (_rec8, 7), 9: (_rec9, 7), 10: (_rec10, 9), 11: (_rec11, 7),
        12: (_rec12, 7), 13: (_rec13, 7)}


def _rec_figure(number, session):
    from ._sessions import recinit
    render, stage = _REC[number]
    recinit(session, stage=stage)
    return render(session)
