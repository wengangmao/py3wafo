"""Explicit, lazy state for the original WAFO and reconstruction papers.

No MATLAB globals, working-directory changes, implicit input prompts or
unrelated figure cleanup are used.  Bundled measurements are loaded on demand.
"""
from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np


@dataclass
class PaperFigure:
    """A paper figure plus the numerical results and disclosed limitations."""
    paper: str
    number: int
    title: str
    figure: object
    data: dict = field(default_factory=dict)
    limitations: tuple = ()

    @property
    def status(self):
        return 'partial' if self.limitations else 'rendered'


@dataclass
class PaperSession:
    """Independent settings, assets, cached calculations and owned figures.

    ``theory`` may supply externally calculated PDF mappings keyed by figure
    number for scientifically unavailable native branches.  No fitted or
    generic curve is substituted when a theoretical result is unavailable.
    ``assets`` overrides bundled datasets for experiments or smaller examples.
    """
    paper: str
    settings: dict
    assets: dict = field(default_factory=dict)
    theory: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)
    figures: list = field(default_factory=list)
    unavailable: dict = field(default_factory=dict)
    menus: list = field(default_factory=list)

    def asset(self, name):
        if name not in self.assets:
            from ..data import load
            self.assets[name] = load(name)
        return np.asarray(self.assets[name], dtype=float).copy()

    def update(self, **settings):
        unknown = set(settings)-set(self.settings)
        if unknown:
            raise TypeError('unknown paper settings: '+', '.join(sorted(unknown)))
        self.settings.update(deepcopy(settings))
        self.cache.clear()
        return self


_WAFO_DEFAULTS = dict(Fs=.95238095238095, Hm0=6., Tp=8., Nsim=1000, Iseed=1000,
    L=80, rate=8, Nrate=8, Jrate=8, sp=15., ma=5., mb=-2.5, u=0.,
    Np=33, NNp=33, JNp=33, Nh=np.linspace(0, 8, 31), Jh=np.linspace(0, 6, 31),
    paramu=(-6, 6, 20), period_horizon=11., kde_inc=128, kernel='epan', hs=None,
    L2=.5, Nhs=None, NL2=(.5, .5), Jhs=None, JL2=(1., .5), theory_options=None)
_REC_DEFAULTS = dict(filename='gfaks89.dat', Nsim=6, L=None, seed=1000, rate=8,
    zcrit=.02, dcrit=None, ddcrit=None, csm1=.95, csm2=.05, param=(-5, 5, 501),
    res=.2, noverlap=0, CSMA=.95, CSMB=.95, kernel='epan', hs=None, L2=(.5, .5),
    kde_inc=128, mc_samples=50, mc_length=36000, conditioning='auto', context=128,
    max_points=8192)


def wafointro():
    """Describe the ISOPE 2000 WAFO paper and its reproduced figure topics."""
    return dict(title='WAFO: a Matlab toolbox for analysis of random waves and loads',
        authors='Brodtkorb, Johannesson, Lindgren, Rychlik, Ryden and Sjo', year=2000,
        venue='10th ISOPE Conference, Seattle',
        topics=('Torsethaugen simulation and spectral estimation', 'Directional spreading',
                'Wave period/amplitude distributions', 'Rainflow and trough-crest cycles'),
        limitations=('Figure 8 was never implemented in the supplied MATLAB source.',
                     'The conditional Tcf/Ac density given Tc (figure 6) requires a supplied result.'))


def recintro():
    """Describe the ISOPE 1999 reconstruction paper and field measurements."""
    return dict(title='Joint distribution of wave height and wave crest velocity from reconstructed data',
        authors='Brodtkorb, Myrhaug and Rue', year=1999, venue='9th ISOPE Conference, III, 66–73',
        dataset='Gullfaks C sensor 219, 24 December 1989, with the 20:00–20:20 interval missing',
        topics=('Transformed-Gaussian reconstruction', 'Gaussian transformation test',
                'Conditional Weibull crest velocity and Rayleigh wave height'),
        limitations=('The Python reconstruction uses its documented conditional-Gaussian estimator, '
                      'not the original whole-record heuristic iteration.',))


def wafoinit(session=None, *, assets=None, theory=None, **settings):
    """Create/update a WAFO paper session and cache its source simulation.

    Field measurements remain lazy.  Source defaults (Hm0=6, Tp=8, 1000
    simulated samples, seed=1000) are retained.  ``L`` selects the source-style
    covariance/Parzen spectral estimator; confidence intervals use its dof.
    """
    if session is None:
        session = PaperSession('wafo', deepcopy(_WAFO_DEFAULTS))
    if not isinstance(session, PaperSession) or session.paper != 'wafo':
        raise TypeError('provide a WAFO PaperSession')
    if assets:
        session.assets.update(assets)
        session.cache.clear()
    if theory:
        session.theory.update(theory)
    if settings:
        session.update(**settings)
    if 'St' not in session.cache:
        from ..spec import torsethaugen
        from ..wsim import spec2sdat
        from ..onedim import dat2spec2
        p = session.settings
        w = np.linspace(0, np.pi*p['Fs'], 257)
        spectrum = torsethaugen(w, [p['Hm0'], p['Tp']])
        simulation = spec2sdat(spectrum, p['Nsim'], seed=p['Iseed'])
        estimate = dat2spec2(simulation, min(p['L'], len(simulation)))
        session.cache.update(St=spectrum, xt=simulation, Ste=estimate)
    return session


def recinit(session=None, *, stage=2, assets=None, theory=None, **settings):
    """Initialize a lazy reconstruction-paper session through a requested stage.

    Stages 1–2 load original measurements/coastline/wind.  Stages 3–6 perform
    transformed-Gaussian reconstruction, spectral estimation and transformation
    fitting.  Stages 7–8 extract velocities/heights outside the original missing
    interval and fit the source Weibull/Rayleigh model; stages >=9 add KDE.
    The original m2 copy/paste error is corrected using the second moment.
    Large missing windows obey the reconstruction backend's max_points limit.
    """
    if session is None:
        session = PaperSession('rec', deepcopy(_REC_DEFAULTS))
    if not isinstance(session, PaperSession) or session.paper != 'rec':
        raise TypeError('provide a reconstruction PaperSession')
    if assets:
        session.assets.update(assets)
        session.cache.clear()
    if theory:
        session.theory.update(theory)
    if settings:
        session.update(**settings)
    p, cache = session.settings, session.cache
    if 'xn' not in cache:
        cache.update(xn=session.asset(p['filename']), map=session.asset('northsea.dat'), wind=session.asset('sfa89.dat'))
    if stage >= 3 and 'xr' not in cache:
        from .. import onedim, trgauss, spec, misc
        xn = cache['xn']
        dt = xn[1, 0]-xn[0, 0]
        bad, good = onedim.findoutliers(xn, p['zcrit'], 5*dt if p['dcrit'] is None else p['dcrit'],
                                       misc.gravity(61)*dt**2 if p['ddcrit'] is None else p['ddcrit'])
        options = dict(csm=p['csm1'], gsm=p['csm2'], param=p['param'])
        xr, reconstruction = onedim.reconstruct(xn, bad, p['Nsim'], p['L'], options=options,
            seed=p['seed'], conditioning=p['conditioning'], context=p['context'],
            max_points=p['max_points'], full_output=True)
        spectrum = onedim.dat2spec2(xr, p['L'])
        m0, m2 = spec.spec2mom(spectrum, 2)
        transform, test, _, _, empirical = trgauss.dat2tr(xr, options=options, full_output=True)
        Hm0, Tm02 = 4*np.sqrt(m0), 2*np.pi*np.sqrt(m0/m2)
        cache.update(xr=xr, inds=bad, good=good, reconstruction=reconstruction, Sr=spectrum,
                     m0=m0, m2=m2, Hm0=Hm0, Tm02=Tm02, Vrms=2*Hm0/Tm02,
                     Hrms=Hm0/np.sqrt(2), g=transform, g2=empirical, test0=float(np.ravel(test)[0]))
    if stage >= 7 and 'phat' not in cache:
        from .. import onedim, wstats
        missing = np.flatnonzero(np.isnan(cache['xn'][:, 1]))
        windows = ([cache['xr'][:missing[0]], cache['xr'][missing[-1]+1:]]
                   if len(missing) else [cache['xr']])
        pairs = [onedim.dat2steep(window, p['rate'], 0) for window in windows if len(window) >= 4]
        if not pairs:
            raise ValueError('no observed windows remain for wave-parameter extraction')
        V, H = (np.concatenate([pair[k] for pair in pairs]) for k in (0, 1))
        v, h = V/cache['Vrms'], H/cache['Hrms']
        phat = wstats.dist2dfit(v, h, ('weibull', 'rayleigh'), [p['res'], p['noverlap']])
        smoothed = wstats.dist2dsmfun2(phat, np.linspace(0, 3.5, 50), [p['CSMA'], p['CSMB']], [1, 1])
        fitted = wstats.dist2dpdf2(np.linspace(0, 4, 50), np.linspace(0, 4, 50), smoothed)
        cache.update(V=V, H=H, v=v, h=h, phat=phat, sphat=smoothed, ft2=fitted)
    if stage >= 9 and 'fkde' not in cache:
        from ..kdetools import kdebin, kdeoptset
        cache['fkde'] = kdebin(np.column_stack((cache['v'], cache['h'])),
            kdeoptset(kernel=p['kernel'], hs=p['hs'], L2=p['L2'], inc=p['kde_inc']))
    return session


def _cleanup(session, paper):
    if not isinstance(session, PaperSession) or session.paper != paper:
        raise TypeError('provide the corresponding PaperSession')
    import matplotlib.pyplot as plt
    figures = {result.figure for result in session.figures}
    for menu in session.menus:
        if hasattr(menu, 'close'):
            menu.close()
        elif hasattr(menu, 'canvas'):
            figures.add(menu)
    for figure in figures:
        plt.close(figure)
    session.figures.clear()
    session.menus.clear()
    session.cache.clear()
    session.unavailable.clear()
    return len(figures)


def wafocleanup(session):
    """Close only this session's figures/menus and clear its calculation cache."""
    return _cleanup(session, 'wafo')


def recleanup(session):
    """Close only this reconstruction session's figures/menus and clear cache."""
    return _cleanup(session, 'rec')


def _settings_controls(session):
    """Editable numeric controls replace the MATLAB globals/settings dialogs."""
    import ast
    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox, Button
    names = (['Hm0', 'Tp', 'Nsim', 'Iseed', 'L', 'Np', 'NNp', 'kde_inc'] if session.paper == 'wafo'
             else ['Nsim', 'L', 'rate', 'res', 'csm1', 'csm2', 'mc_samples', 'mc_length'])
    fig = plt.figure(figsize=(5, 5))
    widgets = {}
    for index, name in enumerate(names):
        ax = fig.add_axes([.35, .85-index*.085, .55, .06])
        widgets[name] = TextBox(ax, name+' ', initial=repr(session.settings[name]))
    status = fig.text(.1, .09, 'Apply clears cached calculations; existing figures remain open.')
    button = Button(fig.add_axes([.35, .02, .4, .06]), 'Apply settings')
    def apply(event):
        try:
            session.update(**{name: ast.literal_eval(widget.text) for name, widget in widgets.items()})
            status.set_text('Settings applied.')
        except (ValueError, SyntaxError, TypeError) as error:
            status.set_text(str(error))
        fig.canvas.draw_idle()
    button.on_clicked(apply)
    fig._wafo_paper_controls = (widgets, button, status)
    session.menus.append(fig)
    return fig


def _dispatch(number, session, paper, *, strict=False, **settings):
    from ._figures import _wafo_figure, _rec_figure
    init = wafoinit if paper == 'wafo' else recinit
    session = init(session, **settings)
    maximum = 10 if paper == 'wafo' else 13
    render = _wafo_figure if paper == 'wafo' else _rec_figure
    if number == -1:
        return _settings_controls(session)
    if number == -3:
        from ..misc import choices
        menu = choices(f'{paper}-figures-{id(session):x}', 'Choose a paper figure',
                       [f'Figure {i}' for i in range(1, maximum+1)],
                       [lambda i=i: render(i, session) for i in range(1, maximum+1)])
        session.menus.append(menu)
        return menu
    if number == -2:
        results = {}
        for i in range(1, maximum+1):
            try:
                results[i] = render(i, session)
            except NotImplementedError as error:
                if strict:
                    raise
                session.unavailable[i] = str(error)
                results[i] = dict(status='unavailable', reason=str(error))
        return results
    if not isinstance(number, (int, np.integer)) or not 1 <= number <= maximum:
        raise ValueError(f'figure number must be 1 through {maximum}, or -1/-2/-3')
    return render(int(number), session)


def wafofig(ix, session=None, *, strict=False, **settings):
    """Render one/all WAFO paper figures; -1 settings, -2 all, -3 choice menu.

    All-figure mode records explicitly unavailable source branches unless
    strict=True.  A returned PaperFigure reports empirical-only limitations.
    """
    return _dispatch(ix, session, 'wafo', strict=strict, **settings)


def recfig(ix, session=None, *, strict=False, **settings):
    """Render one/all reconstruction figures with the same -1/-2/-3 controls."""
    return _dispatch(ix, session, 'rec', strict=strict, **settings)


def _demo(session, paper, menu, settings):
    session = (wafoinit if paper == 'wafo' else recinit)(session, **settings)
    if menu:
        from ..misc import choices
        dispatch = wafofig if paper == 'wafo' else recfig
        cleanup = wafocleanup if paper == 'wafo' else recleanup
        control = choices(f'{paper}-paper-{id(session):x}', 'WAFO paper' if paper == 'wafo' else 'Reconstruction paper',
            ['Change settings', 'Run all figures', 'Choose a figure', 'Clean up'],
            [lambda: dispatch(-1, session), lambda: dispatch(-2, session),
             lambda: dispatch(-3, session), lambda: cleanup(session)])
        session.menus.append(control)
    return session


def wafodemo(session=None, *, menu=True, **settings):
    """Start an independent nonblocking WAFO paper workflow; return its session."""
    return _demo(session, 'wafo', menu, settings)


def recdemo(session=None, *, menu=True, **settings):
    """Start an independent nonblocking reconstruction workflow and menu."""
    return _demo(session, 'rec', menu, settings)


def papermenu():
    """Open callable choices for the two source paper workflows."""
    from ..misc import choices
    return choices('wafo-papers', 'WAFO paper reproductions',
                   ['WAFO toolbox paper', 'Reconstruction paper'], [wafodemo, recdemo])
