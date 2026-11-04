"""Nonblocking teaching figures from WAFO's rainflow demonstrations."""
import numpy as np


def _process(x, name='process'):
    x = np.asarray(x, dtype=float)
    if (x.ndim != 2 or x.shape[1] != 2 or len(x) < 2 or not np.all(np.isfinite(x))
            or np.any(np.diff(x[:, 0]) <= 0)):
        raise ValueError(f'{name} must contain finite increasing time/value pairs')
    return x


def _marked(proc, tp, k, ax):
    proc, tp = _process(proc), _process(tp, 'turning points')
    if not np.isscalar(k) or not np.isfinite(k) or int(k) != k:
        raise ValueError('k must be an integer turning-point index')
    k = int(np.clip(k, 0, len(tp)-1))
    if k < len(tp)-1 and tp[k+1, 1] > tp[k, 1]:
        k += 1
    elif k == len(tp)-1 and tp[k-1, 1] > tp[k, 1]:
        k -= 1
    ax.plot(proc[:, 0], proc[:, 1], label='Load')
    ax.plot(*tp[k], 'k.', markersize=15, label='Selected maximum')
    ax.set(xlabel='Time', ylabel='Load')
    ax.margins(y=.1)
    return proc, tp, k


def democc_markmax(proc, tp, k, diff=0, *, ax=None, full_output=False):
    """Plot a load and mark the neighboring maximum; return its zero-based index.

    ``diff`` is retained for source call compatibility; it was unused there.
    Fixes the source's final-point comparison against a linear matrix element.
    ``full_output`` also returns the axes.  No figure is shown or paused.
    """
    import matplotlib.pyplot as plt
    ax = plt.subplots()[1] if ax is None else ax
    _, _, k = _marked(proc, tp, k, ax)
    return (k, ax) if full_output else k


def _append(cc, low, high):
    cc = np.empty((0, 2)) if cc is None or not np.size(cc) else np.asarray(cc, dtype=float)
    if cc.ndim != 2 or cc.shape[1] != 2 or not np.all(np.isfinite(cc)):
        raise ValueError('previous cycles must be finite endpoint pairs')
    return np.vstack((cc, [low, high]))


def democc_mmdef(proc, tp, point, ccpt=None, *, ax=None, full_output=False):
    """Illustrate the preceding-trough/selected-peak min-max cycle.

    Returns previous cycles with the new endpoint pair appended.  ``point``
    is zero based and is adjusted to a maximum by ``democc_markmax``; the
    source ignored that adjusted index.  Full output also returns the axes.
    """
    import matplotlib.pyplot as plt
    ax = plt.subplots()[1] if ax is None else ax
    _, tp, point = _marked(proc, tp, point, ax)
    if point == 0:
        raise ValueError('the selected maximum has no preceding trough')
    ax.plot(tp[point-1, 0], tp[point-1, 1], 'k.', markersize=15, label='Preceding trough')
    ax.plot(tp[point-1:point+1, 0], tp[point-1:point+1, 1], '--', label='Min-max excursion')
    ax.set_title('Definition of min-max count')
    cycles = _append(ccpt, tp[point-1, 1], tp[point, 1])
    return (cycles, ax) if full_output else cycles


def democc_rfcdef(proc, tp, point, ccrfc=None, *, ax=None, full_output=False):
    """Illustrate the two-sided rainflow search for one selected maximum.

    Horizontal lines run to the previous strictly higher and next equal-or-
    higher load.  The higher of the intervening troughs defines the cycle.
    With no following equal/higher point, the preceding trough is retained,
    matching the source's boundary convention.  Source pauses are removed;
    real sample times replace its integer-time assumption.  Full output is
    ``(cycles, axes, search_details)``.
    """
    import matplotlib.pyplot as plt
    ax = plt.subplots()[1] if ax is None else ax
    proc, tp, point = _marked(proc, tp, point, ax)
    time, value = proc.T
    index = int(np.searchsorted(time, tp[point, 0]))
    if index == len(proc) or not np.allclose(proc[index], tp[point], rtol=1e-10, atol=1e-12):
        raise ValueError('the selected turning point must be an actual process sample')
    peak = value[index]
    earlier = np.flatnonzero(value[:index] > peak)
    left = int(earlier[-1]) if len(earlier) else 0
    left_index = left+int(np.argmin(value[left:index+1]))
    following = np.flatnonzero(value[index+1:] >= peak)
    right = index+1+int(following[0]) if len(following) else len(value)-1
    right_index = index+int(np.argmin(value[index:right+1])) if len(following) else None
    right_depth = value[right_index] if right_index is not None else -np.inf
    trough_index = left_index if value[left_index] > right_depth else right_index
    ax.plot([time[left], time[right]], [peak, peak], '--', color='0.5', label='Peak-level search')
    for i, name in ((left_index, 'Left trough'), (right_index, 'Right trough')):
        if i is not None:
            ax.plot(time[i], value[i], 'o', label=name)
            ax.plot([time[i], time[i]], [value[i], peak], '--', color='0.5')
    ax.plot([time[trough_index], time[trough_index]], [value[trough_index], peak],
            linewidth=2, label='Rainflow range')
    ax.set_title('Definition of rainflow count')
    cycles = _append(ccrfc, value[trough_index], peak)
    details = dict(peak_index=index, left_boundary=left, right_boundary=right,
                   left_trough=left_index, right_trough=right_index, trough_index=trough_index)
    return (cycles, ax, details) if full_output else cycles


def democc_tpdef(L, *, ax=None, full_output=False):
    """Illustrate local turning points with source plateau/end-point rules.

    The complete static figure replaces the source's moving XOR marker.
    Returns axes, or ``(axes, maximum_indices, minimum_indices)`` for full
    output.  Only interior points satisfying its one-sided strict tests are
    marked; all returned indices are zero based.
    """
    import matplotlib.pyplot as plt
    L = _process(L)
    y = L[:, 1]
    maximum = np.flatnonzero((y[:-2] <= y[1:-1]) & (y[1:-1] > y[2:]))+1
    minimum = np.flatnonzero((y[:-2] >= y[1:-1]) & (y[1:-1] < y[2:]))+1
    ax = plt.subplots()[1] if ax is None else ax
    ax.plot(L[:, 0], y, label='Load')
    ax.plot(L[maximum, 0], y[maximum], 'c.', markersize=12, label='Local maxima')
    ax.plot(L[minimum, 0], y[minimum], 'r.', markersize=12, label='Local minima')
    ax.set(title='Definition of turning points', xlabel='Time', ylabel='Load')
    ax.legend()
    return (ax, maximum, minimum) if full_output else ax


def democc_plotmat(demow2=None, ccrfc=None, ccmM=None):
    """Plot rainflow and min-max endpoint clouds side by side; return axes."""
    import matplotlib.pyplot as plt
    from ..cycles import ccplot
    fig = plt.figure() if demow2 is None else (demow2 if hasattr(demow2, 'subplots') else plt.figure(demow2))
    fig.clear()
    axes = fig.subplots(1, 2)
    for ax, cc, title in zip(axes, (ccrfc, ccmM), ('Rainflow cycles', 'Min-max cycles')):
        if cc is not None and np.size(cc):
            ccplot(cc, ax=ax)
            lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
            hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([lo, hi], [lo, hi], '--', color='0.5')
        ax.set(title=title, xlabel='Minimum', ylabel='Maximum')
    return axes


def _length(samples):
    if not np.isscalar(samples) or not np.isfinite(samples) or int(samples) != samples or samples < 4:
        raise ValueError('samples must be an integer of at least four')
    return int(samples)


def rfcdemo1(demoNr=1, P=None, A=None, m=None, s2=None, param=None, *,
             samples=10000, burnin=500, seed=None, plot=True, full_output=False):
    """Switching AR(1) rainflow, spectrum and simulation demonstration.

    Presets 1/2/3 retain the source's coefficients and grid; preset 0 requires
    all model inputs.  Bin probabilities are calculated from ``discar`` rather
    than loaded from precomputed MAT files.  Its finite-state approximation
    renormalizes transition rows on the chosen grid, as ``smc2rfm`` requires.
    The return value is the theoretical rainflow density; full output adds a
    dictionary of simulations, individual models and nonblocking figures.
    No unrelated figure is closed and there are no pauses or warning changes.
    """
    from ._helpers import armaspec, discar
    from .. import cycles, markov, onedim, wsim
    defaults = {
        1: ([[.98, .02], [.01, .99]], [[1, -.5], [1, .5]], [-1, 3], [1, 1], [-8, 8, 64]),
        2: ([[.99, .01], [.02, .98]], [[1, -.5], [1, .5]], [0, 0], [4, 1], [-10, 10, 64]),
        3: ([[.975, .02, .005], [.01, .98, .01], [.005, .02, .975]],
            [[1, -.5], [1, -.3], [1, .5]], [-1, 0, 3], [1, 1, 1.44], [-8, 8, 64])}
    if demoNr not in (0, *defaults):
        raise ValueError('demoNr must be zero, one, two or three')
    if demoNr == 0:
        if any(value is None for value in (P, A, m, s2, param)):
            raise ValueError('custom demo requires P, A, m, s2 and param')
    else:
        P, A, m, s2, param = [default if value is None else value
                              for value, default in zip((P, A, m, s2, param), defaults[demoNr])]
    P, A = markov._transition(P), np.asarray(A, dtype=float)
    r = len(P)
    m, s2 = np.asarray(m, dtype=float), np.asarray(s2, dtype=float)
    if (A.shape != (r, 2) or m.shape != (r,) or s2.shape != (r,) or not np.all(np.isfinite(A))
            or np.any(A[:, 0] == 0) or np.any(np.abs(A[:, 1]/A[:, 0]) >= 1)
            or np.any(s2 <= 0)):
        raise ValueError('provide one stationary AR(1) coefficient pair, mean and positive variance per regime')
    u, samples = cycles.levels(param), _length(samples)
    delta = u[1]-u[0]
    stationary_mean = m/A.sum(axis=1)
    edges = u.copy()
    edges[0] -= .25*(u[-1]-u[0])
    edges[-1] += .25*(u[-1]-u[0])
    matrices = [discar(edges, A[i, 1]/A[i, 0], stationary_mean[i], np.sqrt(s2[i])/abs(A[i, 0]))
                for i in range(r)]
    density, counting = markov.smc2rfm(P, matrices, (2, delta), full_output=True)
    individual = [markov.mc2rfm(q, (2, delta), full_output=True) for q in matrices]
    weights = markov.mc2stat(P)
    x, z, innovations = wsim.sarmasim(np.ones((r, 1)), A, m, s2, P, samples, burnin, seed=seed)
    turning = onedim.dat2tp(np.column_stack((np.arange(samples), x)))
    observed_cycles = cycles.tp2rfc(turning)
    observed = cycles.cc2cmat(param, observed_cycles)/(samples*delta**2)
    crossing = cycles.cc2lc(observed_cycles)
    spectra = [armaspec(1, a, var, 256) for a, var in zip(A, s2)]
    figures = []
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(r, 1, squeeze=False)
        for i, (ax, spectrum) in enumerate(zip(axes[:, 0], spectra)):
            ax.plot(*spectrum.T)
            ax.set(ylabel=f'S(f), regime {i}', xlabel='Frequency, cycles/sample')
            ax.grid()
        fig.suptitle('Individual AR(1) spectra')
        figures.append(fig)
        axes = markov.hmmplot(x[:501], z[:501], Title='Switching AR(1) process', Ylabel='Load', colType=1)
        figures.append(axes[0].figure)
        fig, axes = plt.subplots(1, r, squeeze=False)
        centers = (edges[:-1]+edges[1:])/2
        for i, (ax, q) in enumerate(zip(axes[0], matrices)):
            cycles.cmatplot(centers, centers, q, 3, ax=ax)
            ax.set_title(f'Regime {i}: joint bin probabilities')
        figures.append(fig)
        fig, axes = plt.subplots(2, 2)
        cycles.cmatplot(u, u, density, 3, ax=axes[0, 0])
        axes[0, 0].set_title('Calculated rainflow density')
        cycles.cmatplot(u, u, observed, 3, ax=axes[0, 1])
        axes[0, 1].set_title('Observed rainflow density')
        axes[1, 0].plot(u, np.diag(counting), label='Calculated')
        if len(crossing):
            axes[1, 0].step(crossing[:, 0], crossing[:, 1]/samples, where='post', label='Observed')
        axes[1, 0].set(title='Crossing intensity', xlabel='Level', ylabel='Crossings/sample')
        axes[1, 0].legend()
        axes[1, 1].plot(u, np.diag(counting), label='Switching model')
        weighted = np.zeros(len(u))
        for i, (_, mu) in enumerate(individual):
            term = weights[i]*np.diag(mu)
            weighted += term
            axes[1, 1].plot(u, term, '--', label=f'Weighted regime {i}')
        axes[1, 1].plot(u, weighted, ':', label='Weighted sum')
        axes[1, 1].set(title='Switching versus individual regimes', xlabel='Level')
        axes[1, 1].legend(fontsize='small')
        fig.tight_layout()
        figures.append(fig)
    info = dict(figures=tuple(figures), levels=u, transitions=matrices, regime_transition=P,
                regime_weights=weights, individual=individual, counting=counting,
                samples=x, regimes=z, innovations=innovations, cycles=observed_cycles,
                observed=observed, crossing=crossing, spectra=spectra)
    return (density, info) if full_output else density


def rfcdemo2(demoNr=1, P=None, param=None, x0=None, s=None, lam=None, *,
             samples=5000, seed=None, plot=True, full_output=False):
    """Switching turning-point rainflow, observed cycles and damage demonstration.

    Presets 1/2/11 retain the original elliptical transition models.  Optional
    parameters override those defaults.  Theoretical switching cycles are
    compared with stationary-weighted isolated regimes, simulated cycles,
    their Gaussian-smoothed matrix, crossing spectra and beta=3 damage.
    Returns the theoretical matrix, or ``(matrix, diagnostics)`` with full
    output.  Figures are created without blocking or changing other figures.
    """
    from .. import cycles, damage, markov, wsim
    # The source help advertises rfcdemo2(P), although its body omitted
    # argument dispatch for that shorthand.
    if np.ndim(demoNr) == 2 and P is None:
        P, demoNr = demoNr, 1
    defaults = {
        1: ([[.9, .1], [.05, .95]], [-1, 1, 32], [[-.4, -.3], [.3, .4]], [.15, .15], [1, 1]),
        2: ([[.9, .1], [.05, .95]], [-1, 1, 32], [[-.1, .1], [0, 0]], [.28, .12], [.5, 2]),
        11: ([[.96, .04], [.01, .99]], [-1, 1, 32], [[.2, .3], [-.1, -.1]], [.18, .1], [.5, 2])}
    if demoNr not in defaults:
        raise ValueError('demoNr must be one, two or eleven')
    P, param, x0, s, lam = [default if value is None else value for value, default in
                            zip((P, param, x0, s, lam), defaults[demoNr])]
    P = markov._transition(P)
    r, samples = len(P), _length(samples)
    x0, s, lam = np.asarray(x0, dtype=float), np.asarray(s, dtype=float), np.asarray(lam, dtype=float)
    if x0.shape != (r, 2) or s.shape != (r,) or lam.shape != (r,):
        raise ValueError('provide one ellipse center, scale and orientation per regime')
    u = cycles.levels(param)
    matrices = [markov.mktestmat(param, center, scale, orient, both=True)
                for center, scale, orient in zip(x0, s, lam)]
    weights = markov.mc2stat(P)
    individual = [markov.mctp2rfm(pair, full_output=True) for pair in matrices]
    mixed, counting = markov.smctp2rfm(P, matrices, full_output=True)
    weighted = sum(weight*pair[0] for weight, pair in zip(weights, individual))
    xD, z = wsim.smctpsim(P, matrices, samples, seed=seed)
    x = u[xD]
    observed = cycles.dtp2rfm(xD, len(u))*2/samples
    smoothed = cycles.smoothcmat(observed, 1, .7)
    crossing = cycles.cmat2lc(param, observed)
    mixed_damage, weighted_damage = (damage.cmat2dmat(param, f, 3) for f in (mixed, weighted))
    figures = []
    if plot:
        import matplotlib.pyplot as plt
        axes = markov.hmmplot(x[:501], z[:501], Title='Switching Markov turning points', Ylabel='Load', colType=1)
        figures.append(axes[0].figure)
        fig, axes = plt.subplots(2, 2)
        for ax, matrix, title in zip(axes.ravel(), (mixed, weighted, observed, smoothed),
                                     ('Switching rainflow', 'Weighted isolated regimes',
                                      'Observed rainflow', 'Smoothed observed rainflow')):
            cycles.cmatplot(u, u, matrix, 3, ax=ax)
            ax.set_title(title)
        fig.tight_layout()
        figures.append(fig)
        fig, axes = plt.subplots(1, 2)
        for ax, matrix, title in zip(axes, (mixed_damage, weighted_damage),
                                     ('Switching damage, beta=3', 'Weighted isolated damage, beta=3')):
            cycles.cmatplot(u, u, matrix, 3, ax=ax)
            ax.set_title(title)
        fig.tight_layout()
        figures.append(fig)
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(u, np.diag(counting), label='Calculated')
        axes[0].plot(*crossing.T, label='Observed')
        axes[0].set(title='Calculated versus observed crossings', xlabel='Level', ylabel='Crossings/cycle')
        axes[0].legend(fontsize='small')
        axes[1].plot(u, np.diag(counting), label='Switching model')
        for i, (weight, (_, mu)) in enumerate(zip(weights, individual)):
            axes[1].plot(u, weight*np.diag(mu), '--', label=f'Weighted regime {i}')
        axes[1].plot(u, np.diag(cycles.cmat2nt(weighted)), ':', label='Weighted sum')
        axes[1].set(title='Switching versus isolated regimes', xlabel='Level')
        axes[1].legend(fontsize='small')
        fig.tight_layout()
        figures.append(fig)
    info = dict(figures=tuple(figures), levels=u, transitions=matrices, regime_transition=P,
                regime_weights=weights, individual=individual, weighted=weighted, counting=counting,
                samples=x, discrete_samples=xD, regimes=z, observed=observed, smoothed=smoothed,
                crossing=crossing, damage=mixed_damage, weighted_damage=weighted_damage)
    return (mixed, info) if full_output else mixed
