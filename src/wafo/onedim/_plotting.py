"""Nonblocking time-series and covariance graphics adapted from WAFO MATLAB."""
from __future__ import annotations
import numpy as np


def _axes(ax=None, *, count=1, projection=None):
    import matplotlib.pyplot as plt
    if ax is None:
        if count == 1:
            return [plt.figure().add_subplot(111, projection=projection)]
        return list(plt.subplots(count, 1, squeeze=False)[1][:, 0])
    result = list(np.asarray(ax, dtype=object).reshape(-1))
    if len(result) != count:
        raise ValueError(f'Expected {count} axes')
    return result


def lcplot(lc, plotflag=2, ma=0, sa=None, *, ax=None):
    """Plot crossing counts (1/2) or relative intensity (11/12).

    Modes ending in 2 add the source Gaussian crossing curve. If sa is absent,
    fit a line to the signed square-root log intensity, using interior bins
    when available. Unlike the legacy indexing, short/zero-count spectra are
    handled explicitly. Returns the Matplotlib axes; does not call show().
    """
    a = np.asarray(lc, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2 or not np.all(np.isfinite(a)) or np.any(a[:, 1] < 0) or np.any(np.diff(a[:, 0]) <= 0):
        raise ValueError('lc must contain increasing levels and finite nonnegative crossing counts')
    if plotflag not in (1, 2, 11, 12) or a[:, 1].max() <= 0:
        raise ValueError('Expected plotflag 1, 2, 11, or 12 and at least one positive count')
    i = np.argmax(a[:, 1])
    maximum = a[i, 1]
    if plotflag % 10 == 2:
        if sa is None:
            keep = a[:, 1] > 0
            if len(a) > 22:
                keep[:9] = False
                keep[-9:] = False
            z = np.sqrt(2*np.maximum(0, np.log(maximum/np.maximum(a[:, 1], np.finfo(float).tiny))))
            z[:i] *= -1
            if keep.sum() < 2:
                raise ValueError('At least two positive bins are required to estimate sa')
            slope = np.polyfit(a[keep, 0], z[keep], 1)[0]
            sa = 1/slope if slope > 0 else np.nan
        if not np.isfinite(sa) or sa <= 0 or not np.isfinite(ma):
            raise ValueError('Gaussian comparison requires a finite mean and positive standard deviation')
    axes = _axes(ax)[0]
    counts = a[:, 1]/maximum if plotflag > 2 else a[:, 1]
    axes.step(a[:, 0], counts, where='post', label='Observed crossings')
    if plotflag % 10 == 2:
        logmodel = -(a[:, 0]-ma)**2/(2*sa**2)
        curve = counts[i]*np.exp(logmodel-logmodel[i])
        axes.plot(a[:, 0], curve, 'r--', label='Gaussian crossings')
    axes.set(xlabel='Level', ylabel='Relative crossing intensity' if plotflag > 2 else 'Number of upcrossings', title='Crossing spectrum')
    axes.set_ylim(bottom=0)
    return axes


def covplot(rn, L=None, plotflag=1, lintype='b-', *, ax=None):
    """Plot 1-D covariance by physical lag (1), index (2), or both (3).

    Two-dimensional arrays use row/column coordinate order (x,y,t fields in
    that order): contour (1), surface (2), or zero-lag sections (3). Cartesian
    covariances are supported; a polar map is explicitly unsupported. L limits
    the number of nonnegative lags, including for two-dimensional arrays.
    Returns an axes, or an array of two axes for mode 3; never calls show().
    """
    r = np.asarray(rn['R'], dtype=float)
    if r.ndim not in (1, 2) or not np.all(np.isfinite(r)) or plotflag not in (1, 2, 3):
        raise ValueError('Expected a finite 1-D/2-D covariance and plotflag 1, 2, or 3')
    fields = [key for key in ('x', 'y', 't') if key in rn and np.size(rn[key])]
    title = 'Auto correlation function' if rn.get('norm', 0) else 'Auto covariance function'
    if r.ndim == 1:
        if len(r) < 2:
            raise ValueError('At least two covariance lags are required')
        key = fields[0] if fields else 't'
        grid = np.asarray(rn.get(key, np.arange(len(r))), dtype=float)
        if grid.shape != r.shape or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
            raise ValueError('Covariance coordinates must match and increase')
        limit = len(r)-1 if L is None else int(L)
        if limit < 1:
            raise ValueError('L must be positive')
        limit = min(len(r)-1, limit)
        axes = _axes(ax, count=2 if plotflag == 3 else 1)
        for target, mode in zip(axes, (2, 1) if plotflag == 3 else (plotflag,)):
            coordinate = np.arange(limit+1) if mode == 2 else grid[:limit+1]
            target.plot(coordinate, r[:limit+1], lintype)
            target.axhline(0, color='k', linewidth=.6)
            if 'stdev' in rn and np.size(rn['stdev']):
                sd = np.asarray(rn['stdev'], dtype=float)
                if sd.shape != r.shape or not np.all(np.isfinite(sd)) or np.any(sd < 0):
                    raise ValueError('stdev must contain one nonnegative value per covariance lag')
                target.plot(coordinate, 2*sd[:limit+1], 'r--')
                target.plot(coordinate, -2*sd[:limit+1], 'r--')
            target.set(xlabel='Lag index' if mode == 2 else ('Lag (s)' if key == 't' else 'Lag (m)'), ylabel='ACF', title=title)
            target.grid(True)
        return np.asarray(axes) if plotflag == 3 else axes[0]
    if rn.get('type') == 'polar':
        raise ValueError('Polar covariance graphics require conversion to Cartesian coordinates')
    if len(fields) != 2:
        raise ValueError('Two-dimensional covariance needs two coordinate vectors')
    first, second = [np.asarray(rn[k], dtype=float) for k in fields]
    if r.shape != (len(first), len(second)) or any(not np.all(np.isfinite(v)) or np.any(np.diff(v) <= 0) for v in (first, second)):
        raise ValueError('R rows/columns must match increasing coordinate vectors')
    limits = np.asarray(r.shape)-1 if L is None else np.broadcast_to(np.asarray(L, dtype=int), (2,))
    if np.any(limits < 1):
        raise ValueError('L must be positive')
    nr, nc = np.minimum(limits+1, r.shape)
    r, first, second = r[:nr, :nc], first[:nr], second[:nc]
    axes = _axes(ax, count=2 if plotflag == 3 else 1, projection='3d' if plotflag == 2 else None)
    if plotflag == 1:
        axes[0].contour(first, second, r.T)
    elif plotflag == 2:
        if not hasattr(axes[0], 'plot_surface'):
            raise ValueError('Surface mode requires a 3-D axes')
        xx, yy = np.meshgrid(first, second)
        axes[0].plot_surface(xx, yy, r.T, cmap='viridis')
    else:
        i, j = np.argmin(abs(first)), np.argmin(abs(second))
        if not np.isclose(first[i], 0) or not np.isclose(second[j], 0):
            raise ValueError('Covariance sections require zero on both coordinate axes')
        axes[0].plot(first, r[:, j], lintype)
        axes[1].plot(second, r[i, :], lintype)
        for target, key in zip(axes, fields):
            target.set(xlabel=f'{key} lag', ylabel='ACF', title=title)
        return np.asarray(axes)
    axes[0].set(xlabel=f'{fields[0]} lag', ylabel=f'{fields[1]} lag', title=title)
    return axes[0]


def _data(x):
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = np.column_stack((np.arange(len(a)), a))
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2 or not np.all(np.isfinite(a[:, 0])) or np.any(np.diff(a[:, 0]) <= 0) or np.any(np.isinf(a[:, 1])):
        raise ValueError('Expected values or time/value pairs with increasing finite times')
    return a


def waveplot(x, x2=None, Nsub=None, Nf=1, Sm0=None, v_fact=3, sym1='k-', sym2='k+', *, ax=None):
    """Plot consecutive record segments, including NaN gaps and an overlay.

    Returns a flat axes array. The source's positional string parser and
    integer figure handles are replaced by explicit arguments and Matplotlib
    axes. Nsub defaults to about 20 downcrossing waves per subplot (at most 6
    per figure); Nf controls the figure count. All samples are displayed.
    """
    from . import dat2tc, findcross
    data = _data(x)
    finite = np.isfinite(data[:, 1])
    if not np.any(finite):
        raise ValueError('At least one finite observation is required')
    if Nsub is None:
        crossings = findcross(data[finite, 1])
        Nsub = min(6, max(1, int(np.ceil(len(crossings)/40))))
    for value in (Nsub, Nf):
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
            raise ValueError('Nsub and Nf must be positive integers')
    Nsub, Nf = int(Nsub), int(Nf)
    if Nsub*Nf > len(data)//2:
        raise ValueError('Each subplot needs at least two observations')
    if Sm0 is None:
        Sm0 = float(np.nanstd(data[:, 1], ddof=1))
    if not np.isfinite(Sm0) or Sm0 <= 0 or not np.isfinite(v_fact) or v_fact <= 0:
        raise ValueError('Positive finite Sm0 and v_fact are required')
    if x2 is None and np.all(finite):
        x2 = dat2tc(data, 0, 'tw')
    overlay = None if x2 is None or not np.size(x2) else _data(x2)
    if ax is not None:
        axes = _axes(ax, count=Nsub*Nf)
    else:
        axes = [target for _ in range(Nf) for target in _axes(count=Nsub)]
    for target, indices in zip(axes, np.array_split(np.arange(len(data)), Nsub*Nf)):
        target.plot(data[indices, 0], data[indices, 1], sym1)
        if overlay is not None:
            keep = (overlay[:, 0] >= data[indices[0], 0]) & (overlay[:, 0] <= data[indices[-1], 0])
            target.plot(overlay[keep, 0], overlay[keep, 1], sym2)
        target.axhline(0, color='k', linewidth=.5)
        target.axhline(-2*Sm0, color='k', linestyle='--', linewidth=.5)
        target.axhline(2*Sm0, color='k', linestyle='--', linewidth=.5)
        target.set(xlim=(data[indices[0], 0], data[indices[-1], 0]), ylim=(-v_fact*Sm0, v_fact*Sm0), xlabel='Time (s)', ylabel='Surface elevation', title='Surface elevation from mean water level')
    return np.asarray(axes)


def spwaveplot(x, w_ind, tz_ind=None, x2=None, sym1='k-', sym2='k+', *, ax=None):
    """Plot selected zero-based waves, grouping adjacent wave numbers.

    tz_ind gives alternating beginning/midpoint/end indices of the defining
    waves, as in MATLAB. The default uses trough-to-trough defining crossings
    from dat2tc. Each consecutive group uses a separate axes; returns their
    flat array. Bounds are zero-based and validated before any graphics.
    """
    from . import dat2tc
    data = _data(x)
    waves = np.atleast_1d(w_ind)
    if waves.ndim != 1 or not len(waves) or not np.all(np.isfinite(waves)) or np.any(waves != np.floor(waves)) or np.any(waves < 0) or np.any(np.diff(waves) <= 0):
        raise ValueError('w_ind must contain increasing unique zero-based wave numbers')
    waves = waves.astype(int)
    if tz_ind is None:
        _, _, bounds = dat2tc(data, 0, 'tw', full_output=True)
    else:
        bounds = np.asarray(tz_ind)
    if bounds.ndim != 1 or not np.all(np.isfinite(bounds)) or np.any(bounds != np.floor(bounds)) or np.any(bounds < 0) or np.any(bounds >= len(data)) or np.any(np.diff(bounds) <= 0):
        raise ValueError('tz_ind must contain increasing valid observation indices')
    bounds = bounds.astype(int)
    if 2*waves[-1]+2 >= len(bounds):
        raise ValueError('A selected wave has no complete defining interval')
    groups = np.split(waves, np.flatnonzero(np.diff(waves) > 1)+1)
    axes = _axes(ax, count=len(groups))
    overlay = None if x2 is None else _data(x2)
    for target, group in zip(axes, groups):
        first, last = bounds[2*group[0]], bounds[2*group[-1]+2]
        target.plot(data[first:last+1, 0], data[first:last+1, 1], sym1)
        if overlay is not None:
            keep = (overlay[:, 0] >= data[first, 0]) & (overlay[:, 0] <= data[last, 0])
            target.plot(overlay[keep, 0], overlay[keep, 1], sym2)
        target.axhline(0, color='k', linewidth=.5)
        target.set(xlim=(data[first, 0], data[last, 0]), xlabel='Time (s)', ylabel=f'Wave {group[0]}' if len(group) == 1 else f'Waves {group[0]}–{group[-1]}')
    return np.asarray(axes)
