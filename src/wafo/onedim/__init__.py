"""Time series analysis. All returned indices are zero based."""
from ._reconstruction import reconstruct
from ._plotting import lcplot, covplot, waveplot, spwaveplot
from ._remaining import dat2spec2, dat2steep, mm2lc, bingham, hanning, parzen
import numpy as np
from scipy import signal


def _values(x):
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        return a
    if a.ndim == 2 and a.shape[1] in (1, 2):
        return a[:, -1]
    raise ValueError("Expected values or an N by 2 time/value array")


def findcross(x, v=0):
    """Indices immediately before level crossings; plateaus follow C MEX semantics."""
    a = np.asarray(x, dtype=float).ravel()
    if not np.all(np.isfinite(a)) or not np.isfinite(v):
        raise ValueError("Crossing input must be finite")
    if len(a) < 2:
        return np.empty(0, dtype=int)
    side = np.sign(a-v)
    nz = np.flatnonzero(side)
    if not len(nz):
        return np.empty(0, dtype=int)
    out = []
    first = nz[0]
    if first:
        out.append(first-1)
    previous = side[first]
    for j in range(first+1, len(a)):
        if side[j] and side[j] != previous:
            out.append(j-1)
            previous = side[j]
    return np.asarray(out, dtype=int)


def findextrema(x):
    """Local extrema, using the same plateau rule as derivative crossings."""
    return findcross(np.diff(np.asarray(x, dtype=float).ravel())) + 1


def ecross(t, f, ind, v=0):
    """Linearly interpolated crossing times from preceding indices."""
    t, f, i = np.asarray(t), np.asarray(f), np.asarray(ind, dtype=int)
    return t[i] + (v-f[i])*(t[i+1]-t[i])/(f[i+1]-f[i])


def findrfc(tp, h):
    """Port of source/findrfc.c, including its zero padded terminal minimum.

    Returns sorted indices (possibly repeated), not a hysteresis approximation.
    """
    a = np.asarray(tp, dtype=float).ravel()
    if h <= 0 or not np.isfinite(h):
        raise ValueError("h must be positive")
    if not np.all(np.isfinite(a)):
        raise ValueError("Turning points must be finite")
    if len(a) < 2:
        return np.empty(0, dtype=int)
    offset = int(a[0] > a[1])
    y = np.r_[a[offset:], 0.0]
    nc = (len(a)-offset)//2
    if len(y) >= 3 and ((y[0]>y[1]>y[2]) or (y[0]<y[1]<y[2])):
        raise ValueError("Not a turning point sequence")
    out = []
    for i in range(nc):
        mi, pl = 2*i, 2*i+2
        xm, xp, peak = y[mi], y[pl], y[2*i+1]
        j = i-1
        while j >= 0 and y[2*j+1] <= peak:
            if y[2*j] < xm:
                xm, mi = y[2*j], 2*j
            j -= 1
        if xm < xp:
            j = i+1
            while j < nc and y[2*j+1] < peak:
                if y[2*j+2] <= xp:
                    xp, pl = y[2*j+2], 2*j+2
                j += 1
            if j < nc and xp > xm:
                if peak-xp >= h:
                    out.extend((offset+2*i+1, offset+pl))
                continue
        if peak-xm >= h:
            out.extend((offset+mi, offset+2*i+1))
    return np.sort(np.asarray(out, dtype=int))


def dat2tp(x, h=0, wdef='none', *, return_indices=False):
    """Turning points, with WAFO endpoint and wave-definition conventions."""
    a, y = np.asarray(x, dtype=float), _values(x)
    if h < 0:
        ind = np.arange(len(y))
    else:
        ind = findextrema(y)
        if len(ind) < 2:
            ind = np.empty(0, dtype=int)
        else:
            ind = np.r_[0, ind, len(y)-1] if y[ind[0]]>y[ind[1]] else np.r_[ind, len(y)-1]
            if h > 0:
                ind = ind[findrfc(y[ind], h)]
            if wdef in ('mw', 'Mw') and len(ind) >= 2:
                if (y[ind[0]] > y[ind[1]]) != (wdef == 'Mw'):
                    ind = ind[1:]
                if len(ind) % 2 == 0:
                    ind = ind[:-1]
            elif wdef != 'none':
                raise ValueError("wdef must be none, mw, or Mw")
    return (a[ind], ind) if return_indices else a[ind]


def dat2crossind(x, v=None, wdef='all'):
    """Crossing indices for u/d/all or complete dw/uw/tw/cw wave definitions."""
    y = _values(x)
    v = float(np.mean(y)) if v is None else v
    ind = findcross(y, v)
    if wdef in ('all','none','du') or not len(ind):
        return ind
    up = y[ind+1] > v
    if wdef in ('u','d'):
        return ind[up if wdef=='u' else ~up]
    if wdef not in ('dw','uw','tw','cw'):
        raise ValueError("Unknown crossing/wave definition")
    starts_down = wdef in ('dw','tw')
    if bool(up[0])==starts_down:
        ind = ind[1:]
    desired_parity = 1 if wdef in ('dw','uw') else 0
    if len(ind)%2!=desired_parity:
        ind = ind[:-1]
    return ind


def detrendma(x, L, *, return_trend=False):
    """Port the legacy running-mean update and constant endpoint convention.

    Every column is treated as signal data, as in the MATLAB routine.
    """
    a = np.asarray(x, dtype=float)
    if int(L) != L or L <= 0:
        raise ValueError("L must be a positive integer")
    L = int(L)
    if len(a) < 2*L+1:
        trend = np.broadcast_to(a.mean(axis=0), a.shape).copy()
    else:
        mean = a[:2*L+1].mean(axis=0)
        increments = (a[2*L:]-a[:len(a)-2*L])/(2*L+1)
        middle = mean+np.cumsum(increments, axis=0)
        trend = np.concatenate((np.broadcast_to(mean,(L,)+a.shape[1:]), middle,
                                np.broadcast_to(middle[-1],(L,)+a.shape[1:])),axis=0)
    out = a-trend
    return (out,trend) if return_trend else out


def dat2spec(x, L=None, *, dt=None):
    """Welch PSD as an angular-frequency spectrum (modern estimator).

    Does not reproduce the legacy covariance-window estimator bit for bit.
    """
    from ..spec import createspec
    a, y = np.asarray(x), _values(x)
    if dt is None:
        dt = a[1, 0]-a[0, 0] if a.ndim == 2 and a.shape[1] == 2 else 1.0
    if dt <= 0:
        raise ValueError("dt must be positive")
    f, s = signal.welch(y, fs=1/dt, nperseg=min(len(y), L or 256))
    return createspec(w=2*np.pi*f, S=s/(2*np.pi))


def dat2cov(x, L=None, *, dt=None, flag='biased'):
    """Autocovariance, with NaNs filled by zero after finite-sample demeaning.

    Biased divides by the number of finite observations. Unbiased divides by
    the finite pair count at each lag. Plotting and legacy error bands pending.
    """
    from ..spec import Spectrum
    y = _values(x)
    a = np.asarray(x)
    if dt is None:
        dt = a[1,0]-a[0,0] if a.ndim==2 and a.shape[1]==2 else 1
    n = len(y)
    L = n-1 if L is None else int(L)
    if not 0 <= L < n or dt <= 0 or flag not in ('biased','unbiased'):
        raise ValueError("Invalid lag, sample interval, or scaling flag")
    finite = np.isfinite(y)
    if not finite.any():
        raise ValueError("No finite observations")
    y = np.where(finite,y-np.mean(y[finite]),0)
    raw = signal.correlate(y,y,mode='full',method='fft')[n-1:n+L]
    denom = finite.sum() if flag=='biased' else np.rint(signal.correlate(finite.astype(float),finite.astype(float),mode='full',method='fft')[n-1:n+L])
    with np.errstate(divide='ignore',invalid='ignore'):
        R = raw/denom
    return Spectrum(t=np.arange(L+1)*dt,R=R,type='time',h=np.inf,norm=0)


def dat2cor(x, L=None, *, dt=None, flag='biased', return_variance=False):
    R = dat2cov(x,L,dt=dt,flag=flag)
    variance = R.R[0]
    if variance <= 0:
        raise ValueError("Correlation requires positive variance")
    R.R /= variance
    R.norm = 1
    return (R,variance) if return_variance else R


from ._waves import dat2tc,dat2wa,dat2midind,dat2lc,findoutliers,hs2sign,vc2sign
