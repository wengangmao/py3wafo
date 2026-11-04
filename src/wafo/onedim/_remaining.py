"""Covariance spectra, wave steepness and the original WAFO data windows.

The formulas come from the corresponding ``onedim`` MATLAB routines.  Array
indices are zero based and optional extra outputs use ``full_output=True``.
"""
import warnings

import numpy as np
from scipy.interpolate import CubicSpline


def _window_length(n):
    if not np.isscalar(n) or not np.isfinite(n) or int(n) != n or n < 0:
        raise ValueError('Window length must be a nonnegative integer')
    return int(n)


def bingham(n, p=.2):
    """Original WAFO cosine-tapered rectangular window, of length ``n``.

    ``p=0`` is rectangular and ``p=1`` is the WAFO Hanning window.  The
    positive end samples and denominator ``n+1`` are intentional; this is
    different from a Hann window that includes zero-valued endpoints.
    """
    n = _window_length(n)
    if not np.isscalar(p) or not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Taper fraction p must be between zero and one')
    n1 = int(np.floor(p*n/2))
    edge = .5*(1-np.cos(2*np.pi*np.arange(1, n1+1)/(p*(n+1)))) if n1 else np.empty(0)
    return np.r_[edge, np.ones(n-2*n1), edge[::-1]]


def hanning(n):
    """WAFO's positive-endpoint Hanning window: ``bingham(n, 1)``."""
    return bingham(n, 1)


def parzen(n, fs=1, *, full_output=False):
    """Original positive-endpoint piecewise cubic Parzen window.

    With ``full_output=True``, also return the source bandwidth estimate
    ``1.33*fs/(n+1)`` in the same frequency units as ``fs``.
    """
    n = _window_length(n)
    if not np.isscalar(fs) or not np.isfinite(fs) or fs <= 0:
        raise ValueError('Sampling frequency must be positive and finite')
    tau = 2*np.abs(np.arange(1, n+1)/(n+1)-.5)
    w = np.where(tau <= .5, 1-6*tau**2+6*tau**3, 2*(1-tau)**3)
    return (w, 1.33*fs/(n+1)) if full_output else w


def _signals(x, *, dt=None, missing=False, uniform=False):
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        y = a[:, None]
        t = np.arange(len(y), dtype=float)
    elif a.ndim == 2 and a.shape[1] == 1:
        y = a.copy()
        t = np.arange(len(y), dtype=float)
    elif a.ndim == 2 and a.shape[1] >= 2:
        t, y = a[:, 0], a[:, 1:]
    else:
        raise ValueError('Expected a value vector or time followed by signal columns')
    if len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('At least two finite, increasing sample times are required')
    if np.any(np.isinf(y)) or (not missing and not np.all(np.isfinite(y))):
        raise ValueError('Signal samples must be finite' if not missing else 'Signal samples cannot be infinite')
    if dt is None:
        dt = t[1]-t[0]
        if uniform and not np.allclose(np.diff(t), dt, rtol=1e-7, atol=1e-12*max(1., abs(dt))):
            raise ValueError('Spectral estimation requires uniformly sampled data')
    if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Sampling interval must be positive and finite')
    return t, y, float(dt)


def dat2spec2(xn, L=None, g=None, plotflag=0, wdef='parzen', dT=None,
              chopOffHighFreq=False):
    """One-sided angular spectrum from a Parzen/Hanning-windowed covariance.

    ``xn`` is a value vector or time followed by signal columns.  For several
    columns the covariances are averaged, producing one spectrum, as in the
    source.  ``g`` is an optional physical-to-Gaussian transformation table.
    NaNs are supported without a transformation, using the existing biased
    finite-observation covariance estimator.  ``L`` counts covariance samples
    including lag zero.  The default uses the source two-standard-deviation
    cutoff (at most 300 lags), with its Parzen adjustment.

    The source window, FFT interpolation, ``1e-12`` covariance nugget, absolute
    value of negative estimates, bandwidth and degree-of-freedom formulas are
    retained.  Linear covariance replaces the old insufficiently padded FFT's
    circular covariance.  Short records and constant signals have defined
    defaults rather than the source indexing failures.  Hanning estimates can
    be negative before taking their absolute value; this emits a warning.

    Plotting and the experimental high-frequency truncation option are not
    supported.  They must be performed explicitly on the returned spectrum.
    """
    from . import dat2cov
    from ..spec import createspec

    if plotflag:
        raise NotImplementedError('Plot the returned spectrum explicitly')
    if chopOffHighFreq:
        raise NotImplementedError('The experimental irregularity-matching frequency truncation is not ported')
    _, y, dt = _signals(xn, dt=dT, missing=True, uniform=True)
    n = len(y)
    if isinstance(wdef, str):
        window = {'p': 1, 'h': 2}.get(wdef.lower()[:1])
    else:
        window = wdef
    if window not in (1, 2):
        raise ValueError('Window must be parzen/1 or hanning/2')
    if g is not None and np.size(g):
        from ..trgauss import tranproc
        if not np.all(np.isfinite(y)):
            raise ValueError('Transformation of missing samples is not supported')
        # tranproc treats a matrix as a process and its derivatives; these
        # columns are independent signals and must be transformed separately.
        y = np.column_stack([tranproc(values, g) for values in y.T])
    covariances, deviations = [], []
    for values in y.T:
        count = np.count_nonzero(np.isfinite(values))
        if not count:
            raise ValueError('Each signal column requires finite observations')
        r = dat2cov(values, n-1, dt=dt).R
        # Bartlett's large-lag standard deviation, avoiding normalization by
        # r(0) so a constant process also has a well-defined estimate.
        variance = np.r_[0., r[0]**2, r[0]**2+2*np.cumsum(r[1:-1]**2)]/count
        covariances.append(r)
        deviations.append(np.sqrt(variance))
    r = np.mean(covariances, axis=0)
    std = np.mean(deviations, axis=0)
    if L is None:
        limit = min(300, n-1)
        significant = np.flatnonzero(abs(r[:limit]) > 2*std[:limit])
        L = min(significant[-1]+2, limit) if len(significant) else 1
        if window == 1:
            L = min(int(np.floor(4*L/3)), max(1, n-2))
    if not np.isscalar(L) or not np.isfinite(L) or int(L) != L or not 1 <= L <= n:
        raise ValueError('L must be an integer between one and the sample count')
    L = int(L)
    win = parzen(2*L-1) if window == 1 else hanning(2*L-1)
    nf = 1 << int(np.ceil(np.log2(max(1, 2*L-2))))
    nfft = 2*nf
    rwin = (r[:L]+1e-12)*win[L-1:]
    extension = np.r_[rwin, np.zeros(nfft-(2*L-1)), rwin[:0:-1]]
    raw = np.fft.fft(extension).real
    negative = bool(np.min(raw) < -64*np.finfo(float).eps*max(1., np.max(abs(raw))))
    if negative:
        warnings.warn('Negative covariance-window spectral estimates were folded by absolute value, as in WAFO', RuntimeWarning, stacklevel=2)
    S = createspec(w=np.arange(nf+1)/nf*np.pi/dt, S=abs(raw[:nf+1])*dt/np.pi)
    S.tr = [] if g is None else g
    S.L = L
    S.Bw = 2*np.pi*(1.33 if window == 1 else 1)/(L*dt)
    S.dof = int(np.floor((3.71 if window == 1 else 2.67)*n/L))
    S.note = 'dat2spec2: covariance-window estimate'
    S.negative_estimates = negative
    return S


def mm2lc(mm, definition=2, plotflag=0, sa=None, *, full_output=False):
    """Level-crossing spectrum from min-to-max cycles; optionally return alpha.

    Definitions 1/2/3/4 include neither endpoint / maxima / both / minima.
    Definition 4 retains the source's repeated final count.  Reversed pairs
    are discarded, as in MATLAB.  ``full_output=True`` additionally returns
    ``alpha=max(crossing_count)/number_of_valid_cycles``.  ``sa`` only affected
    the original plot and does not affect either numerical output.
    Zero-span cycles contribute only to definition 3; this corrects the
    original definition-1 formula's negative count at their common endpoint.
    """
    if plotflag:
        raise NotImplementedError('Plot the returned crossing spectrum explicitly')
    if definition not in (1, 2, 3, 4):
        raise ValueError('definition must be 1, 2, 3 or 4')
    pairs = np.asarray(mm, dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.all(np.isfinite(pairs)):
        raise ValueError('Expected finite min/max pairs with two columns')
    pairs = pairs[pairs[:, 0] <= pairs[:, 1]]
    if not len(pairs):
        raise ValueError('No valid min-to-max cycles')
    u, inverse = np.unique(pairs, return_inverse=True)
    inverse = inverse.reshape(pairs.shape)
    minima = np.bincount(inverse[:, 0], minlength=len(u))
    maxima = np.bincount(inverse[:, 1], minlength=len(u))
    counts = np.cumsum(minima-maxima)
    if definition in (1, 2):
        counts -= minima
    if definition == 1:
        zero = inverse[pairs[:, 0] == pairs[:, 1], 0]
        counts += np.bincount(zero, minlength=len(u))
    if definition in (2, 3):
        counts += maxima
    if definition == 4 and len(counts) > 1:
        counts[-1] = counts[-2]
    lc = np.column_stack((u, counts))
    return (lc, float(np.max(counts)/len(pairs))) if full_output else lc


def dat2steep(xx, rate=1, method=1, *, full_output=False):
    """Wave steepness/speed and heights from complete zero-level waves.

    Return ``(S, H)`` by default; full output is
    ``(S,H,Ac,At,Tcf,Tcb,crossing_indices,interpolated_signal)``.
    ``At`` has one extra boundary trough per signal.  Multiple signal columns
    are processed independently and concatenated.  For these, crossing indices
    retain the source packed layout ``[count,indices...,count,indices...]``.

    Methods 1/-1 give crest front/back speeds, 2/-2 their steepnesses, 3/-3
    total down/up-crossing steepnesses; 0 selects the greater crest speed and
    its corresponding height.  ``rate>1`` uses not-a-knot cubic splines, as in
    MATLAB; ``rate<=1`` uses the original samples.  Gravity uses WAFO's default
    latitude.  Crossing times are linearly interpolated between samples.
    """
    from . import dat2tc, ecross
    from ..misc import gravity

    if method not in (0, 1, -1, 2, -2, 3, -3):
        raise ValueError('Unknown steepness method; use 0, +/-1, +/-2 or +/-3')
    if not np.isscalar(rate) or not np.isfinite(rate):
        raise ValueError('Interpolation rate must be finite')
    t, columns, dt = _signals(xx)
    if rate > 1:
        if not np.allclose(np.diff(t), dt, rtol=1e-7, atol=1e-12*max(1., dt)):
            raise ValueError('Interpolation rate requires uniformly sampled input')
        dt /= rate
        count = int(np.floor((t[-1]-t[0])/dt+1e-10))+1
        ti = t[0]+np.arange(count)*dt
        columns = CubicSpline(t, columns, axis=0)(ti)
        t = ti
    interpolated = np.column_stack((t, columns))
    steepness, heights, crests, troughs, fronts, backs, crossings = ([] for _ in range(7))
    for y in columns.T:
        tc, _, cross = dat2tc(np.column_stack((t, y)), 0, 'tw', full_output=True)
        if columns.shape[1] > 1:
            crossings.append(np.r_[len(cross), cross])
        else:
            crossings.append(cross)
        ac, at = tc[1::2, 1], -tc[::2, 1]
        tu = ecross(t, y, cross[1:-1:2], 0)
        td = ecross(t, y, cross[2::2], 0)
        tf, tb = tc[1::2, 0]-tu, td-tc[1::2, 0]
        tf, tb = np.where(tf == 0, dt, tf), np.where(tb == 0, dt, tb)
        hd, hu = ac+at[:-1], ac+at[1:]
        if method == 0:
            s = ac/np.minimum(tf, tb)
            h = np.where(tf <= tb, hd, hu)
        elif abs(method) == 1:
            s, h = (ac/tf, hd) if method > 0 else (ac/tb, hu)
        else:
            period = np.diff(ecross(t, y, cross[::2] if method > 0 else cross[1::2], 0))
            h = hd if method > 0 else hu
            s = 2*np.pi*(ac/(tf if method > 0 else tb)/period if abs(method) == 2 else h/period**2)/gravity()
        steepness.append(s); heights.append(h); crests.append(ac); troughs.append(at)
        fronts.append(tf); backs.append(tb)
    result = tuple(np.concatenate(v) for v in (steepness, heights, crests, troughs, fronts, backs, crossings))
    return (*result, interpolated) if full_output else result[:2]
