"""Numerical helpers from the ITM course and switching-rainflow demos."""
import numpy as np
from scipy import integrate, linalg, signal, special


def _crack_data(v):
    v = np.asarray(v, dtype=float)
    if (v.ndim != 2 or min(v.shape) < 2 or not np.all(np.isfinite(v))
            or np.any(np.diff(v[:, 0]) <= 0)):
        raise ValueError('v must contain increasing crack lengths and experiment columns')
    return v


def alevel(v, l):
    """Slice Virkler-style experiments at crack length ``l``.

    Returns the arithmetic mean of the two measurements bracketing ``l`` in
    each experiment, retaining the source's midpoint approximation rather
    than linear interpolation.  The upper bracket is strict, so the last
    measured crack length has no slice and returns an empty vector.
    """
    v = _crack_data(v)
    if not np.isscalar(l) or not np.isfinite(l):
        raise ValueError('l must be a finite scalar')
    intervals = np.flatnonzero((v[:-1, 0] <= l) & (v[1:, 0] > l))
    if not len(intervals):
        return np.empty(0)
    return ((v[intervals, 1:] + v[intervals+1, 1:])/2).T.ravel()


def nlevel(v, N):
    """Interpolate crack length at cycle count ``N`` for each experiment.

    Experiments not bracketing the count are omitted, as in the source's
    upper-tail behavior.  Exact first measurements are handled explicitly,
    fixing the original attempt to access MATLAB row zero.  Counts below
    an experiment's first observation are omitted instead of extrapolated.
    """
    v = _crack_data(v)
    if not np.isscalar(N) or not np.isfinite(N) or np.any(np.diff(v[:, 1:], axis=0) < 0):
        raise ValueError('N must be finite and each experiment must have nondecreasing cycle counts')
    result = []
    for cycles in v[:, 1:].T:
        if N < cycles[0]:
            continue
        i = int(np.searchsorted(cycles, N))
        if i == 0:
            result.append(v[0, 0])
        elif i < len(v):
            fraction = (N-cycles[i-1])/(cycles[i]-cycles[i-1])
            result.append(v[i-1, 0] + fraction*(v[i, 0]-v[i-1, 0]))
    return np.asarray(result)


def load2ud(param, L):
    """Strict up/down crossing counts and levels for a sampled load.

    Returns ``(Nup, Ndown, u)``.  ``L`` is a value vector or N-by-2 time/value
    matrix.  An interval counts only when its endpoints are strictly on
    opposite sides of a level; touching a level or traversing its plateau
    is excluded, matching this teaching routine's source semantics.
    """
    param = np.asarray(param, dtype=float)
    if (param.shape != (3,) or not np.all(np.isfinite(param)) or param[0] > param[1]
            or param[2] < 1 or int(param[2]) != param[2]):
        raise ValueError('param must be [a,b,n] with a<=b and positive integer n')
    L = np.asarray(L, dtype=float)
    if L.ndim == 1:
        values = L
    elif L.ndim == 2 and L.shape[1] == 2:
        values = L[:, 1]
    else:
        raise ValueError('L must be values or an N-by-2 time/value matrix')
    if not np.all(np.isfinite(L)):
        raise ValueError('L must be finite')
    n = int(param[2])
    u = np.linspace(param[0], param[1], n)
    previous, following = values[:-1, None], values[1:, None]
    up = np.count_nonzero((previous < u) & (following > u), axis=0)
    down = np.count_nonzero((previous > u) & (following < u), axis=0)
    return up, down, u


def splitload(x0, tz):
    """Split a switching load into regime arrays and aligned sample labels.

    ``tz`` has transition times and integer labels; its last time terminates
    the observation window.  Intervals include the left endpoint and exclude
    the right endpoint.  Returns ``(segments, x, z)`` where ``segments`` is a
    dict keyed by the actual labels, allowing zero-based/nonconsecutive labels
    without the original MATLAB cell-index restriction.  Empty regimes remain
    present.  The last row's label is unused, as in the source.
    """
    x0, tz = np.asarray(x0, dtype=float), np.asarray(tz, dtype=float)
    if (x0.ndim != 2 or x0.shape[1] != 2 or tz.ndim != 2 or tz.shape[1] != 2
            or len(tz) < 2 or not np.all(np.isfinite(x0)) or not np.all(np.isfinite(tz))
            or np.any(np.diff(x0[:, 0]) < 0) or np.any(np.diff(tz[:, 0]) <= 0)
            or np.any(tz[:, 1] != np.floor(tz[:, 1]))):
        raise ValueError('load and transitions need ordered finite time/value pairs and integer labels')
    x = x0[(x0[:, 0] >= tz[0, 0]) & (x0[:, 0] < tz[-1, 0])].copy()
    interval = np.searchsorted(tz[:, 0], x[:, 0], side='right')-1
    z = tz[interval, 1].astype(int)
    segments = {int(label): x[z == label].copy() for label in np.unique(tz[:-1, 1])}
    return segments, x, z


def _arma(C, A, s2):
    C, A = np.atleast_1d(np.asarray(C, dtype=float)), np.atleast_1d(np.asarray(A, dtype=float))
    if (C.ndim != 1 or A.ndim != 1 or not len(C) or not len(A)
            or not np.all(np.isfinite(C)) or not np.all(np.isfinite(A)) or A[0] == 0
            or not np.isscalar(s2) or not np.isfinite(s2) or s2 < 0):
        raise ValueError('finite coefficient vectors, nonzero A[0] and nonnegative variance are required')
    if np.any(np.abs(np.roots(A)) >= 1):
        raise ValueError('the AR polynomial must describe a stationary causal model')
    return C, A, float(s2)


def armacov(C, A, s2=1, m=50):
    """Exact stationary ARMA autocovariances at integer lags zero through m.

    Coefficients use ``A(q)x = C(q)sqrt(s2)e`` with the constant coefficient
    first.  Implements the source's finite covariance equations using linear
    solves, then the AR recurrence; no truncation of the impulse response is
    required.  Unstable AR models are rejected.
    """
    C, A, s2 = _arma(C, A, s2)
    if not np.isscalar(m) or not np.isfinite(m) or int(m) != m or m < 0:
        raise ValueError('m must be a nonnegative integer')
    m = int(m)
    p, q = len(A)-1, len(C)-1
    n = max(p, q)
    a, c = np.pad(A, (0, n-p)), np.pad(C, (0, n-q))
    A1 = linalg.toeplitz(a, np.r_[A[0], np.zeros(n)])
    A2 = np.zeros_like(A1)
    C1 = np.zeros((n+1, q+1))
    for j in range(1, n+1):
        A2[:n+1-j, j] = a[j:]
    for j in range(q+1):
        C1[:n+1-j, j] = c[j:]
    initial = np.linalg.solve(A1+A2, C1 @ np.linalg.solve(A1[:q+1, :q+1], C))
    r = np.zeros(max(m+1, n+1))
    r[:n+1] = initial
    for k in range(n+1, m+1):
        if p:
            r[k] = -np.dot(r[k-np.arange(1, p+1)], A[1:])/A[0]
    return s2*r[:m+1]


def armaspec(C, A, s2=1, n=512):
    """ARMA spectral density on n frequencies in [0, 1/2) cycles/sample.

    Returns an N-by-2 frequency/density array, matching the source's actual
    output.  Density is ``s2*abs(C/A)**2``: the positive half of a two-sided
    spectrum, so its integral is half the variance for real coefficients.
    """
    C, A, s2 = _arma(C, A, s2)
    if not np.isscalar(n) or not np.isfinite(n) or int(n) != n or n < 1:
        raise ValueError('n must be a positive integer')
    w, H = signal.freqz(C, A, worN=int(n))
    return np.column_stack((w/(2*np.pi), s2*np.abs(H)**2))


def _ar_parameters(a1, m, s):
    if (not all(np.isscalar(value) and np.isfinite(value) for value in (a1, m, s))
            or abs(a1) >= 1 or s <= 0):
        raise ValueError('require |a1|<1, finite stationary mean m, and innovation standard deviation s>0')


def f_ar(y, u0, u1, a1, m, s):
    """Stationary AR(1) density times a destination-bin conditional probability.

    Follows the actual source model ``Xnext=m-a1*(X-m)+s*e``; ``m`` is its
    stationary mean, correcting the accompanying demo's innovation-mean
    documentation.  Integrating over a source bin gives a *joint* bin mass.
    """
    _ar_parameters(a1, m, s)
    if np.isnan(u0) or np.isnan(u1) or u0 >= u1:
        raise ValueError('destination bounds must increase')
    y = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError('source coordinates must be finite')
    sx = s/np.sqrt(1-a1*a1)
    next_mean = m-a1*(y-m)
    lower, upper = (u0-next_mean)/s, (u1-next_mean)/s
    # Complementary tails prevent cancellation for far-right bins.
    probability = np.where(lower > 0, special.ndtr(-lower)-special.ndtr(-upper),
                           special.ndtr(upper)-special.ndtr(lower))
    return np.exp(-.5*((y-m)/sx)**2)/(np.sqrt(2*np.pi)*sx)*probability


def discar(u, a1, m=0, s=1, *, conditional=False, full_output=False, epsabs=1e-11):
    """Discretize the stationary AR(1) model used by :func:`f_ar`.

    Like the executable source, the default matrix contains joint bin
    probabilities, despite its original conditional-probability help text.
    ``conditional=True`` divides each row by its stationary source-bin mass;
    rows may sum below one when the finite grid excludes destination tails.
    ``full_output=True`` returns ``(matrix, bin_centers)``.  Infinite outer
    edges are supported, with infinite outer midpoints (the stationary mean
    labels a single whole-real-line bin).  Accurate adaptive
    integration replaces the source's very loose quadrature tolerance.
    """
    _ar_parameters(a1, m, s)
    u = np.asarray(u, dtype=float)
    if (u.ndim != 1 or len(u) < 2 or np.any(np.isnan(u)) or np.any(u[1:] <= u[:-1])
            or not np.isfinite(epsabs) or epsabs <= 0):
        raise ValueError('u must be increasing bin edges and epsabs must be positive')
    n = len(u)-1
    matrix = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = integrate.quad(f_ar, u[i], u[i+1],
                                          args=(u[j], u[j+1], a1, m, s),
                                          epsabs=epsabs, epsrel=1e-10)[0]
    if conditional:
        sx = s/np.sqrt(1-a1*a1)
        bounds = (u-m)/sx
        mass = np.where(bounds[:-1] > 0,
                        special.ndtr(-bounds[:-1])-special.ndtr(-bounds[1:]),
                        special.ndtr(bounds[1:])-special.ndtr(bounds[:-1]))
        if np.any(mass == 0):
            raise ValueError('source-bin mass underflows; use less extreme grid edges')
        matrix /= mass[:, None]
    centers = np.array([m]) if n == 1 and np.all(np.isinf(u)) else (u[:-1]+u[1:])/2
    return (matrix, centers) if full_output else matrix
