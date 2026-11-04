"""Source-based cross-validation, plug-in and solve-the-equation selectors.

These estimators can fail to produce a positive pilot curvature on difficult
samples.  Such failures are reported, never replaced with a reference rule.
"""
from functools import lru_cache
import warnings

import numpy as np
from scipy import optimize, signal, special, stats

from ._estimation import _sample, _normal_derivative_sum
from ._kernels import kernelstats, gridcount, qlevels, qlevels2


def _positive(x, name):
    if not np.all(np.isfinite(x)) or np.any(np.asarray(x) <= 0):
        raise RuntimeError(name+' must be positive and finite; this pilot estimate is unsuitable')
    return x


def _ratio(kernel, dimension=1):
    mu, R, _ = kernelstats(kernel)
    return (R**dimension/mu**2/(1/(2*np.sqrt(np.pi)))**dimension)**(1/(dimension+4))


def _candidates(A, kernel, hvec):
    from . import hos
    a = _sample(A)
    if a.shape[1] != 1:
        raise ValueError('This cross-validation rule requires one data column')
    if hvec is None:
        upper = float(hos(a, kernel)[0])
        hs = np.linspace(.25*upper, upper, 100)
    else:
        hs = np.asarray(hvec, dtype=float)
    if hs.ndim != 1 or not len(hs) or not np.all(np.isfinite(hs)) or np.any(hs <= 0):
        raise ValueError('Candidate bandwidths must be a nonempty positive finite vector')
    return a[:, 0], hs, hs/_ratio(kernel)


def hboot(A, kernel='gaussian', hvec=None, *, full_output=False):
    """Bootstrap cross-validation using the original three-Gaussian objective.

    Non-Gaussian kernels use the source AMISE rescaling of Gaussian bandwidths.
    ``full_output`` returns ``(bandwidth,candidates,scores)``.  Pair sums are
    exact and chunked, so no resampling randomness or binning is involved.
    """
    x, hs, gh = _candidates(A, kernel, hvec)
    n = len(x)
    scores = []
    for h in gh:
        total = 0.
        for start in range(0, n, 128):
            z = (x[start:start+128, None]-x[None, :])/h
            y = -z*z/4
            terms = np.exp(y/2)-4*np.exp(y/1.5)/np.sqrt(3)+np.sqrt(2)*np.exp(y)
            indices = np.arange(min(128, n-start))
            terms[indices, start+indices] = 0
            total += terms.sum()
        scores.append((total+n*np.sqrt(2))/(2*n*n*h*np.sqrt(2*np.pi)))
    scores = np.asarray(scores)
    chosen = float(hs[np.argmin(scores)])
    return (chosen, hs, scores) if full_output else chosen


def _normal_reference(order, sigma):
    half = order//2
    return (-1)**half*special.gamma(order+1)/(special.gamma(half+1)*np.sqrt(np.pi)*(2*sigma)**(order+1))


def _kzero(order):
    return special.eval_hermitenorm(order, 0)/np.sqrt(2*np.pi)


def _grid_size(inc):
    if not np.isscalar(inc) or not np.isfinite(inc) or int(inc) != inc or not 8 <= inc <= 2048:
        raise ValueError('inc must be an integer from 8 through 2048')
    return int(inc)


def _binned_functional(x, inc):
    lo, hi = x.min(), x.max()
    width = hi-lo
    grid = np.linspace(lo-width/8, hi+width/8, inc)
    counts = gridcount(x, grid)
    lags = np.arange(1-inc, inc)*(grid[1]-grid[0])
    def functional(h, order, denominator=None):
        _positive(h, 'Pilot bandwidth')
        z = lags/h
        weights = special.eval_hermitenorm(order, z)*np.exp(-z*z/2)/np.sqrt(2*np.pi)
        convolved = signal.fftconvolve(counts, weights, mode='same')
        return counts@convolved/((len(x)**2 if denominator is None else denominator)*h**(order+1))
    return functional


def hscv(A, kernel='gaussian', hvec=None, *, inc=128, full_output=False):
    """Smoothed CV with the source four pilot derivative estimates.

    Linear binning and FFT convolution estimate psi6/psi10, then psi4/psi8.
    The original Gaussian difference objective is evaluated exactly.  The
    normal-reference psi12 constant is corrected to its Gaussian integral.
    Non-Gaussian kernels retain the source AMISE bandwidth rescaling.
    Full output is ``(bandwidth,candidates,scores)``.
    """
    from . import hns
    x, hs, gh = _candidates(A, kernel, hvec)
    inc = _grid_size(inc)
    n = len(x)
    sigma = float(hns(x, 'gaussian')[0]/(4/(3*n))**.2)
    psi = _binned_functional(x, inc)
    g1 = _positive(-2*_kzero(6)/(_normal_reference(8, sigma)*n), 'psi8 pilot')**(1/9)
    g2 = _positive(-2*_kzero(10)/(_normal_reference(12, sigma)*n), 'psi12 pilot')**(1/13)
    p6, p10 = psi(g1, 6), psi(g2, 10)
    g3 = _positive(-2*_kzero(4)/(p6*n), 'psi6 pilot')**(1/7)
    g4 = _positive(-2*_kzero(8)/(p10*n), 'psi10 pilot')**(1/11)
    p4, p8 = _positive(psi(g3, 4), 'psi4'), _positive(psi(g4, 8), 'psi8')
    C = (441/(64*np.pi))**(1/18)*(4*np.pi)**(-1/5)*p4**(-2/5)*p8**(-1/9)
    scores = []
    for h in gh:
        g = C*n**(-23/45)*h**(-2)
        sigmas = [np.sqrt(2*h*h+2*g*g), np.sqrt(h*h+2*g*g), np.sqrt(2*g*g)]
        term = sum(coef*_normal_derivative_sum(x, s, 0) for coef, s in zip((1, -2, 1), sigmas))
        scores.append(1/(2*np.sqrt(np.pi)*n*h)+term/n**2)
    scores = np.asarray(scores)
    chosen = float(hs[np.argmin(scores)])
    return (chosen, hs, scores) if full_output else chosen


def _iteration_options(maxit, releps, abseps):
    if int(maxit) != maxit or maxit < 1 or not np.all(np.isfinite([maxit, releps, abseps])) or releps < 0 or abseps < 0:
        raise ValueError('Need positive integer maxit and finite nonnegative tolerances')
    return int(maxit)


def hste(A, kernel='gaussian', h=None, inc=128, maxit=100, releps=.01,
         abseps=0., *, full_output=False):
    """Two-stage solve-the-equation bandwidth with binned Gaussian pilots.

    Implements the source psi4/psi6 pilot equation and fixed-point iteration.
    ``inc`` controls linear binning; Gaussian AMISE rescaling supplies other
    kernels.  Nonconvergence emits a warning.  Full output adds iteration and
    convergence arrays, so a returned last iterate is never a success claim.
    """
    from . import hns
    a = _sample(A); n, d = a.shape
    inc = _grid_size(inc); maxit = _iteration_options(maxit, releps, abseps)
    factor = _ratio(kernel)
    current = hns(a, kernel) if h is None else np.broadcast_to(np.asarray(h, dtype=float), (d,)).copy()
    if not np.all(np.isfinite(current)) or np.any(current <= 0):
        raise ValueError('Initial bandwidths must be positive and finite')
    sigma = hns(a, 'gaussian')/(4/(3*n))**.2
    iterations, convergence = [], []
    for j in range(d):
        psi = _binned_functional(a[:, j], inc)
        g1 = (-2*_kzero(4)/(_normal_reference(6, sigma[j])*n))**(1/7)
        g2 = (-2*_kzero(6)/(_normal_reference(8, sigma[j])*n))**(1/9)
        p4 = _positive(psi(g1, 4, n*(n-1)), 'psi4')
        p6 = psi(g2, 6, n*(n-1))
        _positive(-p6, '-psi6')
        h1 = current[j]/factor
        converged = False
        for count in range(1, maxit+1):
            pilot = (2*_kzero(4)*p4*h1**5/(-p6/(2*np.sqrt(np.pi))))**(1/7)
            curvature = _positive(psi(pilot, 4, n*(n-1)), 'Iterated psi4')
            updated = (1/(2*np.sqrt(np.pi)*n*curvature))**.2
            converged = abs(updated-h1) <= max(releps*updated, abseps/factor)
            h1 = updated
            if converged:
                break
        current[j] = h1*factor
        iterations.append(count); convergence.append(converged)
    if not all(convergence):
        warnings.warn('hste reached maxit without convergence', RuntimeWarning, stacklevel=2)
    info = dict(iterations=np.asarray(iterations), converged=np.asarray(convergence), inc=inc)
    return (current, info) if full_output else current


def hstt(A, kernel='gaussian', inc=64, maxit=100, releps=.01, abseps=0., *, full_output=False):
    """Scott-Tapia-Thompson bandwidth from integrated squared KDE curvature.

    Uses the source finite-difference/quadrature iteration on an ``inc`` point
    grid.  Kernel evaluation is exact rather than FFT binned.  Convergence
    failures are warned and exposed through ``full_output`` diagnostics.
    """
    from . import hns
    from ._estimation import kdefun
    a = _sample(A); n, d = a.shape
    inc = _grid_size(inc); maxit = _iteration_options(maxit, releps, abseps)
    mu2, R, _ = kernelstats(kernel)
    current = hns(a, kernel)
    iterations, convergence = [], []
    for j in range(d):
        x = a[:, j]; width = np.ptp(x)
        grid = np.linspace(x.min()-width/4, x.max()+width/4, inc)
        dx = grid[1]-grid[0]
        h1 = current[j]; converged = False
        for count in range(1, maxit+1):
            f = kdefun(x, {'kernel': kernel, 'hs': h1}, grid)
            curvature = _positive(dx*np.sum(np.diff(f, 2)**2)/dx**4, 'Squared KDE curvature')
            updated = (R/(mu2**2*n*curvature))**.2
            converged = abs(updated-h1) <= max(releps*updated, abseps)
            h1 = updated
            if converged:
                break
        current[j] = h1
        iterations.append(count); convergence.append(converged)
    if not all(convergence):
        warnings.warn('hstt reached maxit without convergence', RuntimeWarning, stacklevel=2)
    info = dict(iterations=np.asarray(iterations), converged=np.asarray(convergence), inc=inc)
    return (current, info) if full_output else current


def _normal_mixed(order, cov):
    """Integral f*D^order f for a centered bivariate Gaussian reference."""
    C = np.linalg.inv(2*cov)
    @lru_cache(None)
    def moment(r, s):
        if r < 0 or s < 0:
            return 0.
        if r+s == 0:
            return 1.
        if r:
            return (r-1)*C[0, 0]*moment(r-2, s)+s*C[0, 1]*moment(r-1, s-1)
        return (s-1)*C[1, 1]*moment(0, s-2)
    r, s = order
    return (-1)**((r+s)//2)*moment(r, s)/(4*np.pi*np.sqrt(np.linalg.det(cov)))


def _mixed_sum(x, h, order):
    r, s = order
    total = 0.
    for start in range(0, len(x), 64):
        z = (x[start:start+64, None, :]-x[None, :, :])/h
        total += np.sum(special.eval_hermitenorm(r, z[:, :, 0])*special.eval_hermitenorm(s, z[:, :, 1])*
                        np.exp(-np.sum(z*z, axis=2)/2)/(2*np.pi))
    return total/(len(x)**2*h**(r+s+2))


def _mixed_fft(x, inc):
    width = np.ptp(x, axis=0)
    axes = np.linspace(x.min(axis=0)-width/4, x.max(axis=0)+width/4, inc)
    counts = gridcount(x, axes)
    lag = np.arange(1-inc, inc)[:, None]*(axes[1]-axes[0])
    xx, yy = np.meshgrid(lag[:, 0], lag[:, 1])
    def functional(h, order):
        r, s = order
        z1, z2 = xx/h, yy/h
        weights = special.eval_hermitenorm(r, z1)*special.eval_hermitenorm(s, z2)*np.exp(-(z1*z1+z2*z2)/2)/(2*np.pi)
        convolved = signal.fftconvolve(counts, weights, mode='same')
        return np.sum(counts*convolved)/(len(x)**2*h**(r+s+2))
    return functional


def _dpi2(A, kernel, L, fft, inc, full_output):
    x = _sample(A)
    if x.shape[1] != 2:
        raise ValueError('Bivariate DPI requires exactly two data columns')
    if not np.isscalar(L) or not np.isfinite(L) or int(L) != L or not 0 <= L <= 3:
        raise ValueError('L must be 0, 1, 2 or 3')
    n = len(x); scale = np.std(x, axis=0, ddof=1)
    x = (x-x.mean(axis=0))/scale
    cov = np.cov(x, rowvar=False)
    if np.linalg.eigvalsh(cov).min() <= 1e-12:
        raise ValueError('Bivariate DPI requires nonsingular sample covariance')
    estimate = _mixed_fft(x, _grid_size(inc)) if fft else lambda h, order: _mixed_sum(x, h, order)
    q = 2*int(L)+4
    psi = {(r, q-r): _normal_mixed((r, q-r), cov) for r in range(0, q+1, 2)}
    for q in range(2*int(L)+2, 3, -2):
        new = {}
        for r in range(0, q+1, 2):
            s = q-r
            k0 = _kzero(r)*_kzero(s)
            pilot = _positive(-2*k0/((psi[r+2, s]+psi[r, s+2])*n), 'Mixed derivative pilot')**(1/(q+4))
            new[r, s] = estimate(pilot, (r, s))
        psi = new
    p40, p04, p22 = psi[4, 0], psi[0, 4], psi[2, 2]
    _positive([p40, p04, np.sqrt(max(0., p40*p04))+p22], 'Bivariate curvature')
    mu2, R, _ = kernelstats(kernel)
    h1 = (p04**.75*R**2/(n*mu2**2*p40**.75*(np.sqrt(p40*p04)+p22)))**(1/6)
    result = np.asarray([h1, h1*(p40/p04)**.25])*scale
    info = dict(psi40=p40, psi04=p04, psi22=p22, stages=int(L), method='linear-binned FFT' if fft else 'exact pair sums')
    return (result, info) if full_output else result


def hldpi2(A, kernel='gaussian', L=2, *, full_output=False):
    """Bivariate L-stage direct plug-in selector with exact derivative sums.

    Implements the source pilot chain and diagonal AMISE solution.  Corrects
    source sign/normal-reference errors and the missing derivative bandwidth
    powers in its ``psim`` helper.  Normal references use the full covariance;
    arbitrary diagonal changes of units are respected.  No complex bandwidths
    are silently reduced to their real part.
    """
    return _dpi2(A, kernel, L, False, None, full_output)


def hldpi2fft(A, kernel='gaussian', L=2, *, inc=128, full_output=False):
    """Bivariate DPI with linear binning and genuine FFT derivative convolution.

    Uses the same corrected pilot chain as ``hldpi2``.  Every mixed derivative
    is estimated by FFT, retaining its sign.  ``inc`` controls approximation
    accuracy; this repairs the source's partial psi40-only FFT overwrite.
    """
    return _dpi2(A, kernel, L, True, inc, full_output)


def _bcv2_objective(x, h):
    n = len(x); total = 0.
    for start in range(0, n, 64):
        z = (x[start:start+64, None, :]-x[None, :, :])/h
        t = np.sum(z*z, axis=2)
        terms = (t*t-8*t+8)*np.exp(-t/2)/(2*np.pi)
        indices = np.arange(min(64, n-start)); terms[indices, start+indices] = 0
        total += terms.sum()
    return 1/(4*np.pi*n*np.prod(h))+total/(4*n*(n-1)*np.prod(h))


def hbcv2(A, kernel='gaussian', *, full_output=False):
    """Bivariate biased-CV criterion, minimized over positive diagonal widths.

    Retains the source Gaussian radial derivative objective and non-Gaussian
    AMISE rescaling.  Bounded log-bandwidth optimization replaces the source
    fragile finite-difference line search.  Bounds are 0.03 to 20 times the
    standardized Gaussian reference widths; full output reports diagnostics.
    """
    from . import hns
    x = _sample(A)
    if x.shape[1] != 2:
        raise ValueError('Bivariate BCV requires exactly two data columns')
    scale = np.std(x, axis=0, ddof=1); x = (x-x.mean(axis=0))/scale
    initial = hns(x, 'gaussian')
    bounds = np.log(np.column_stack((.03*initial, 20*initial)))
    objective = lambda logh: _bcv2_objective(x, np.exp(logh))
    results = [optimize.minimize(objective, np.log(initial*factor), method='L-BFGS-B', bounds=bounds,
                                options={'ftol': 1e-13, 'gtol': 1e-8, 'maxiter': 200}) for factor in (.5, 1, 2)]
    best = min(results, key=lambda r: r.fun)
    gaussian = np.exp(best.x)
    result = gaussian*scale*_ratio(kernel, 2)
    info = dict(converged=bool(best.success), objective=float(best.fun), iterations=int(best.nit),
                gaussian_standardized=gaussian, bounds=np.exp(bounds), message=str(best.message))
    if not best.success:
        warnings.warn('Bivariate BCV optimizer did not converge: '+str(best.message), RuntimeWarning, stacklevel=2)
    return (result, info) if full_output else result


def _density_values(f):
    a = np.asarray(f['f'] if isinstance(f, dict) else f, dtype=float).ravel()
    if not len(a) or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError('Density evaluations must be nonempty, nonnegative and finite')
    return a


def _chi2(fs, ft, sampled, full_output):
    observed, reference = _density_values(fs), _density_values(ft)
    n = len(observed)
    if n <= 120:
        raise ValueError('The source asymptotic test requires more than 120 observations')
    k = max(int(np.ceil(np.sqrt(n))), 8)
    p = np.arange(k-1, 0, -1)*100/k
    if sampled:
        divisions = qlevels2(reference, p)
    else:
        density = np.asarray(ft['f'] if isinstance(ft, dict) else ft, dtype=float)
        coordinates = ft.get('x') if isinstance(ft, dict) else None
        if coordinates is not None and len(coordinates) in (1, 2):
            divisions = qlevels(density, p, *coordinates)
        else:
            divisions = qlevels(density, p)
    if np.any(np.diff(divisions) <= 0):
        raise ValueError('Repeated density-height quantiles cannot form equal-probability groups')
    counts = np.bincount(np.searchsorted(divisions, observed, side='left'), minlength=k)
    statistic = np.sum((counts-n/k)**2)/(n/k)
    pvalue = float(stats.chi2.sf(statistic, k-1))
    return (pvalue, float(statistic), k-1) if full_output else pvalue


def chi2gof(fs, ft, *, full_output=False):
    """Pearson test of density heights against an equidistant PDF grid.

    Groups have equal probability under grid-density level sets, as in WAFO.
    This tests density-height distributions, not every aspect of distributional
    equality.  Requires more than 120 observed values and distinct thresholds.
    Optional PDF ``x`` coordinates support nonuniform 1D/2D grids.  Full output
    is ``(pvalue,statistic,degrees_of_freedom)``; no fitted-parameter correction
    is inferred.
    """
    return _chi2(fs, ft, False, full_output)


def chi2gof2(fs, ft, *, full_output=False):
    """Pearson density-height test using simulated reference evaluations.

    Equal-probability groups come from reference quantiles rather than a PDF
    grid.  Shares ``chi2gof``'s sample-size and continuous-threshold limits.
    """
    return _chi2(fs, ft, True, full_output)
