"""Crossing-tail and rainflow extrapolation from the WAFO source algorithms.

GPD shape follows WAFO: positive k gives a finite endpoint and survival
``(1-k*x/s)**(1/k)``.  Frequency weights replace repeated sample expansion.
Figures are optional, nonblocking, and are returned in diagnostic dictionaries.
"""
import numpy as np
from scipy import optimize, stats


def _crossings(lc):
    lc = np.asarray(lc, dtype=float)
    if (lc.ndim != 2 or lc.shape[1] != 2 or len(lc) < 3 or not np.all(np.isfinite(lc))
            or np.any(np.diff(lc[:, 0]) <= 0) or np.any(lc[:, 1] < 0) or lc[:, 1].max() <= 0):
        raise ValueError('lc must have increasing finite levels and nonnegative nonzero crossing counts')
    return lc


def _matrix(param, F):
    from . import levels
    grid = levels(param)
    F = np.asarray(F, dtype=float)
    if (F.shape != (len(grid), len(grid)) or not np.all(np.isfinite(F))
            or np.any(F < 0) or np.any(np.tril(F) != 0) or F.sum() <= 0):
        raise ValueError('F must be a nonzero nonnegative strictly upper-triangular cycle matrix on param')
    return grid, F


def _method(method, discrete=False):
    method = 'gpd,ml' if method is None else method.lower()
    if ',' not in method:
        method += ',ml'
    shape, estimator = method.split(',', 1)
    allowed = {'gpd': {'ml', 'ml0', 'mom', 'pwm', 'pkd'}, 'exp': {'ml'}, 'ray': {'ml'}}
    if discrete:
        allowed['exp'] |= {'mld', 'ls1', 'ls2', 'ls3', 'ls4', 'wls1', 'wls2', 'wls3', 'wls4'}
    if shape not in allowed or estimator not in allowed[shape]:
        raise ValueError('unsupported tail family/estimator: '+method)
    return shape, estimator, method


def _sf(x, k, s):
    return stats.genpareto.sf(np.asarray(x), -k, scale=s)


def _covariance(k, s, n, method):
    if method in ('ml', 'ml0'):
        return ((1-k)/n*np.array([[1-k, s], [s, 2*s*s]])
                if k < .5 else np.full((2, 2), np.nan))
    if method == 'mom' and k > -.25:
        factor = (1+k)**2/((1+2*k)*(1+3*k)*(1+4*k)*n)
        return factor*np.array([[(1+2*k)**2*(1+k+6*k*k), s*(1+2*k)*(1+4*k+12*k*k)],
                                [s*(1+2*k)*(1+4*k+12*k*k), 2*s*s*(1+6*k+12*k*k)]])
    if method == 'pwm' and k > -.5:
        factor = 1/((1+2*k)*(3+2*k)*n)
        cross = s*(2+k)*(2+6*k+7*k*k+2*k**3)
        return factor*np.array([[(1+k)*(2+k)**2*(1+k+2*k*k), cross],
                                [cross, s*s*(7+18*k+11*k*k+2*k**3)]])
    return np.full((2, 2), np.nan)


def _gpd_fit(values, weights, method='ml'):
    """Weighted source MOM/PWM/Pickands and profile-likelihood estimation."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    n = weights.sum()
    mean = np.average(values, weights=weights)
    if len(values) < 2 or n <= 1 or values[0] < 0 or mean <= 0:
        raise ValueError('GPD fitting requires at least two distinct positive tail observations and mass >1')
    variance = np.dot(weights, (values-mean)**2)/(n-1)
    if variance <= 0:
        raise ValueError('a constant tail cannot identify GPD shape and scale')
    ratio = mean*mean/variance
    mom = ((ratio-1)/2, mean*(ratio+1)/2)
    used = method
    if method == 'mom':
        k, s = mom
    elif method == 'pwm':
        before = np.r_[0, np.cumsum(weights)[:-1]]
        a1 = np.dot(values, weights*(n-.15-before-.5*weights))/(n*n)
        denominator = mean-2*a1
        if denominator <= 0:
            raise ValueError('PWM equations have no positive scale for this tail')
        k, s = mean/denominator-2, 2*mean*a1/denominator
    elif method == 'pkd':
        if not np.all(weights == np.floor(weights)) or n < 4:
            raise ValueError('Pickands estimation requires integer frequencies totaling at least four')
        desc, cumulative = values[::-1], np.cumsum(weights[::-1])
        maximum = int(n//4)
        candidates = np.unique(np.r_[(1, *(np.floor(cumulative/d)+1 for d in (1, 2, 4)))]).astype(int)
        candidates = candidates[(candidates >= 1) & (candidates <= maximum)]
        before = np.r_[0, np.cumsum(weights)[:-1]]
        left, right = (before+.5)/n, (np.cumsum(weights)-.5)/n
        best = None
        for rank in candidates:
            x1, x2, x4 = desc[np.searchsorted(cumulative, [rank, 2*rank, 4*rank])]
            if x1 <= x2 or x2 <= x4:
                continue
            shape = -np.log((x1-x2)/(x2-x4))/np.log(2)
            scale = ((x2-x4)/np.log(2) if abs(shape) < 1e-4
                     else (x2-x4)*shape/(-np.expm1(-shape*np.log(2))))
            F = 1-_sf(values, shape, scale)
            distance = max(np.max(abs(F-left)), np.max(abs(F-right)))
            if best is None or distance < best[0]:
                best = distance, shape, scale
        if best is None:
            raise ValueError('Pickands order-statistic spacings are all zero')
        _, k, s = best
    elif method in ('ml', 'ml0'):
        from ._remaining import wgpdfit_mld
        positive = values[values > 0]
        smallest, largest = positive[0], values[-1]
        t_lower = 2*(smallest-mean)/(smallest*smallest)
        lower = np.log(max(1/largest-t_lower, 1e-6/mean))
        upper = np.log(1e-6/mean)
        grid = np.linspace(lower, upper, 160)
        data = np.column_stack((values, weights))
        score = np.array([wgpdfit_mld(x, data) for x in grid])
        roots = []
        for a, b, fa, fb in zip(grid[:-1], grid[1:], score[:-1], score[1:]):
            if np.signbit(fa) == np.signbit(fb):
                continue
            root = optimize.brentq(lambda x: wgpdfit_mld(x, data), min(a, b), max(a, b))
            _, shape, scale = wgpdfit_mld(root, data, full_output=True)
            if scale > 0 and shape < 1 and np.all(1-shape*values/scale > 0):
                likelihood = np.dot(weights, stats.genpareto.logpdf(values, -shape, scale=scale))
                roots.append((likelihood, shape, scale))
        if roots:
            _, k, s = max(roots)
        else:
            # This is the documented fallback in cmat2extralc.m, rather than
            # a failed numerical root or an invented tail with zero shape.
            k, s = mom
            used = 'mom (no interior ML root)'
    else:
        raise ValueError('unknown GPD estimator')
    if not np.isfinite(k) or not np.isfinite(s) or s <= 0:
        raise ValueError('tail estimation did not produce a positive finite scale')
    covariance = _covariance(k, s, n, 'mom' if used.startswith('mom') else method)
    return dict(k=float(k), s=float(s), cov=covariance, estimator=used, n=float(n))


def _qq_mse(values, weights, estimate, offset=0):
    p = (np.cumsum(weights)-.5*weights)/weights.sum()
    if estimate['family'] == 'ray':
        predicted = np.sqrt(offset*offset-estimate['s']**2*np.log1p(-p))-offset
    else:
        predicted = stats.genpareto.ppf(p, -estimate.get('k', 0), scale=estimate['s'])
    return float(np.average((values-predicted)**2, weights=weights))


def _tail_sample(lc, threshold):
    # The source retains record-increasing intensities when traversing a
    # possibly nonmonotone observed tail from its far end toward the threshold.
    retained, record = [], -np.inf
    for level, count in lc[lc[:, 0] > threshold][::-1]:
        if count > record:
            retained.append((level, count))
            record = count
    if not retained:
        raise ValueError('threshold has no observed tail levels')
    retained = np.asarray(retained)
    weights = np.diff(np.r_[0, retained[:, 1]])
    keep = weights > 0
    values = retained[keep, 0]-threshold
    weights = weights[keep]
    order = np.argsort(values)
    if not len(weights) or weights.sum() <= 0:
        raise ValueError('threshold has no positive tail crossing observations')
    return values[order], weights[order]


def _continuous_tail(lc, threshold, method, offset, excess_grid):
    family, estimator, method = _method(method)
    x, weights = _tail_sample(lc, threshold)
    n = weights.sum()
    if family == 'gpd':
        estimate = _gpd_fit(x, weights, estimator)
        survival = _sf(excess_grid, estimate['k'], estimate['s'])
    elif family == 'exp':
        scale = float(np.average(x, weights=weights))
        estimate = dict(k=0., s=scale, cov=scale*scale/n, estimator=estimator, n=float(n))
        survival = np.exp(-excess_grid/scale)
    else:
        scale = float(np.sqrt(np.average((x+offset)**2-offset**2, weights=weights)))
        estimate = dict(s=scale, cov=scale*scale/(4*n), estimator=estimator, n=float(n))
        survival = np.exp(-((excess_grid+offset)**2-offset**2)/(scale*scale))
    estimate.update(family=family, method=method, threshold=float(threshold), offset=float(offset),
                    lcu=float(np.interp(threshold, lc[:, 0], lc[:, 1])+1),
                    observations=np.column_stack((x, weights)))
    if estimate.get('k', 0) > 0:
        estimate['UpperLimit'] = threshold+estimate['s']/estimate['k']
    fitted = estimate['lcu']*survival
    result = np.column_stack((excess_grid+threshold, fitted))
    if family == 'gpd' and np.all(np.isfinite(estimate['cov'])):
        eig, basis = np.linalg.eigh(estimate['cov'])
        angles = np.linspace(0, 2*np.pi, 17)
        circle = np.sqrt(-2*np.log(.1))*np.array([np.sin(angles), np.cos(angles)])
        ellipse = np.array([estimate['k'], estimate['s']])[:, None]+basis@(np.sqrt(np.maximum(eig, 0))[:, None]*circle)
        curves = np.array([_sf(excess_grid, k, s) for k, s in ellipse.T if s > 0])
        if len(curves):
            uncertainty = 1.64*np.sqrt(estimate['lcu'])
            lower = np.minimum(fitted, max(estimate['lcu']-uncertainty, 0)*curves.min(axis=0))
            upper = np.maximum(fitted, (estimate['lcu']+uncertainty)*curves.max(axis=0))
            result = np.column_stack((result, lower, upper))
    return result, estimate


def extralc(lc, u, method='gpd,ml', plotflag=0, *, tail_grid=None, full_output=False):
    """Extrapolate lower/upper crossing tails with GPD, exponential or Rayleigh.

    ``u=[low,high]`` must bracket a crossing-spectrum maximum.  Default tail
    distances are 0:.01:4, as in the MATLAB source; ``tail_grid`` supplies a
    nonnegative increasing distance grid.  Returns ``{'Low','High'}`` arrays;
    full output adds parameter dictionaries with WAFO shape/scale conventions.
    GPD arrays include 90% parameter/Poisson envelopes when covariance exists.
    The source's threshold count +1 adjustment is retained.  Exponential
    variance is corrected from s/n to s**2/n and invalid negative Poisson
    lower limits are clipped at zero.  Plotting is off by default.
    """
    lc = _crossings(lc)
    u = np.asarray(u, dtype=float)
    peak = lc[np.argmax(lc[:, 1]), 0]
    if u.shape != (2,) or not np.all(np.isfinite(u)) or not lc[0, 0] < u[0] <= peak <= u[1] < lc[-1, 0] or u[0] >= u[1]:
        raise ValueError('thresholds must lie inside the observed range and bracket its crossing maximum')
    grid = np.arange(401)/100 if tail_grid is None else np.asarray(tail_grid, dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.all(np.isfinite(grid)) or grid[0] < 0 or np.any(np.diff(grid) <= 0):
        raise ValueError('tail_grid must contain increasing nonnegative distances')
    high, eh = _continuous_tail(lc, u[1], method, u[1]-peak, grid)
    reflected = np.column_stack((-lc[::-1, 0], lc[::-1, 1]))
    low, el = _continuous_tail(reflected, -u[0], method, peak-u[0], grid)
    low = np.column_stack((-low[::-1, 0], low[::-1, 1:]))
    el['threshold'] = float(u[0])
    if 'UpperLimit' in el:
        el['LowerLimit'] = -el.pop('UpperLimit')
    estimates = dict(High=eh, Low=el, method=_method(method)[2], figures=())
    if plotflag:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.semilogx(lc[:, 1], lc[:, 0], label='Observed')
        ax.semilogx(high[:, 1], high[:, 0], label='Upper tail')
        ax.semilogx(low[:, 1], low[:, 0], label='Lower tail')
        ax.set(xlabel='Crossings', ylabel='Level', title='Extrapolated crossing spectrum')
        ax.legend()
        estimates['figures'] = (fig,)
    result = dict(High=high, Low=low)
    return (result, estimates) if full_output else result


def _discrete_tail(counts, offset, method):
    family, estimator, method = _method(method, True)
    counts = np.asarray(counts, dtype=float)
    # Differencing a tail after removing noncrossing cycles is equivalent to
    # the source's reversed cumulative dN construction, including half bins.
    weights = counts[:-1]-counts[1:]
    tolerance = 1e-10*max(1, counts.max())
    if np.any(weights < -tolerance):
        raise ValueError('conditioned crossing tail must be nonincreasing')
    keep = weights > tolerance
    x, weights = np.arange(len(weights))[keep].astype(float), weights[keep]
    if not len(x) or weights.sum() <= 0:
        raise ValueError('tail has no positive crossing decrements')
    n, xf = weights.sum(), np.arange(len(counts), dtype=float)
    diagnostics, mse = {}, None
    if family == 'gpd':
        estimate = _gpd_fit(x+.5, weights, estimator)
        survival = _sf(xf, estimate['k'], estimate['s'])
    elif family == 'ray':
        scale = float(np.sqrt(np.average((x+offset+.5)**2-offset*offset, weights=weights)))
        estimate = dict(s=scale, cov=scale*scale/(4*n), estimator=estimator, n=float(n))
        survival = np.exp(-((xf+offset)**2-offset*offset)/scale**2)
    elif estimator in ('ml', 'mld'):
        mean = float(np.average(x+(.5 if estimator == 'ml' else 0), weights=weights))
        scale = mean if estimator == 'ml' else (1/np.log1p(1/mean) if mean > 0 else 0.)
        if scale <= 0:
            raise ValueError('a tail concentrated in the first bin has no positive discrete exponential scale')
        estimate = dict(k=0., s=scale, cov=np.diag([0., scale*scale/n]), estimator=estimator, n=float(n))
        survival = np.exp(-xf/scale)
    else:
        weighted = estimator.startswith('w')
        selector = int(estimator[-1])
        cumulative = np.cumsum(weights[::-1])[::-1]
        y = np.log(cumulative)
        if selector in (1, 4):
            y -= np.log(cumulative[0])
        X = {1: x[:, None], 2: np.column_stack((np.ones(len(x)), x)),
             3: np.column_stack((np.ones(len(x)), x, x*x)),
             4: np.column_stack((x, x*x))}[selector]
        w = weights if weighted else np.ones(len(x))
        design, response = np.sqrt(w[:, None])*X, np.sqrt(w)*y
        theta, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
        if rank < X.shape[1]:
            raise ValueError('too few distinct tail decrements for this regression')
        linear = theta[0] if selector in (1, 4) else theta[1]
        quadratic = theta[-1] if selector in (3, 4) else 0.
        logarithm = linear*xf+quadratic*xf*xf
        if np.any(np.diff(logarithm) > 1e-10) or logarithm.max() > 1e-10:
            raise ValueError('unconstrained source regression yields an increasing survival tail')
        estimate = dict(k=0., s=float(-1/linear) if linear != 0 else np.inf,
                        cov=np.full((2, 2), np.nan), estimator=estimator, n=float(n))
        if selector in (3, 4):
            estimate['s2'] = float(-1/quadratic) if quadratic != 0 else np.inf
        if selector in (2, 3):
            estimate['lcu'] = float(np.exp(theta[0]))
        survival = np.exp(logarithm)
        mse = float(np.dot(w, (X@theta-y)**2)/w.sum())
        diagnostics = dict(X=X, Y=y, weights=w, theta=theta)
    estimate.update(family=family, method=method, offset=float(offset), observations=np.column_stack((x+.5, weights)))
    estimate.setdefault('lcu', float(counts[0]))
    if mse is None:
        mse = _qq_mse(x+.5, weights, estimate, offset)
    fitted = estimate['lcu']*survival
    fitted[-1] = 0  # No crossings beyond the matrix's largest discrete level.
    if estimate.get('k', 0) > 0:
        estimate['UpperLimit'] = estimate['s']/estimate['k']
    return fitted, estimate, diagnostics, mse


def cmat2extralc(param, F, u, method='gpd,ml', plotflag=0, *, full_output=False):
    """Extrapolate cycle-matrix crossing tails using the source's bin estimators.

    Conditions on cycles spanning the low/high thresholds, differences the
    resulting monotone tails and fits weighted half-bin exceedances.  Supports
    GPD ML/MOM/PWM/Pickands, exponential ML/discrete ML and LS/WLS selectors
    1--4, and shifted-Rayleigh ML.  Results include ``Low,High,lc,D``; ``D``
    uses zero-based classes.  Full output is ``(result,estimates,internal,MSE)``.
    Estimates retain source ``D`` parameters and add physical ``Low/High``
    scales.  MSE contains actual weighted-midrank QQ or log-regression error,
    replacing the source's unimplemented zero ML diagnostics.  Covariance
    estimates use corrected scale-squared units.  No sample expansion occurs.
    """
    from . import cmat2lc
    grid, F = _matrix(param, F)
    u = np.asarray(u, dtype=float)
    n, step = len(grid), grid[1]-grid[0]
    if u.shape != (2,) or not np.all(np.isfinite(u)) or not grid[0] < u[0] < u[1] < grid[-1]:
        raise ValueError('ordered thresholds must lie strictly inside the grid')
    high = int(np.searchsorted(grid, u[1], side='right'))
    low = int(np.searchsorted(grid, u[0], side='left')-1)
    if high >= n-1 or low < 1:
        raise ValueError('leave at least two grid levels beyond each threshold')
    conditioned = F.copy()
    conditioned[high:] = 0
    conditioned[:, :low+1] = 0
    lc = cmat2lc(param, conditioned)
    peak = int(np.argmax(lc[:, 1]))
    if not low < peak < high:
        raise ValueError('thresholds must bracket the conditioned crossing-spectrum maximum')
    fh, eh, dh, mh = _discrete_tail(lc[high:, 1], high-peak, method)
    fl, el, dl, ml = _discrete_tail(lc[:low+1, 1][::-1], peak-low, method)
    indices = np.arange(n)
    D = dict(High=np.column_stack((indices[high:], fh)),
             Low=np.column_stack((indices[:low+1], fl[::-1])))
    D['lc'] = np.column_stack((indices, lc[:, 1].copy()))
    D['lc'][:low+1, 1], D['lc'][high:, 1] = fl[::-1], fh
    eh['threshold'], el['threshold'] = high, low
    if 'UpperLimit' in eh:
        eh['UpperLimit'] += high
    if 'UpperLimit' in el:
        el['LowerLimit'] = low-el.pop('UpperLimit')
    result = {key: np.column_stack((grid[value[:, 0].astype(int)], value[:, 1])) for key, value in D.items()}
    result['D'] = D
    def physical(estimate):
        out = dict(estimate)
        out['s'] *= step
        out['threshold'] = grid[int(estimate['threshold'])]
        out['offset'] *= step
        if 's2' in out:
            out['s2'] *= step*step
        if np.shape(out['cov']) == (2, 2):
            out['cov'] = np.diag([1, step])@out['cov']@np.diag([1, step])
        else:
            out['cov'] *= step*step
        for name in ('UpperLimit', 'LowerLimit'):
            if name in out:
                out[name] = grid[0]+estimate[name]*step
        out['observations'] = estimate['observations']*np.array([step, 1])
        return out
    estimates = dict(D=dict(High=eh, Low=el), High=physical(eh), Low=physical(el),
                     method=_method(method, True)[2], figures=())
    internal = dict(lc0=cmat2lc(param, F), lc1=lc, F1=conditioned,
                    diagnostics=dict(High=dh, Low=dl), threshold_indices=np.array([low, high]))
    mse = dict(High=mh, Low=ml)
    if plotflag:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1)
        for ax, key in zip(axes, ('High', 'Low')):
            ax.semilogy(lc[:, 0], lc[:, 1], label='Conditioned crossings')
            ax.semilogy(*result[key].T, label='Fitted '+key.lower()+' tail')
            ax.set(xlabel='Level', ylabel='Crossings', title=_method(method, True)[2])
            ax.legend()
        estimates['figures'] = (fig,)
    return (result, estimates, internal, mse) if full_output else result


def rfmextrapolate(F, Pin=None, plotflag=0, *, full_output=False):
    """Source extreme-tail/kernel rainflow combination on zero-based classes.

    ``Pin`` accepts ``method,u_lev,LCfrac,h,beta,Lim,LimRelDam,param,PL``.
    Manual ``u_lev`` and ``Lim.min/max`` are zero-based class indices; range
    limits remain class distances.  Automatic thresholds require at least
    eight distinct crossing-tail jumps, as the source needs three tail jumps
    on each side.  ML uses a documented moment fallback when no interior root
    exists.  No normalization forces total mass: extrapolation intentionally
    changes extreme-cycle frequencies, while kernel smoothing preserves mass.
    Full output is ``(Fest,Pout,Fextreme,Fsmooth,Fest0)`` as in MATLAB.
    """
    from . import (cmat2lc, cmat2amp, lc2rfmextreme, smoothcmat, smoothcmat_hnorm,
                   cmatcombine)
    from ..damage import cmat2dmat
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError('F must be a square cycle matrix')
    n = len(F)
    options = dict(param=(0, n-1, n), method='gpd,ml', u_lev=None, LCfrac=None,
                   h=-1., Lim=None, LimRelDam=None, beta=7., PL=[10, 30, 50, 70, 90, 99, 99.9, 99.99, 99.999])
    if Pin is not None:
        unknown = set(Pin)-set(options)
        if unknown:
            raise TypeError('unknown extrapolation options: '+', '.join(sorted(unknown)))
        options.update(Pin)
    grid, F = _matrix(options['param'], F)
    if not np.isfinite(options['beta']) or options['beta'] <= 0 or plotflag not in (0, 1, 2):
        raise ValueError('beta must be positive and plotflag zero, one or two')
    if options['u_lev'] is None:
        fraction = .05 if options['LCfrac'] is None else options['LCfrac']
        if not np.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError('LCfrac must lie strictly between zero and one')
        options['LCfrac'] = fraction
        current, previous, visited = F.copy(), None, set()
        for _ in range(2*n+1):
            counts = cmat2lc((0, n-1, n), current)[:, 1]
            peak = int(np.argmax(counts))
            if counts[peak] <= 0:
                raise ValueError('no crossings remain for threshold selection')
            active = np.flatnonzero(counts > fraction*counts[peak])
            jumps = np.r_[np.diff(np.r_[0, counts[:peak]]), 0,
                          np.diff(np.r_[0, counts[peak+1:][::-1]])[::-1]]
            changes = np.flatnonzero(jumps)
            if len(changes) < 8:
                raise ValueError('automatic thresholds need eight crossing jumps; specify u_lev manually')
            low, high = max(active[0], changes[3]), min(active[-1], changes[-4])
            if low >= high:
                raise ValueError('automatic tail thresholds overlap; specify u_lev manually')
            pair = int(low), int(high)
            if pair == previous:
                break
            if pair in visited:
                raise ValueError('automatic threshold iteration cycles; specify u_lev manually')
            visited.add(pair)
            previous = pair
            current = F.copy()
            current[high+1:] = 0
            current[:, :low] = 0
        else:
            raise ValueError('automatic threshold iteration did not converge')
        options['u_lev'] = np.array(pair)
    thresholds = np.asarray(options['u_lev'], dtype=float)
    if thresholds.shape != (2,) or not np.all(np.isfinite(thresholds)) or np.any(thresholds != np.floor(thresholds)):
        raise ValueError('u_lev must contain two zero-based integer class indices')
    fitted, estimates, _, _ = cmat2extralc((0, n-1, n), F, thresholds, options['method'], full_output=True)
    padded = np.vstack(([-1, 0], fitted['lc'], [n, 0]))
    extreme = lc2rfmextreme(padded)[1:-1, 1:-1]
    h = options['h']
    if not np.isscalar(h) or not np.isfinite(h) or (h < 0 and h != -1):
        raise ValueError('h must be nonnegative or -1 for automatic bandwidth')
    first_diagonal = next(k for k in range(1, n) if np.diag(F, k).sum() > 0)
    if h == -1:
        h = .5*smoothcmat_hnorm(F, first_diagonal-1)
    smooth = F.copy() if h == 0 else smoothcmat(F, 1, h, first_diagonal-1)
    options['h'] = h
    observed_damage = cmat2amp((0, n-1, n), cmat2dmat((0, n-1, n), F, options['beta']))[:, 1]
    extreme_damage = cmat2amp((0, n-1, n), cmat2dmat((0, n-1, n), extreme, options['beta']))[:, 1]
    relative = np.divide(extreme_damage, observed_damage, out=np.full(n, 1e10), where=observed_damage > 0)
    relative[(observed_damage == 0) & (extreme_damage == 0)] = 0
    limits = {} if options['Lim'] is None else dict(options['Lim'])
    if 'range' not in limits:
        cutoff = .95 if options['LimRelDam'] is None else options['LimRelDam']
        if not np.isfinite(cutoff) or cutoff < 0:
            raise ValueError('LimRelDam must be nonnegative')
        options['LimRelDam'] = cutoff
        candidates = np.flatnonzero((observed_damage > 0) & (extreme_damage > 0) & (relative > cutoff))
        limits['range'] = int(candidates[0]+1) if len(candidates) else n
    peak = int(np.argmax(fitted['lc'][:, 1]))
    gap = int(np.floor(.02*n))+1
    limits.setdefault('min', peak-gap)
    limits.setdefault('max', peak+gap)
    combined, limits, _, _ = cmatcombine(extreme, smooth, limits, full_output=True)
    rows, cols = np.flatnonzero(extreme.sum(axis=1) > 0), np.flatnonzero(extreme.sum(axis=0) > 0)
    if not len(rows) or not len(cols):
        raise ValueError('fitted crossing tails yield no extreme-cycle support')
    imin, jmax = int(rows[0]), int(cols[-1])
    result = combined.copy()
    result[:imin] = 0
    result[:, jmax+1:] = 0
    options.update(Lim=limits, imin=imin, jmax=jmax, estimates=estimates,
                   crossing=fitted['lc'], relative_damage=relative, figures=())
    if plotflag:
        import matplotlib.pyplot as plt
        from . import cmatplot
        percentages = np.asarray(options['PL'], dtype=float)
        if percentages.ndim != 1 or np.any((percentages <= 0) | (percentages >= 100)):
            raise ValueError('PL contour percentages must lie strictly between zero and 100')
        sorted_cells = np.sort(result.ravel())[::-1]
        cumulative = np.cumsum(sorted_cells)/sorted_cells.sum()*100
        contour_levels = np.unique(np.interp(percentages, cumulative, sorted_cells))
        contour_levels = contour_levels[(contour_levels > result.min()) & (contour_levels < result.max())]
        fig, ax = plt.subplots()
        if len(contour_levels):
            ax.contour(grid, grid, F.T, levels=contour_levels, colors='b')
            ax.contour(grid, grid, result.T, levels=contour_levels, colors='r')
        ax.set(xlabel='Minimum', ylabel='Maximum', title='Observed (blue) and extrapolated (red) rainflow')
        figures = [fig]
        if plotflag == 2:
            fig, axes = plt.subplots(2, 2)
            for ax, matrix, title in zip(axes.ravel(), (F, extreme, smooth, result),
                                         ('Observed', 'Extreme approximation', 'Kernel smoothed', 'Combined')):
                cmatplot(grid, grid, matrix, 3, ax=ax)
                ax.set_title(title)
            figures.append(fig)
            fig, ax = plt.subplots()
            ax.semilogy(grid, cmat2lc(options['param'], F)[:, 1], label='Observed')
            ax.semilogy(grid, fitted['lc'][:, 1], label='Tail extrapolation')
            ax.set(xlabel='Level', ylabel='Crossings')
            ax.legend()
            figures.append(fig)
        options['figures'] = tuple(figures)
    return (result, options, extreme, smooth, combined) if full_output else result


def tpextrapolate(tp, N, Pin=None, plotflag=0, *, seed=None, conservative=True, full_output=False):
    """Generate N blocks with rank-preserving extreme turning-point simulation.

    Implements the source's sorted exponential/GPD exceedances, conservative
    envelope and optional amplitude limits, then rainflow-filters block joins.
    ``Pin`` accepts ``method,LCfrac,u_lev,lim``.  Manual thresholds use physical
    values.  ``N<=0`` estimates parameters without simulation.  Empty tails are
    supported; GPD fitting needs at least two distinct exceedances per nonempty
    tail.  Timed input receives consecutive block time offsets, fixing repeated
    MATLAB timestamps.  Full output is ``(tpe,Pout,I,tpe0)`` with zero-based
    extrapolated indices; ``tpe0`` precedes clipping/filtering.  A seed makes
    simulation reproducible.  Conservative replacement is enabled by default.
    """
    from . import rfcfilter, tp2lc
    a = np.asarray(tp, dtype=float)
    vector = a.ndim == 1
    rows = a[:, None] if vector else a.copy()
    if (rows.ndim != 2 or rows.shape[1] not in (1, 2) or len(rows) < 2
            or not np.all(np.isfinite(rows)) or np.any(np.diff(rows[:, -1]) == 0)
            or np.any(np.diff(rows[:, -1])[:-1]*np.diff(rows[:, -1])[1:] >= 0)):
        raise ValueError('tp must contain a finite strictly alternating sequence of turning points')
    if rows.shape[1] == 2 and np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError('turning-point times must increase')
    if not np.isscalar(N) or not np.isfinite(N) or int(N) != N or plotflag not in (0, 1, 2):
        raise ValueError('N must be an integer and plotflag zero, one or two')
    options = dict(method='exp', LCfrac=None, u_lev=None, lim=None)
    if Pin is not None:
        unknown = set(Pin)-set(options)
        if unknown:
            raise TypeError('unknown turning-point extrapolation options: '+', '.join(sorted(unknown)))
        options.update(Pin)
    method = options['method'].lower()
    if method not in ('exp', 'gpd'):
        raise ValueError('method must be exp or gpd')
    if options['u_lev'] is None:
        crossing = tp2lc(rows)
        maximum = crossing[:, 1].max() if len(crossing) else 0
        fraction = 1/np.sqrt(maximum) if options['LCfrac'] is None and maximum > 0 else options['LCfrac']
        if fraction is None or not np.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError('automatic thresholds need more crossings; specify u_lev manually')
        selected = crossing[crossing[:, 1] > maximum*fraction, 0]
        if len(selected) < 2:
            raise ValueError('automatic thresholds have no distinct tail levels; specify u_lev manually')
        options.update(LCfrac=fraction, u_lev=np.array([selected[0], selected[-1]]))
    thresholds = np.asarray(options['u_lev'], dtype=float)
    if thresholds.shape != (2,) or not np.all(np.isfinite(thresholds)) or thresholds[0] >= thresholds[1]:
        raise ValueError('u_lev must contain two ordered finite thresholds')
    values = rows[:, -1]
    maxima = np.arange(int(values[0] < values[1]), len(rows), 2)
    minima = np.arange(int(values[0] > values[1]), len(rows), 2)
    imax, imin = maxima[values[maxima] > thresholds[1]], minima[values[minima] < thresholds[0]]
    tails = dict(max=values[imax]-thresholds[1], min=thresholds[0]-values[imin])
    fits = {}
    for name, excess in tails.items():
        options['a_'+name] = float(excess.mean()) if len(excess) else 0.
        options['n_'+name] = len(excess)
        if method == 'gpd':
            fits[name] = _gpd_fit(excess, np.ones(len(excess)), 'ml') if len(excess) else None
            options['gpd_'+name] = np.array([fits[name]['k'], fits[name]['s']]) if fits[name] else None
            options['cov_'+name] = fits[name]['cov'] if fits[name] else None
            options['ci_k_'+name] = (fits[name]['k']+np.array([-2, 2])*np.sqrt(fits[name]['cov'][0, 0])
                                      if fits[name] else None)
    limits = None if options['lim'] is None else np.asarray(options['lim'], dtype=float)
    if limits is not None and (limits.shape != (2,) or np.any(np.isnan(limits)) or limits[0] >= limits[1]):
        raise ValueError('lim must contain ordered lower/upper bounds')
    rng, blocks = np.random.default_rng(seed), []
    period = rows[-1, 0]-rows[0, 0]+np.median(np.diff(rows[:, 0])) if rows.shape[1] == 2 else 0
    for block in range(max(0, int(N))):
        copied = rows.copy()
        for name, indices, threshold, sign in (('max', imax, thresholds[1], 1), ('min', imin, thresholds[0], -1)):
            if not len(indices):
                continue
            excess = tails[name]
            simulated = (rng.exponential(options['a_'+name], len(indices)) if method == 'exp' else
                         stats.genpareto.ppf(rng.random(len(indices)), -fits[name]['k'], scale=fits[name]['s']))
            ordered = np.argsort(excess, kind='stable')
            replacement = np.sort(simulated)
            if conservative:
                replacement = np.maximum(replacement, excess[ordered])
            copied[indices[ordered], -1] = threshold+sign*replacement
        if rows.shape[1] == 2:
            copied[:, 0] += block*period
        blocks.append(copied)
    raw = np.vstack(blocks) if blocks else np.empty((0, rows.shape[1]))
    bounded = raw.copy()
    if limits is not None:
        bounded[:, -1] = np.clip(bounded[:, -1], *limits)
    result = rfcfilter(bounded, 0, 1)
    if len(result) >= 2:
        maxstart = int(result[0, -1] < result[1, -1])
        ismax = np.arange(len(result)) % 2 == maxstart
        indices = dict(max=np.flatnonzero(ismax & (result[:, -1] > thresholds[1])),
                       min=np.flatnonzero(~ismax & (result[:, -1] < thresholds[0])))
    else:
        indices = dict(max=np.empty(0, int), min=np.empty(0, int))
    options.update(Zmin=tails['min'], Zmax=tails['max'], figures=())
    if plotflag:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2)
        for row, name in enumerate(('max', 'min')):
            excess = np.sort(tails[name])
            axes[row, 0].plot(np.arange(len(excess)), excess, '.')
            axes[row, 0].set(title=name+'imum exceedances', xlabel='Rank', ylabel='Excess')
            if len(excess):
                p = (np.arange(len(excess))+.5)/len(excess)
                quantile = (-options['a_'+name]*np.log1p(-p) if method == 'exp' else
                            stats.genpareto.ppf(p, -fits[name]['k'], scale=fits[name]['s']))
                axes[row, 1].plot(excess, quantile, '.')
                high = max(excess[-1], quantile[-1])
                axes[row, 1].plot([0, high], [0, high], '--')
            axes[row, 1].set(xlabel='Observed excess', ylabel='Fitted quantile', title=method+' QQ plot')
        figures = [fig]
        if plotflag == 2:
            fig, axes = plt.subplots(2, 1)
            center = values.mean()
            for ax, name, source in ((axes[0], 'max', values[maxima]), (axes[1], 'min', -values[minima])):
                signed_center = center if name == 'max' else -center
                levels = np.linspace(signed_center, source.max(), 100)
                mean, error = [], []
                for level in levels:
                    excess = source[source > level]-level
                    average = excess.mean() if len(excess) else 0.
                    mean.append(average)
                    error.append(average/np.sqrt(len(excess)) if len(excess) else 0.)
                mean, error = np.array(mean), np.array(error)
                ax.plot(levels, mean)
                ax.fill_between(levels, np.maximum(mean-2*error, 0), mean+2*error, alpha=.2)
                ax.set(title=name+'imum mean excess', xlabel='Signed threshold', ylabel='Mean excess')
                options['U'+name], options['A'+name] = levels, mean
            figures.append(fig)
        options['figures'] = tuple(figures)
    if vector:
        result, raw = result[:, 0], raw[:, 0]
    return (result, options, indices, raw) if full_output else result
