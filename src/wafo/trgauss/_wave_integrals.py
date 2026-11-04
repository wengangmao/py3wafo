"""Portable Gaussian Rice-integral wave distributions.

These replace the *capabilities* of WAFO's sp2Acdf, sp2mM/sp2mmtpdf,
sp2thpdf, sp2tccpdf and minmax executable families. The construction follows
the Xt/Xd/Xc indicator/Jacobian/conditioning vectors documented in those
Fortran drivers. Covariances come directly from spectral quadrature and RIND
performs the weighted conditional Gaussian integral; no native WAFO code runs.

Path restrictions hold at the requested discrete time grid. This is a
discretization of the continuous-path Rice formula, not a claim of exact
continuous wave distributions. Refine the grid and extend the time horizon to
check convergence. Returned ``err`` estimates cover numerical RIND sampling
only, not time-grid error, omitted tail mass, or spectral quadrature error.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import simpson, trapezoid
from ..spec import Spectrum, _validate
from ._rind import rind


def _spectrum(S):
    if S.get('type', 'freq').lower() != 'freq' or 'w' not in S:
        raise NotImplementedError('Rice wave drivers currently require a 1D angular-frequency spectrum')
    if S.get('tr') is not None and np.size(S.get('tr')):
        raise NotImplementedError('Transformed-Gaussian wave drivers are not implemented; use an untransformed Gaussian spectrum')
    w, s = _validate(S)
    if not np.all(np.isfinite(w)):
        raise ValueError('Frequency grid must be finite')
    moments = np.array([simpson(s*w**i, x=w) for i in (0, 2, 4)])
    if np.any(moments <= 0):
        raise ValueError('Wave distributions require positive m0, m2, and m4')
    return w, s, moments


def _covariance(w, s, times, orders):
    """Cov[X^(p)(ti), X^(q)(tj)] from a one-sided angular-frequency PSD."""
    t, d = np.asarray(times, dtype=float), np.asarray(orders, dtype=int)
    if t.ndim != 1 or t.shape != d.shape or np.any((d < 0) | (d > 2)):
        raise ValueError('Invalid derivative variables')
    delta = t[:, None]-t[None, :]
    order = d[:, None]+d[None, :]
    phase = (d[:, None]-d[None, :])*np.pi/2
    C = simpson(s[None, None, :]*w[None, None, :]**order[..., None]
                *np.cos(delta[..., None]*w+phase[..., None]), x=w, axis=-1)
    return (C+C.T)/2


def _time_grid(paramt, moments):
    mean_period = 2*np.pi*np.sqrt(moments[0]/moments[1])
    paramt = [0, 1.5*mean_period, 13] if paramt is None else np.asarray(paramt, dtype=float)
    if len(paramt) != 3:
        raise ValueError('paramt must be [first_time, last_time, number_of_grid_points]')
    start, stop, count = paramt
    if not np.all(np.isfinite(paramt)) or start < 0 or stop <= 0 or stop < start or count < 3 or count != int(count):
        raise ValueError('Invalid period grid; at least three points and a positive horizon are required')
    t = np.linspace(0, stop, int(count))
    return t, t >= start


def _options(options):
    # Small explicit default grids and integration budgets keep the default
    # calculation inspectable. Users control convergence through these options.
    out = dict(maxpts=2048, replications=8, seed=1729, abseps=1e-5, releps=.02)
    if options is not None:
        if not isinstance(options, dict):
            raise TypeError('options must be a mapping of Python RIND options')
        out.update(options)
    return out


def _rice(w, s, xt, xd, xc, values, lower, upper, options):
    """Xt/Xd/Xc layout used by the original drivers, without dummy variables."""
    variables = [*xt, *xd, *xc]
    C = _covariance(w, s, [a[0] for a in variables], [a[1] for a in variables])
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    lo, up = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    result, error, _, info = rind(C, np.zeros(len(variables)), lo, up,
        xc=values, Nt=len(xt), options=options, full_output=True)
    return np.asarray(result), np.asarray(error), info


def _normalization(moments, level=0, extrema=False):
    m0, m2, m4 = moments
    return (2*np.pi*np.sqrt(m2/m4) if extrema else
            2*np.pi*np.sqrt(m0/m2)*np.exp(level*level/(2*m0)))


def _pdf(f, axes, labels, error, title, extra=None):
    result = Spectrum(f=np.asarray(f), x=[np.asarray(x) for x in axes],
        labx=list(labels), err=np.asarray(error), title=title, indexing='ij',
        method='discrete-grid Gaussian Rice integral with randomized Sobol RIND',
        note='err excludes path-grid, time-tail, and spectral-quadrature errors; no automatic normalization is applied')
    if extra:
        result.update(extra)
    return result


def _amplitudes(h, scale):
    h = np.linspace(0, 4*scale, 7) if h is None else np.atleast_1d(h).astype(float)
    if h.ndim != 1 or not len(h) or np.any(np.isnan(h)) or np.any(h < 0):
        raise ValueError('Amplitudes must be a nonempty vector of nonnegative values')
    return h


def _half_periods(S, level, definition, paramt, h, options):
    w, s, moments = _spectrum(S)
    t, mask = _time_grid(paramt, moments)
    cap = _amplitudes(h, np.sqrt(moments[0]))
    crest = definition.lower() == 'tc'
    if definition.lower() not in ('tc', 'tt'):
        raise NotImplementedError('Only temporal Tc/Tt half-wave definitions are implemented')
    f, err = np.zeros((len(t), len(cap))), np.zeros((len(t), len(cap)))
    opt, norm = _options(options), _normalization(moments, level)
    for j in range(1, len(t)):
        interior = [(x, 0) for x in t[1:j]]
        derivatives = [(0, 1), (t[j], 1)]
        condition = [(0, 0), (t[j], 0)]
        for k, height in enumerate(cap):
            if height == 0:
                continue
            lower = [level if crest else level-height]*len(interior)+([0, -np.inf] if crest else [-np.inf, 0])
            upper = [level+height if crest else level]*len(interior)+([np.inf, 0] if crest else [0, np.inf])
            value, error, _ = _rice(w, s, interior, derivatives, condition,
                [level, level], lower, upper, opt)
            f[j, k], err[j, k] = value[0]*norm, error[0]*norm
    return t, mask, cap, f, err


def spec2Acdf(spec, utc=0, definition='Tc', paramt=None, h=None, *, options=None):
    """Crest/trough amplitude CDF from finite-horizon half-wave Rice integrals.

    Replaces sp2Acdf51/70 capabilities for an untransformed temporal Gaussian
    process. The signature uses Python ``definition`` instead of MATLAB ``def``.
    Returns an attribute-access mapping: ``x=[h]``, ``f=CDF``, plus the period
    subdensity, period grid, and numerical error estimates. The result is not
    clipped or renormalized; excessive mass indicates insufficient path-grid
    resolution. ``utc=None`` chooses zero, the most frequently crossed level.
    """
    level = 0.0 if utc is None else float(utc)
    t, _, cap, density, err = _half_periods(spec, level, definition, paramt, h, options)
    # Integrate the entire supplied horizon even if paramt starts above zero.
    cdf = trapezoid(density, t, axis=0)
    error = trapezoid(err, t, axis=0)
    return _pdf(cdf, [cap], ['crest amplitude' if definition.lower() == 'tc' else 'trough amplitude'],
        error, 'Gaussian wave amplitude CDF', dict(t=t, period_subdensity=density, horizon=float(t[-1])))


def spec2mmtpdf(spec, utc=0, definition='Mm', paramt=None, paramu=None, options=None, *, bound=0):
    """Joint maximum/following-minimum density, optionally with elapsed period.

    Supported definitions are ``Mm`` and ``MmTMm``. Condition on X'(0)=X'(T)=0,
    X(0)=M, X(T)=m; require X'<0 at interior nodes and opposite endpoint
    curvature signs. Normalize by the mean local-maximum intensity as in
    sp2mmtpdf.f. Output axes are [minimum,maximum] or [minimum,maximum,time].
    Other level-separated/rainflow/transformed/space definitions are rejected.
    """
    if bound:
        raise NotImplementedError('Continuous-path upper/lower bounds are not implemented')
    if definition.lower() not in ('mm', 'mmtmm'):
        raise NotImplementedError('Only Mm and MmTMm are implemented; other native driver branches remain pending')
    w, s, moments = _spectrum(spec)
    t, mask = _time_grid(paramt, moments)
    if paramu is None:
        paramu = [-4*np.sqrt(moments[0]), 4*np.sqrt(moments[0]), 7]
    lo, hi, n = np.asarray(paramu, dtype=float)
    if not np.all(np.isfinite(paramu)) or hi <= lo or n < 2 or n != int(n):
        raise ValueError('paramu must be [minimum_level, maximum_level, count]')
    u = np.linspace(lo, hi, int(n))
    pairs = [(i,j) for j in range(len(u)) for i in range(j)]
    f = np.zeros((len(u),len(u),len(t)))
    err = np.zeros_like(f)
    norm, opt = _normalization(moments, extrema=True), _options(options)
    values = np.asarray([[0,0,u[j],u[i]] for i,j in pairs]).T
    for k in range(1,len(t)):
        interior = [(x,1) for x in t[1:k]]
        derivatives = [(0,2),(t[k],2)]
        condition = [(0,1),(t[k],1),(0,0),(t[k],0)]
        lower = [-np.inf]*len(interior)+[-np.inf,0]
        upper = [0]*len(interior)+[0,np.inf]
        value,error,_ = _rice(w,s,interior,derivatives,condition,values,lower,upper,opt)
        for index,(i,j) in enumerate(pairs):
            f[i,j,k],err[i,j,k] = value[index]*norm,error[index]*norm
    if definition.lower() == 'mm':
        return _pdf(trapezoid(f,t,axis=-1),[u,u],['minimum','maximum'],trapezoid(err,t,axis=-1),
            'Maximum and following minimum',dict(horizon=float(t[-1])))
    return _pdf(f[...,mask],[u,u,t[mask]],['minimum','maximum','period'],err[...,mask],
        'Maximum, following minimum, and elapsed period')


def spec2thpdf(spec, utc=0, definition='Tc', paramt=None, h=None, options=None, *, plotflag=0):
    """Half-wave period/amplitude density from sp2thpdf's Rice construction.

    Supports Tc/Tt, TcAc/TtAt and TcfAc/TtbAt. Full half-wave/amplitude
    densities integrate the interior crest/trough time over the same discrete
    grid; front/back densities use one crossing and one extremum. TAc/TAt,
    spatial definitions and transformed Gaussian processes remain unsupported.
    """
    if plotflag:
        raise NotImplementedError('Plot the returned density explicitly; automatic legacy plotting is not implemented')
    d = definition.lower()
    level = 0.0 if utc is None else float(utc)
    if d in ('tc','tt'):
        t,mask,_,f,e = _half_periods(spec,level,d,paramt,[np.inf],options)
        return _pdf(f[mask,0],[t[mask]],['period'],e[mask,0],'Gaussian half-wave period')
    if d not in ('tcac','ttat','tcfac','ttbat'):
        raise NotImplementedError('Supported definitions: Tc,Tt,TcAc,TtAt,TcfAc,TtbAt')
    crest = d in ('tcac','tcfac')
    front = d in ('tcfac','ttbat')
    w,s,moments = _spectrum(spec)
    t,mask = _time_grid(paramt,moments)
    cap = _amplitudes(h,np.sqrt(moments[0]))
    if not np.all(np.isfinite(cap)):
        raise ValueError('Joint amplitude densities need finite amplitudes')
    f,err = np.zeros((len(t),len(cap))),np.zeros((len(t),len(cap)))
    norm,opt = _normalization(moments,level),_options(options)
    for j in range(1,len(t)):
        peaks = [j] if front else range(1,j)
        for k in peaks:
            tau,T = t[k],t[j]
            interior = [(x,0) for ix,x in enumerate(t[1:j],1) if ix!=k]
            derivatives = ([(0,1),(T,2)] if front else [(tau,2),(0,1),(T,1)])
            condition = ([(0,0),(T,0),(T,1)] if front else [(tau,0),(0,0),(T,0),(tau,1)])
            for l,height in enumerate(cap):
                if height == 0:
                    continue
                amplitude = level+height if crest else level-height
                lower = [min(level,amplitude)]*len(interior)
                upper = [max(level,amplitude)]*len(interior)
                if front:
                    lower += [0,-np.inf] if crest else [-np.inf,0]
                    upper += [np.inf,0] if crest else [0,np.inf]
                    values = [level,amplitude,0]
                else:
                    lower += [-np.inf,0,-np.inf] if crest else [0,-np.inf,0]
                    upper += [0,np.inf,0] if crest else [np.inf,0,np.inf]
                    values = [amplitude,level,level,0]
                value,error,_ = _rice(w,s,interior,derivatives,condition,values,lower,upper,opt)
                weight = 1 if front else t[1]-t[0]
                f[j,l] += value[0]*norm*weight
                err[j,l] += error[0]*norm*weight
    return _pdf(f[mask],[t[mask],cap],['period','crest amplitude' if crest else 'trough amplitude'],
        err[mask], 'Gaussian wave period and amplitude density')


def spec2tccpdf(spec, utc=0, definition='t<', paramt=None, h1=np.inf, h2=np.inf, *, options=None):
    """Full upcrossing-wave period subdensity with crest/trough amplitude limits.

    The native driver sums over the intermediate downcrossing time. This
    implementation performs that same finite-grid Rice integral. ``t<`` keeps
    both amplitudes below h1,h2. ``t>`` uses inclusion-exclusion to keep both
    above the thresholds. Densities are subdensities per upcrossing, not
    conditional distributions normalized by the amplitude-selection event.
    """
    d = definition.lower()
    if d not in ('t<','t>'):
        raise NotImplementedError('Only temporal t< and t> wave definitions are implemented')
    if np.isnan(h1) or np.isnan(h2) or h1 < 0 or h2 < 0:
        raise ValueError('Amplitude thresholds must be nonnegative scalars')
    w,s,moments = _spectrum(spec)
    level = 0.0 if utc is None else float(utc)
    t,mask = _time_grid(paramt,moments)
    opt,norm = _options(options),_normalization(moments,level)
    f,error = np.zeros(len(t)),np.zeros(len(t))
    terms = [(h1,h2,1)] if d=='t<' else [(np.inf,np.inf,1),(h1,np.inf,-1),(np.inf,h2,-1),(h1,h2,1)]
    for j in range(2,len(t)):
        for k in range(1,j):
            tau,T = t[k],t[j]
            interior = [(x,0) for ix,x in enumerate(t[1:j],1) if ix != k]
            derivatives = [(tau,1),(0,1),(T,1)]
            condition = [(tau,0),(0,0),(T,0)]
            for c1,c2,sign in terms:
                # A differentiable positive/negative excursion has strictly
                # positive amplitude even when the coarse path grid has no
                # interior sample in one of its two segments.
                if c1 == 0 or c2 == 0:
                    continue
                lower = [level if x<tau else level-c2 for x,_ in interior]+[-np.inf,0,0]
                upper = [level+c1 if x<tau else level for x,_ in interior]+[0,np.inf,np.inf]
                value,err,_ = _rice(w,s,interior,derivatives,condition,[level]*3,lower,upper,opt)
                weight = norm*(t[1]-t[0])
                f[j] += sign*value[0]*weight
                error[j] += err[0]*weight
    return _pdf(f[mask],[t[mask]],['period'],error[mask], 'Full Gaussian wave period subdensity',
        dict(crest_limit=h1,trough_limit=h2,selection=d))


def wminmax(spec, nit=None, paramu=None, t=None, *, options=None):
    """Minimum/following-maximum density replacing the old minmax driver.

    A zero-mean stationary Gaussian process has sign-reversal symmetry. Apply
    the maximum/following-minimum Rice density to -X and reverse both level
    axes. Returns a PDF mapping instead of the old (matrix,paramu) tuple.
    ``nit`` is rejected when supplied: legacy integration-order selectors do
    not describe the randomized Sobol backend.
    """
    if nit is not None:
        raise NotImplementedError('Use Python options rather than legacy nit integration-order selectors')
    if t is None:
        paramt = None
    else:
        t = np.asarray(t,dtype=float)
        if len(t)<3 or t[0]!=0 or np.any(np.diff(t)<=0) or not np.allclose(np.diff(t),t[1]-t[0]):
            raise ValueError('t must be an increasing uniform time grid starting at zero')
        paramt = [0,t[-1],len(t)]
    reflected = None if paramu is None else [-paramu[1],-paramu[0],paramu[2]]
    result = spec2mmtpdf(spec,definition='Mm',paramt=paramt,paramu=reflected,options=options)
    # Rows of Mm are minima, columns maxima. A sign change swaps those roles.
    result.f = result.f[::-1,::-1].T
    result.err = result.err[::-1,::-1].T
    result.x = [-result.x[1][::-1],-result.x[0][::-1]]
    result.labx = ['minimum','following maximum']
    result.title = 'Minimum and following maximum'
    return result
