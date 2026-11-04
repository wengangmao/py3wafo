"""Breaking-limited Tayfun and source fitted climate/velocity models.

Numerical and empirical domains are deliberately explicit.  In particular,
fractional powers of the oscillatory Tayfun characteristic function are not
silently promoted to probability distributions.
"""
from functools import lru_cache

import numpy as np
from scipy import integrate, special, stats
from scipy.interpolate import CubicSpline

from ..spec import Spectrum
from ..wstats import wtweibpdf, wtweibcdf
from ._empirical import jhnlwparfun
from ._tables import _data, _sea, _scales


def _tayfun_input(x, a, b, tol):
    x, a, b = np.broadcast_arrays(np.asarray(x, dtype=float), a, b)
    if np.any(~np.isfinite(a)) or np.any(a <= 0) or np.any(np.isnan(b)) or np.any(b <= 0):
        raise ValueError('Require a positive finite scale and positive limiting parameter')
    if not np.isscalar(tol) or not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError('tol must lie strictly between zero and one')
    finite = np.isfinite(b)
    n = b[finite]**2
    if np.any(~np.isfinite(n)) or np.any(n < 6-1e-10) or np.any(abs(n-np.rint(n)) > 1e-9*np.maximum(1, n)):
        raise NotImplementedError('Finite Tayfun probabilities require integer B**2>=6; general fractional Bessel powers need a separately validated probability law')
    return x, a, b


def _hankel(r, B, cdf, tol):
    """Finite-radius phasor law; polynomial Bessel tail bounds plus quadrature."""
    n = int(round(B*B)); p = n/2
    rmax = float(np.max(r)); eps = tol/4
    logC = p*np.log(2*B/np.pi)
    if cdf:
        logU = (np.log(rmax)+logC-np.log(p-1)-np.log(eps))/(p-1)
    else:
        logU = (.5*np.log(2*rmax/np.pi)+logC-np.log(p-1.5)-np.log(eps))/(p-1.5)
    U = max(12., float(np.exp(logU)))
    segments = max(1, int(np.ceil(U*max(rmax, 1/B)/np.pi)))
    if segments > 20000:
        raise RuntimeError('Requested Tayfun tolerance exceeds the quadrature budget; increase tol or use a larger integer B**2')
    # |J0(t)| <= sqrt(2/(pi*t)) follows from the Sonin energy of
    # sqrt(t)*J0(t).  For the CDF the weaker |J1|<=1 is sufficient.
    if cdf:
        tail = r*np.exp(logC+(1-p)*np.log(U))/(p-1)
    else:
        tail = np.sqrt(2*r/np.pi)*np.exp(logC+(1.5-p)*np.log(U))/(p-1.5)
    def quadrature(order):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        dx = U/segments
        u = ((np.arange(segments)[:, None]+.5)*dx+nodes[None, :]*dx/2).ravel()
        w = np.tile(weights*dx/2, segments)*special.j0(u/B)**n
        if not cdf:
            w *= u
        answer = np.empty(len(r))
        for start in range(0, len(r), 32):
            ru = r[start:start+32, None]*u[None, :]
            answer[start:start+32] = r[start:start+32]*((special.j1(ru) if cdf else special.j0(ru))@w)
        return answer
    coarse, fine = quadrature(16), quadrature(32)
    error = abs(fine-coarse)+tail
    if np.any(error > tol):
        refined = quadrature(64)
        error = abs(refined-fine)+tail
        fine = refined
    if np.any(error > tol):
        raise RuntimeError('Tayfun Hankel quadrature did not meet the requested tolerance')
    if np.any(fine < -error) or (cdf and np.any(fine > 1+error)):
        raise RuntimeError('Numerical Hankel result violates the probability bounds')
    return np.clip(fine, 0, 1) if cdf else np.maximum(fine, 0), error


def _tayfun(x, a, b, tol, cdf, full_output):
    x, a, b = _tayfun_input(x, a, b, tol)
    out, error = np.zeros(x.shape), np.zeros(x.shape)
    out[np.isnan(x)] = np.nan; error[np.isnan(x)] = np.nan
    infinite = np.isposinf(b)
    if cdf:
        out[infinite] = stats.rayleigh.cdf(x[infinite], scale=a[infinite]/np.sqrt(2))
    else:
        out[infinite] = stats.rayleigh.pdf(x[infinite], scale=a[infinite]/np.sqrt(2))
    if cdf:
        out[(~infinite) & (x >= a*b)] = 1
    active = (~infinite) & (x > 0) & (x < a*b)
    for B in np.unique(b[active]):
        mask = active & (b == B)
        values, errors = _hankel((x[mask]/a[mask]).ravel(), float(B), cdf, tol)
        out[mask] = values if cdf else values/a[mask]
        error[mask] = errors if cdf else errors/a[mask]
    return (out, error) if full_output else out


def tay81pdf(x, a, b, tol=1e-3, *, full_output=False):
    """Tayfun Bessel-integral wave-height PDF on a validated finite-B domain.

    ``b=inf`` gives Rayleigh(a/sqrt(2)).  For finite ``b``, require integer
    ``b**2>=6``: the source Bessel power is then the characteristic function
    of that many independent planar phasors of length1/b.  The exact radial
    support is0..a*b; the source angular correction beyond a*b has zero mass.
    Noninteger powers are not guaranteed positive-definite and are rejected.

    ``tol`` controls absolute error of the dimensionless Hankel integral,
    unlike the source's unvalidated relative stopping heuristic.  Full output
    adds estimated quadrature error plus an analytic absolute tail bound,
    divided by a.  Finite-budget failures raise; values are never renormalized.
    """
    return _tayfun(x, a, b, tol, False, full_output)


def tay81cdf(x, a, b, tol=1e-3, *, full_output=False):
    """CDF consistent with ``tay81pdf`` on its validated integer-B² domain.

    Integrates the radial law by ``r*integral(J1(r*u)*J0(u/b)**(b*b),du)``.
    This corrects the source's repeated B squaring and extra u factor in its
    unused CDF integrand.  Full output adds numerical/tail error estimates.
    """
    return _tayfun(x, a, b, tol, True, full_output)


def _parameters(values, defaults):
    result = np.asarray(defaults, dtype=float)
    if values is not None:
        supplied = np.asarray(values, dtype=float).ravel()[:len(result)]
        good = np.flatnonzero(~np.isnan(supplied))
        result[good] = supplied[good]
    if not np.all(np.isfinite(result)):
        raise ValueError('Model parameters must be finite or NaN placeholders')
    return result


def ltwcpdf(t, h, Pt=None, Ph=None, definition=0, condon=0, *, normalize=False, full_output=False):
    """Haver/Nyhus long-term climate density ``f(T|Hs)*f(Hs)``.

    Pt=[A1,A2,A3,B1,B2,B3,B4] defines a lognormal conditional, with mean of
    log(T) A1+A2*log(Hs+A3) for even definitions or A1+A2*Hs**A3 for odd ones,
    and variance B1+B2*exp(-B3*Hs**B4).  Ph=[C1,C2,C3,Hc,M1,S1] splices a
    lognormal below Hc with shifted Weibull (definitions0/1) or generalized
    gamma (2/3) above.  S1 and the conditional S are log variances, not SDs.
    Missing parameters/NaNs use source Northern North Sea defaults.

    condon0/1/3/4/5/10 returns joint density / conditional density / T or T²
    times conditional density / f(Hs)*P(T|Hs) / height marginal.  The source
    splice need not integrate exactly to one (defaults:0.99965405765).
    ``normalize=True`` divides its marginal by its exact total mass.  Full
    output adds that mass; conditional outputs are unchanged by normalization.
    """
    if definition not in (0, 1, 2, 3) or condon not in (0, 1, 3, 4, 5, 10):
        raise ValueError('Use definition0..3 and condon0,1,3,4,5,10')
    pt = _parameters(Pt, [1.59, .42, 2, .005, .09, .13, 1.34])
    ph = _parameters(Ph, [2.82, 1.547, 0, 3.27, .836, .376])
    c1, c2, c3, split, mean_h, var_h = ph
    if c1 <= 0 or c2 <= 0 or split < 0 or var_h <= 0 or (definition > 1 and c3 <= 0):
        raise ValueError('Invalid marginal height distribution parameters')
    low = stats.lognorm(np.sqrt(var_h), scale=np.exp(mean_h))
    upper = stats.weibull_min(c2, loc=c3, scale=c1) if definition <= 1 else stats.gengamma(c1, c3, scale=c2)
    mass = float(low.cdf(split)+upper.sf(split))
    if mass <= 0 or not np.isfinite(mass):
        raise ValueError('Height splice has no finite positive mass')
    t, h = np.broadcast_arrays(np.asarray(t, dtype=float), h)
    result = np.zeros(t.shape)
    positive = h > 0
    safe = np.where(positive & np.isfinite(h), h, 1.)
    a1, a2, a3, b1, b2, b3, b4 = pt
    if definition % 2 == 0:
        if np.any(safe+a3 <= 0):
            raise ValueError('Hs+A3 must be positive for the log-mean parametrization')
        mean = a1+a2*np.log(safe+a3)
    else:
        mean = a1+a2*safe**a3
    with np.errstate(over='ignore'):
        variance = b1+b2*np.exp(-b3*safe**b4)
    if np.any(variance[positive] <= 0) or not np.all(np.isfinite(variance[positive])):
        raise ValueError('Conditional log variance is nonpositive/nonfinite at requested heights')
    conditional = stats.lognorm(np.sqrt(np.maximum(variance, np.finfo(float).tiny)), scale=np.exp(mean))
    if condon == 10:
        result = np.where(positive, np.where(h <= split, low.pdf(h), upper.pdf(h)), 0.)
    else:
        result = np.where(positive, conditional.cdf(t) if condon == 5 else conditional.pdf(t), 0.)
        if condon in (3, 4):
            with np.errstate(invalid='ignore'):
                result = np.where(np.isinf(t), 0., t**(condon-2)*result)
        if condon in (0, 5):
            result *= np.where(h <= split, low.pdf(h), upper.pdf(h))
    if normalize and condon in (0, 5, 10):
        result /= mass
    result = np.where(np.isnan(t) | np.isnan(h), np.nan, result)
    return (result, mass) if full_output else result


@lru_cache(maxsize=64)
def _velocity_curve(gam, key):
    data = _data('jhvnlpar.mat')
    height, gamma = data['h2'].ravel(), data['gam'].ravel()
    values = CubicSpline(gamma, data[key], axis=1)(gam)
    return CubicSpline(height, values, bc_type='natural'), height


def _nonlinear_state(Hm0, Tp, gam):
    H, T, gam = _sea(Hm0, Tp, gam)
    if np.any((T <= 3.6*np.sqrt(H)) | (T >= 5*np.sqrt(H))):
        raise ValueError('Nonlinear JONSWAP fit requires 3.6*sqrt(Hm0)<Tp<5*sqrt(Hm0)')
    return H, T, gam


def _velocity_parameters(h, gam):
    h, gam = np.broadcast_arrays(h, gam)
    values = []
    for key in ('A11s', 'B11s'):
        result = np.empty(h.shape)
        for g in np.unique(gam):
            mask = gam == g
            curve, grid = _velocity_curve(float(g), key)
            point = np.maximum(h[mask], 0.)
            result[mask] = (curve(np.clip(point, grid[0], grid[-1]))+
                            np.minimum(point-grid[0], 0)*curve(grid[0], 1)+
                            np.maximum(point-grid[-1], 0)*curve(grid[-1], 1))
        if np.any(result <= 0) or not np.all(np.isfinite(result)):
            raise ValueError('Height extrapolation of the nonlinear velocity fit gives invalid parameters')
        values.append(result)
    return tuple(values)


def _parameter_pdf(key):
    data = _data('jhvnlpar.mat')
    return Spectrum(f=data[key].copy(), x=[data['gam'].ravel().copy(), data['h2'].ravel().copy()],
                    labx=['Gamma', 'Hd/Hrms'], title='Source conditional Weibull parameter '+key)


def jhvnlpdf(Hd, Vcf, Hm0=6, Tp=10, gam=None, normalizedInput=False, condon=0, *, full_output=False):
    """Nonlinear JONSWAP height/front-velocity model from original fit values.

    Uses the gamma-curve branch that actually matches jhvnlpar.mat.  Its active
    MATLAB branch incorrectly requests Hm0/Tp axes absent from that file.
    Conditional Weibull scale/shape use cubic interpolation in gamma and a
    natural cubic in height, with the source linear extension past h=2.3.
    Invalid extrapolated parameters raise.  Requires gamma1..7 and
    3.6*sqrt(Hm0)<Tp<5*sqrt(Hm0); defaults6,10 satisfy this fitted domain.

    condon0/1/3/4/5/6/7 means joint PDF / conditional velocity PDF / V or V²
    times that PDF / f(H)*P(V|H) / P(V|H) / f(H)*P(V>v|H).  Jacobians and
    moments use physical units, correcting the source common scaling bug.
    Full output returns (f,Hrms,Vrms,scale_table,shape_table).
    """
    if normalizedInput not in (False, True) or condon not in (0, 1, 3, 4, 5, 6, 7):
        raise ValueError('Unsupported normalization flag or conditioning mode')
    H, T, gam = _nonlinear_state(Hm0, Tp, gam)
    Hd, V, H, T, gam = np.broadcast_arrays(Hd, Vcf, H, T, gam)
    hs, vs = _scales('jhv', H, T, gam, normalizedInput)
    h, v = Hd/hs, V/vs
    if condon in (1, 3, 4, 6) and np.any(~np.isfinite(h)):
        raise ValueError('Conditional velocity evaluation requires finite heights')
    a, b = _velocity_parameters(np.where(np.isfinite(h), h, 0.), gam)
    distribution = stats.weibull_min(b, scale=a)
    with np.errstate(invalid='ignore'):
        density = np.where(np.isposinf(v), 0., distribution.pdf(v)/vs)
    cumulative, survival = distribution.cdf(v), distribution.sf(v)
    marginal = wtweibpdf(h, *jhnlwparfun(H, T, gam))/hs
    if condon == 0:
        result = marginal*density
    elif condon == 1:
        result = density
    elif condon in (3, 4):
        with np.errstate(invalid='ignore'):
            result = np.where(np.isinf(V), 0., V**(condon-2)*density)
    elif condon == 5:
        result = marginal*cumulative
    elif condon == 6:
        result = cumulative
    else:
        result = marginal*survival
    result = np.where(h < 0, 0., result)
    result = np.where(np.isnan(h) | np.isnan(v), np.nan, result)
    return (result, hs, vs, _parameter_pdf('A11s'), _parameter_pdf('B11s')) if full_output else result


def jhvnlcdf(Hd, Vcf, Hm0=6, Tp=10, gam=None, tail=0, *, tol=1e-8, full_output=False):
    """Nonlinear joint CDF or upper survival using the actual gamma fit table.

    Integrates over marginal height probability without the source hard h=5
    cutoff.  Same sea-state domain as jhvnlpdf; full output adds quadrature
    error estimates.  Very high extrapolated heights can be invalid for some
    gamma values and raise rather than substituting invented tail fits.
    """
    if tail not in (0, 1) or not np.isfinite(tol) or tol <= 0:
        raise ValueError('Require tail0/1 and positive finite tolerance')
    H, T, gam = _nonlinear_state(Hm0, Tp, gam)
    Hd, V, H, T, gam = np.broadcast_arrays(Hd, Vcf, H, T, gam)
    out, error = np.zeros(H.shape), np.zeros(H.shape)
    for idx in np.ndindex(H.shape):
        height, velocity, H0, T0, g = Hd[idx], V[idx], H[idx], T[idx], gam[idx]
        if np.isnan(height) or np.isnan(velocity):
            out[idx] = error[idx] = np.nan; continue
        hs, vs = _scales('jhv', H0, T0, g, False)
        a0, b0, c0 = (float(p) for p in jhnlwparfun(H0, T0, g))
        cutoff = float(wtweibcdf(max(0., height)/hs, a0, b0, c0))
        if tail == 0 and (height <= 0 or velocity <= 0):
            continue
        if tail == 1 and (np.isposinf(height) or np.isposinf(velocity)):
            continue
        if tail == 0 and np.isposinf(velocity):
            out[idx] = cutoff; continue
        if tail == 1 and velocity <= 0:
            out[idx] = 1-cutoff; continue
        lower, upper = (0., cutoff) if tail == 0 else (cutoff, 1.)
        def integrand(probability):
            h = a0*((c0/a0)**b0-np.log1p(-probability))**(1/b0)-c0
            a1, b1 = (float(p) for p in _velocity_parameters(h, g))
            conditional = stats.weibull_min(b1, scale=a1)
            return float(conditional.cdf(velocity/vs) if tail == 0 else conditional.sf(velocity/vs))
        out[idx], error[idx] = integrate.quad(integrand, lower, upper, epsabs=tol, epsrel=tol, limit=150)
    return (out, error) if full_output else out


def jhvnlpdf2(Hd, Vcf, Hm0=6, Tp=10, gam=None, normalizedInput=False, *, full_output=False):
    """Nonlinear height/front-velocity PDF on meshgrid(Vcf,Hd)."""
    h, v = np.asarray(Hd, dtype=float), np.asarray(Vcf, dtype=float)
    if h.ndim != 1 or v.ndim != 1 or not np.all(np.isfinite(h)) or not np.all(np.isfinite(v)) or np.any(np.diff(h) <= 0) or np.any(np.diff(v) <= 0):
        raise ValueError('PDF grid coordinates must be finite increasing vectors')
    vv, hh = np.meshgrid(v, h)
    result, hs, vs, scale, shape = jhvnlpdf(hh, vv, Hm0, Tp, gam, normalizedInput, full_output=True)
    pdf = Spectrum(f=result, x=[v, h], norm=int(normalizedInput),
                   labx=['Vcf' if normalizedInput else 'Vcf [m/s]', 'Hd' if normalizedInput else 'Hd [m]'],
                   title='Nonlinear JONSWAP height/front-velocity distribution')
    return (pdf, hs, vs, scale, shape) if full_output else pdf
