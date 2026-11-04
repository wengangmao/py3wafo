"""Coefficient transforms and packed-correlation native gateway replacements."""
import numpy as np
from scipy import optimize
from scipy.integrate import trapezoid

from ._probability import mvnormprb_full


def _abscissas(x, mean, sigma):
    if not np.isscalar(mean) or not np.isfinite(mean) or not np.isscalar(sigma) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Mean must be finite and standard deviation positive')
    x = np.linspace(mean-5*sigma, mean+5*sigma, 513) if x is None or np.size(x) == 0 else np.asarray(x, dtype=float).ravel()
    if len(x) < 2 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError('Transformation abscissas must be a finite increasing vector')
    return x, (x-mean)/sigma


def hermitefun(coef, x=None, ma=0., sa=1., gdef=1, *, full_output=False):
    """Direct-coefficient Hermite transform from the private MATLAB helper.

    ``coef=[c3,c4]``.  For ``gdef<0``, apply the hardening polynomial directly.
    Otherwise invert ``K*(y+c3*(y**2-1)+c4*(y**3-3*y))``, with
    ``K=(1+2*c3**2+6*c4**2)**(-1/2)``.  The increasing branch containing zero
    is used; points beyond its turning points raise instead of switching
    branches.  Bracketed real inversion avoids source complex cube roots.
    Full output adds ``integral((standardized_x-g(x))**2,standardized_x)``.
    """
    coefficients = np.asarray(coef)
    if np.iscomplexobj(coefficients) or coefficients.size != 2:
        raise ValueError('coef must contain two real coefficients')
    c3, c4 = np.asarray(coefficients, dtype=float).ravel()
    if not np.all(np.isfinite([c3, c4])) or not np.isscalar(gdef) or not np.isfinite(gdef):
        raise ValueError('Hermite coefficients and definition must be finite')
    x, z = _abscissas(x, ma, sa)
    if gdef < 0:
        p = np.asarray([-c4, -c3, 1+3*c4, c3])
        derivative = np.polyder(p)
        candidates = [z[0], z[-1]]
        if derivative[0] != 0:
            vertex = -derivative[1]/(2*derivative[0])
            if z[0] < vertex < z[-1]:
                candidates.append(vertex)
        if np.min(np.polyval(derivative, candidates)) < -1e-12:
            raise ValueError('Hardening polynomial is not increasing over the requested interval')
        values = np.polyval(p, z)
    else:
        K = 1/np.sqrt(1+2*c3*c3+6*c4*c4)
        p = np.trim_zeros(K*np.asarray([c4, c3, 1-3*c4, -c3]), 'f')
        derivative = np.polyder(p)
        # A double zero of the derivative is a stationary inflection, not a
        # branch boundary (e.g. a pure cubic).  Retain turning points only.
        critical = np.roots(derivative)
        critical = critical.real[abs(critical.imag) < 1e-9]
        turning = []
        for root in critical:
            eps = 1e-5*max(1., abs(root))
            if np.polyval(derivative, root-eps)*np.polyval(derivative, root+eps) < 0:
                turning.append(root)
        turning = np.asarray(turning)
        if np.polyval(derivative, 0) < 0 or np.any(abs(turning) < 1e-12):
            raise ValueError('Hermite polynomial has no increasing central branch')
        left = np.max(turning[turning < 0], initial=-np.inf)
        right = np.min(turning[turning > 0], initial=np.inf)
        if (np.isfinite(left) and np.any(z <= np.polyval(p, left))) or (np.isfinite(right) and np.any(z >= np.polyval(p, right))):
            raise ValueError('Requested values extend beyond the increasing central Hermite branch')
        values = np.empty_like(z)
        for i, target in enumerate(z):
            lo = left if np.isfinite(left) else -1.
            hi = right if np.isfinite(right) else 1.
            if not np.isfinite(left):
                while np.polyval(p, lo) > target:
                    lo *= 2
            if not np.isfinite(right):
                while np.polyval(p, hi) < target:
                    hi *= 2
            values[i] = optimize.brentq(lambda y: np.polyval(p, y)-target, lo, hi, xtol=1e-13)
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
        raise ValueError('Hermite transformation is not finite and strictly increasing')
    table = np.column_stack((x, values))
    departure = float(trapezoid((z-values)**2, z))
    return (table, departure) if full_output else table


def ochifun(y=None, data=None, *, full_output=False):
    """Ochi exponential transform with direct six-parameter normalization.

    ``data=[gamma_a,gamma_b,sigma,mean,sigma2,mean2]`` defaults to
    ``[.1,.15,1,0,1,0]``; NaNs leave defaults unchanged.  Gamma_a applies above
    the mean and gamma_b below.  Zero gamma uses its exact linear limit;
    ``expm1`` preserves small-gamma accuracy.  Full output adds the original
    integrated squared departure on the standardized coordinate.
    """
    parameters = np.asarray([.1, .15, 1., 0., 1., 0.])
    if data is not None:
        supplied = np.asarray(data, dtype=float).ravel()[:6]
        valid = ~np.isnan(supplied)
        indices = np.flatnonzero(valid)
        parameters[indices] = supplied[indices]
    if not np.all(np.isfinite(parameters)) or parameters[2] <= 0 or parameters[4] <= 0:
        raise ValueError('Ochi parameters must be finite with positive sigma and sigma2')
    ga, gb, sigma, mean, sigma2, mean2 = parameters
    y, z = _abscissas(y, mean, sigma)
    gamma = np.where(z >= 0, ga, gb)
    transformed = z.copy()
    active = gamma != 0
    with np.errstate(over='ignore'):
        transformed[active] = -np.expm1(-gamma[active]*z[active])/gamma[active]
    values = (transformed-mean2)/sigma2
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
        raise ValueError('Ochi transformation overflows or loses strict monotonicity on this interval')
    table = np.column_stack((y, values))
    departure = float(trapezoid((z-values)**2, z))
    return (table, departure) if full_output else table


def _unpack_correlation(correl, n):
    packed = np.asarray(correl, dtype=float)
    if packed.ndim > 2 or (packed.ndim == 2 and min(packed.shape, default=0) > 1):
        raise ValueError('correl must be a packed vector, not a full matrix')
    packed = packed.ravel()
    if packed.size != n*(n-1)//2 or not np.all(np.isfinite(packed)) or np.any(abs(packed) > 1):
        raise ValueError('Packed correlation length/values do not match the bounds')
    correlation = np.eye(n)
    indices = np.tril_indices(n, -1)
    correlation[indices] = packed
    correlation[indices[1], indices[0]] = packed
    return correlation


def _packed_probability(correl, A, B, abseps, releps, maxpoints, limit, seed, minpts, replications):
    a, b = np.asarray(A, dtype=float).ravel(), np.asarray(B, dtype=float).ravel()
    if a.shape != b.shape or np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(a > b):
        raise ValueError('Bounds must have the same length and satisfy A<=B without NaNs')
    n = len(a)
    if not 1 <= n <= limit:
        return 0., 1., 2
    correlation = _unpack_correlation(correl, n)
    # These are the actual directional sentinel tests in the MEX gateway.
    a, b = np.where(a <= -37, -np.inf, a), np.where(b >= 37, np.inf, b)
    budget = max(256, 1000*n) if maxpoints is None else maxpoints
    return mvnormprb_full(correlation, a, b, abseps, releps, budget, seed=seed,
                         minpts=minpts, replications=replications)


def mexmvnprb(correl, A, B, abseps=1e-4, releps=1e-3, maxpoints=None, *,
              seed=None, minpts=256, replications=8):
    """Packed-correlation MVN probability, returning ``(value,error,inform)``.

    Packing follows the FORTRAN row formula: ``[r21,r31,r32,r41,r42,r43,...]``
    (one-based matrix labels).  Use ``R[np.tril_indices(n,-1)]``.  The MATLAB
    help's column-major ``find(tril(...))`` note is inconsistent for n>=4.
    Lower bounds <=-37 and upper bounds >=37 are treated as infinite.

    The Python sequential replicated-Sobol engine replaces Korobov/Richtmeyer
    rules.  ``error`` is its estimated 99% replicate interval, not a rigorous
    bound or native parity.  ``inform=0/1`` means estimated tolerance met /
    budget exhausted; dimension outside1..100 returns ``(0,1,2)``.
    """
    return _packed_probability(correl, A, B, abseps, releps, maxpoints, 100, seed, minpts, replications)


def mexmvnprb2(correl, A, B, abseps=1e-4, releps=1e-3, maxpoints=None, *,
               seed=None, minpts=256, replications=8):
    """The 500-dimensional packed gateway, using the Python Sobol engine.

    Packing, directional infinity sentinels, outputs and error interpretation
    are those of ``mexmvnprb``.  Honors the documented intended limit500;
    this repairs the supplied gateway's stale 100-element work arrays.
    """
    return _packed_probability(correl, A, B, abseps, releps, maxpoints, 500, seed, minpts, replications)


def mexGenzMvnPrb(correl, A, B, abseps=1e-4, releps=1e-3, maxpoints=None,
                  method='sobol', *, seed=None, minpts=256, replications=8):
    """Seven-argument packed MVN gateway with explicit Python method selection.

    ``method='sobol'`` uses the shared engine and limit100.  Original numeric
    codes1..4 select SADAPT, Korobov, antithetic crude MC and Deak algorithms;
    they are explicitly unsupported, rather than silently relabeled Sobol.
    The return tuple, packing and estimated error follow ``mexmvnprb``.
    """
    if method != 'sobol':
        raise NotImplementedError("Native method codes1..4 are unavailable; use method='sobol' for the Python backend")
    return _packed_probability(correl, A, B, abseps, releps, maxpoints, 100, seed, minpts, replications)
