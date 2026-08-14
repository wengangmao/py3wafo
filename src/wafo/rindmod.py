"""Pure-Python compatibility implementation of WAFO's RIND extension.

The original module is a Fortran extension.  This implementation uses SciPy
for conditional Gaussian calculations and scrambled Sobol integration for the
Jacobian-weighted expectations.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from scipy.special import ndtri
from scipy.stats import multivariate_normal, norm, qmc


_OPTIONS = {
    "method": 3,
    "xcscale": 0.0,
    "abseps": 0.0,
    "releps": 1e-3,
    "coveps": 1e-10,
    "maxpts": 40000,
    "minpts": 0,
    "nit": 1000,
    "xcutoff": 8.5,
    "nc1c2": 2,
    "quadno": 2,
    "xsplit": 1.5,
}

IS_PYTHON_FALLBACK = True


def set_constants(method=3, xcscale=0.0, abseps=0.0, releps=1e-3,
                  coveps=1e-10, maxpts=40000, minpts=0, nit=1000,
                  xcutoff=8.5, nc1c2=2, quadno=2, xsplit=1.5):
    """Set integration options accepted by the historical extension."""
    values = {
        name: value[0] if isinstance(value, tuple) and len(value) == 1 else value
        for name, value in locals().items()
    }
    _OPTIONS.update(values)


def show_constants():
    """Print the current compatibility-integrator options."""
    for name, value in _OPTIONS.items():
        print(f"{name}={value}")


def _full_covariance(cov):
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance matrix must be square")
    # RIND historically accepts an upper-triangular covariance matrix.
    if np.allclose(np.tril(cov, -1), 0.0):
        cov = cov + np.triu(cov, 1).T
    return (cov + cov.T) * 0.5


def _limits(blo, bup, indices, xc):
    n = int(indices[-1])
    lower = np.empty(n, dtype=float)
    upper = np.empty(n, dtype=float)
    for j, (start, stop) in enumerate(zip(indices[:-1], indices[1:])):
        lo = blo[0, j]
        hi = bup[0, j]
        coefficients = min(xc.size, blo.shape[0] - 1)
        if coefficients:
            lo += np.dot(blo[1:coefficients + 1, j], xc[:coefficients])
            hi += np.dot(bup[1:coefficients + 1, j], xc[:coefficients])
        lower[int(start):int(stop)] = lo
        upper[int(start):int(stop)] = hi
    return lower, upper


def _conditional(cov, mean, xc):
    nc = xc.size
    if not nc:
        return cov, mean, 1.0
    split = len(mean) - nc
    c_tt = cov[:split, :split]
    c_tc = cov[:split, split:]
    c_cc = cov[split:, split:]
    delta = xc - mean[split:]
    gain = solve(c_cc, c_tc.T, assume_a="sym").T
    conditional_mean = mean[:split] + gain @ delta
    conditional_cov = c_tt - gain @ c_tc.T
    density = multivariate_normal.pdf(xc, mean=mean[split:], cov=c_cc,
                                      allow_singular=True)
    density *= np.exp(float(_OPTIONS["xcscale"]))
    return conditional_cov, conditional_mean, float(density)


def _rectangle_probability(mean, cov, lower, upper):
    value = multivariate_normal.cdf(
        upper, mean=mean, cov=cov, lower_limit=lower,
        maxpts=int(_OPTIONS["maxpts"]),
        abseps=max(float(_OPTIONS["abseps"]), 1e-10),
        releps=float(_OPTIONS["releps"]),
        allow_singular=True,
    )
    return float(np.clip(value, 0.0, 1.0))


def _weighted_expectation(mean, cov, lower, upper, nt, seed):
    dimension = len(mean)
    maxpts = max(int(_OPTIONS["maxpts"]), 1024)
    exponent = min(22, max(10, int(np.floor(np.log2(maxpts)))))
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    uniforms = np.clip(sampler.random_base2(exponent), 1e-14, 1 - 1e-14)
    standard = ndtri(uniforms)
    values, vectors = np.linalg.eigh((cov + cov.T) * 0.5)
    factor = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
    samples = mean + standard @ factor.T
    inside = np.all((samples >= lower) & (samples <= upper), axis=1)
    weights = np.prod(np.abs(samples[:, nt:]), axis=1)
    integrand = weights * inside
    value = float(np.mean(integrand))
    # Batch variation gives a stable sampling-error estimate for scrambled QMC.
    batches = np.array_split(integrand, 16)
    batch_means = np.asarray([np.mean(batch) for batch in batches])
    error = 2.576 * float(np.std(batch_means, ddof=1)) / np.sqrt(len(batches))
    return value, error


def rind(big, ex, xc, nt, indi, blo, bup, infin, seed):
    """Compute conditional Gaussian probabilities or RIND expectations.

    The signature and three-array return value match the f2py extension.
    """
    del infin  # Infinite limits are already represented by Blo/Bup.
    covariance = _full_covariance(big)
    mean = np.asarray(ex, dtype=float).reshape(-1)
    xc = np.asarray(xc, dtype=float)
    if xc.ndim == 1:
        xc = xc[:, None]
    nx = max(xc.shape[1] if xc.ndim == 2 else 0, 1)
    blo = np.atleast_2d(np.asarray(blo, dtype=float))
    bup = np.atleast_2d(np.asarray(bup, dtype=float))
    indices = np.asarray(indi, dtype=int).reshape(-1)

    values = np.empty(nx)
    errors = np.empty(nx)
    truncation = np.full(nx, max(float(_OPTIONS["coveps"]), 1e-10))
    for column in range(nx):
        condition = xc[:, column] if xc.size else np.empty(0)
        cov_i, mean_i, density = _conditional(covariance, mean, condition)
        lower, upper = _limits(blo, bup, indices, condition)
        if int(nt) == len(mean_i):
            values[column] = density * _rectangle_probability(
                mean_i, cov_i, lower, upper)
            errors[column] = max(float(_OPTIONS["abseps"]), 1e-8)
        else:
            value, error = _weighted_expectation(
                mean_i, cov_i, lower, upper, int(nt), int(seed) + column)
            values[column] = density * value
            errors[column] = density * error
    return values, errors, truncation
