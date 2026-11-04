"""Conditional Gaussian reconstruction of missing time-series observations.

Derived from onedim/reconstruct.m. The transform and covariance estimation
are separated from conditional prediction; observed values are preserved.
"""
from __future__ import annotations
import numpy as np


def _series(x):
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        t, y = np.arange(len(a), dtype=float), a
    elif a.ndim == 2 and a.shape[1] in (1, 2):
        t = a[:, 0] if a.shape[1] == 2 else np.arange(len(a), dtype=float)
        y = a[:, -1]
    else:
        raise ValueError('Expected values, an N by 1 array, or an N by 2 time/value array')
    if len(y) < 3 or not np.all(np.isfinite(t)) or np.any(np.isinf(y)):
        raise ValueError('At least three samples, finite times, and values finite or NaN are required')
    dt = t[1]-t[0]
    if dt <= 0 or not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-10*dt):
        raise ValueError('Time samples must be increasing and uniformly spaced')
    return a, t, y.copy(), float(dt)


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) or int(value) != value or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def reconstruct(x, inds=None, Nsim=20, L=None, definition='nonlinear', options=None,
                *, covariance=None, mean=None, sample=False, seed=None,
                conditioning='auto', context=None, max_points=1024,
                tolerance=1e-4, full_output=False):
    """Fill missing/spurious values with a conditional transformed-Gaussian model.

    ``inds`` contains zero-based indices or a Boolean mask; existing NaNs are
    always missing. ``definition`` is linear, nonlinear (crossing transform),
    or mnonlinear (marginal transform). ``options`` are dat2tr transform options.
    Nsim bounds covariance-refinement iterations; L is the maximum covariance
    lag. Estimated covariance uses masked, biased autocovariance and a Bartlett
    lag taper, keeping the covariance positive semidefinite. A supplied
    covariance mapping (R, optional t) is in physical units for linear models
    and latent standard-normal units otherwise. ``mean`` specifies a known
    physical mean for linear models. sample=False returns conditional latent
    predictions back-transformed to physical units; sample=True draws once
    from the final conditional model, reproducibly with seed.

    Conditioning is exact for at most max_points samples. For longer series,
    auto uses separate windows around contiguous missing runs; local explicitly
    requests this approximation. Each window conditions only on original known
    observations within context samples of the run. Draws in different windows
    are independent; they do not represent the full joint conditional law.
    Oversized windows are rejected rather than silently thinning observations.

    full_output returns (filled_data, diagnostics), including g, empirical_g,
    missing indices, latent conditional mean/std, covariance, method and change
    history. These diagnostics replace the seven positional MATLAB outputs.
    This estimator preserves known observations and does not reproduce the
    legacy whole-record detrending or heuristic convergence procedure. A
    back-transformed latent mean is generally not a physical conditional mean.
    """
    from ..trgauss import dat2tr, tranproc
    from ..wsim import cov2csdat
    from . import dat2cov
    a, t, y, dt = _series(x)
    n = len(y)
    Nsim = _integer(Nsim, 'Nsim')
    max_points = _integer(max_points, 'max_points', 3)
    L = min(n-1, 200) if L is None else _integer(L, 'L')
    if L >= n:
        raise ValueError('L must be smaller than the series length')
    context = min(L, 128) if context is None else _integer(context, 'context', 0)
    if conditioning not in ('auto', 'exact', 'local'):
        raise ValueError('conditioning must be auto, exact, or local')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be positive and finite')
    missing = np.isnan(y)
    if inds is not None:
        ind = np.asarray(inds)
        if ind.dtype.kind == 'b':
            if ind.shape != y.shape:
                raise ValueError('Boolean inds must have one entry per observation')
            missing |= ind
        else:
            if ind.ndim > 1 or not np.all(np.isfinite(ind)) or np.any(ind != np.floor(ind)) or np.any(ind < 0) or np.any(ind >= n):
                raise ValueError('Invalid missing indices')
            missing[ind.astype(int)] = True
    known = ~missing
    index = np.flatnonzero(missing)
    if known.sum() < 3:
        raise ValueError('At least three known observations are required')
    definition = str(definition).lower()
    if definition not in ('linear', 'nonlinear', 'mnonlinear'):
        raise ValueError('definition must be linear, nonlinear, or mnonlinear')
    if mean is not None and (definition != 'linear' or not np.isfinite(mean)):
        raise ValueError('mean must be finite and is only available for linear models')
    if definition == 'linear':
        center = float(np.mean(y[known])) if mean is None else float(mean)
        scale = float(np.std(y[known], ddof=1))
        if covariance is not None:
            variance = float(np.asarray(covariance['R'])[0])
            if not np.isfinite(variance) or variance <= 0:
                raise ValueError('Supplied covariance must have positive finite variance')
            scale = np.sqrt(variance)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('A positive-variance model is required')
        grid = center+scale*np.linspace(-8, 8, 401)
        g = np.column_stack((grid, (grid-center)/scale))
        empirical = g.copy()
        departure = 0.
    else:
        g, departure, _, _, empirical = dat2tr(np.column_stack((t[known], y[known])), definition, options, full_output=True)
        departure = float(np.asarray(departure).reshape(-1)[0])
        center, scale = 0., 1.
    latent = np.full(n, np.nan)
    latent[known] = tranproc(y[known], g)
    physical_cov = None
    if covariance is not None:
        physical_cov = dict(covariance)
        physical_cov['R'] = np.asarray(covariance['R'], dtype=float)/scale**2
        if 't' in physical_cov:
            lag = np.asarray(physical_cov['t'])
            if len(lag) > 1 and not np.isclose(lag[1]-lag[0], dt):
                raise ValueError('Covariance and signal sample intervals must match')
    exact = conditioning == 'exact' or (conditioning == 'auto' and n <= max_points)
    if exact and n > max_points:
        raise ValueError('Exact conditioning exceeds max_points; select local or increase the explicit limit')
    if exact:
        windows = [(0, n, index)] if len(index) else []
    else:
        runs = np.split(index, np.flatnonzero(np.diff(index) > 1)+1) if len(index) else []
        windows = [(max(0, run[0]-context), min(n, run[-1]+context+1), run) for run in runs]
        if any(stop-start > max_points for start, stop, _ in windows):
            raise ValueError('A local reconstruction window exceeds max_points; reduce context or increase the limit')
    rng = np.random.default_rng(seed)

    def predict(R, draw):
        predictions, std, draws = np.empty(len(index)), np.empty(len(index)), np.empty(len(index))
        positions = {int(k): i for i, k in enumerate(index)}
        for start, stop, targets in windows:
            values, mu, sigma, local = cov2csdat(latent[start:stop], R, seed=rng, full_output=True)
            lookup = {int(k)+start: j for j, k in enumerate(local)}
            source = [lookup[int(k)] for k in targets]
            dest = [positions[int(k)] for k in targets]
            predictions[dest], std[dest] = mu[source], sigma[source]
            draws[dest] = values[source, 0] if draw else mu[source]
        return predictions, std, draws

    filled = np.interp(np.arange(n), np.flatnonzero(known), latent[known])
    change = []
    mu, std = np.empty(0), np.empty(0)
    R = physical_cov
    for iteration in range(Nsim if len(index) else 0):
        if physical_cov is None:
            # First iteration uses observed pairs only. Later iterations refine
            # with the conditional predictions, never with random imputations.
            R = dict(dat2cov(latent if iteration == 0 else filled, L, dt=dt))
            R['R'] = np.asarray(R['R'])*(1-np.arange(L+1)/(L+1))
            R['R'][0] += max(R['R'][0], np.finfo(float).tiny)*1e-9
        mu, std, _ = predict(R, False)
        delta = float(np.max(abs(filled[index]-mu)))
        change.append(delta)
        filled[index] = mu
        if physical_cov is not None or delta <= tolerance:
            break
    if len(index) and sample:
        _, _, filled[index] = predict(R, True)
    result = a.copy()
    physical = y.copy()
    if len(index):
        physical[index] = tranproc(filled[index], g[:, ::-1])
    if result.ndim == 1:
        result[:] = physical
    else:
        result[:, -1] = physical
    diagnostics = dict(g=g, empirical_g=empirical, departure=departure,
                       inds=index, latent_mean=mu, latent_std=std,
                       covariance=R, method='exact' if exact else 'local',
                       iterations=len(change), change_history=np.asarray(change),
                       sampled=bool(sample))
    return (result, diagnostics) if full_output else result
