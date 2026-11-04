"""Capability reconstruction of the source-missing down2cc executable.

The supplied MATLAB wrapper specifies horizontal slicing of a downcrossing
intensity, but its Fortran source is absent. This implementation decomposes a
piecewise-linear intensity into nested cycles via its superlevel sets. It uses
midpoint slices and does not claim the unknown executable's endpoint convention.
"""
import numpy as np


def down2cc(mu, n=500):
    """Return (cycle_min_max_pairs, slice_weight) from a crossing intensity.

    Each connected superlevel interval at a midpoint slice is one cycle of
    weight delta=max(mu[:,1])/n. The intensity must be zero at both boundary
    levels, so no open intervals are silently truncated. Multiple disconnected
    intervals per slice are supported. Increase n to check discretization error.
    """
    mu = np.asarray(mu, dtype=float)
    if (mu.ndim != 2 or mu.shape[1] != 2 or len(mu) < 3
            or not np.all(np.isfinite(mu)) or np.any(np.diff(mu[:,0]) <= 0)
            or np.any(mu[:,1] < 0)):
        raise ValueError("Expected finite, increasing levels and nonnegative crossing intensities")
    if not np.isscalar(n) or not np.isfinite(n) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    u, intensity = mu.T
    if intensity[0] != 0 or intensity[-1] != 0:
        raise ValueError("Crossing intensity must vanish at both boundary levels")
    delta = float(intensity.max()/n)
    if delta == 0:
        return np.empty((0,2)), 0.0
    cycles = []
    for level in (np.arange(int(n))+.5)*delta:
        above = intensity > level
        changes = np.flatnonzero(above[:-1] != above[1:])
        intersections = u[changes]+(level-intensity[changes])*(u[changes+1]-u[changes])/(intensity[changes+1]-intensity[changes])
        cycles.extend(intersections.reshape(-1,2))
    return np.asarray(cycles).reshape(-1,2), delta


def lc2dplus(cross, beta, num_slice=500):
    """Damage of the nested-cycle reconstruction of a crossing intensity.

    Uses amplitudes abs(max-min)/2 and equal midpoint slice weights. This fixes
    the missing parentheses and undocumented final 0.75 weight in lc2dplus.m.
    For this source-missing backend, the name preserves the WAFO capability;
    no claim of bitwise parity or a rigorous finite-slice upper bound is made.
    For convex damage powers, refine num_slice to assess integration error.
    """
    beta = np.atleast_1d(np.asarray(beta,dtype=float))
    if not np.all(np.isfinite(beta)) or np.any(beta < 1):
        raise ValueError("Nested-cycle damage reconstruction requires beta >= 1")
    cycles, delta = down2cc(cross,num_slice)
    amplitudes = (cycles[:,1]-cycles[:,0])/2
    return delta*np.sum(amplitudes[:,None]**beta,axis=0)
