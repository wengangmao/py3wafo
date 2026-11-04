"""Palmgren–Miner fatigue damage."""
import numpy as np
from ._remaining import damint,snplot
from ._crossing_bounds import down2cc, lc2dplus
from ._models import cmat2dmat,spec2dplus,roadspec,ftf,sphdam


def cc2dam(cc, beta, K=1):
    """K times sum of cycle amplitudes raised to each beta."""
    c = np.asarray(cc, dtype=float)
    if c.size == 0:
        c = np.empty((0,2))
    if c.ndim != 2 or c.shape[1] < 2:
        raise ValueError("Expected cycles with minima and maxima in the first two columns")
    amp = np.abs(c[:, 1]-c[:, 0])/2
    return K*np.sum(amp[:, None]**np.atleast_1d(beta), axis=0)


def cmat2dam(param, F, beta, K=1):
    """Damage of a cycle matrix on the supplied level grid."""
    from ..cycles import levels
    u = levels(param)
    F = np.asarray(F, dtype=float)
    if F.shape != (len(u), len(u)) or np.any(F < 0):
        raise ValueError("Expected a nonnegative square cycle matrix")
    amp = abs(u[:, None]-u[None, :])/2
    return np.asarray([K*np.sum(np.triu(F)*amp**b) for b in np.atleast_1d(beta)])
