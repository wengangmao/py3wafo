"""Randomized cubature capability shared by legacy intmodule/MVBRVMOD.

Uses scrambled Sobol replicas instead of the original Korobov rules.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import qmc, t


@dataclass(frozen=True)
class IntegrationResult:
    value: object
    error: object
    evaluations: int
    converged: bool


def integrate_box(fun, lower, upper, *, maxpts=32768, replicates=8,
                  abseps=1e-6, releps=1e-3, seed=None):
    """Integrate a vectorized scalar/vector function on a finite rectangle.

    fun receives (npoints,dimension) and returns (npoints,) or
    (npoints,noutputs). Independent scrambled Sobol replicates give a 99%
    Student-t estimated error, not a deterministic bound. The point budget is
    never exceeded. `converged=False` reports unmet requested tolerances.
    """
    a,b = np.asarray(lower,dtype=float),np.asarray(upper,dtype=float)
    if a.ndim!=1 or a.size==0 or b.shape!=a.shape or not np.all(np.isfinite([a,b])) or np.any(b<a):
        raise ValueError("Expected finite ordered vectors of rectangle bounds")
    if not np.isscalar(replicates) or not np.isfinite(replicates) or int(replicates)!=replicates or replicates<2:
        raise ValueError("replicates must be an integer >= 2")
    if not np.isscalar(maxpts) or not np.isfinite(maxpts) or int(maxpts)!=maxpts or maxpts<replicates:
        raise ValueError("maxpts must be an integer >= replicates")
    if not np.isfinite(abseps) or not np.isfinite(releps) or abseps<0 or releps<0:
        raise ValueError("Tolerances must be finite and nonnegative")
    m = int(np.floor(np.log2(maxpts/replicates)))
    estimates = []
    rng = np.random.default_rng(seed)
    volume = np.prod(b-a)
    for _ in range(int(replicates)):
        unit = qmc.Sobol(a.size,scramble=True,seed=int(rng.integers(0,2**32-1))).random_base2(m)
        values = np.asarray(fun(a+unit*(b-a)),dtype=float)
        if values.ndim not in (1,2) or values.shape[0]!=len(unit) or not np.all(np.isfinite(values)):
            raise ValueError("Integrand must return finite values with one row per sample")
        estimates.append(values.mean(axis=0)*volume)
    estimates = np.asarray(estimates)
    value = estimates.mean(axis=0)
    error = t.ppf(.995,replicates-1)*estimates.std(axis=0,ddof=1)/np.sqrt(replicates)
    return IntegrationResult(value,error,int(replicates)*2**m,
                             bool(np.all(error<=np.maximum(abseps,releps*abs(value)))))
