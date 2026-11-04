"""Gaussian probabilities, weighted expectations and finite-grid Rice densities.

Native algorithms are consolidated into modern Python/SciPy/Sobol backends.
See individual functions for supported options and numerical error limits.
"""
from ._plotting import trplot, pdfplot
from ._quadratic_moments import (holmquist1, holmquist2, holmquist3,
    holmberg1, holmberg2, holmberg3, kchitwo, rfadd)
from ._quadratic_geometry import mindist, mindist3, chitwo2lc_sorm
from ._quadratic_saddlepoint import (ewwdir, dirsp2chitwo, chitwo2lc_sp, specq2lc,
    rqlf_asympt, writecov, mctp2tc)
import numpy as np
from ._compatibility import hermitefun, ochifun, mexmvnprb, mexmvnprb2, mexGenzMvnPrb
from scipy import stats
from ._rind import rind, rindoptset
from ._probability import mvnormprb_full, mvnormpcprb_full, mvtprb, mvnortpcprb
from ._wave_integrals import spec2Acdf, spec2mmtpdf, spec2thpdf, spec2tccpdf, wminmax
from ._transformations import (createpdf,trmak,trunmak,trangood,tranproc,
    hermitetr,ochitr,troptset,cdf2tr,lc2tr,dat2tr)


def bvnormcdf(b1, b2, r):
    a, b, rho = np.broadcast_arrays(b1, b2, r)
    if np.any(abs(rho) > 1):
        raise ValueError("Correlation must lie in [-1,1]")
    out = np.empty(a.shape)
    for i in np.ndindex(a.shape):
        out[i] = stats.multivariate_normal.cdf([a[i], b[i]], cov=[[1, rho[i]], [rho[i], 1]], allow_singular=True)
    return out


def bvnormprb(a, b, r):
    return mvnormprb([[1,r],[r,1]], a, b)


def mvnormprb(correl, A, B, abseps=1e-4, releps=1e-3, maxpts=None, *, seed=None, full_output=False):
    """Normal rectangle probability using SciPy's integration implementation.

    Default returns a probability. full_output=True selects the replicated
    Sobol backend and returns (probability, estimated error, convergence flag).
    Its estimates are documented in mvnormprb_full; they are not legacy bounds.
    """
    if full_output:
        return mvnormprb_full(correl,A,B,abseps,releps,65536 if maxpts is None else maxpts,seed=seed)
    C, a, b = np.asarray(correl), np.asarray(A), np.asarray(B)
    if C.shape != (a.size, a.size) or a.shape != b.shape or np.any(a > b):
        raise ValueError("Inconsistent covariance/bounds")
    # The frozen form takes a seed on every supported SciPy release; it stores
    # its integration controls as attributes (SciPy 1.15's cdf takes no rng).
    law = stats.multivariate_normal(mean=np.zeros(a.size), cov=C, allow_singular=True,
        seed=np.random.default_rng(seed))
    if maxpts is not None:
        law.maxpts = maxpts
    law.abseps, law.releps = abseps, releps
    return law.cdf(b, lower_limit=a)


def mvnormpcprb(rho, a, b, **kwargs):
    """Normal rectangle probability with off-diagonal correlations rho_i*rho_j."""
    rho = np.asarray(rho)
    if np.any(abs(rho) > 1):
        raise ValueError("Absolute correlation factors must not exceed one")
    C = np.outer(rho, rho)
    np.fill_diagonal(C, 1)
    return mvnormprb(C, a, b, **kwargs)


def lomaxcdf(x, alpha, m0):
    if np.any((np.asarray(alpha) < 0) | (np.asarray(alpha) > 1)) or np.any(np.asarray(m0) <= 0):
        raise ValueError("Require 0 <= alpha <= 1 and m0 > 0")
    x, alpha, m0 = np.broadcast_arrays(x, alpha, m0)
    with np.errstate(divide='ignore', invalid='ignore'):
        c1 = x/np.sqrt(m0*(1-alpha**2))
        out = stats.norm.cdf(c1)-alpha*np.exp(-x*x/(2*m0))*stats.norm.cdf(alpha*c1)
    return np.where(alpha == 1, np.where(x >= 0, -np.expm1(-x*x/(2*m0)), 0), out)


def dat2gaus(x, g):
    a = np.asarray(x, dtype=float).copy()
    a[:, 1:] = tranproc(a[:, 1:], g)
    return a


def gaus2dat(x, g):
    return dat2gaus(x, np.asarray(g)[:, ::-1])


from ._remaining import (mc2rfc,mctp2rfc,iter,iter_mc,nt2fr,th2vhpdf,
    spec2tpdf,spec2tpdf2,spec2cmat,spec2AcAt,initoptions,lc2tr2,mctrtest,spec2skew)
