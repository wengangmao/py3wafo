"""Kernel density estimation and normalized multivariate kernel utilities."""
from ._bandwidth import hbcv2, hboot, hldpi2, hldpi2fft, hscv, hste, hstt, chi2gof, chi2gof2
import numpy as np
from scipy import stats
from scipy.special import gamma
from ._sampling import sample,ssample,mkernel2,mkernelrnd,vsph,ffndgrid
from ._kernels import kernelstats, mkernel, deriv, deriv2, bincount, gridcount, qlevels, qlevels2


def hns(A, kernel='gauss'):
    """Robust normal-reference univariate bandwidth per column."""
    a = np.asarray(A, dtype=float)
    a = a[:, None] if a.ndim == 1 else a
    n = len(a)
    if a.ndim!=2 or n < 2 or not np.all(np.isfinite(a)):
        raise ValueError("At least two finite observations are required")
    sd = np.std(a, axis=0, ddof=1)
    q = np.diff(np.quantile(a, [.25, .75], axis=0,method='hazen'), axis=0)[0]
    scale = np.where(q > 0, np.minimum(sd, q/1.349), sd)
    mu2,R,_ = kernelstats(kernel)
    return scale*(8*np.sqrt(np.pi)*R/(3*mu2**2*n))**.2


def kde(data, points=None, hs=None):
    """Gaussian KDE. Columns are dimensions; returns estimator or evaluated PDF.

    hs is an absolute bandwidth for 1D data; multidimensional data uses Scott's
    rule. Legacy transformation/kernel option structures are not supported.
    """
    a = np.asarray(data, dtype=float)
    samples = a if a.ndim == 1 else a.T
    if hs is not None and a.ndim != 1:
        raise NotImplementedError("Explicit bandwidth is currently univariate")
    factor = None if hs is None else float(hs)/np.std(a, ddof=1)
    estimate = stats.gaussian_kde(samples, bw_method=factor)
    if points is None:
        return estimate
    p = np.asarray(points)
    return estimate(p if a.ndim == 1 else p.T)


def hos(A,kernel='gaussian'):
    """Oversmoothing bandwidth, source hns/0.93 rule."""
    return hns(A,kernel)/.93


def hmns(A,kernel='gaussian'):
    """Normal-reference multivariate bandwidth matrix, following hmns.m."""
    a = np.asarray(A,dtype=float)
    a = a[:,None] if a.ndim==1 else a
    if a.ndim!=2 or len(a)<2 or not np.all(np.isfinite(a)):
        raise ValueError("Expected at least two finite observations")
    n,d = a.shape
    if d==1:
        return hns(a,kernel)
    name = kernel.lower()[:4]
    if name in ('gaus','norm'):
        constant = (4/(d+2))**(1/(d+4))
    elif name=='epan':
        volume = np.pi**(d/2)/gamma(d/2+1)
        constant = (8*(d+4)*(2*np.sqrt(np.pi))**d/volume)**(1/(d+4))
    elif name in ('biwe','triw') and d==2:
        constant = 2.7779 if name=='biwe' else 3.12
    else:
        raise ValueError("This kernel/dimension has no source normal-reference rule")
    eigenvectors = np.linalg.eigh(np.cov(a,rowvar=False))
    values,vectors = eigenvectors
    if np.any(values<=0):
        raise ValueError("Bandwidth needs nonsingular sample covariance")
    return constant*(vectors*np.sqrt(values))@vectors.T*n**(-1/(d+4))


from ._estimation import kdeoptset,kdefun,kdebin,evalpdf,fftce,hldpi,hbcv,hlscv
from ._gui import kde1dgui, kde2dgui, kdedemo1, kdedemo2
