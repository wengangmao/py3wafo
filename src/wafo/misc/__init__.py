"""Numerical utilities and physical constants."""
from ._interactive import (choices, choicex, cltext, fcolorbar, figtext, findpeaks,
                           fwaitbar, maximizefigs, mkcontents, wafostamp)
import numpy as np
from scipy import integrate, special
from ..cycles import levels
from ._extended import parseoptions,qrule2d,gaussq2d,ccquad,savgol,convlv,smooth,clevels,geth1line,wafoversion


def gravity(phi=45):
    """Normal gravity at latitude phi in degrees (WAFO formula)."""
    p = np.deg2rad(phi)
    return 9.78049*(1+0.0052884*np.sin(p)**2-0.0000059*np.sin(2*p)**2)


def wdensity(S=35, T=4, P=0):
    """UNESCO 1980 seawater density; salinity psu, temperature C, pressure dbar.

    Fixes an undefined local variable in the supplied MATLAB pressure branch.
    """
    S, T, P = np.broadcast_arrays(np.asarray(S, dtype=float), T, P)
    if np.any(S < 0):
        raise ValueError("Salinity must be nonnegative")
    sr = np.sqrt(S)
    rho = np.polyval([6.536332e-9, -1.120083e-6, 1.001685e-4, -9.095290e-3, 6.793952e-2, 999.842594], T)
    rho += (np.polyval([5.3875e-9, -8.2467e-7, 7.6438e-5, -4.0899e-3, .824493], T)
            + np.polyval([-1.6546e-6, 1.0227e-4, -.00572466], T)*sr + .00048314*S)*S
    K = np.polyval([-5.155288e-5, .01360477, -2.327105, 148.4206, 19652.21], T)
    K += (np.polyval([-6.167e-5, .0109987, -.603459, 54.6746], T)
          + np.polyval([-.00053009, .016483, .07944], T)*sr)*S
    A = np.polyval([-5.77905e-7, .000116092, .00143713, 3.239908], T)
    A += (np.polyval([-1.6078e-6, -1.0981e-5, .0022838], T)+.000191075*sr)*S
    B = np.polyval([5.2787e-8, -6.12293e-6, 8.50935e-5], T)
    B += np.polyval([9.1697e-10, 2.0816e-8, -9.9348e-7], T)*S
    p = P/10
    K += (A+B*p)*p
    return rho/(1-p/K)


def csort(a, amin=None, amax=None):
    """Stable integer sort, returning (values, zero-based permutation)."""
    a = np.asarray(a).ravel()
    if not np.all(np.isfinite(a)) or np.any(a != np.floor(a)):
        raise ValueError("csort requires finite integers")
    if amin is not None and np.any(a < amin) or amax is not None and np.any(a > amax):
        raise ValueError("Values outside requested bounds")
    ind = np.argsort(a, kind='stable')
    return a[ind], ind


def binom(n, m):
    return special.comb(n, m)


def sinc(x):
    return np.sinc(x)


def hypgf(a, b, c, z):
    return special.hyp2f1(a, b, c, z)


def simpson(x, y, dim=-1):
    return integrate.simpson(y, x=x, axis=dim)


def gaussq(fun, a, b, rtol=1e-8, atol=1e-10):
    """Adaptive quadrature: returns (integral, estimated absolute error)."""
    return integrate.quad(fun, a, b, epsrel=rtol, epsabs=atol)


def qrule(n, wfun=1, alpha=0, beta=0):
    """Gaussian nodes and weights using the original WAFO selectors 1–10."""
    def shifted(a, b):
        x, w = special.roots_jacobi(n, b, a)
        return (x+1)/2, w/2**(a+b+1)
    funcs = {1: lambda: special.roots_legendre(n), 2: lambda: special.roots_chebyt(n),
             3: lambda: special.roots_chebyu(n), 4: lambda: shifted(.5, -.5),
             5: lambda: shifted(0, -.5), 6: lambda: shifted(0, .5),
             7: lambda: special.roots_jacobi(n, beta, alpha),
             8: lambda: special.roots_genlaguerre(n, alpha),
             9: lambda: special.roots_hermite(n), 10: lambda: special.roots_legendre(n)}
    if wfun not in funcs:
        raise ValueError("wfun must be 1 through 10")
    return funcs[wfun]()


def genchol(A, tol=1e-12):
    """Pivoted PSD Cholesky: returns (L, permutation, rank)."""
    A = np.asarray(A, dtype=float)
    if not np.allclose(A, A.T):
        raise ValueError("Matrix must be symmetric")
    if np.any(np.linalg.eigvalsh(A) < -tol):
        raise ValueError("Matrix must be positive semidefinite")
    n = len(A)
    p = np.arange(n)
    L = np.zeros_like(A)
    d = np.diag(A).copy()
    rank = 0
    for j in range(n):
        pivot = j+np.argmax(d[j:])
        p[[j, pivot]] = p[[pivot, j]]
        d[[j, pivot]] = d[[pivot, j]]
        L[[j, pivot], :] = L[[pivot, j], :]
        if d[j] <= tol:
            break
        L[j, j] = np.sqrt(d[j])
        L[j+1:, j] = (A[p[j+1:], p[j]]-L[j+1:, :j]@L[j, :j])/L[j, j]
        d[j+1:] -= L[j+1:, j]**2
        rank += 1
    return L, p, rank


def loaddata(filename):
    return np.loadtxt(filename)


def comnsize(*args):
    """NumPy broadcasting of arguments (raises ValueError on mismatch)."""
    return tuple(np.broadcast_arrays(*args))
