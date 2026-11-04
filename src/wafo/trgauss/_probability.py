"""Normal and noncentral-t integration capabilities of the native gateways.

The t parameter is numerator noncentrality: (Z+delta)/sqrt(V/nu), with one
shared independent chi-square V. A shifted central-t distribution is different.
Legacy quadrature/method-selection codes are not reproduced.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.stats import chi2, nct, norm
from ._rind import (_covariance, _normal_mass, _pivoted_factor,
                    _sequential_integrand, _sobol_estimate, rind)


def _vectors(A,B):
    a,b = np.asarray(A,dtype=float).reshape(-1),np.asarray(B,dtype=float).reshape(-1)
    if not len(a) or a.shape!=b.shape or np.any(np.isnan(a)) or np.any(np.isnan(b)):
        raise ValueError("Bounds must be nonempty vectors of equal length, without NaNs")
    if np.any(a>b):
        raise ValueError("Lower bounds must not exceed upper bounds")
    return a,b


def mvnormprb_full(correl,A,B,abseps=1e-6,releps=1e-3,maxpts=65536,*,seed=None,
                   minpts=256,replications=8,coveps=1e-10):
    """Normal rectangle probability: (value, estimated absolute error, inform).

    This is the RIND Sobol backend with no derivative weight. inform=0 means
    estimated tolerance met; 1 means the budget was reached. The error is an
    estimated 99% replicate interval, not a rigorous or legacy error bound.
    """
    a,b = _vectors(A,B)
    S = _covariance(correl,coveps)
    if S.shape!=(len(a),len(a)):
        raise ValueError("Covariance dimension must match the bounds")
    result,error,_,info = rind(S,np.zeros(len(a)),a,b,Nt=len(a),full_output=True,
        abseps=abseps,releps=releps,maxpts=maxpts,minpts=minpts,replications=replications,
        seed=seed,coveps=coveps)
    return float(result[0]),float(error[0]),int(info['inform'][0])


def mvnormpcprb_full(rho,a,b,tol=1e-6,*,releps=1e-6):
    """Product-correlation normal probability by a common-factor integral.

    Returns (value, quadrature error estimate, inform). Coefficients equal to
    +/-1 impose deterministic bounds on the common factor and are supported.
    """
    a,b = _vectors(a,b)
    rho = np.asarray(rho,dtype=float).reshape(-1)
    if rho.shape!=a.shape or not np.all(np.isfinite(rho)) or np.any(abs(rho)>1):
        raise ValueError("rho must match bounds and lie in [-1,1]")
    if tol<=0 or releps<0 or not np.isfinite(tol+releps):
        raise ValueError("Invalid integration tolerance")
    if np.any(a==b):
        return 0.,0.,0
    low,high = -np.inf,np.inf
    deterministic = abs(rho)==1
    for aa,bb,rr in zip(a[deterministic],b[deterministic],rho[deterministic]):
        left,right = (aa,bb) if rr>0 else (-bb,-aa)
        low,high = max(low,left),min(high,right)
    if low>=high:
        return 0.,0.,0
    r = rho[~deterministic]
    sd = np.sqrt(1-r*r)
    lower,upper = a[~deterministic],b[~deterministic]
    if not len(r):
        return float(_normal_mass(low,high)),0.,0
    independent = r==0
    fixed = np.prod(_normal_mass(lower[independent],upper[independent]))
    if np.all(independent):
        return float(fixed*_normal_mass(low,high)),0.,0
    r,sd,lower,upper = r[~independent],sd[~independent],lower[~independent],upper[~independent]
    def integrand(z):
        return float(norm.pdf(z)*np.prod(_normal_mass((lower-r*z)/sd,(upper-r*z)/sd)))
    # Split at transition locations to avoid missing narrow common-factor peaks.
    breaks = sorted(set([low,high]+[float(x) for x in np.r_[lower/r,upper/r] if low<x<high and np.isfinite(x)]))
    # Infinite endpoints are handled by QUAD rather than a finite normal cutoff.
    value,error = 0.,0.
    for aa,bb in zip(breaks[:-1],breaks[1:]):
        v,e = quad(integrand,aa,bb,epsabs=tol/(max(fixed,np.finfo(float).tiny)*max(1,len(breaks)-1)),epsrel=releps,limit=200)
        value,error = value+v,error+e
    value,error = value*fixed,error*fixed
    return float(value),float(error),int(error>max(tol,releps*abs(value)))


def mvtprb(cov,A,B,nu,delta=None,constr=None,*,abseps=1e-6,releps=1e-3,
           maxpts=65536,minpts=256,replications=8,seed=None,coveps=1e-10,
           full_output=False):
    """P(A < (constr @ Z + delta)/sqrt(V/nu) < B).

    Z~N(0,cov), V~chi-square(nu), independent; nu<=0 gives the normal case
    with denominator one. ``delta`` has one entry per constraint, matching the
    native mvdist.f90 contract (its noncentrality lies inside the scale mixture).
    The default constr is identity. Singular transformed covariance is allowed.

    Returns a scalar, or (value,error,inform) with full_output=True. Errors for
    the general t integrator are estimated 99% independent-Sobol replicate
    intervals; these are not rigorous bounds. No legacy method code is accepted.
    """
    a,b = _vectors(A,B)
    S = np.asarray(cov,dtype=float)
    if S.ndim==2 and np.all(np.triu(S,1)==0):
        S = np.tril(S)+np.tril(S,-1).T
    S = _covariance(S,coveps)
    C = np.eye(len(S)) if constr is None else np.asarray(constr,dtype=float)
    if C.shape!=(len(a),len(S)) or not np.all(np.isfinite(C)):
        raise ValueError("constr must be finite with shape (number of bounds, covariance dimension)")
    d = np.zeros(len(a)) if delta is None else np.asarray(delta,dtype=float).reshape(-1)
    if d.shape!=a.shape or not np.all(np.isfinite(d)) or not np.isfinite(nu):
        raise ValueError("Invalid noncentrality or degrees of freedom")
    transformed = _covariance(C@S@C.T,coveps)
    if np.any(a==b):
        result = (0.,0.,0)
    elif nu<=0:
        result = mvnormprb_full(transformed,a-d,b-d,abseps,releps,maxpts,
            seed=seed,minpts=minpts,replications=replications,coveps=coveps)
    elif len(a)==1 and transformed[0,0]>0:
        sigma = np.sqrt(transformed[0,0])
        dist = nct(nu,d[0]/sigma)
        aa,bb = a[0]/sigma,b[0]/sigma
        value = dist.sf(aa)-dist.sf(bb) if aa>0 else dist.cdf(bb)-dist.cdf(aa)
        result = (float(value),0.,0)  # Univariate SciPy analytic evaluator; no QMC sampling error.
    else:
        L,p,rank = _pivoted_factor(transformed,coveps)
        def integrand(u):
            scale = np.sqrt(chi2.ppf(np.clip(u[:,0],np.finfo(float).tiny,np.nextafter(1.,0.)),nu)/nu)
            lower = scale[:,None]*a[None,p]-d[None,p]
            upper = scale[:,None]*b[None,p]-d[None,p]
            return _sequential_integrand(u[:,1:],np.zeros(len(a)),L,lower,upper,np.zeros(len(a),bool),rank)
        value,error,flag,_ = _sobol_estimate(integrand,rank+1,maxpts=maxpts,minpts=minpts,
            replications=replications,abseps=abseps,releps=releps,seed=seed,
            uncertain_zero=rank<len(a))
        result = value,error,flag
    return result if full_output else result[0]


def mvnortpcprb(RHO,A,B,D=None,NDF=0,abseps=1e-4,IERC=0,HNC=.24,*,
               releps=1e-3,maxpts=65536,seed=None,full_output=False):
    """Normal or noncentral-t rectangle probability with product correlation.

    D is numerator noncentrality. The normal case uses adaptive common-factor
    quadrature; the t case uses a shared chi-square scale mixture. IERC!=0 and
    nondefault HNC select native algorithms and are explicitly unsupported.
    Returns probability; full_output=True returns (value,error,inform).
    """
    rho = np.asarray(RHO,dtype=float).reshape(-1)
    a,b = _vectors(A,B)
    if rho.shape!=a.shape or not np.all(np.isfinite(rho)) or np.any(abs(rho)>1):
        raise ValueError("rho must match bounds and lie in [-1,1]")
    if IERC!=0 or HNC!=.24:
        raise NotImplementedError("Legacy derivative error bounds/Simpson start widths are not implemented")
    d = np.zeros(len(a)) if D is None else np.asarray(D,dtype=float).reshape(-1)
    if d.shape!=a.shape or not np.all(np.isfinite(d)):
        raise ValueError("D must match the bounds and be finite")
    if NDF<=0:
        result = mvnormpcprb_full(rho,a-d,b-d,abseps,releps=releps)
    else:
        cov = np.outer(rho,rho)
        np.fill_diagonal(cov,1.)
        result = mvtprb(cov,a,b,NDF,d,abseps=abseps,releps=releps,maxpts=maxpts,seed=seed,full_output=True)
    return result if full_output else result[0]
