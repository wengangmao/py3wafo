"""Portable Gaussian expectations replacing the RIND Fortran capability.

The mathematical integral follows ``rind.m`` and ``source/jacobmod.f``. The
backend is a new randomized Sobol implementation, not a translation of the
legacy integration algorithms. Errors are estimated 99% replicate intervals,
not deterministic bounds. Singular *integrated* covariance is supported;
singular conditioning covariance has no ordinary joint density and is rejected.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc, t as student_t


def _covariance(S, coveps=1e-10):
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or not np.all(np.isfinite(S)):
        raise ValueError("Covariance must be finite and square")
    scale = max(np.max(np.abs(np.diag(S)), initial=0), np.finfo(float).tiny)
    tolerance = coveps*scale
    if not np.allclose(S, S.T, rtol=1e-12, atol=tolerance):
        # The legacy RIND interface explicitly allows upper triangle only.
        if np.all(np.tril(S, -1) == 0):
            S = np.triu(S)+np.triu(S, 1).T
        else:
            raise ValueError("Covariance must be symmetric or upper triangular")
    S = (S+S.T)/2
    if np.linalg.eigvalsh(S).min(initial=0) < -tolerance:
        raise ValueError("Covariance is not positive semidefinite")
    return S


def _pivoted_factor(S, coveps):
    """Low-rank Cholesky S[p,p]~L L.T, with diagonal residual <= tolerance."""
    n = len(S)
    L, p = np.zeros_like(S), np.arange(n)
    diagonal = np.diag(S).copy()
    tolerance = coveps*max(diagonal.max(initial=0), np.finfo(float).tiny)
    rank = 0
    for j in range(n):
        pivot = j+int(np.argmax(diagonal[j:]))
        p[[j,pivot]] = p[[pivot,j]]
        diagonal[[j,pivot]] = diagonal[[pivot,j]]
        L[[j,pivot]] = L[[pivot,j]]
        if diagonal[j] <= tolerance:
            break
        L[j,j] = np.sqrt(diagonal[j])
        L[j+1:,j] = (S[p[j+1:],p[j]]-L[j+1:,:j]@L[j,:j])/L[j,j]
        diagonal[j+1:] -= L[j+1:,j]**2
        rank += 1
    return L,p,rank


def _normal_mass(a,b):
    """P(a<Z<b), avoiding cancellation in positive normal tails."""
    a,b = np.broadcast_arrays(a,b)
    return np.maximum(0,np.where(a>0,ndtr(-a)-ndtr(-b),ndtr(b)-ndtr(a)))


def _truncated_abs_moment(center, sigma, low, high):
    """Integral |center+sigma*z| phi(z) over standardized bounds."""
    split = -center/sigma
    bneg = np.minimum(high,split)
    apos = np.maximum(low,split)
    phi = lambda z: np.exp(-z*z/2)/np.sqrt(2*np.pi)
    neg = np.where(bneg>low, center*_normal_mass(low,bneg)+sigma*(phi(low)-phi(bneg)),0)
    pos = np.where(high>apos,center*_normal_mass(apos,high)+sigma*(phi(apos)-phi(high)),0)
    return np.maximum(0,pos-neg)


def _sequential_integrand(points, mean, L, lower, upper, weighted, rank):
    """Genz-style conditional truncated normals with optional |Xd| product.

    Final nonsingular variable is integrated analytically to reduce variance.
    ``lower/upper`` may be per-sample arrays for noncentral-t mixtures.
    """
    count,n = len(points),len(mean)
    z = np.zeros((count,n))
    weights = np.ones(count)
    lower = np.broadcast_to(lower,(count,n))
    upper = np.broadcast_to(upper,(count,n))
    tiny = np.finfo(float).tiny
    for j in range(n):
        center = mean[j]+z[:,:j]@L[j,:j]
        sigma = L[j,j]
        if j >= rank:
            weights *= (lower[:,j]<center)&(center<upper[:,j])
            if weighted[j]:
                weights *= abs(center)
            continue
        a,b = (lower[:,j]-center)/sigma,(upper[:,j]-center)/sigma
        mass = _normal_mass(a,b)
        if j == n-1:
            weights *= _truncated_abs_moment(center,sigma,a,b) if weighted[j] else mass
            continue
        u = points[:,j]
        # In positive tails invert survival probabilities, not 1-small_value.
        tail = ndtr(-a)-u*mass
        normal = ndtr(a)+u*mass
        z[:,j] = np.where(a>0,-ndtri(np.clip(tail,tiny,np.nextafter(1.,0.))),
                         ndtri(np.clip(normal,tiny,np.nextafter(1.,0.))))
        weights *= mass
        if weighted[j]:
            weights *= abs(center+sigma*z[:,j])
    return weights


def _sobol_estimate(integrand, dimensions, *, maxpts, minpts, replications,
                    abseps, releps, seed, uncertain_zero=False):
    """Adaptive replicated Sobol estimate; evaluation count never exceeds maxpts."""
    for name,value in [('maxpts',maxpts),('minpts',minpts),('replications',replications)]:
        if int(value)!=value or value<0:
            raise ValueError(f"{name} must be a nonnegative integer")
    maxpts,minpts,replications = int(maxpts),int(minpts),int(replications)
    if replications<2 or maxpts<2*replications or minpts>maxpts:
        raise ValueError("Need >=2 replications, >=2 points each, and minpts<=maxpts")
    if abseps<0 or releps<0 or not np.isfinite(abseps+releps):
        raise ValueError("Tolerances must be finite and nonnegative")
    cap = int(np.floor(np.log2(maxpts//replications)))
    start = min(cap,5)
    rng = np.random.default_rng(seed)
    engines = [qmc.Sobol(max(1,dimensions),scramble=True,seed=int(rng.integers(2**32))) for _ in range(replications)]
    totals = np.zeros(replications)
    done = 0
    critical = student_t.ppf(.995,replications-1)
    for power in range(start,cap+1):
        target = 2**power
        for j,engine in enumerate(engines):
            values = np.asarray(integrand(engine.random(target-done)),dtype=float)
            if values.shape!=(target-done,) or not np.all(np.isfinite(values)):
                raise FloatingPointError("Nonfinite or invalid integrand; integral may overflow")
            totals[j] += values.sum()
        done = target
        means = totals/done
        value = float(means.mean())
        error = float(critical*means.std(ddof=1)/np.sqrt(replications))
        # Singular deterministic constraints can produce missed rare events.
        # A zero sampled indicator is not a proof of zero probability.
        if uncertain_zero and np.all(means==0):
            error = float('inf')
        evaluations = done*replications
        if evaluations>=minpts and error<=max(abseps,releps*abs(value)):
            return value,error,0,evaluations
    return value,error,1,evaluations


_DEFAULTS = dict(method='sobol',abseps=1e-6,releps=1e-3,coveps=1e-10,
                 maxpts=65536,minpts=256,replications=8,seed=None,xcscale=0.)


def rindoptset(options=None, **kwargs):
    """Create options for the Python RIND Sobol engine (legacy selectors differ)."""
    out = _DEFAULTS.copy()
    if options is not None:
        out.update(options)
    out.update(kwargs)
    unknown = set(out)-set(_DEFAULTS)
    if unknown:
        raise TypeError('Unsupported RIND options: '+', '.join(sorted(unknown)))
    if out['method']!='sobol':
        raise NotImplementedError("Python RIND implements method='sobol'; native method selectors are not equivalent")
    if not np.isfinite(out['coveps']) or out['coveps']<=0:
        raise ValueError("coveps must be positive and finite")
    if not np.isfinite(out['xcscale']):
        raise ValueError("xcscale must be finite")
    return out


def _barriers(B, ntd, nc, indI):
    b = np.asarray(B,dtype=float)
    if b.ndim==0:
        b = b.reshape(1,1)
    elif b.ndim==1:
        b = b[None,:]
    if b.ndim!=2 or b.shape[0]>nc+1 or np.any(np.isnan(b)):
        raise ValueError("Barriers must have at most Nc+1 rows and contain no NaNs")
    if indI is None:
        if b.shape[1]!=ntd:
            raise ValueError("Without indI, provide one barrier column per integrated variable")
        indices = np.arange(ntd+1)
    else:
        indices = np.asarray(indI)
        if (indices.ndim!=1 or len(indices)!=b.shape[1]+1 or indices[0]!=0 or
            indices[-1]!=ntd or np.any(indices!=np.floor(indices)) or np.any(np.diff(indices)<0)):
            raise ValueError("indI must be nondecreasing cumulative counts starting 0 and ending Nt+Nd")
    b = np.repeat(b,np.diff(indices).astype(int),axis=1)
    if b.shape[0]>1 and not np.all(np.isfinite(b[1:])):
        raise ValueError("Affine barrier coefficients must be finite")
    return b


def rind(S,m,Blo,Bup,indI=None,xc=None,Nt=None,options=None,*,full_output=False,**kwargs):
    """E[|product Xd| I(Blo<Xt,Xd<Bup)|Xc=xc] times f_Xc(xc).

    Variables are ordered [Xt,Xd,Xc]. ``indI`` are cumulative counts starting at
    zero (same convention as MATLAB); barrier row0 is constant and later rows
    multiply conditioning values. ``xc`` has shape (Nc,Nx). The default output
    is an array of Nx estimates. ``full_output=True`` returns
    ``(values,errors,elapsed,info)``; info contains convergence flags and actual
    evaluations. Errors estimate 99% replicate uncertainty, not rigorous bounds.
    Maxpts applies per conditioning point. Native algorithm selectors, special
    Jacobians and singular conditioning densities are not silently approximated.
    """
    start = time.perf_counter()
    opts = rindoptset(options,**kwargs)
    S = _covariance(S,opts['coveps'])
    m = np.asarray(m,dtype=float).reshape(-1)
    n = len(S)
    if len(m)!=n or not np.all(np.isfinite(m)):
        raise ValueError("Mean must be a finite vector matching covariance")
    if xc is None or np.size(xc)==0:
        xc = np.empty((0,1))
    else:
        xc = np.asarray(xc,dtype=float)
        if xc.ndim==1:
            xc = xc[:,None]
        if xc.ndim!=2 or not np.all(np.isfinite(xc)):
            raise ValueError("xc must be a finite Nc by Nx matrix")
    nc,nx = xc.shape
    ntd = n-nc
    Nt = ntd if Nt is None else Nt
    if np.ndim(Nt)!=0:
        raise NotImplementedError("Python RIND accepts scalar Nt and the absolute derivative-product Jacobian")
    if int(Nt)!=Nt or not 0<=Nt<=ntd:
        raise ValueError("Nt must be an integer in [0,Nt+Nd]")
    Nt = int(Nt)
    lo = _barriers(Blo,ntd,nc,indI)
    hi = _barriers(Bup,ntd,nc,indI)
    # Barrier matrices may independently omit zero affine-coefficient rows.
    lo_values = np.broadcast_to(lo[0,:,None],(ntd,nx)).copy()
    hi_values = np.broadcast_to(hi[0,:,None],(ntd,nx)).copy()
    if len(lo)>1:
        lo_values += lo[1:].T@xc[:len(lo)-1]
    if len(hi)>1:
        hi_values += hi[1:].T@xc[:len(hi)-1]
    means = np.broadcast_to(m[:ntd,None],(ntd,nx)).copy()
    density = np.full(nx,np.exp(opts['xcscale']))
    cov = S[:ntd,:ntd].copy()
    if nc:
        cc = S[ntd:,ntd:]
        try:
            chol = np.linalg.cholesky(cc)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Singular conditioning covariance has no ordinary joint density") from exc
        if np.linalg.eigvalsh(cc).min()<=opts['coveps']*max(np.diag(cc).max(),np.finfo(float).tiny):
            raise ValueError("Conditioning covariance is numerically singular at coveps")
        displacement = xc-m[ntd:,None]
        solved = np.linalg.solve(cc,displacement)
        means += S[:ntd,ntd:]@solved
        # The Schur complement is symmetric by construction; keep it so in
        # floating point, where conditional variances can be tiny.
        W = np.linalg.solve(chol,S[ntd:,:ntd])
        cov -= W.T@W
        cov = (cov+cov.T)/2
        standardized = np.linalg.solve(chol,displacement)
        density = np.exp(-.5*np.sum(standardized**2,axis=0)-np.log(np.diag(chol)).sum()-nc/2*np.log(2*np.pi)+opts['xcscale'])
    if not np.all(np.isfinite(density)):
        raise FloatingPointError("Conditioning density or xcscale overflows double precision")
    cov = _covariance(cov,opts['coveps'])
    L,p,rank = _pivoted_factor(cov,opts['coveps'])
    weighted = p>=Nt
    values,errors = np.zeros(nx),np.zeros(nx)
    flags,evals = np.zeros(nx,dtype=int),np.zeros(nx,dtype=int)
    seed_rng = np.random.default_rng(opts['seed'])
    for ix in range(nx):
        low,high,mean = lo_values[p,ix],hi_values[p,ix],means[p,ix]
        if density[ix]==0 or np.any(low>=high):
            continue
        if rank==0:
            values[ix] = density[ix]*np.prod(abs(mean[weighted]))*np.all((low<mean)&(mean<high))
            continue
        with np.errstate(divide='ignore',over='ignore'):
            abseps = opts['abseps']/density[ix]
        if not np.isfinite(abseps):
            # A subnormal conditioning density contributes below double precision.
            continue
        integrand = lambda u: _sequential_integrand(u,mean,L,low,high,weighted,rank)
        val,err,flag,count = _sobol_estimate(integrand,rank,
            maxpts=opts['maxpts'],minpts=opts['minpts'],replications=opts['replications'],
            abseps=abseps,releps=opts['releps'],
            seed=int(seed_rng.integers(2**32)),uncertain_zero=rank<ntd)
        values[ix],errors[ix],flags[ix],evals[ix] = val*density[ix],err*density[ix],flag,count
    residual = float(np.max(abs(cov[np.ix_(p,p)]-L@L.T),initial=0))
    info = dict(opts,inform=flags,evaluations=evals,rank=rank,covariance_residual=residual,
                requested_tolerance=np.maximum(opts['abseps'],opts['releps']*abs(values)),
                error_description='Estimated 99% interval across independently scrambled Sobol replicates; not a rigorous bound')
    return (values,errors,time.perf_counter()-start,info) if full_output else values
