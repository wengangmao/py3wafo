"""Additional source-grounded WAFO statistical capabilities.

Fitting routines use explicit modern optimization and return estimates only.
No plotting or undocumented method selector is silently ignored.
"""
from __future__ import annotations
import numpy as np
from scipy import stats, special, optimize, integrate


def _sample(x, positive=False):
    x = np.asarray(x,dtype=float).ravel()
    if x.size<2 or not np.all(np.isfinite(x)) or (positive and np.any(x<=0)):
        raise ValueError('Need at least two finite observations'+(' greater than zero' if positive else ''))
    return x


def walpharnd(a, *, size=None, seed=None):
    """Symmetric alpha-stable variates with characteristic function exp(-|t|^a)."""
    a = np.asarray(a,dtype=float)
    if not np.all(np.isfinite(a)) or np.any((a<=0)|(a>2)):
        raise ValueError('alpha must lie in (0,2]')
    return stats.levy_stable.rvs(a,0,size=size,random_state=np.random.default_rng(seed))


def wtweibcdf(x,a,b,c=0):
    """CDF of W-c given W>c for W~Weibull(scale=a,shape=b); c is absolute."""
    x,a,b,c=np.broadcast_arrays(x,a,b,np.abs(c))
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        z=-np.expm1(-((np.maximum(x,0)+c)/a)**b+(c/a)**b)
    return np.where((a<=0)|(b<=0),np.nan,np.where(x>=0,z,0))


def wtweibpdf(x,a,b,c=0):
    x,a,b,c=np.broadcast_arrays(x,a,b,np.abs(c))
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        z=(np.maximum(x,0)+c)/a
        f=np.exp(np.log(b/a)+(b-1)*np.log(z)-z**b+(c/a)**b)
    f=np.where((x==0)&(c==0)&(b==1),1/a,f)
    f=np.where(np.isposinf(x),0,f)
    return np.where((a<=0)|(b<=0),np.nan,np.where(x>=0,f,0))


def wtraylcdf(x,b,c=0,a=2):
    """Legacy truncated Rayleigh/Weibull CDF, requiring c<=0 for a valid law."""
    if np.any(np.asarray(c)>0):
        raise ValueError('Truncation location c must be nonpositive')
    return wtweibcdf(x,np.asarray(b)*2**(1/np.asarray(a)),a,-np.asarray(c))


def wtraylpdf(x,b,c=0):
    if np.any(np.asarray(c)>0):
        raise ValueError('Truncation location c must be nonpositive')
    return wtweibpdf(x,np.asarray(b)*np.sqrt(2),2,-np.asarray(c))


def _fit_truncated(x, weibull):
    x=_sample(x,True)
    def objective(z):
        a,b,c=(np.exp(z[0]),np.exp(z[1]),np.exp(z[2])) if weibull else (np.exp(z[0]),2,np.exp(z[1]))
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            v=np.log(b/a)+(b-1)*np.log((x+c)/a)-((x+c)/a)**b+(c/a)**b
        return -v.sum() if np.all(np.isfinite(v)) else 1e100
    starts=([np.log(np.mean(x)),np.log(1.5),np.log(np.mean(x)*.2)],
            [np.log(np.mean(x)*2),np.log(2),np.log(np.mean(x))]) if weibull else (
            [np.log(np.mean(x)),np.log(np.mean(x)*.2)],
            [np.log(np.mean(x)*2),np.log(np.mean(x))])
    candidates=[optimize.minimize(objective,z,method='Nelder-Mead',options={'maxiter':4000}) for z in starts]
    best=min(candidates,key=lambda r:r.fun)
    if not np.isfinite(best.fun) or not best.success:
        raise RuntimeError('Truncated-distribution fit did not converge')
    p=np.exp(best.x)
    return p if weibull else np.array([p[0]/np.sqrt(2),-p[1]])


def wtweibfit(x):
    return _fit_truncated(x,True)


def wtraylfit(x):
    return _fit_truncated(x,False)


def winvgfit(x):
    x=_sample(x,True)
    m=x.mean()
    denominator=np.mean(1/x)-1/m
    if denominator<=0:
        raise ValueError('Inverse Gaussian fit requires nonconstant data')
    return np.array([m,1/denominator])


def wggamfit(x, *, exponent=None):
    """Generalized gamma MLE [a,b,c], with positive exponent, scale b."""
    x=_sample(x,True)
    def objective(z):
        a,b,c=np.exp(z) if exponent is None else (*np.exp(z),float(exponent))
        value=stats.gengamma.logpdf(x,a,c,scale=b)
        return -value.sum() if np.all(np.isfinite(value)) else 1e100
    if exponent is not None and (not np.isfinite(exponent) or exponent<=0):
        raise ValueError('exponent must be positive')
    starts=([0,np.log(x.mean()),0], [np.log(2),np.log(x.mean()/2),0],
            [0,np.log(x.mean()),np.log(.5)]) if exponent is None else ([0,np.log(x.mean())],)
    results=[optimize.minimize(objective,start,method='Nelder-Mead',options={'maxiter':6000}) for start in starts]
    result=min(results,key=lambda v:v.fun)
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('Generalized gamma MLE did not converge')
    p=np.exp(result.x)
    return p if exponent is None else np.r_[p,exponent]


def wgpdfit_ml(x):
    """Generalized Pareto MLE with fixed location zero; WAFO shape sign."""
    x=_sample(x)
    if np.any(x<0):
        raise ValueError('GPD observations must be nonnegative')
    c,_,scale=stats.genpareto.fit(x,floc=0)
    return np.array([-c,scale,0.])


def loglike(phat,*args,return_covariance=False):
    """Negative log likelihood for named WAFO PDF or callable.

    Example loglike([2,3], samples, 'wweibpdf'). Optional covariance uses a
    central finite-difference observed information matrix and may be singular.
    """
    if len(args)<2:
        raise TypeError('Provide data arrays followed by a PDF name/callable')
    dist=args[-1]
    if isinstance(dist,str):
        from . import __dict__ as functions
        if not dist.endswith('pdf') or not callable(functions.get(dist)):
            raise ValueError('Unknown WAFO PDF')
        dist=functions[dist]
    if not callable(dist):
        raise TypeError('PDF must be a callable or WAFO PDF name')
    p=np.atleast_1d(np.asarray(phat,dtype=float))
    def fun(parameters):
        values=np.asarray(dist(*args[:-1],*parameters),dtype=float)
        return -np.log(values).sum() if np.all(np.isfinite(values)) and np.all(values>0) else np.inf
    value=fun(p)
    if not return_covariance:
        return float(value)
    steps=np.cbrt(np.finfo(float).eps)*(1+abs(p))
    H=np.empty((len(p),len(p)))
    for i in np.arange(len(p)):
        ei=np.zeros(len(p));ei[i]=steps[i]
        H[i,i]=(fun(p+ei)-2*value+fun(p-ei))/steps[i]**2
        for j in np.arange(i):
            ej=np.zeros(len(p));ej[j]=steps[j]
            H[i,j]=H[j,i]=(fun(p+ei+ej)-fun(p+ei-ej)-fun(p-ei+ej)+fun(p-ei-ej))/(4*steps[i]*steps[j])
    if not np.all(np.isfinite(H)) or np.any(np.linalg.eigvalsh(H)<=0):
        raise ValueError('Observed information is not positive definite at these parameters')
    return float(value),np.linalg.inv(H)


def wgevlike(p,sample,*,return_covariance=False):
    return loglike(p,sample,'wgevpdf',return_covariance=return_covariance)


def wgamafit(a,A,G):
    return np.log(a)-special.digamma(a)-np.log(np.asarray(A)/G)


def wggambfit(b,data,F,definition=1):
    """Source generalized-gamma likelihood/least-squares/moment objectives."""
    if definition==1:
        x=_sample(data,True);ld=np.asarray(F,dtype=float)
        if ld.shape!=x.shape or not np.all(np.isfinite(ld)) or b<=0:
            raise ValueError('Need scalar b>0 and log-data with matching sample shape')
        weights=np.exp(b*ld-np.max(b*ld))
        a=-1/(b*(ld.mean()-np.dot(weights,ld)/weights.sum()))
        return np.log(len(x)*a)-special.digamma(a)+b*ld.mean()-special.logsumexp(b*ld)
    if definition==2:
        a,scale,c=np.asarray(b,dtype=float)
        expected=-np.log1p(-np.asarray(F))
        actual=-stats.gengamma.logsf(data,a,c,scale=scale)
        return np.mean((np.sqrt(expected)-np.sqrt(actual))**2)
    if definition in (3,4):
        a,c=np.asarray(b,dtype=float)
        if a<=0 or c<=0:
            return np.inf
        logg=lambda k:special.gammaln(a+k/c)
        ratio23=np.exp(.5*logg(0)+logg(3)-1.5*logg(2))
        if definition==3:
            ratio12=np.exp(logg(0)+logg(2)-2*logg(1))
            return float((ratio12-data)**2+(ratio23-F)**2)
        ratio24=np.exp(logg(0)+logg(4)-2*logg(2))
        return float((ratio23-data)**2+(ratio24-F)**2)
    raise ValueError('definition must be 1,2,3,4')


def wgumbafit(a,data):
    x=_sample(data)
    if a<=0:
        return np.nan
    w=np.exp(-(x-x.min())/a)
    return a-x.mean()+np.sum(x*w)/w.sum()


def wweibcfit(c,data):
    x=_sample(data,True)
    l=np.log(x)
    weights=np.exp(c*l-np.max(c*l))
    return 1/c-np.sum(weights*l)/weights.sum()+l.mean()


def wtweibfun(phat,x,F,definition=1,monitor=False):
    """Source least-squares objectives 1–4 for truncated Weibull estimation."""
    if monitor:
        raise ValueError('Plotting monitor is unavailable; evaluate the objective directly')
    p=np.atleast_1d(phat)
    a,b,c=p[0],p[1] if len(p)>1 else 2,p[2] if len(p)>2 else 0
    if a<=0 or b<=0:
        return np.inf
    x,F=np.broadcast_arrays(x,F)
    if np.any((F<0)|(F>=1)):
        raise ValueError('F must lie in [0,1)')
    t=-np.log1p(-F)
    predicted=((x+abs(c))/a)**b-(abs(c)/a)**b
    if definition==1:
        residual=np.sqrt(t)-np.sqrt(predicted)
    elif definition==2:
        residual=t-predicted
    elif definition==3:
        residual=t**(1/b)-predicted**(1/b)
    elif definition==4:
        residual=(t+(abs(c)/a)**b)**(1/b)-(x+abs(c))/a
    else:
        raise ValueError('Only documented objective definitions 1–4 are supported')
    return float(np.mean(residual**2)+np.size(F)*abs(c)*(c<0))


def range(x,dim=0):
    return np.ptp(x,axis=dim)


def var(x,flag=0,dim=0):
    if flag not in (0,1):
        raise ValueError('flag must be 0 (sample) or 1 (population)')
    return np.var(x,axis=dim,ddof=1-flag)


def comnsize(*args):
    """Common-size broadcasting; raises for incompatible sizes, like wafo.misc."""
    return tuple(np.broadcast_arrays(*args))
