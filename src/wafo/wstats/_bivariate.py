"""Bivariate Weibull and Plackett capabilities from the supplied WAFO sources."""
import numpy as np
from scipy import stats, special, integrate, optimize


def _parameters(param,others):
    p=np.asarray((param,*others) if others else param,dtype=float).ravel()
    if p.shape!=(5,) or not np.all(np.isfinite(p)) or np.any(p[:4]<=0) or abs(p[4])>=1:
        raise ValueError('Parameters must be [A1,B1,A2,B2,C12], first four positive and |C12|<1')
    return p


def weib2dpdf(x1,x2,param,*parameters,condon=0):
    """2D Weibull density, or conditional density selected by condon=1/2.

    Param is the five-element vector; alternatively pass five scalar parameters.
    Conditional densities use the exact noncentral-chi-square representation.
    """
    a,b,c,d,r=_parameters(param,parameters)
    if condon not in (0,1,2,3,4):
        raise ValueError('condon must be 0,1,2,3,4')
    x,y=np.broadcast_arrays(np.asarray(x1,dtype=float),np.asarray(x2,dtype=float))
    n=1-r*r
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        u=(np.maximum(x,0)/a)**b;v=(np.maximum(y,0)/c)**d
        t=2*abs(r)*np.sqrt(u*v)/n
        common=-(u+v)/n+t+np.log(special.i0e(t))-np.log(n)
        logdx=np.log(b/a)+(b-1)*np.log(np.maximum(x,0)/a)
        logdy=np.log(d/c)+(d-1)*np.log(np.maximum(y,0)/c)
        logdx=np.where((x==0)&(b==1),np.log(b/a),logdx)
        logdy=np.where((y==0)&(d==1),np.log(d/c),logdy)
        if condon==0:
            f=np.exp(common+logdx+logdy)
        elif condon==1:
            f=np.exp(common+u+logdy)
        else:
            f=np.exp(common+v+logdx)
            if condon in (3,4):
                f*=x**(condon-2)
    f=np.where(np.isinf(x)|np.isinf(y),0,f)
    return np.where((x>=0)&(y>=0),f,0)


def weib2dcdf(x1,x2,param,*parameters,condon=0,tol=1e-8):
    a,b,c,d,r=_parameters(param,parameters)
    x,y=np.broadcast_arrays(np.asarray(x1,dtype=float),np.asarray(x2,dtype=float))
    u=(np.maximum(x,0)/a)**b;v=(np.maximum(y,0)/c)**d;n=1-r*r
    if condon==1:
        result=stats.ncx2.cdf(2*v/n,2,2*r*r*u/n)
    elif condon==2:
        result=stats.ncx2.cdf(2*u/n,2,2*r*r*v/n)
    elif condon==0:
        result=np.zeros(x.shape)
        for index in np.ndindex(x.shape):
            if x[index]>0 and y[index]>0:
                if r==0:
                    result[index]=-np.expm1(-u[index])*-np.expm1(-v[index])
                else:
                    result[index]=integrate.quad(lambda z:np.exp(-z)*stats.ncx2.cdf(2*v[index]/n,2,2*r*r*z/n),0,u[index],epsabs=tol,epsrel=tol)[0]
    else:
        raise ValueError('condon must be 0,1,2')
    return np.where((x>=0)&(y>=0),result,0)


def weib2dcinv(x1,p,param,*parameters):
    a,b,c,d,r=_parameters(param,parameters)
    x,p=np.broadcast_arrays(x1,p)
    if np.any(x<0) or np.any((p<0)|(p>1)):
        raise ValueError('Require x1>=0 and probabilities in [0,1]')
    n=1-r*r
    return c*(n/2*stats.ncx2.ppf(p,2,2*r*r*(x/a)**b/n))**(1/d)


def weib2drnd(param, *, size=1, seed=None):
    a,b,c,d,r=_parameters(param,())
    rng=np.random.default_rng(seed)
    x,y,z,w=rng.normal(size=(4,)+(() if size is None else (size,) if np.isscalar(size) else tuple(size)))
    u=(x*x+y*y)/2
    v=((r*x+np.sqrt(1-r*r)*z)**2+(r*y+np.sqrt(1-r*r)*w)**2)/2
    return a*u**(1/b),c*v**(1/d)


def weib2dstat(param,condon=0,cvar=None):
    """Return mean and covariance (or conditional mean/variance).

    Conditional moments use the noncentral chi-square fractional-moment identity.
    """
    a,b,c,d,r=_parameters(param,())
    m1=a*special.gamma(1+1/b);m2=c*special.gamma(1+1/d)
    v1=a*a*special.gamma(1+2/b)-m1*m1
    v2=c*c*special.gamma(1+2/d)-m2*m2
    if condon==0:
        covariance=m1*m2*(special.hyp2f1(-1/b,-1/d,1,r*r)-1)
        return np.array([m1,m2]),np.array([[v1,covariance],[covariance,v2]])
    if condon not in (1,2) or cvar is None or np.any(np.asarray(cvar)<0):
        raise ValueError('Conditional moments require condon=1/2 and nonnegative cvar')
    if condon==2:
        a,b,c,d=c,d,a,b
    u=(np.asarray(cvar)/a)**b;n=1-r*r
    mean=c*n**(1/d)*special.gamma(1+1/d)*special.hyp1f1(-1/d,1,-r*r*u/n)
    second=c*c*n**(2/d)*special.gamma(1+2/d)*special.hyp1f1(-2/d,1,-r*r*u/n)
    return mean,np.maximum(0,second-mean*mean)


def weib2dprb(param,x1lo,x1up,x2lo,x2up):
    if np.any(np.asarray(x1lo)>x1up) or np.any(np.asarray(x2lo)>x2up):
        raise ValueError('Lower rectangle bounds must not exceed upper bounds')
    return (weib2dcdf(x1up,x2up,param)-weib2dcdf(x1lo,x2up,param)
            -weib2dcdf(x1up,x2lo,param)+weib2dcdf(x1lo,x2lo,param))


def weib2dlike(param,data1,data2):
    f=weib2dpdf(data1,data2,param)
    return float(-np.log(f).sum()) if np.all(np.isfinite(f)) and np.all(f>0) else np.inf


def weib2dfit(data1,data2):
    x,y=np.asarray(data1,dtype=float).ravel(),np.asarray(data2,dtype=float).ravel()
    if len(x)!=len(y) or len(x)<5 or np.any(x<=0) or np.any(y<=0) or not np.all(np.isfinite([x,y])):
        raise ValueError('Need matching finite positive samples of at least five pairs')
    b,_,a=stats.weibull_min.fit(x,floc=0)
    d,_,c=stats.weibull_min.fit(y,floc=0)
    def objective(z):
        p=np.r_[np.exp(z[:4]),np.tanh(z[4])]
        return weib2dlike(p,x,y)
    initial=np.r_[np.log([a,b,c,d]),.5]
    result=optimize.minimize(objective,initial,method='Nelder-Mead',options={'maxiter':5000})
    if not result.success:
        raise RuntimeError('Bivariate Weibull maximum likelihood did not converge')
    return np.r_[np.exp(result.x[:4]),abs(np.tanh(result.x[4]))]


def weib2dpdf2(x1,x2,param,*,condon=0):
    """PDF mapping on a mesh; axes x1 (columns), x2 (rows)."""
    from ..spec import Spectrum
    X,Y=np.meshgrid(x1,x2)
    return Spectrum(f=weib2dpdf(X,Y,param,condon=condon),x=[np.asarray(x1),np.asarray(x2)],
                    labx=['X1','X2'],note='2D Weibull',phat=np.asarray(param))


def _marginal(name,parameters):
    from . import _FAMILIES
    key=name.lower()[:2]
    codes={'we':'weib','ra':'rayl','ga':'gam','gg':'ggam','lo':'logn','gu':'gumb','no':'norm','ex':'exp'}
    p=np.atleast_1d(parameters)
    if key=='tg':
        return _FAMILIES['gumb'](*p,1)
    if key not in codes:
        raise ValueError('Unsupported marginal distribution '+name)
    return _FAMILIES[codes[key]](*p)


def _plackett(phat):
    x=phat['x']; dist=phat['dist']
    psi=float(x[2])
    if not np.isfinite(psi) or psi<=0 or len(dist)!=2:
        raise ValueError('Need two marginal families and positive finite interaction psi')
    return _marginal(dist[0],x[0]),_marginal(dist[1],x[1]),psi


def _copula(u,v,psi,conditional=0):
    if psi==1:
        return u*v if conditional==0 else (v if conditional==1 else u)
    q=psi-1
    A=1+q*(u+v)
    root=np.sqrt(np.maximum(0,A*A-4*psi*q*u*v))
    if conditional==1:
        return .5*(1-(A-2*psi*v)/root)
    if conditional==2:
        return .5*(1-(A-2*psi*u)/root)
    # Rationalized form avoids cancellation when q is small.
    return 2*psi*u*v/(A+root)


def _copula_density(u,v,psi):
    """Plackett density with its cancellation-resistant discriminant."""
    q=psi-1
    middle=u+v-2*u*v
    discriminant=1+2*q*middle+q*q*(u-v)**2
    return psi*(1+q*middle)/discriminant**1.5


def mdist2dcdf(x1,x2,phat,condon=0):
    if condon not in (0,1,2):
        raise ValueError('condon must be 0,1,2')
    d1,d2,psi=_plackett(phat)
    return _copula(d1.cdf(x1),d2.cdf(x2),psi,condon)


def mdist2dpdf(x1,x2,phat,condon=0):
    if condon not in (0,1,2):
        raise ValueError('condon must be 0,1,2')
    d1,d2,psi=_plackett(phat)
    u,v=d1.cdf(x1),d2.cdf(x2)
    density=_copula_density(u,v,psi)
    if condon!=1:
        density*=d1.pdf(x1)
    if condon!=2:
        density*=d2.pdf(x2)
    return density


def mdist2dcinv(x1,p,phat):
    d1,d2,psi=_plackett(phat)
    u,p=np.broadcast_arrays(d1.cdf(x1),p)
    if np.any((p<0)|(p>1)):
        raise ValueError('Probability must lie in [0,1]')
    v=np.empty(u.shape)
    for i in np.ndindex(u.shape):
        v[i]=p[i] if psi==1 or p[i] in (0,1) else optimize.brentq(lambda z:_copula(u[i],z,psi,1)-p[i],0,1)
    return d2.ppf(v)


def mdist2drnd(phat, *, size=1, seed=None):
    d1,_,_=_plackett(phat)
    rng=np.random.default_rng(seed)
    x=d1.ppf(rng.random(size))
    return x,mdist2dcinv(x,rng.random(size),phat)


def mdist2dlike(phat,x1,x2):
    pdf=mdist2dpdf(x1,x2,phat)
    return -np.log(pdf).sum() if np.all(np.isfinite(pdf)) and np.all(pdf>0) else np.inf


def mdist2dpdf2(x1,x2,phat,condon=0):
    from ..spec import Spectrum
    X,Y=np.meshgrid(x1,x2)
    return Spectrum(f=mdist2dpdf(X,Y,phat,condon),x=[np.asarray(x1),np.asarray(x2)],
                    labx=['X1','X2'],note='Plackett bivariate distribution',phat=phat)
