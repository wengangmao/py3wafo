"""Analytic wave models from the supplied WAFO MATLAB source.

Ochi's PDF is repaired to differentiate the general CDF coherently: the source
PDF has a scaling error and its equal-scale Rayleigh shortcut is discontinuous.
The coherent law is the sum of independent Rayleigh(a/sqrt(2)),Rayleigh(b/sqrt(2)).
"""
import numpy as np
from scipy import integrate,optimize,stats
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr,i0e,jv,betaln
from ..spec import Spectrum


def _pdf(values,axes,labels,title):
    return Spectrum(f=np.asarray(values),x=[np.asarray(x) for x in axes],labx=labels,title=title)


def _positive_scales(x,a,b):
    x,a,b=np.broadcast_arrays(np.asarray(x,dtype=float),a,b)
    return x,a,b


def ochi98cdf(x,a,b):
    """Continuous repaired source CDF for a sum of two Rayleigh amplitudes.

    Allows either scale to vanish, yielding the corresponding Rayleigh limit.
    a==b uses the continuous general law, not the broken source shortcut.
    """
    x,a,b=_positive_scales(x,a,b)
    out=np.zeros(x.shape)
    valid=(a>=0)&(b>=0)&((a+b)>0)
    active=valid&(a>0)&(b>0)&(x>0)&np.isfinite(x)
    aa,bb,xx=a[active],b[active],x[active]
    r1,r2=aa*aa,bb*bb; R=r1+r2
    D=ndtr(np.sqrt(2*r2/(r1*R))*xx)-ndtr(-np.sqrt(2*r1/(r2*R))*xx)
    term=(r1*np.exp(-xx*xx/r1)+r2*np.exp(-xx*xx/r2)+2*xx*np.sqrt(np.pi*r1*r2/R)*np.exp(-xx*xx/R)*D)/R
    out[active]=np.clip(1-term,0,1)
    single=valid&((a==0)|(b==0))&(x>=0)
    out[single]=-np.expm1(-x[single]**2/np.maximum(a[single],b[single])**2)
    out[valid&np.isposinf(x)]=1
    return np.where(valid&~np.isnan(x),out,np.nan)


def ochi98pdf(x,a,b):
    """Derivative of repaired ochi98cdf; normalized and continuous at a==b."""
    x,a,b=_positive_scales(x,a,b)
    out=np.zeros(x.shape)
    valid=(a>=0)&(b>=0)&((a+b)>0)
    active=valid&(a>0)&(b>0)&(x>0)&np.isfinite(x)
    aa,bb,xx=a[active],b[active],x[active]
    r1,r2=aa*aa,bb*bb; R=r1+r2
    D=ndtr(np.sqrt(2*r2/(r1*R))*xx)-ndtr(-np.sqrt(2*r1/(r2*R))*xx)
    value=2*xx/R**2*(r1*np.exp(-xx*xx/r1)+r2*np.exp(-xx*xx/r2))
    value+=2*np.sqrt(np.pi*r1*r2/R)/R*np.exp(-xx*xx/R)*(2*xx*xx/R-1)*D
    out[active]=np.maximum(value,0)
    single=valid&((a==0)|(b==0))&(x>=0)&np.isfinite(x)
    scale=np.maximum(a[single],b[single])
    out[single]=2*x[single]/scale**2*np.exp(-x[single]**2/scale**2)
    return np.where(valid&~np.isnan(x),out,np.nan)


def ochi98fit(data):
    """Positive-scale maximum likelihood for the repaired model, parameters only."""
    data=np.asarray(data,dtype=float).ravel()
    if len(data)<3 or not np.all(np.isfinite(data)) or np.any(data<=0):
        raise ValueError('At least three positive finite wave heights are required')
    total=2*data.mean()/np.sqrt(np.pi)
    product=(total*total-np.mean(data*data))/(2-np.pi/2)
    difference=np.sqrt(max(0,total*total-4*product))
    initial=np.maximum([(total+difference)/2,(total-difference)/2],total*1e-3)
    def objective(logp):
        values=ochi98pdf(data,*np.exp(logp))
        return -np.log(values).sum() if np.all(values>0) else np.inf
    result=optimize.minimize(objective,np.log(initial),method='Nelder-Mead',options={'maxiter':2000,'xatol':1e-9,'fatol':1e-8})
    if not result.success:
        raise RuntimeError('Ochi parameter likelihood did not converge: '+result.message)
    return np.sort(np.exp(result.x))[::-1]


def braylfit(data,c=None):
    """Beta-Rayleigh MLE, parameter-only output.

    Supplying c fixes the upper endpoint and fits a,b. With c=None the free
    endpoint likelihood is constrained to b>1 to avoid an unbounded endpoint
    density; this constraint is explicit rather than a failed unconstrained MLE.
    """
    data=np.asarray(data,dtype=float).ravel()
    if len(data)<3 or np.any(data<=0) or not np.all(np.isfinite(data)):
        raise ValueError('At least three positive finite heights are required')
    if c is not None:
        if not np.isfinite(c) or c<=max(data):
            raise ValueError('Fixed endpoint c must exceed every observation')
        a,b,_,_=stats.beta.fit((data/c)**2,floc=0,fscale=1)
        return np.array([a,b,c])
    endpoint=max(data)
    initial=braylfit(data,np.sqrt(2)*endpoint)
    def decode(p):
        return np.exp(p[0]),1+np.exp(p[1]),endpoint+np.exp(p[2])
    def objective(p):
        a,b,c=decode(p); z=(data/c)**2
        logpdf=np.log(2/c)+(2*a-1)*np.log(data/c)+(b-1)*np.log1p(-z)-betaln(a,b)
        return -logpdf.sum() if np.all(np.isfinite(logpdf)) else np.inf
    start=np.log([initial[0],max(initial[1]-1,.1),initial[2]-endpoint])
    result=optimize.minimize(objective,start,method='Nelder-Mead',options={'maxiter':3000})
    if not result.success:
        raise RuntimeError('Beta-Rayleigh likelihood did not converge')
    return np.array(decode(result.x))


def _transform(g,variance=1):
    if g is None:
        return (lambda x:np.asarray(x)/np.sqrt(variance),
                lambda x:np.ones_like(np.asarray(x),dtype=float)/np.sqrt(variance),
                lambda z:np.asarray(z)*np.sqrt(variance),None)
    g=np.asarray(g,dtype=float)
    if g.ndim!=2 or g.shape[1]!=2 or len(g)<2 or not np.all(np.isfinite(g)) or np.any(np.diff(g,axis=0)<=0):
        raise ValueError('Transformation must be a strictly increasing finite two-column table')
    forward=PchipInterpolator(g[:,0],g[:,1],extrapolate=False)
    inverse=PchipInterpolator(g[:,1],g[:,0],extrapolate=False)
    return forward,forward.derivative(),inverse,g


def trraylpdf(x=None,definition='Ac',gn=None,utc=None):
    """Transformed Rayleigh amplitude/range approximation as a PDF mapping.

    A nonzero Gaussian reference level yields the source approximation's
    subdensity, not an automatically renormalized conditional law.
    """
    forward,derivative,inverse,table=_transform(gn)
    utc=float(inverse(0)) if utc is None else float(utc)
    definition=definition.lower()
    if definition not in ('ac','at','acat'):
        raise ValueError('definition must be Ac,At,or AcAt')
    if x is None:
        limit=5 if table is None else min(table[-1,1],-table[0,1])
        maxvalue=float(inverse(limit)-utc) if definition=='ac' else float(utc-inverse(-limit))
        if definition=='acat':maxvalue=float(inverse(limit)-inverse(-limit))
        x=np.linspace(0,maxvalue,101)
    x=np.asarray(x,dtype=float)
    if np.any(x<0) or not np.all(np.isfinite(x)):
        raise ValueError('Amplitude coordinates must be finite and nonnegative')
    if definition in ('ac','at'):
        values=utc+x if definition=='ac' else utc-x
        z=forward(values); jac=abs(derivative(values))
        z=z if definition=='ac' else -z
        if not np.all(np.isfinite(z)):
            raise ValueError('Amplitude outside the transformation table')
        density=jac*np.maximum(z,0)*np.exp(-z*z/2)
    elif table is None:
        density=x/4*np.exp(-x*x/8)
    else:
        limit=min(table[-1,1],-table[0,1])
        maxrange=float(inverse(limit)-inverse(-limit))
        if np.any(x>maxrange):raise ValueError('Range outside the transformation table')
        inverse_derivative=inverse.derivative()
        density=np.empty_like(x)
        for idx in np.ndindex(x.shape):
            r=optimize.brentq(lambda r:float(inverse(r)-inverse(-r)-x[idx]),0,limit)
            density[idx]=r*np.exp(-r*r/2)/(inverse_derivative(r)+inverse_derivative(-r))
    return _pdf(density,[x],[definition+' [m]'],'Transformed Rayleigh approximation')


def _joint_inputs(t,h,mom,g,moment_kind):
    mom=np.asarray(mom,dtype=float)
    if mom.shape!=(3,) or np.any(mom<=0) or not np.all(np.isfinite(mom)):
        raise ValueError('Three positive finite spectral moments are required')
    m0,middle,last=mom
    if middle*middle>=m0*last:
        raise ValueError('Moments must describe a nondegenerate spectral bandwidth')
    period=2*np.pi*np.sqrt(m0/(middle if moment_kind=='024' else last))
    t=np.linspace(0,1.7*period,51) if t is None else np.asarray(t,dtype=float)
    forward,derivative,inverse,table=_transform(g,m0)
    h=np.linspace(0,1.3*float(inverse(4)-inverse(0)),41) if h is None else np.asarray(h,dtype=float)
    if t.ndim!=1 or h.ndim!=1 or np.any(t<0) or np.any(h<0) or not np.all(np.isfinite(t)) or not np.all(np.isfinite(h)):
        raise ValueError('Time and amplitude grids must be finite nonnegative vectors')
    z,jac=forward(h),abs(derivative(h))
    if not np.all(np.isfinite(z)):
        raise ValueError('Amplitude grid outside transformation table')
    return t,h,z,jac,mom


def cav76pdf(t=None,h=None,mom=None,g=None):
    """Cavanie 1976 approximation; axes [period,amplitude], f shape (H,T)."""
    t,h,z,jac,mom=_joint_inputs(t,h,mom,g,'024')
    m0,m2,m4=mom; alpha=m2/np.sqrt(m0*m4)
    constant=2*np.pi**4*2/(1+alpha)/(np.sqrt(2*np.pi)*m4*np.sqrt(1-alpha*alpha))
    result=np.zeros((len(h),len(t)))
    positive=t>0; T=t[positive][None,:]; H=(np.sqrt(m0)*z)[:,None]
    exponent=-.5*(H/T**2)**2*((T*T-np.pi**2*m2/m4)**2/(m0*(1-alpha*alpha))+np.pi**4/m4)
    result[:,positive]=constant*jac[:,None]*H**2/T**5*np.exp(exponent)
    return _pdf(result,[t,h],['Tc','Ac'],'Cavanie et al. (1976) joint approximation')


def lh83pdf(t=None,h=None,mom=None,g=None):
    """Longuet-Higgins 1983 period/amplitude approximation."""
    t,h,z,jac,mom=_joint_inputs(t,h,mom,g,'012')
    m0,m1,m2=mom; mean_period=2*np.pi*m0/m1
    eps=np.sqrt(m0*m2/m1**2-1)
    constant=4/(np.sqrt(np.pi)*eps*(1+1/np.sqrt(1+eps*eps)))
    result=np.zeros((len(h),len(t))); positive=t>0
    T=(2*t[positive]/mean_period)[None,:]; H=(z/np.sqrt(2))[:,None]
    result[:,positive]=constant*jac[:,None]*(H/T)**2*np.exp(-H*H*(1+((1-1/T)/eps)**2))/(mean_period*np.sqrt(2)/2)
    return _pdf(result,[t,h],['Tc','Ac'],'Longuet-Higgins (1983) joint approximation')


def tay90fun(u,x,b,pdfstr='pdf'):
    """Source smooth integrand (the original quadrature supplied a weight).

    The unweighted PDF integral is tay90fun(u,x,b)*sqrt(u/(1-u)).
    """
    if pdfstr!='pdf':raise ValueError('Only the documented PDF integrand is supported')
    u,x,b=np.broadcast_arrays(u,x,b)
    if np.any((u<0)|(u>1)) or np.any(abs(b)>=1):raise ValueError('Require 0<=u<=1 and abs(b)<1')
    A=.25/(1-b*b); arg=u*x*x*b*A
    return np.sqrt(u)/2*x**3*A*np.exp(-A*x*x*(2-u-abs(u*b)))*i0e(arg)


def tay90pdf(x,a,b,tol=1e-6,*,full_output=False):
    """Tayfun 1990 density; stable endpoint substitution removes quadrature weight."""
    x,a,b=_positive_scales(x,a,b)
    if tol<=0:raise ValueError('tol must be positive')
    out,error=np.zeros(x.shape),np.zeros(x.shape)
    valid=(a>0)&(abs(b)<=1)
    for idx in np.ndindex(x.shape):
        if not valid[idx] or np.isnan(x[idx]):out[idx]=np.nan;continue
        if x[idx]<=0 or np.isposinf(x[idx]):continue
        if abs(b[idx])==1:
            out[idx]=stats.rayleigh.pdf(x[idx],scale=2*a[idx]);continue
        height=x[idx]/a[idx];rho=b[idx];A=.25/(1-rho*rho)
        def integrand(s):
            u=1-s*s;arg=A*height*height*rho*u
            return height**3*A*u*np.exp(-A*height*height*(2-u-abs(rho*u)))*i0e(arg)
        out[idx],error[idx]=integrate.quad(integrand,0,1,epsabs=tol*a[idx],epsrel=tol,limit=200)
        out[idx]/=a[idx];error[idx]/=a[idx]
    return (out,error) if full_output else out


def tay90cdf(x,a,b,tol=1e-6,*,full_output=False):
    """Tayfun 1990 CDF; numerical integral of the supported density."""
    x,a,b=_positive_scales(x,a,b)
    out,error=np.zeros(x.shape),np.zeros(x.shape)
    for idx in np.ndindex(x.shape):
        if a[idx]<=0 or abs(b[idx])>1 or np.isnan(x[idx]):out[idx]=np.nan;continue
        if x[idx]<=0:continue
        if np.isposinf(x[idx]):out[idx]=1;continue
        if abs(b[idx])==1:
            out[idx]=stats.rayleigh.cdf(x[idx],scale=2*a[idx]);continue
        out[idx],error[idx]=integrate.quad(lambda h:float(tay90pdf(h,a[idx],b[idx],tol/10)),0,x[idx],epsabs=tol,epsrel=tol,limit=200)
    return (out,error) if full_output else out


def tay81fun(u,x,b,pdfstr='pdf'):
    """Source Bessel integrand, with b equal to the squared limiting parameter.

    Finite-limit PDF/CDF integration is not silently approximated here: the
    source's oscillatory cutoff scheme requires separate convergence analysis.
    """
    u,x,b=np.broadcast_arrays(u,x,b)
    if np.any(b<=0) or not np.all(np.isfinite(b)):raise ValueError('b must be positive and finite')
    power=np.real(np.asarray(jv(0,u/np.sqrt(b)),dtype=complex)**b)
    if pdfstr=='pdf':return jv(0,u*x)*u*power
    if pdfstr=='cdf':return jv(1,u*x)*u*x*power
    raise ValueError('pdfstr must be pdf or cdf')
