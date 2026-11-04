"""Conditional-parameter bivariate distributions from the DIST2D sources.

Models factor as f(x1|x2)*f(x2). ``phat['dist']`` is [conditional,marginal]
and ``phat['x']`` is [conditional_table,marginal_parameters]. Each table row
contains [conditioning_coordinate,parameters...]. This implementation removes
the MATLAB globals, integrates actual supports rather than truncating to
mean+10*std, and completes the previously unimplemented marginal moments.
"""
from copy import deepcopy
from datetime import datetime,timezone
import numpy as np
from scipy import integrate,stats


def _family(name):
    name=str(name).lower().replace('_','')
    groups={'rayl':('ray','ra'),'weib':('we','wwe'),'gumb':('gu','wgu'),
        'tgumb':('tg',),'logn':('lo','wlo'),'ggam':('gg','wgg'),
        'gam':('ga','wga'),'trayl':('tr',),'tweib':('tw',),'norm':('no','wno')}
    for family,prefixes in groups.items():
        if any(name.startswith(p) for p in prefixes):return family
    raise ValueError(f'Unknown DIST2D family {name!r}')


_NPAR={'rayl':1,'weib':2,'gumb':2,'tgumb':2,'logn':2,'ggam':3,
       'gam':2,'trayl':2,'tweib':3,'norm':2}
_POSITIVE={'rayl':(0,),'weib':(0,1),'gumb':(0,),'tgumb':(0,),
           'logn':(1,),'ggam':(0,1,2),'gam':(0,1),'trayl':(0,),
           'tweib':(0,1),'norm':(1,)}


def _model(phat):
    if not isinstance(phat,dict) or len(phat.get('x',[]))!=2 or len(phat.get('dist',[]))!=2:
        raise ValueError("phat must contain x=[table,parameters], dist=[conditional,marginal]")
    conditional,marginal=map(_family,phat['dist'])
    table=np.asarray(phat['x'][0],dtype=float)
    if table.ndim!=2 or table.shape[1]<1+_NPAR[conditional]:
        raise ValueError('Conditional table has too few parameter columns')
    table=table[np.all(np.isfinite(table),axis=1)]
    if not len(table):raise ValueError('Conditional table contains no finite rows')
    table=table[np.argsort(table[:,0],kind='stable')]
    if np.any(np.diff(table[:,0])<=0):raise ValueError('Conditioning coordinates must be distinct')
    parameters=np.asarray(phat['x'][1],dtype=float).ravel()
    if parameters.size!=_NPAR[marginal] or not np.all(np.isfinite(parameters)):
        raise ValueError('Marginal parameter count does not match the selected distribution')
    if any(parameters[index]<=0 for index in _POSITIVE[marginal]):
        raise ValueError('Marginal scales, shapes and variances must be positive')
    if marginal=='trayl' and parameters[1]>0:
        raise ValueError('Truncated Rayleigh location must be nonpositive')
    return conditional,marginal,table,parameters


def _choices(value,default):
    value=np.asarray(default if value is None or np.size(value)==0 else value,dtype=float).ravel()
    if not len(value) or len(value)>3:raise ValueError('Smoothing/extrapolation arrays have length 1..3')
    out=np.ones(3);out[:len(value)]=value
    return out


def _smoothed(phat,points=None,csm=None,lin=None):
    family,_,table,_=_model(phat)
    points=table[:,0] if points is None else np.asarray(points,dtype=float)
    if not np.all(np.isfinite(points)):raise ValueError('Conditioning values must be finite')
    smoothing=_choices(csm,1);linear=_choices(lin,1)
    if np.any((smoothing<0)|(smoothing>1)) or np.any(~np.isin(linear,[0,1])):
        raise ValueError('Require 0<=csm<=1 and lin selectors equal to 0 or 1')
    count=min(table.shape[1]-1,3)
    values=[]
    from ..misc import smooth
    for column in range(count):
        y=table[:,column+1]
        if column in _POSITIVE[family] and np.any(y<=0):
            raise ValueError('Scale, shape, and variance parameters must be positive')
        if len(table)==1:
            result=np.full(np.shape(points),y[0])
        else:
            result=smooth(table[:,0],y,smoothing[column],points,int(linear[column]))
            if column in _POSITIVE[family] and np.any(result<=0):
                result=np.exp(smooth(table[:,0],np.log(y),smoothing[column],points,int(linear[column])))
        values.append(result)
    values += [np.zeros(np.shape(points))]*(3-len(values))
    return family,values


def dist2dsmfun(phat,h2=None,CSMA=None,lin_extrap=None):
    """Smooth/interpolate conditional parameters, returning (A,B,C).

    Uses the source roughness objective through ``misc.smooth`` with uniform
    observation weights. Legacy CI/statistics-specific reweighting is not
    reproduced; positive parameters use log interpolation if extrapolation
    would otherwise make them nonpositive. For a Rayleigh law B is zero and C
    is its optional location column (the old file indexed that column wrongly).
    """
    family,values=_smoothed(phat,h2,CSMA,lin_extrap)
    if family=='rayl':
        return values[0],np.zeros_like(values[0]),values[1]
    return tuple(values)


def dist2dsmfun2(phat,h2=None,csm=None,lin=None,visual=0,f0=None):
    """Return a new model with evaluated/smoothed conditional parameter rows.

    Visual point picking and gamma-moment/CI weighting are not implemented;
    use explicit data/weights to create a parameter table instead.
    """
    if visual or f0 is not None:raise NotImplementedError('Interactive visual fitting is not supported')
    family,_,table,_=_model(phat)
    h2=table[:,0] if h2 is None else np.asarray(h2,dtype=float).ravel()
    _,values=_smoothed(phat,h2,csm,lin)
    result=deepcopy(phat)
    result['x']=list(result['x'])
    count=min(table.shape[1]-1,3)
    result['x'][0]=np.column_stack((h2,*values[:count]))
    result.pop('CI',None)
    result.update(csm=csm,lin=lin,date=datetime.now(timezone.utc).isoformat())
    return result


def _law(family,parameters,operation,x=None):
    """Unified scalar/array family operations; location C is external except GG/TW/TR."""
    from . import __dict__ as functions
    p=list(parameters)
    n=_NPAR[family]
    shift=0 if family in ('ggam','tweib','trayl') or len(p)<=n else p[n]
    p=p[:n]
    if family=='tgumb':
        p.append(1);prefix='gumb'
    else:prefix=family
    if family in ('tweib','trayl') and operation in ('inv','stat'):
        a,b,c=(p[0],p[1],abs(np.asarray(p[2]))) if family=='tweib' else (np.asarray(p[0])*np.sqrt(2),2,-np.asarray(p[1]))
        if np.any(np.asarray(a)<=0) or np.any(np.asarray(b)<=0) or np.any(c<0):raise ValueError('Invalid truncated distribution parameters')
        if operation=='inv':
            probability=np.asarray(x)
            with np.errstate(divide='ignore',invalid='ignore'):
                result=a*((c/a)**b-np.log1p(-probability))**(1/b)-c
            return np.where((probability>=0)&(probability<=1),result,np.nan)
        # E[W^r|W>c], with incomplete gamma and exponential survival factor.
        from scipy.special import gammaincc,gammaln
        z=(c/a)**b
        def raw(r):return a**r*np.exp(gammaln(1+r/b)+z)*gammaincc(1+r/b,z)
        m=raw(1)-c
        return m,raw(2)-2*c*raw(1)+c*c-m*m
    function=functions['w'+prefix+operation]
    if operation=='stat':
        mean,var=function(*p)
        return mean+shift,var
    argument=np.asarray(x)-shift if operation in ('pdf','cdf') else x
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        result=function(argument,*p)
    return result+shift if operation=='inv' else result


def _conditional(phat,h,operation,x=None,csm=None,lin=None):
    family,values=_smoothed(phat,h,csm,lin)
    if family=='rayl':parameters=[values[0],values[1]]
    else:parameters=values
    return _law(family,parameters,operation,x)


def _marginal(phat,operation,x=None):
    _,family,_,parameters=_model(phat)
    return _law(family,parameters,operation,x)


def _support(phat):
    _,family,_,_=_model(phat)
    return (-np.inf,np.inf) if family in ('gumb','norm') else (0.,np.inf)


def dist2dfun(H,V,phat,condon=0):
    """Explicit-model version of the MATLAB global integrand.

    condon=0: f(H)*F(V|H); 1: f(H)*f(V|H); 2: F(V|H); 3:f(V|H).
    """
    if condon not in (0,1,2,3):raise ValueError('condon must be 0,1,2,3')
    V,H=np.broadcast_arrays(np.asarray(V,dtype=float),np.asarray(H,dtype=float))
    if condon in (2,3) and not np.all(np.isfinite(H)):
        raise ValueError('Conditional evaluation requires finite conditioning values')
    result=np.zeros(V.shape)
    finite=np.isfinite(H)
    if np.any(finite):
        values=_conditional(phat,H[finite],'cdf' if condon in (0,2) else 'pdf',V[finite])
        result[finite]=values*_marginal(phat,'pdf',H[finite]) if condon in (0,1) else values
    return np.where(np.isnan(V)|np.isnan(H),np.nan,result)


def dist2dpdf(v1,h1,phat):
    """Joint density f(X1=v1|X2=h1)*f(X2=h1), with broadcasting."""
    return dist2dfun(h1,v1,phat,1)


def _quad(function,a,b,tol):
    if a>=b:return 0.,0.
    return integrate.quad(function,a,b,epsabs=tol,epsrel=tol,limit=200)


def dist2dcdf(V,H,phat,condon=0,*,tol=1e-7,full_output=False):
    """Joint or conditional CDF. full_output returns (value,error_estimate).

    condon=0 joint; 1 X2|X1; 2 X1|X2. Reverse conditioning integrates the
    actual infinite support instead of a finite heuristic cutoff.
    """
    if condon not in (0,1,2) or tol<=0:raise ValueError('Invalid conditioning selector or tolerance')
    V,H=np.broadcast_arrays(np.asarray(V,dtype=float),np.asarray(H,dtype=float))
    out,error=np.zeros(V.shape),np.zeros(V.shape)
    low,high=_support(phat)
    for index in np.ndindex(V.shape):
        v,h=float(V[index]),float(H[index])
        if np.isnan(v) or np.isnan(h):out[index]=np.nan;continue
        if condon==2:
            out[index]=_conditional(phat,h,'cdf',v)
            continue
        if h<=low:continue
        stop=min(h,high)
        mode=0 if condon==0 else 1
        integrand=lambda level:float(dist2dfun(level,v,phat,mode))
        value,err=_quad(integrand,low,stop,tol)
        if condon==1:
            denominator,denerr=_quad(integrand,low,high,tol)
            if denominator<=0:out[index]=np.nan;error[index]=np.inf;continue
            error[index]=err/denominator+abs(value)*denerr/denominator**2
            out[index]=value/denominator
        else:out[index],error[index]=value,err
    return (out,error) if full_output else out


def dist2dprb(phat,x1lo,x1up,x2lo,x2up,*,tol=1e-7,full_output=False):
    """Rectangle probability via conditional-CDF differences and 1D quadrature."""
    arrays=np.broadcast_arrays(x1lo,x1up,x2lo,x2up)
    out,error=np.empty(arrays[0].shape),np.empty(arrays[0].shape)
    lo,hi=_support(phat)
    for index in np.ndindex(out.shape):
        a,b,c,d=(float(v[index]) for v in arrays)
        if a>b or c>d:raise ValueError('Rectangle lower bounds must not exceed upper bounds')
        def integrand(h):return float((_conditional(phat,h,'cdf',b)-_conditional(phat,h,'cdf',a))*_marginal(phat,'pdf',h))
        out[index],error[index]=_quad(integrand,max(c,lo),min(d,hi),tol)
    return (out,error) if full_output else out


def dist2drnd(N,phat,csm=None,lin=None,*,seed=None):
    """Return N conditional/marginal random pairs (X1,X2), seeded by NumPy."""
    if not isinstance(N,(int,np.integer)) or N<0:raise ValueError('N must be a nonnegative integer')
    rng=np.random.default_rng(seed)
    h=_marginal(phat,'inv',rng.random(N))
    v=_conditional(phat,h,'inv',rng.random(N),csm,lin)
    return v,h


def dist2dstat(phat,condon=0,cvar=None,csm=None,lin=None,*,tol=1e-6):
    """Mean and covariance, or conditional mean and variance.

    condon=0 completes the old unfinished marginal calculation using total
    expectation/variance. Returns (mean[2],covariance[2,2]). condon=1/2 returns
    mean/variance arrays for X2|X1 and X1|X2 respectively.
    """
    if condon not in (0,1,2):raise ValueError('condon must be 0,1,2')
    low,high=_support(phat)
    if condon==2:
        if cvar is None:
            _,_,table,_=_model(phat);cvar=table[:,0]
        return _conditional(phat,cvar,'stat',None,csm,lin)
    if condon==1:
        if cvar is None:raise ValueError('Provide X1 values for reverse conditional moments')
        cvar=np.asarray(cvar,dtype=float);means=np.empty(cvar.shape);variances=np.empty(cvar.shape)
        for index in np.ndindex(cvar.shape):
            weight=lambda h:float(dist2dpdf(cvar[index],h,phat))
            denominator=_quad(weight,low,high,tol)[0]
            if denominator<=0:means[index]=variances[index]=np.nan;continue
            mean=_quad(lambda h:h*weight(h),low,high,tol)[0]/denominator
            means[index]=mean
            variances[index]=_quad(lambda h:(h-mean)**2*weight(h),low,high,tol)[0]/denominator
        return means,variances
    m2,v2=_marginal(phat,'stat')
    def expectation(power,cross=False):
        def fun(h):
            mean,var=_conditional(phat,h,'stat',None,csm,lin)
            value=mean if power==1 else var+mean*mean
            return float(value*(h if cross else 1)*_marginal(phat,'pdf',h))
        return _quad(fun,low,high,tol)[0]
    m1=expectation(1);v1=expectation(2)-m1*m1
    covariance=expectation(1,True)-m1*m2
    return np.array([m1,m2]),np.array([[v1,covariance],[covariance,v2]])


def dist2dpdf2(v1,h1,phat):
    """Joint density on a meshgrid, returning an attribute-access PDF mapping."""
    from ..spec import Spectrum
    v,h=np.asarray(v1),np.asarray(h1)
    if v.ndim!=1 or h.ndim!=1:raise ValueError('Grid coordinates must be vectors')
    vv,hh=np.meshgrid(v,h,indexing='xy')
    return Spectrum(f=dist2dpdf(vv,hh,phat),x=[v,h],phat=deepcopy(phat),title='Conditional-parameter joint density')


def _fit_sample(sample,family,method):
    from . import __dict__ as functions
    sample=np.asarray(sample,dtype=float)
    if method!='MLE':raise NotImplementedError('Only MLE fitting is implemented; MOM is not silently substituted')
    if family=='tgumb':
        from scipy.optimize import minimize
        if np.any(sample<0):raise ValueError('Truncated Gumbel observations must be nonnegative')
        scale=max(np.std(sample)*np.sqrt(6)/np.pi,np.finfo(float).eps)
        def objective(z):
            values=functions['wgumbpdf'](sample,np.exp(z[0]),z[1],1)
            return -np.log(values).sum() if np.all(values>0) else np.inf
        result=minimize(objective,[np.log(scale),np.mean(sample)-np.euler_gamma*scale],method='Nelder-Mead')
        if not result.success:raise RuntimeError('Truncated Gumbel MLE did not converge')
        return np.array([np.exp(result.x[0]),result.x[1]])
    return np.atleast_1d(functions['w'+family+'fit'](sample))


def dist2dfit(V,H,dist=('tgumbel','rayleigh'),res=None,method=('MLE','MLE'),monitor=0,chat0=None):
    """Fit a marginal law and binned conditional parameter table.

    res is width or [width,overlap,location_offset]. Bins with fewer than six
    observations are skipped. Returns point estimates and bin statistics;
    confidence intervals are not fabricated. Native interactive monitoring and
    generalized-gamma fixed-parameter chat0 selectors remain unsupported.
    """
    if monitor or chat0 is not None:raise NotImplementedError('Monitoring and fixed-parameter chat0 fitting are not implemented')
    v,h=np.asarray(V,dtype=float).ravel(),np.asarray(H,dtype=float).ravel()
    if v.shape!=h.shape or len(v)<12 or not np.all(np.isfinite([v,h])):raise ValueError('Need at least twelve finite paired observations')
    dist=list(dist)
    if len(dist)==1:dist.append('rayleigh')
    if len(dist)!=2:raise ValueError('dist must contain one or two distribution names')
    conditional,marginal=map(_family,dist)
    if isinstance(method,str):method=[method,'MLE']
    methods=[str(x).upper() for x in method]
    if len(methods)==1:methods.append('MLE')
    if len(methods)!=2:raise ValueError('method must have one or two entries')
    settings=np.atleast_1d(np.ptp(h)/12 if res is None else res).astype(float)
    width=settings[0];overlap=0 if len(settings)<2 else settings[1]
    if width<=0 or not np.isfinite(width) or overlap<0 or int(overlap)!=overlap:raise ValueError('Require positive width and nonnegative integer overlap')
    if len(settings)>3:raise ValueError('res supports width, overlap and location offset')
    offset=None if len(settings)<3 else settings[2]
    if offset is not None and (not np.isfinite(offset) or offset<0):raise ValueError('Location offset must be nonnegative')
    if offset is not None and conditional in ('ggam','tweib','trayl'):raise NotImplementedError('External location offsets are not defined for this family')
    origin=np.floor(h.min()/width)*width
    group=np.floor((h-origin)/width).astype(int)
    rows,statsrows=[],[]
    for index in np.unique(group):
        selected=abs(group-index)<=overlap
        if selected.sum()<6:continue
        sample=v[selected]
        shift=0 if offset is None else max(float(sample.min()-offset),0)
        parameters=_fit_sample(sample-shift,conditional,methods[0])
        row=[origin+(index+.5)*width,*parameters]
        if offset is not None:row.append(shift)
        rows.append(row)
        statsrows.append([row[0],float(sample.mean()),float(sample.var(ddof=1)),int(len(sample))])
    if not rows:raise ValueError('No conditional bin contains enough observations; increase width/overlap')
    return dict(x=[np.asarray(rows),_fit_sample(h,marginal,methods[1])],dist=dist,method=methods,
        stats1=np.asarray(statsrows),res=width,noverlap=int(overlap),C=offset,
        note='Binned conditional MLE; no confidence intervals; Python support-aware integration',date=datetime.now(timezone.utc).isoformat())


def mdist2dfit(V,H,dist,alpha=None,method='SML',*,interaction_log_bound=14):
    """Plackett marginal or simultaneous maximum-likelihood estimation.

    MML fits each marginal then its copula interaction. SML jointly refines
    every parameter with positivity-preserving coordinates. Confidence-interval
    alpha requests are rejected; no unvalidated bounds are returned. The MML
    interaction search is bounded by +/-interaction_log_bound in log(psi);
    hitting that numerical boundary raises rather than declaring convergence.
    """
    from scipy.optimize import minimize,minimize_scalar
    from ._bivariate import _marginal as frozen,_copula_density,mdist2dlike
    if alpha is not None:raise NotImplementedError('Confidence intervals for joint Plackett fits are not implemented')
    method=str(method).upper()
    if method not in ('MML','SML'):raise ValueError('method must be MML or SML')
    if not np.isfinite(interaction_log_bound) or not 0<interaction_log_bound<=30:
        raise ValueError('interaction_log_bound must lie in (0,30]')
    x,y=np.asarray(V,dtype=float).ravel(),np.asarray(H,dtype=float).ravel()
    if x.shape!=y.shape or len(x)<8 or not np.all(np.isfinite([x,y])):raise ValueError('Need at least eight finite paired observations')
    if len(dist)!=2:raise ValueError('Provide two marginal distribution names')
    families=list(map(_family,dist))
    if any(f in ('tweib','trayl') for f in families):raise NotImplementedError('This Plackett marginal family is not implemented')
    parameters=[_fit_sample(data,family,'MLE') for data,family in zip((x,y),families)]
    first,second=(frozen(name,p) for name,p in zip(dist,parameters))
    u,v=first.cdf(x),second.cdf(y)
    def objective_logpsi(logpsi):
        density=_copula_density(u,v,np.exp(logpsi))
        return -np.log(density).sum() if np.all(density>0) and np.all(np.isfinite(density)) else np.inf
    copula=minimize_scalar(objective_logpsi,bounds=(-interaction_log_bound,interaction_log_bound),method='bounded',options={'xatol':1e-8})
    if not copula.success:raise RuntimeError('Plackett interaction fit failed')
    if abs(copula.x)>interaction_log_bound-.001:
        raise RuntimeError('Plackett interaction reached the numerical search boundary; increase interaction_log_bound or investigate near-deterministic dependence')
    result=dict(x=[*parameters,float(np.exp(copula.x))],dist=list(dist),method=method)
    if method=='SML':
        sizes=list(map(len,parameters))
        encode=[]
        for p,family in zip(parameters,families):
            p=p.copy();positive=list(_POSITIVE[family]);p[positive]=np.log(p[positive]);encode.extend(p)
        encode.append(copula.x)
        def decode(z):
            decoded=[];offset=0
            for n,family in zip(sizes,families):
                p=z[offset:offset+n].copy();positive=list(_POSITIVE[family]);p[positive]=np.exp(p[positive]);decoded.append(p);offset+=n
            return dict(x=[*decoded,float(np.exp(z[-1]))],dist=list(dist))
        def objective(z):
            with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                try:value=mdist2dlike(decode(z),x,y)
                except (ValueError,FloatingPointError):return 1e100
            return float(value) if np.isfinite(value) else 1e100
        optimized=minimize(objective,np.asarray(encode),method='Nelder-Mead',options={'maxiter':6000,'xatol':1e-7,'fatol':1e-7})
        if not optimized.success or optimized.fun>=1e100:raise RuntimeError('Simultaneous Plackett MLE did not converge')
        result=decode(optimized.x);result['method']=method
    result.update(date=datetime.now(timezone.utc).isoformat(),note='Point estimates only; no confidence intervals')
    return result


def mdist2dstat(phat,condon=0,cvar=None,tol=1e-6,*,quadrature=128):
    """Plackett marginal/conditional moments using actual marginal quantiles.

    Returns (mean[2],cov[2,2]) for condon=0; conditional mean/variance arrays
    for condon=1 (X2|X1) or 2 (X1|X2). Covariance uses tensor Gauss-Legendre
    quadrature on copula coordinates, replacing the source's incorrect use of
    uniform-marginal Spearman correlation as general Pearson correlation.
    ``quadrature`` controls that deterministic approximation; refine it for
    tail-heavy marginals. Conditional moments use adaptive 1D quadrature.
    Modal values and legacy extra output arrays are not fabricated.
    """
    from ._bivariate import _plackett,_copula_density
    first,second,psi=_plackett(phat)
    if condon not in (0,1,2) or tol<=0 or not np.isfinite(tol):raise ValueError('Invalid conditional selector/tolerance')
    if not isinstance(quadrature,(int,np.integer)) or quadrature<8:raise ValueError('quadrature must be an integer >=8')
    m1,v1=first.stats(moments='mv');m2,v2=second.stats(moments='mv')
    if not np.all(np.isfinite([m1,v1,m2,v2])):raise ValueError('Finite first/second marginal moments are required')
    if condon==0:
        if psi==1:covariance=0.
        else:
            z,weights=np.polynomial.legendre.leggauss(quadrature)
            p,weights=(z+1)/2,weights/2
            x,y=first.ppf(p)-m1,second.ppf(p)-m2
            density=_copula_density(p[:,None],p[None,:],psi)
            covariance=float(np.sum((weights*x)[:,None]*(weights*y)[None,:]*density))
        return np.array([m1,m2]),np.array([[v1,covariance],[covariance,v2]])
    if cvar is None:raise ValueError('Provide conditioning values')
    cvar=np.asarray(cvar,dtype=float)
    if not np.all(np.isfinite(cvar)):raise ValueError('Conditioning values must be finite')
    known,unknown=(first,second) if condon==1 else (second,first)
    if psi==1:
        mean,variance=unknown.stats(moments='mv')
        return np.full(cvar.shape,mean),np.full(cvar.shape,variance)
    means,variances=np.empty(cvar.shape),np.empty(cvar.shape)
    for index in np.ndindex(cvar.shape):
        probability=float(known.cdf(cvar[index]))
        if probability<=0 or probability>=1:raise ValueError('Conditioning value is outside the numerically resolvable marginal support')
        def moment(power):
            def integrand(p):
                q=float(unknown.ppf(p));return q**power*float(_copula_density(probability,p,psi))
            return integrate.quad(integrand,0,1,epsabs=tol,epsrel=tol,limit=200)[0]
        mean=moment(1);means[index]=mean;variances[index]=moment(2)-mean*mean
    return means,variances
