"""Source-based rainflow inversions, density changes and Gaussian diagnostics.

The legacy Rychlik kernels use an anti-diagonal matrix convention. Those
functions retain that convention; public cycle-matrix wrappers document their
conversion to ordinary min-row/max-column matrices.
"""
from copy import deepcopy
import numbers
import time
import numpy as np
from scipy.integrate import trapezoid,cumulative_trapezoid
from scipy.stats import norm
from ..spec import Spectrum


def _matrix(F):
    F=np.asarray(F,dtype=float)
    if F.ndim!=2 or F.shape[0]!=F.shape[1] or len(F)<2 or not np.all(np.isfinite(F)) or np.any(F<0):
        raise ValueError('Expected a finite nonnegative square frequency matrix of size >=2')
    return F


def _solve(A,b):
    try:
        return np.linalg.solve(A,b)
    except np.linalg.LinAlgError as exc:
        # Only consistent singular hitting-probability systems have a useful
        # minimum-norm solution. Do not turn an incompatible kernel into counts.
        x=np.linalg.lstsq(A,b,rcond=None)[0]
        if not np.allclose(A@x,b,rtol=1e-8,atol=1e-12):
            raise ValueError('Singular, inconsistent Markov hitting-probability system') from exc
        return x


def _nonnegative_result(result,scale):
    if np.min(result,initial=0)<-1e-9*max(1,float(scale)):
        raise ValueError('Kernel is inconsistent with a nonnegative rainflow frequency matrix')
    return np.maximum(result,0)


def mctp2rfc(f_mM,f_Mm=None,paramm=None,paramM=None):
    """Rychlik source min/max Markov-to-rainflow recursion.

    Retains the original anti-diagonal representation. Grid parameters affected
    only the removed MATLAB plots, not the numerical recursion. For the modern
    upper-triangular convention use wafo.markov.mctp2rfm instead.
    """
    F=_matrix(f_mM); G=F if f_Mm is None else _matrix(f_Mm)
    if G.shape!=F.shape:raise ValueError('Both Markov matrices must have the same shape')
    n=len(F); maximum=F.sum(axis=1); minimum=F.sum(axis=0)
    out=np.zeros_like(F)
    out[n-2,0]=maximum[n-2];out[0,n-2]=minimum[n-2]
    for k in range(3,n):
        for i in range(2,k):
            rows=slice(n-k,n-k+i-1);cols=slice(k-i,k-1)
            A=F[rows,cols];A1=G[rows,cols];R=out[rows,cols]
            MA=maximum[rows];mA=minimum[cols]
            adjustment=A.sum()-R.sum()
            remaining=max(0,min(mA[0]-R[:,0].sum(),MA[0]-R[0].sum()))
            if remaining>1e-6*max(MA[0],mA[0]):
                NN=MA-A.sum(axis=1)
                e=(mA-A.sum(axis=0))[::-1].copy()
                B=np.rot90(A).copy()
                denom=mA[::-1]
                B=np.divide(B,denom[:,None],out=np.zeros_like(B),where=denom[:,None]!=0)
                e=np.divide(e,denom,out=np.zeros_like(e),where=denom!=0)
                correction=0.
                if max(abs(e),default=0)>1e-6 and max(abs(NN),default=0)>1e-6*max(MA[0],mA[0]):
                    C=np.divide(A1,MA[:,None],out=np.zeros_like(A1),where=MA[:,None]!=0)[:,::-1]
                    correction=float(NN@(C@_solve(np.eye(len(A))-B@C,e)))
                out[n-k,k-i]=adjustment+correction
        low=max(0,minimum[0]-out[n-k+1:,0].sum())
        high=max(0,maximum[n-k]-out[n-k,1:k].sum())
        out[n-k,0]=min(low,high)
    for k in range(2,n+1):
        high=max(0,maximum[0]-out[0,n-k+1:].sum())
        low=max(0,minimum[n-k]-out[1:k,n-k].sum())
        out[0,n-k]=min(low,high)
    return _nonnegative_result(out,F.sum())


def mc2rfc(f_xy,paramv=None,paramu=None):
    """Rychlik source general Markov-to-rainflow recursion, anti-diagonal form.

    Plotting side effects are removed. Pure diagonal mass, which made the old
    0/0 expressions undefined, contributes directly without an extra loop.
    """
    F=_matrix(f_xy);n=len(F)
    dd=np.diag(np.rot90(F));plus=F.sum(axis=1);minus=F.sum(axis=0)[::-1]
    normal=minus+plus-dd
    divisor=normal-dd
    maximum=np.divide((plus-dd)**2,divisor,out=np.zeros(n),where=divisor>0)
    minimum=np.divide((minus-dd)**2,divisor,out=np.zeros(n),where=divisor>0)
    out=np.zeros_like(F)
    out[n-2,0]=maximum[n-2];out[0,n-2]=minimum[1]
    for k in range(3,n):
        for i in range(2,k):
            rows=slice(n-k,n-k+i);cols=slice(k-i,k)
            A=F[rows,cols];R=out[rows,cols]
            MA=plus[rows];mA=minus[rows];normA=normal[rows];Mrfc=maximum[rows]
            adjustment=A.sum()-MA.sum()-R.sum()+Mrfc.sum()
            remaining=max(0,Mrfc[0]-R[0].sum())
            if remaining>1e-6*Mrfc[0]:
                NN=MA-A.sum(axis=1)
                e=(mA[::-1]-A.sum(axis=0))[::-1]
                B=np.rot90(A+np.flipud(np.rot90(A,-1)))
                B=B-.5*np.diag(np.diag(B))
                B=np.divide(B,normA[:,None],out=np.zeros_like(B),where=normA[:,None]!=0)
                e=np.divide(e,normA,out=np.zeros_like(e),where=normA!=0)
                correction=0.
                if max(abs(e),default=0)>1e-7 and max(abs(NN),default=0)>1e-7*Mrfc[0]:
                    correction=float(NN@_solve(np.eye(len(A))-B,e))
                out[n-k,k-i]=adjustment+correction
        low=max(0,minimum[n-1]-out[n-k+1:,0].sum())
        high=max(0,maximum[n-k]-out[n-k,1:k].sum())
        out[n-k,0]=min(low,high)
    for k in range(2,n+1):
        high=max(0,maximum[0]-out[0,n-k+1:].sum())
        low=max(0,minimum[k-1]-out[1:k,n-k].sum())
        out[0,n-k]=min(low,high)
    return _nonnegative_result(out+np.rot90(np.diag(dd),-1),F.sum())


def _inverse(target,initial,k,epsilon,forward,full_output):
    target=_matrix(target)
    estimate=target.copy() if initial is None else _matrix(initial).copy()
    if estimate.shape!=target.shape or not isinstance(k,numbers.Integral) or k<0 or not np.isfinite(epsilon) or epsilon<0:
        raise ValueError('Invalid initial matrix, iteration count, or convergence tolerance')
    target,estimate=target[:,::-1],estimate[:,::-1]
    history=[]
    for _ in range(k):
        updated=np.maximum(0,estimate+target-forward(estimate))
        change=float(abs(updated-estimate).sum());estimate=updated;history.append(change)
        if change<=epsilon:break
    reconstructed=forward(estimate)
    result=estimate[:,::-1],reconstructed[:,::-1]
    if full_output:
        return (*result,dict(changes=np.asarray(history),iterations=len(history),
            residual=float(abs(target-reconstructed).sum()),
            converged=bool(history and history[-1]<=epsilon)))
    return result


def iter(f0,f=None,k=1,epsilon=1e-5,*,full_output=False):
    """Source nonnegative fixed-point inversion of turning-point rainflow.

    Returns (Markov estimate,reconstructed rainflow), optionally diagnostics.
    The method is iterative, not a promise of convergence for every target.
    """
    return _inverse(f0,f,k,epsilon,mctp2rfc,full_output)


def iter_mc(f0,f=None,k=1,epsilon=1e-5,*,full_output=False):
    """Source fixed-point inversion for general Markov rainflow kernels."""
    return _inverse(f0,f,k,epsilon,mc2rfc,full_output)


def nt2fr(M,definition=0):
    """Source mixed finite difference of a counting-distribution matrix."""
    M=np.asarray(M,dtype=float)
    if M.ndim!=2 or M.shape[0]!=M.shape[1] or not np.all(np.isfinite(M)) or definition not in (0,1):
        raise ValueError('Expected square finite counts and definition 0 or 1')
    D=M[:-1,:-1]+M[1:,1:]-M[:-1,1:]-M[1:,:-1]
    out=np.zeros_like(M)
    if definition==0:out[:-1,:-1]=D
    else:out[1:,1:]=D
    return out


def th2vhpdf(f,v=None):
    """Transform joint T/H density to V=H/T,H with Jacobian H/V**2.

    Accepts source meshgrid-xy PDF layout (height,time), or the time-first
    layout returned by Python spec2thpdf when indicated by its labels. Values
    outside the time grid are zero. No spurious-density clipping is applied.
    """
    if len(f.get('x',[]))!=2:raise ValueError('Expected a two-coordinate T/H PDF')
    t,h=map(lambda x:np.asarray(x,dtype=float),f['x'])
    density=np.asarray(f['f'],dtype=float)
    if min(len(t),len(h))<2 or np.any(np.diff(t)<=0) or np.any(np.diff(h)<=0) or t[0]<0 or h[0]<0 or not np.all(np.isfinite(density)) or np.any(density<0):
        raise ValueError('Expected increasing nonnegative coordinates and a finite nonnegative density')
    # Python Rice output is (time,height); most original PDF structures are xy.
    labels=f.get('labx',[])
    time_first=density.shape==(len(t),len(h)) and (density.shape!=(len(h),len(t)) or 'period' in str(labels).lower())
    if time_first:density=density.T
    if density.shape!=(len(h),len(t)):raise ValueError('PDF values do not match the coordinate grids')
    v=np.linspace(0,4,len(t)) if v is None else np.asarray(v,dtype=float)
    if v.ndim!=1 or not np.all(np.isfinite(v)) or np.any(v<0) or np.any(np.diff(v)<=0):
        raise ValueError('Velocities must be a nonnegative increasing vector')
    result=Spectrum(deepcopy(dict(f)));out=np.zeros((len(h),len(v)))
    valid=v>0
    for j,height in enumerate(h):
        if height<=0:continue
        time=height/v[valid]
        out[j,valid]=np.interp(time,t,density[j],left=0,right=0)*height/v[valid]**2
    result.f=out;result.x=[v,h];result.labx=['velocity','height']
    result.pop('indexing',None)  # output is (height,velocity) meshgrid-xy storage
    result.title='Joint velocity and height density';result.note='T=H/V; finite time support and linear interpolation'
    if 'err' in result:
        error=np.asarray(f['err']);error=error.T if time_first else error
        transformed=np.zeros_like(out)
        for j,height in enumerate(h):
            if height>0:transformed[j,valid]=np.interp(height/v[valid],t,error[j],left=0,right=0)*height/v[valid]**2
        result.err=transformed
    return result


def spec2tpdf2(spec,utc=0,definition='Tc',paramt=None,options=None):
    """Gaussian temporal half-wave period PDF using the shared Rice driver.

    Spatial/transformed branches remain unsupported by that driver. The output
    method and finite-grid/tail caveats are retained rather than hidden.
    """
    from ._wave_integrals import spec2thpdf
    if str(definition).lower() not in ('tc','tt'):
        raise NotImplementedError('Only temporal Tc/Tt definitions are implemented')
    return spec2thpdf(spec,utc,definition,paramt,options=options)


def spec2tpdf(spec,utc=0,definition='Tc',paramt=None,h=0,nit=None,speed=None,bound=0,plotflag=0,*,options=None):
    """Half-wave period subdensity selected by crest/trough amplitude >h.

    The source amplitude-selected output is per reference crossing (not divided
    by P(amplitude>h)). Legacy integration selectors and continuous bounds are
    explicitly rejected. h=0 matches spec2tpdf2.
    """
    if nit is not None or speed is not None or bound or plotflag:
        raise NotImplementedError('Use Python integration options and explicit plotting; native selectors/bounds unsupported')
    if not np.isscalar(h) or not np.isfinite(h) or h<0:raise ValueError('h must be finite and nonnegative')
    base=spec2tpdf2(spec,utc,definition,paramt,options)
    if h==0:return base
    from ._wave_integrals import _half_periods
    _,mask,_,bounded,errors=_half_periods(spec,0. if utc is None else float(utc),definition,paramt,[h],options)
    base.f=base.f-bounded[mask,0]
    base.err=base.err+errors[mask,0]
    base.selection=f'amplitude > {h}'
    return base


def spec2cmat(spec,utc=0,definition='rfc',paramt=None,paramu=None,nit=None,*,options=None,full_output=False):
    """Discrete Gaussian extrema/rainflow probability matrix.

    The Rice density is multiplied by level spacing squared, as in the source.
    Rainflow uses a Markov turning-point approximation. Transformed, crest/trough
    and legacy native integration-selector branches are not silently mapped.
    """
    from ._wave_integrals import spec2mmtpdf
    if nit is not None:raise NotImplementedError('Use Python integration options instead of native nit')
    if definition.lower() not in ('mm','rfc'):raise NotImplementedError('Only Mm and rfc cycle definitions are implemented')
    density=spec2mmtpdf(spec,utc,'Mm',paramt,paramu,options)
    u=np.asarray(density.x[0]);step=u[1]-u[0]
    mm=Spectrum(deepcopy(dict(density)));mm.f=density.f*step**2;mm.err=density.err*step**2
    mm.note='Discrete min-row/max-column cycle mass from finite-grid Rice density'
    result=Spectrum(deepcopy(dict(mm)))
    if definition.lower()=='rfc':
        result.f=mctp2rfc(mm.f[:,::-1])[:,::-1]
        result.title='Markov approximation to Gaussian rainflow probability matrix'
        # Propagated RF uncertainty is not available; do not retain false bounds.
        result.pop('err',None)
        result.note+='; Rice-to-Markov uncertainty is not propagated'
    return (result,mm) if full_output else result


def spec2AcAt(spec,utc=0,definition='Tcc',paramtc=None,paramtt=None,paramt=None,h1=None,h2=None,nit=None,speed=None,bound=0,*,options=None):
    """Finite-horizon Gaussian crest/trough joint survival via Rice period PDF.

    The shared Python driver uses a single grid for total period and its
    intermediate crossing. Separate paramtc/paramtt grids are not ignored.
    """
    from ._wave_integrals import spec2tccpdf
    if str(definition).lower()!='tcc' or paramtc is not None or paramtt is not None or nit is not None or speed is not None or bound:
        raise NotImplementedError('Use temporal Tcc with one paramt grid and Python options')
    h1=np.asarray([0.] if h1 is None else h1,dtype=float).reshape(-1)
    h2=np.asarray([0.] if h2 is None else h2,dtype=float).reshape(-1)
    if np.any(h1<0) or np.any(h2<0) or not np.all(np.isfinite([*h1,*h2])):
        raise ValueError('Amplitude thresholds must be finite and nonnegative')
    values=np.zeros((len(h2),len(h1)));error=np.zeros_like(values)
    for j,trough in enumerate(h2):
        for i,crest in enumerate(h1):
            density=spec2tccpdf(spec,utc,'t>',paramt,crest,trough,options=options)
            values[j,i]=trapezoid(density.f,x=density.x[0])
            error[j,i]=trapezoid(density.err,x=density.x[0])
    return Spectrum(f=values,err=error,x=[h1,h2],labx=['crest threshold','trough threshold'],
        title='Finite-horizon joint crest/trough survival',note='No omitted-tail or path-grid error bound')


def initoptions(speed,options=None):
    """Map source accuracy speeds 1..11 to Python Sobol tolerances/budgets.

    Native quadrature-order and normal-cutoff fields do not describe the new
    engine and are omitted, so the returned mapping can be passed to rind.
    """
    from ._rind import rindoptset
    if not isinstance(speed,numbers.Integral) or not 1<=speed<=11:
        raise ValueError('speed must be an integer from 1 through 11')
    result=rindoptset(options)
    eps=1e-4 if speed<=3 else 1e-3 if speed<=6 else 1e-2 if speed<=10 else .1
    exponent=((max(11-speed,1)+1)%3)+1
    result.update(abseps=eps,releps=min(eps,.01),coveps=eps*.1**exponent,
                  maxpts=30000 if speed<=3 else 20000 if speed<=6 else 10000)
    return result


def lc2tr2(cross,ma=0,sa=1,options=None,*,full_output=False):
    """Cumulative-crossing version of the source Gaussian transformation fit.

    Integrates the crossing curve, applies the source half-count correction,
    estimates its central mode, then smooths Gaussian quantiles. Uses the same
    Python smoothing objective/options as lc2tr, not the square-root-log method.
    """
    from ._transformations import _options,_grid,_variance,_enabled,_finish,troptset
    from ..misc import smooth
    if isinstance(cross,str) and cross=='defaults':
        return troptset(csm=.9,gsm=.05,ntr=1000)
    opt=_options(options)
    c=np.asarray(cross,dtype=float)
    if c.ndim!=2 or c.shape[1]!=2 or not np.all(np.isfinite(c)) or np.any(np.diff(c[:,0])<=0) or np.any(c[:,1]<0) or c[:,1].sum()<=0:
        raise ValueError('Expected increasing levels and a finite nonnegative crossing intensity')
    ne=int(opt['ne'])
    if len(c)<=2*ne+2:raise ValueError('Not enough crossing points after endpoint trimming')
    u,grid=_grid(ma,sa,opt)
    correction_low=correction_high=0.
    if opt['ntr'] and len(c)>opt['ntr']:
        if ne:
            correction_low=trapezoid(c[:ne+1,1],x=c[:ne+1,0])
            correction_high=trapezoid(c[-ne-1:,1],x=c[-ne-1:,0])
        xx=np.linspace(c[ne,0],c[-ne-1,0] if ne else c[-1,0],int(opt['ntr']))
        c=np.column_stack((xx,np.interp(xx,c[:,0],c[:,1])))
        ne=0
    cumulative=cumulative_trapezoid(c[:,1],x=c[:,0],initial=0)+correction_low
    probabilities=(cumulative+.5)/(cumulative[-1]+correction_high+1)
    z=(c[:,0]-ma)/sa
    first=int(np.argmin(abs(probabilities-.15)));last=int(np.argmin(abs(probabilities-.85)))
    if last-first<2:raise ValueError('Too little central crossing mass to estimate a mode')
    central=slice(first,last+1)
    mode_values=smooth(z[central],c[central,1],opt['csm'],z[central],_enabled(opt['linextrap']),_variance(opt['cvar'],last-first+1))
    peak=int(np.argmax(mode_values));mode=z[first+peak]
    # The source chooses a grid node, introducing a half-cell translation on
    # even grids. A local concave quadratic removes that artificial shift.
    if 0<peak<len(mode_values)-1:
        xx=z[first+peak-1:first+peak+2]
        a,b,_=np.polyfit(xx,mode_values[peak-1:peak+2],2)
        candidate=-b/(2*a) if a<0 else mode
        if xx[0]<=candidate<=xx[-1]:mode=candidate
    gaussian=norm.ppf(probabilities)-mode
    empirical=np.column_stack((c[:,0],gaussian))
    selection=slice(ne,len(c)-ne if ne else None)
    fitted=smooth(z[selection],gaussian[selection],opt['gsm'],u,
                  _enabled(opt['linextrap']),_variance(opt['gvar'],len(z[selection])))
    return _finish(grid,fitted,u,empirical,opt,full_output)


def mctrtest(S,Np,test0=None,definition='nonlinear',options=None,*,seed=None,full_output=False):
    """Monte Carlo transformation-departure test under a Gaussian spectrum.

    Np is a sample count or [sample_count,simulation_count] (default100 runs).
    Returns exactly the requested number of independent statistics; source
    batching/remainder bugs are avoided. Optional diagnostics give the finite
    Monte Carlo tail probability (count+1)/(runs+1), not an automatic rejection.
    """
    from ._transformations import dat2tr,troptset
    from ..wsim import spec2sdat
    sizes=np.atleast_1d(Np)
    if len(sizes) not in (1,2) or np.any(sizes!=np.floor(sizes)) or sizes[0]<3 or (len(sizes)==2 and sizes[1]<1):
        raise ValueError('Np must be a sample count >=3 or [sample_count,positive_runs]')
    n,runs=int(sizes[0]),100 if len(sizes)==1 else int(sizes[1])
    if test0 is not None and (not np.isscalar(test0) or not np.isfinite(test0) or test0<0):
        raise ValueError('Observed departure must be finite and nonnegative')
    if S.get('tr') is not None and np.size(S.get('tr')):
        raise ValueError('The null spectrum must be untransformed Gaussian')
    opt=troptset({} if options is None else options,multip=1)
    rng=np.random.default_rng(seed);tests=[]
    block=max(1,200000//n)
    for start in range(0,runs,block):
        count=min(block,runs-start)
        simulation=spec2sdat(S,n,cases=count,seed=int(rng.integers(2**32)))
        _,departure,*_=dat2tr(simulation,definition,opt,full_output=True)
        tests.extend(np.asarray(departure).reshape(-1))
    tests=np.asarray(tests)
    info=dict(simulations=runs,sample_count=n,seed=seed,
              pvalue=None if test0 is None else float((1+np.count_nonzero(tests>=test0))/(runs+1)))
    return (tests,info) if full_output else tests


def _qtf(w,h,g):
    from ..spec import w2k
    k=w2k(w,h=h,g=g)[0]
    k1,k2=np.meshgrid(k,k,indexing='xy')
    if np.isinf(h):
        return .25*(abs(k1)+abs(k2)),-.25*abs(abs(k1)-abs(k2)),np.zeros_like(w)
    w1,w2=np.meshgrid(w,w,indexing='xy')
    wp,wm=w1+w2,w1-w2;kp,km=k1+k2,abs(k1-k2);ww=w1*w2
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        first=.5*g*k1*k2/ww
        second=.25/g*(w1*w1+w2*w2+ww)
        Hs=(first-second+.25*g*(w1*k2*k2+w2*k1*k1)/(ww*wp))/(1-g*kp/wp**2*np.tanh(kp*h))+second-.5*first
        second=.25/g*(w1*w1+w2*w2-ww)
        Hd=(first-second-.25*g*(w1*k2*k2-w2*k1*k1)/(ww*wm))/(1-g*km/wm**2*np.tanh(km*h))+second-.5*first
        th=np.tanh(k*h)
        group=.5*g*(th+k*h*(1-th*th))/w
        diagonal=.5*(.5*g*(k/w)**2-.5*w*w/g+g*k/(w*group))/(1-g*h/group**2)-.5*k/np.sinh(2*k*h)
    np.fill_diagonal(Hd,diagonal)
    # As in the source, zero-frequency 0/0 expressions contribute zero.
    Hs=np.where(np.isnan(Hs),0,Hs);Hd=np.where(np.isnan(Hd),0,Hd);diagonal=np.where(np.isnan(diagonal),0,diagonal)
    if not np.all(np.isfinite(Hs)) or not np.all(np.isfinite(Hd)) or not np.all(np.isfinite(diagonal)):
        raise FloatingPointError('Resonant/nonfinite quadratic transfer coefficients')
    return Hs,Hd,diagonal


def spec2skew(S,h=None,method='approximate',*,g=9.81,full_output=False):
    """Second-order wave moments from source quadratic transfer functions.

    Supports approximate, quasi, and eigenvalue formulas. Full eigendecomposition
    replaces the source's changing eigenpair truncation; general frequency grids
    use trapezoidal weights. Returns (skew,kurtosis,mean,sigma), optionally Hs,Hd.
    """
    from ..spec import _validate
    w,density=_validate(S)
    h=S.get('h',np.inf) if h is None else h
    if np.isnan(h) or h<=0 or not np.isfinite(g) or g<=0:raise ValueError('Positive depth/gravity required')
    if S.get('type','freq')!='freq' or S.get('tr') is not None and np.size(S.get('tr')):
        raise NotImplementedError('Second-order moments require an untransformed1D angular-frequency spectrum')
    m0=trapezoid(density,x=w)
    if m0<=0:raise ValueError('Spectrum must have positive variance')
    Hs,Hd,diagonal=_qtf(w,h,g)
    interaction=trapezoid(trapezoid((Hs+Hd)*density[:,None]*density[None,:],x=w,axis=1),x=w)
    mean=2*trapezoid(diagonal*density,x=w)
    first=str(method).lower()[:1]
    if first=='a':
        sigma=np.sqrt(m0);skew=6*interaction/sigma**3;kurt=3+(4*skew/3)**2
    elif first in ('q','e'):
        half=np.diff(w)/2;weights=np.r_[half,0]+np.r_[0,half]
        mass=density*weights
        factor=np.sqrt(mass[:,None]*mass[None,:])
        hs,hd=Hs*factor,Hd*factor
        eigenvalues,vectors=np.linalg.eigh(np.block([[hd,hs],[hs,hd]]))
        if first=='q':
            sigma=np.sqrt(m0+2*np.sum(eigenvalues**2));skew=6*interaction/sigma**3;kurt=3+(4*skew/3)**2
        else:
            linear=np.sqrt(mass/2);coefficients=vectors.T@np.r_[linear,linear]
            e2,c2=eigenvalues**2,coefficients**2
            mean=eigenvalues.sum();sigma=np.sqrt(c2.sum()+2*e2.sum())
            skew=np.sum((6*c2+8*e2)*eigenvalues)/sigma**3
            kurt=3+48*np.sum((c2+e2)*e2)/sigma**4
    else:raise ValueError('Unknown second-order moment method')
    output=(float(skew),float(kurt),float(mean),float(sigma))
    return (*output,Hs,Hd) if full_output else output
