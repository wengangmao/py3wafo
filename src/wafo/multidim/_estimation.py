"""Directional spectra from measured sensor cross-spectra.

MLM and IMLM follow the steering/cross-spectrum formulas of the supplied
multidim/private sources. MEM fits their normalized exponential moment model
using a convex dual optimizer instead of the old unchecked Newton iterations.
Welch estimates replace MATLAB's obsolete psd/csd functions. The separate
advanced module implements EMEM's Fourier/AIC model-order selection.
"""
from copy import deepcopy
import numbers
import numpy as np
from scipy import optimize,signal
from scipy.integrate import trapezoid
from scipy.special import logsumexp


_DEFAULTS = dict(dflag='mean',ftype='w',thtype='radians',noverlap=None,
    window='hann',nharm=2,gravity=9.81,wdensity=1028.,bet=1,igam=1,
    x_axisdir=90.,y_axisdir=0.,maxiter=100,coefabstol=1e-7,errortol=1e-5,
    relax=1.4,diagonal_loading=1e-6,calibrate=False,plotflag=0,
    minmodelorder=1,maxmodelorder=None,maxcoef=10000.,aictol=.1,
    estimate_transfer=False)


def specoptset(*args,**kwargs):
    """Create directional-estimator options; names accept unique prefixes.

    Function names dat2dspec/dat2spec can prefix the arguments. Only implemented
    Python options are accepted; unsupported GUI selectors are not ignored.
    """
    from ..misc import parseoptions
    if args and isinstance(args[0],str) and args[0].lower() in ('dat2dspec','dat2spec'):
        args=args[1:]
    normalized=[]
    for arg in args:
        normalized.append({str(k).lower():v for k,v in arg.items()} if isinstance(arg,dict)
                          else arg.lower() if isinstance(arg,str) else arg)
    return parseoptions(_DEFAULTS,*normalized,**{k.lower():v for k,v in kwargs.items()})


def _angles(theta):
    theta=np.asarray(theta,dtype=float)
    if theta.ndim!=1 or len(theta)<3 or not np.all(np.isfinite(theta)) or np.any(np.diff(theta)<=0):
        raise ValueError('theta must be a finite increasing vector of at least three angles')
    if not np.isclose(theta[-1]-theta[0],2*np.pi,rtol=1e-7,atol=1e-10):
        raise ValueError('A complete 2*pi directional grid including both endpoints is required')
    widths=np.diff(theta)/2
    weights=np.r_[widths,0]+np.r_[0,widths]
    return theta,weights


def normspfn(DS,thetai):
    """Normalize nonnegative directional columns with stable trapezoid weights."""
    theta,weights=_angles(thetai)
    d=np.asarray(DS,dtype=float)
    vector=d.ndim==1
    d=d[:,None] if vector else d
    if d.ndim!=2 or d.shape[0]!=len(theta) or not np.all(np.isfinite(d)):
        raise ValueError('Directional columns must be finite and match theta')
    d=np.maximum(d,0)
    scale=d.max(axis=0)
    scaled=np.divide(d,scale[None,:],out=np.zeros_like(d),where=scale[None,:]>0)
    mass=weights@scaled
    result=np.divide(scaled,mass[None,:],out=np.full_like(d,1/(2*np.pi)),where=mass[None,:]>0)
    return result[:,0] if vector else result


def _inputs(Sxy,Gwt,thetai,k):
    theta,weights=_angles(thetai)
    S=np.asarray(Sxy,dtype=complex)
    G=np.asarray(Gwt,dtype=complex)
    if S.ndim!=3 or S.shape[0]!=S.shape[1] or G.ndim!=3 or G.shape!=(len(S),len(theta),S.shape[2]):
        raise ValueError('Expected Sxy=(M,M,Nf), Gwt=(M,Ntheta,Nf)')
    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(G)):
        raise ValueError('Cross-spectra and transfer functions must be finite')
    if not np.allclose(S,S.swapaxes(0,1).conj(),atol=1e-10,rtol=1e-7):
        raise ValueError('Cross-spectral matrices must be Hermitian')
    active=np.arange(S.shape[2]) if k is None else np.asarray(k)
    if active.ndim!=1 or np.any(active!=np.floor(active)) or np.any(active<0) or np.any(active>=S.shape[2]):
        raise ValueError('Active frequency indices must be zero-based integers in range')
    active=np.unique(active.astype(int))
    for index in active:
        values=np.linalg.eigvalsh(S[:,:,index])
        if values.min()<-1e-8*max(1.,abs(values).max()):
            raise ValueError('Cross-spectral matrix is not positive semidefinite')
    return S,G,theta,weights,active


def getcrossspectra(thetai,Gwt,DS):
    """Reconstruct normalized cross-spectra int Gi*conj(Gj)*D dtheta."""
    theta,weights=_angles(thetai)
    G=np.asarray(Gwt,dtype=complex)
    D=np.asarray(DS,dtype=float)
    if G.ndim!=3 or G.shape[1:]!=D.shape or G.shape[1]!=len(theta) or not np.all(np.isfinite(G)) or not np.all(np.isfinite(D)) or np.any(D<0):
        raise ValueError('Expected finite Gwt=(M,Ntheta,Nf), DS=(Ntheta,Nf)')
    return np.einsum('itf,jtf,tf,t->ijf',G,G.conj(),D,weights,optimize=True)


def mlm(Sxy,Gwt,thetai,fi=None,k=None,opt=None,*,full_output=False):
    """Maximum-likelihood/Capon directional density from inverse cross spectra.

    Uses relative diagonal loading before a Hermitian pseudoinverse. The native
    source used an absolute sqrt(eps) loading. Zero spectra return uniform D.
    """
    S,G,theta,weights,active=_inputs(Sxy,Gwt,thetai,k)
    options=specoptset(opt or {})
    loading=options['diagonal_loading']
    if not np.isfinite(loading) or loading<0:
        raise ValueError('diagonal_loading must be finite and nonnegative')
    D=np.full((len(theta),S.shape[2]),1/(2*np.pi))
    ranks=np.zeros(S.shape[2],dtype=int)
    for index in active:
        matrix=S[:,:,index]
        power=max(float(np.trace(matrix).real/len(matrix)),0.)
        if power==0:
            continue
        loaded=matrix+np.eye(len(matrix))*loading*power
        inverse=np.linalg.pinv(loaded,hermitian=True)
        ranks[index]=np.linalg.matrix_rank(loaded)
        if loading==0 and ranks[index]<len(matrix):
            raise ValueError('Singular sensor cross-spectra need positive diagonal_loading')
        denominator=np.einsum('it,ij,jt->t',G[:,:,index].conj(),inverse,G[:,:,index],optimize=True).real
        floor=max(np.max(abs(denominator))*np.finfo(float).eps,np.finfo(float).tiny)
        D[:,index]=1/np.maximum(denominator,floor)
    D=normspfn(D,theta)
    info=dict(method='MLM',rank=ranks,diagonal_loading=loading)
    return (D,info) if full_output else D


def imlm(Sxy,Gwt,thetai,fi=None,k=None,opt=None,*,full_output=False):
    """Source IMLM signed-square corrections with normalized angular columns.

    Reports iteration convergence; budget exhaustion is not silently certified.
    """
    S,G,theta,_,active=_inputs(Sxy,Gwt,thetai,k)
    options=specoptset(opt or {})
    count=options['maxiter']; relax=options['relax']; tolerance=options['errortol']
    if not isinstance(count,numbers.Integral) or count<1 or not np.isfinite(relax) or not 0<relax<=1.5 or not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError('Require maxiter>=1, 0<relax<=1.5 and positive errortol')
    D0=mlm(S,G,theta,fi,active,options)
    D=D0.copy(); previous=np.inf;converged=False
    for iteration in range(1,int(count)+1):
        modeled=getcrossspectra(theta,G,D)
        difference=D0-mlm(modeled,G,theta,fi,active,options)
        updated=normspfn(D+relax*np.sign(difference)*difference**2,theta)
        change=float(np.max(abs(updated-D)))
        D=updated
        if iteration>5 and min(change,3*abs(change-previous))<tolerance:
            converged=True;break
        previous=change
    info=dict(method='IMLM',iterations=iteration,converged=converged,last_change=change)
    return (D,info) if full_output else D


def _features(S,G,weights):
    features=[];targets=[];constant_residual=0.
    for i in range(len(S)):
        for j in range(i,len(S)):
            response=G[i]*G[j].conj()
            for component in (np.real,np.imag):
                feature=component(response)
                if np.ptp(feature)>1e-10:
                    features.append(feature)
                    targets.append(component(S[i,j]))
                else:
                    constant_residual=max(constant_residual,abs(float(component(S[i,j]))-float(feature[0])))
    if not features:
        return np.empty((len(weights),0)),np.empty(0),constant_residual
    H=np.asarray(features).T
    target=np.asarray(targets)
    # Remove constant and dependent constraints without altering the represented
    # angular moment space. Preserve any infeasibility as a residual diagnostic.
    mean=weights@H/(2*np.pi)
    centered=H-mean
    U,s,Vh=np.linalg.svd(np.sqrt(weights)[:,None]*centered,full_matrices=False)
    rank=int(np.sum(s>max(s[0],1.)*1e-10))
    basis=Vh[:rank].T
    reduced=centered@basis
    desired=(target-mean)@basis
    discarded=(target-mean)-desired@basis.T
    return reduced,desired,max(constant_residual,float(np.max(abs(discarded),initial=0)))


def mem(Sxy,Gwt,thetai,fi=None,k=None,opt=None,*,full_output=False):
    """Maximum-entropy density fitting the source cross-spectrum moments.

    D(theta)=exp(H(theta)@lambda)/Z. Uses the convex log-partition dual and
    analytic gradient/Hessian, replacing unstable native Newton updates. For
    noisy/infeasible moments, inspect residual/convergence in full output;
    no unsupported EMEM/AIC model-order selection is implied.
    """
    S,G,theta,weights,active=_inputs(Sxy,Gwt,thetai,k)
    options=specoptset(opt or {})
    maxiter=options['maxiter']; tolerance=options['errortol']
    if not isinstance(maxiter,numbers.Integral) or maxiter<1 or tolerance<=0 or not np.isfinite(tolerance):
        raise ValueError('Require a positive iteration budget and error tolerance')
    D=np.full((len(theta),S.shape[2]),1/(2*np.pi))
    residual=np.zeros(S.shape[2]); converged=np.ones(S.shape[2],bool);iterations=np.zeros(S.shape[2],int)
    logweights=np.log(weights)
    for index in active:
        H,target,discarded=_features(S[:,:,index],G[:,:,index],weights)
        if H.shape[1]==0:
            residual[index]=discarded
            converged[index]=discarded<=tolerance
            continue
        def evaluate(coef):
            exponent=H@coef
            logZ=logsumexp(logweights+exponent)
            probabilities=np.exp(logweights+exponent-logZ)
            moment=probabilities@H
            value=logZ-coef@target
            gradient=moment-target
            centered=H-moment
            hessian=centered.T@(probabilities[:,None]*centered)
            return value,gradient,hessian
        result=optimize.minimize(lambda c:evaluate(c)[0],np.zeros(H.shape[1]),
            jac=lambda c:evaluate(c)[1],hess=lambda c:evaluate(c)[2],method='trust-exact',
            options=dict(maxiter=int(maxiter),gtol=min(tolerance,options['coefabstol'])))
        exponent=H@result.x
        D[:,index]=np.exp(exponent-logsumexp(logweights+exponent))
        residual[index]=max(discarded,float(np.max(abs(evaluate(result.x)[1]),initial=0)))
        converged[index]=residual[index]<=tolerance
        iterations[index]=result.nit
    info=dict(method='MEM',converged=converged,residual=residual,iterations=iterations)
    return (D,info) if full_output else D


def bfsspec(Sf,Hw,pos,bfs=None):
    """Best surface-elevation frequency spectrum from selected sensor autospectra.

    Direction-independent sensors are divided by squared frequency response.
    Orthogonal x/y sensors must be present in pairs; curvatures require xx,yy,xy.
    No empirical recalibration of theoretical sensor response is applied.
    """
    Sf,Hw,pos=np.asarray(Sf,dtype=float),np.asarray(Hw,dtype=float),np.asarray(pos,dtype=float)
    if Sf.ndim!=2 or Hw.shape!=Sf.shape or pos.shape!=(len(Sf),5) or not np.all(np.isfinite(Sf)) or np.any(Sf<0) or not np.all(np.isfinite(Hw)) or np.any(Hw<0) or not np.all(np.isfinite(pos)):
        raise ValueError('Expected nonnegative autospectra, matching responses and M by5 positions')
    if bfs is None:
        selected=np.flatnonzero(pos[:,4]>.5)
    else:
        selected=np.asarray(bfs)
        if selected.dtype==bool:
            selected=np.flatnonzero(selected)
        if selected.ndim!=1 or np.any(selected!=np.floor(selected)) or np.any(selected<0) or np.any(selected>=len(pos)):
            raise ValueError('bfs indices must be zero-based sensor indices')
        selected=selected.astype(int)
    if not len(selected):
        raise ValueError('At least one sensor must contribute a frequency spectrum')
    normalized=np.divide(Sf,Hw**2,out=np.zeros_like(Sf),where=Hw>0)
    candidates=[]
    independent=(1,2,3,9,12,15,18)
    for index in selected:
        sensor=int(pos[index,3])
        if sensor in independent:
            candidates.append(normalized[index]);continue
        family=next((pair for pair in ((4,5),(10,11),(13,14),(16,17)) if sensor in pair),None)
        if family is None and sensor not in (6,7,8):
            raise ValueError('Unsupported frequency-spectrum sensor type')
        types=family if family is not None else (6,7,8)
        members=[np.flatnonzero(pos[:,3]==kind) for kind in types]
        if any(not len(item) for item in members):
            raise ValueError('Frequency estimation needs complete orthogonal sensor pairs/triples')
        count=min(map(len,members))
        total=np.zeros(Sf.shape[1])
        for j,group in enumerate(members):
            total+=(2 if family is None and j==2 else 1)*normalized[group[:count]].mean(axis=0)
        candidates.append(total)
    return np.mean(candidates,axis=0)


def dat2dspec(xn,pos,h=np.inf,nfft=256,nt=101,method='MLM',options=None,*,full_output=False):
    """Estimate directional spectrum from uniformly sampled physical sensors.

    xn columns are time and sensor values; pos rows are [x,y,z,type,bfs].
    Methods MLM, IMLM, MEM and Fourier/AIC EMEM are implemented. Welch PSD/CSD has physical density units;
    unlike the legacy code, no extra obsolete psd scaling factors are applied.
    Theoretical sensor calibration is preserved by default (calibrate=False,
    estimate_transfer=False). estimate_transfer=True enables the source's
    autospectral common-response calibration after estimating the best spectrum.

    Returns a spectral mapping. full_output=True adds (D,Sw,Fcof,Gwt,Sxy,info).
    Gwt excludes the frequency factor; Sxy is the normalized cross-spectrum.
    info also contains the unnormalized measured cross-spectral matrices.
    """
    from ..spec import Spectrum
    from . import tran,emem,hwestimate
    options=specoptset(options or {})
    x=np.asarray(xn,dtype=float);pos=np.asarray(pos,dtype=float)
    if x.ndim!=2 or x.shape[1]<4 or len(x)<4 or not np.all(np.isfinite(x)) or pos.shape!=(x.shape[1]-1,5) or not np.all(np.isfinite(pos)):
        raise ValueError('Expected finite time plus >=3 sensor columns and matching M by5 positions')
    if np.any(pos[:,3]!=np.floor(pos[:,3])) or np.any((pos[:,3]<1)|(pos[:,3]>18)):
        raise ValueError('Sensor type IDs must be integers 1 through 18')
    times=x[:,0];steps=np.diff(times)
    if np.any(steps<=0) or not np.allclose(steps,steps[0],rtol=1e-7,atol=1e-12):
        raise ValueError('Sampling times must be uniformly increasing')
    if not isinstance(nfft,numbers.Integral) or nfft<4 or not isinstance(nt,numbers.Integral) or nt<5:
        raise ValueError('Require integer nfft>=4 and nt>=5')
    nfft=min(int(nfft)+(int(nfft)%2),len(x)-len(x)%2)
    if options['ftype'] not in ('w','f') or str(options['thtype']).lower() not in ('r','radians','d','degrees'):
        raise ValueError('ftype must be w/f and thtype radians/degrees')
    method=method.lower()
    if method not in ('mlm','imlm','mem','emem'):
        raise NotImplementedError('Implemented estimators are MLM, IMLM, MEM and EMEM; BDM/FEM are distinct algorithms')
    if options['plotflag'] not in (0,False,'off','none'):
        raise ValueError('Plot returned spectra explicitly; automatic estimator plotting is omitted')
    if not isinstance(options['calibrate'],(bool,np.bool_)):
        raise ValueError('calibrate must be a boolean')
    if not isinstance(options['estimate_transfer'],(bool,np.bool_)):
        raise ValueError('estimate_transfer must be a boolean')
    if not np.isclose(np.cos(np.deg2rad(options['x_axisdir']-options['y_axisdir'])),0,atol=1e-8):
        raise ValueError('Frequency estimation currently requires orthogonal x/y sensor axes')
    data=x[:,1:].copy()
    detrending=options['dflag']
    if detrending in ('mean','constant'):
        data-=data.mean(axis=0)
    elif detrending=='linear':
        data=signal.detrend(data,axis=0)
    elif detrending=='ma':
        from ..onedim import detrendma
        data=detrendma(data,2*nfft)
    elif detrending!='none':
        raise ValueError('dflag must be mean,linear,ma,or none')
    if options['calibrate']:
        for kind in np.unique(pos[:,3]):
            group=np.flatnonzero(pos[:,3]==kind)
            sd=data[:,group].std(axis=0,ddof=1)
            rms=np.sqrt(np.mean(sd**2))
            if np.any(sd<=0):
                raise ValueError('Cannot recalibrate a constant sensor channel')
            data[:,group]/=sd/rms
    overlap=nfft//2 if options['noverlap'] is None else options['noverlap']
    if not isinstance(overlap,numbers.Integral) or not 0<=overlap<nfft:
        raise ValueError('noverlap must be an integer in [0,nfft)')
    parameters=dict(fs=1/steps[0],window=options['window'],nperseg=nfft,noverlap=int(overlap),nfft=nfft,detrend=False,scaling='density')
    frequency,auto=signal.welch(data,axis=0,**parameters)
    auto=auto.T
    m,nf=auto.shape
    raw=np.zeros((m,m,nf),complex)
    for i in range(m):
        raw[i,i]=auto[i]
        for j in range(i+1,m):
            _,raw[i,j]=signal.csd(data[:,i],data[:,j],**parameters)
            raw[j,i]=raw[i,j].conj()
    theta=np.linspace(-np.pi,np.pi,int(nt))
    Hw=np.zeros((m,nf));G=np.zeros((m,len(theta),nf),complex)
    for i in range(m):
        Hw[i],G[i]=tran(2*np.pi*frequency,theta,pos[i,:3],int(pos[i,3]),h,
            options['gravity'],options['wdensity'],options['bet'],options['igam'],
            options['x_axisdir'],options['y_axisdir'])
    best=bfsspec(auto,Hw,pos)
    best[best<max(best.max(initial=0),np.finfo(float).tiny)*1e-5]=0
    transfer_info=None
    if options['estimate_transfer']:
        Hw,transfer_info=hwestimate(auto,best,Hw,pos,full_output=True)
    denominator=Hw[:,None,:]*Hw[None,:,:]*best[None,None,:]
    normalized=np.divide(raw,denominator,out=np.zeros_like(raw),where=denominator>0)
    active=np.flatnonzero(best>0)
    function={'mlm':mlm,'imlm':imlm,'mem':mem,'emem':emem}[method]
    D,info=function(normalized,G,theta,frequency,active,options,full_output=True)
    nharm=options['nharm']
    if not isinstance(nharm,numbers.Integral) or not 0<=nharm<len(theta):
        raise ValueError('nharm must be an integer in [0,nt)')
    coefficients=trapezoid(D[None,:,:]*np.exp(1j*np.arange(nharm+1)[:,None,None]*theta[None,:,None]),x=theta,axis=1)
    key=options['ftype'];coordinate=2*np.pi*frequency if key=='w' else frequency
    spectral=best/(2*np.pi) if key=='w' else best
    degrees=str(options['thtype']).lower() in ('d','degrees')
    angles=np.rad2deg(theta) if degrees else theta
    spreading=D*np.pi/180 if degrees else D
    sd=Spectrum(type='dir',S=spreading*spectral[None,:],theta=angles,h=h,norm=0,
        theta_units='degrees' if degrees else 'radians',note='Welch sensor spectra with '+method.upper()+' direction estimation',**{key:coordinate})
    distribution=Spectrum(deepcopy(dict(sd)));distribution.S=spreading
    sw=Spectrum(type='freq',S=spectral,h=h,norm=0,note='Best sensor-derived elevation spectrum',**{key:coordinate})
    info.update(autospectra=auto,measured_cross_spectra=raw,frequency_hz=frequency,
        calibrated=bool(options['calibrate']),normalization='physical Welch density',active_frequencies=active,
        transfer_amplitudes=Hw,transfer_estimation=transfer_info)
    return (sd,distribution,sw,coefficients,G,normalized,info) if full_output else sd


def testmeasurements(pos,type,thet0=0,f0=.1,N=5000,dt=.5,amp=1,d=np.inf,g=9.81,
                     thetx=90,thety=0,*,noise=.0001,seed=None):
    """Source linear-wave measurement fixture; direction degrees counterclockwise from east.

    noise is uniform additive noise as in the source, with explicit seed and
    optional noise=0 for exact sensor identities. Output has one time column.
    """
    from . import tran,sensortypeid
    positions=np.asarray(pos,dtype=float)
    types=np.atleast_1d(sensortypeid(type))
    if positions.ndim!=2 or positions.shape!=(len(types),3) or not np.all(np.isfinite(types)) or not np.all(np.isfinite(positions)):
        raise ValueError('Expected M positions and M valid sensor types')
    if not isinstance(N,numbers.Integral) or N<2 or dt<=0 or f0<=0 or amp<0 or noise<0 or not np.all(np.isfinite([dt,f0,amp,noise,thet0])):
        raise ValueError('Invalid measurement-fixture sampling or wave parameters')
    times=np.arange(int(N))*dt
    carrier=amp*np.exp(-1j*2*np.pi*f0*times)
    data=np.empty((len(times),len(types)))
    rng=np.random.default_rng(seed)
    for i,identifier in enumerate(types):
        amplitude,direction=tran([2*np.pi*f0],[np.deg2rad(thet0)],positions[i],int(identifier),d,g,
            thx=thetx,thy=thety)
        data[:,i]=np.real(amplitude[0]*direction[0,0]*carrier)+noise*abs(amp)*rng.random(len(times))
    return np.column_stack((times,data))


def testsurf(N,dt,amp,f0,thet0,x,y,d=np.inf,g=9.81,thetx=90,thety=0,*,noise=None,seed=None):
    """Surface-array fixture; bearing degrees clockwise from north, as source."""
    from ..spec import w2k
    x,y=np.broadcast_arrays(np.atleast_1d(x).astype(float),np.atleast_1d(y).astype(float))
    if x.ndim!=1 or not np.all(np.isfinite([x,y])) or not isinstance(N,numbers.Integral) or N<2 or dt<=0 or amp<0 or f0<=0 or not np.all(np.isfinite([dt,amp,f0,thet0,thetx,thety])):
        raise ValueError('Invalid surface-array positions or sampling')
    k=float(w2k([2*np.pi*f0],h=d,g=g)[0][0])
    times=np.arange(int(N))*dt
    phase=k*(x*np.cos(np.deg2rad(thet0-thetx))+y*np.cos(np.deg2rad(thet0-thety)))
    result=amp*np.cos(phase[None,:]-2*np.pi*f0*times[:,None])
    scale=.01*abs(amp)/np.sqrt(2000) if noise is None else noise
    if scale<0 or not np.isfinite(scale):raise ValueError('noise must be finite and nonnegative')
    result+=scale*np.random.default_rng(seed).normal(size=result.shape)
    return np.column_stack((times,result))


def testbuoy(N,dt,amp,f0,thet0,x=0,y=0,d=np.inf,g=9.81,thetx=90,thety=0,*,noise=None,seed=None):
    """Elevation and x/y slope fixture; bearing degrees clockwise from north."""
    from ..spec import w2k
    surface=testsurf(N,dt,amp,f0,thet0,x,y,d,g,thetx,thety,noise=0,seed=seed)
    x,y=np.broadcast_arrays(np.atleast_1d(x).astype(float),np.atleast_1d(y).astype(float))
    k=float(w2k([2*np.pi*f0],h=d,g=g)[0][0])
    cx,cy=np.cos(np.deg2rad(thet0-thetx)),np.cos(np.deg2rad(thet0-thety))
    phase=k*(x*cx+y*cy)[None,:]-2*np.pi*f0*surface[:,0,None]
    data=np.column_stack((surface[:,1:],-amp*k*cx*np.sin(phase),-amp*k*cy*np.sin(phase)))
    scale=.01*abs(amp)/np.sqrt(200) if noise is None else noise
    if scale<0 or not np.isfinite(scale):raise ValueError('noise must be finite and nonnegative')
    data+=scale*np.random.default_rng(seed).normal(size=data.shape)
    return np.column_stack((surface[:,0],data))
