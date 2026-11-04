"""Spectral constructors, change of variables, and covariance transforms.

The implementations use explicit quadrature/interpolation rather than invoking
MATLAB. Encounter and Cartesian directional conversions are not implicit.
"""
from copy import deepcopy
from itertools import product
import numpy as np
from scipy.integrate import simpson, trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.special import gammaln, i0e, i1e
from scipy.optimize import brentq
from . import Spectrum, _validate, _grid, createspec, jonswap, w2k, k2w, spec2mom, spec2cov


def _copy(S):
    return Spectrum(deepcopy(dict(S)))


def _frequency(S):
    fields = [key for key in ('w','f','k') if key in S and np.size(S[key])]
    if len(fields)!=1:
        raise ValueError('Spectrum must contain one populated w, f, or k coordinate')
    key = fields[0]
    x,y = np.asarray(S[key],dtype=float),np.asarray(S['S'],dtype=float)
    if (x.ndim!=1 or len(x)<2 or np.any(~np.isfinite(x)) or x[0]<0
            or np.any(np.diff(x)<=0) or y.shape[-1:]!=(len(x),)
            or np.any(~np.isfinite(y)) or np.any(y<0)):
        raise ValueError('Invalid frequency coordinate or spectral density')
    return key,x,y


def freqtype(S):
    """Frequency-coordinate name: w (rad/s), f (Hz), or k (rad/m)."""
    keys=[k for k in ('w','f','k') if k in S]
    populated=[k for k in keys if np.size(S[k])]
    keys=populated or keys
    if len(keys)!=1:
        raise ValueError('Expected one spectrum coordinate field w, f, or k')
    return keys[0]


def lagtype(R):
    """Populated covariance coordinate names, ordered x,y,t."""
    populated = ''.join(key for key in ('x','y','t') if key in R and np.size(R[key]))
    if not populated:
        raise ValueError('Covariance has no populated x, y, or t coordinate')
    return populated


def createcov(nr=0, vari='t', ctype='none'):
    """Covariance-field constructor, including all mixed derivatives up to nr."""
    if not isinstance(nr,(int,np.integer)) or nr<0:
        raise ValueError('nr must be a nonnegative integer')
    vari = ''.join(v for v in 'xyt' if v in vari.lower())
    if not vari:
        raise ValueError('vari must contain x, y, or t')
    R = Spectrum(R=np.array([]),h=np.inf,tr=[],phi=0,type=ctype,norm=0,note='')
    for v in vari:
        R[v] = np.array([])
    if len(vari)==1:
        R.stdev = np.array([])
    if ctype.startswith('enc'):
        R.v=0
    for powers in product(range(nr+1),repeat=len(vari)):
        if 1<=sum(powers)<=nr:
            R['R'+''.join(v*p for v,p in zip(vari,powers))] = np.array([])
    return R


def phi1(w,h,g=9.81):
    """Finite-depth TMA transfer factor from the supplied phi1.m formula."""
    w = np.asarray(w,dtype=float)
    if np.any(~np.isfinite(w)) or np.any(w<0) or not np.isfinite(g) or g<=0 or np.isnan(h) or h<=0:
        raise ValueError('Require finite nonnegative frequencies and positive depth/gravity')
    if np.isinf(h):
        return np.ones_like(w)
    kinf=w*w/g
    k = w2k(w,h=h,g=g)[0]
    kh=k*h
    th=np.tanh(kh)
    denominator=th+kh*(1-th*th)
    return np.divide(np.divide(kinf,k,out=np.zeros_like(w),where=k>0)**3,
                     denominator,out=np.zeros_like(w),where=denominator>0)


def tmaspec(w=None,sdata=(7,11),h=42):
    """JONSWAP multiplied by finite-depth transfer; no hidden renormalization."""
    S=jonswap(w,sdata)
    S.S *= phi1(S.w,h)
    S.h=h
    S.note='TMA finite-depth spectrum'
    return S


def ttspec(S,ftype=None,thtype='radians'):
    """Convert frequency units Hz/rad-s and directional angle units."""
    key,x,y=_frequency(S)
    target=('f' if key=='w' else 'w' if key=='f' else 'k') if ftype is None else str(ftype).lower()[0]
    if target not in ('f','w','k'):
        raise ValueError('ftype must be f, w, or k')
    if (key=='k')!=(target=='k'):
        raise ValueError('Use spec2spec for frequency/wavenumber change of variables')
    out=_copy(S)
    if target!=key:
        del out[key]
        out[target]=x*(2*np.pi if target=='w' else 1/(2*np.pi))
        out.S=y/(2*np.pi) if target=='w' else y*2*np.pi
    if 'theta' in out and np.size(out.theta):
        old=out.get('theta_units','degrees' if np.ptp(out.theta)>3*np.pi else 'radians')
        target_angle='degrees' if thtype.lower().startswith('d') else 'radians'
        if old!=target_angle:
            factor=180/np.pi if target_angle=='degrees' else np.pi/180
            out.theta=np.asarray(out.theta)*factor
            out.S=np.asarray(out.S)/factor
            out.phi=out.get('phi',0)*factor
        out.theta_units=target_angle
    return out


def scalespec(S,m0n=1,m2n=1):
    """Scale spectral amplitude/frequency so m0=m0n and m2=m2n."""
    if not np.isfinite(m0n+m2n) or m0n<=0 or m2n<=0:
        raise ValueError('Target spectral moments must be positive')
    key,x,y=_frequency(S)
    if y.ndim!=1:
        raise NotImplementedError('Moment scaling currently supports one-dimensional spectra')
    angular=x*2*np.pi if key=='f' else x
    density=y/(2*np.pi) if key=='f' else y
    m0=simpson(density,x=angular)
    m2=simpson(density*angular**2,x=angular)
    if m0<=0 or m2<=0:
        raise ValueError('Input must have positive zero and second moments')
    A=np.sqrt(m0*m2n/(m2*m0n))
    B=m0n/(m0*A)
    out=_copy(S)
    out[key]=x*A
    out.S=y*B
    out.norm=int(m0n==1 and m2n==1)
    out.note=str(out.get('note',''))+' Scaled'
    return out


def wnormspec(S,m0=None,m2=None):
    """Normalize spectral m0 and m2 to one; optional moments are input moments."""
    if m0 is None and m2 is None:
        return scalespec(S)
    if m0 is None or m2 is None or m0<=0 or m2<=0:
        raise ValueError('Supply both positive input moments or neither')
    key,x,y=_frequency(S)
    if y.ndim!=1:
        raise NotImplementedError('Only one-dimensional normalization is supported')
    A=np.sqrt(m0/m2)
    out=_copy(S)
    out[key]=x*A
    out.S=y/(m0*A)
    out.norm=1
    return out


def specinterp(S,dt=None):
    """Interpolate a spectrum onto a power-of-two frequency grid for dt."""
    key,x,y=_frequency(S)
    natural=1/(2*x[-1]) if key=='f' else np.pi/x[-1]
    dt=natural if dt is None else float(dt)
    if not np.isfinite(dt) or dt<=0:
        raise ValueError('dt must be positive and finite')
    intervals=2**int(np.ceil(np.log2(len(x)-1)))
    while natural*(len(x)-1)/intervals>dt:
        intervals*=2
        if intervals>2**20:
            raise ValueError('Requested interpolation grid is too large')
    new=np.linspace(0,(.5 if key=='f' else np.pi)/dt,intervals+1)
    out=_copy(S)
    out[key]=new
    if y.ndim==1:
        out.S=np.interp(new,x,y,left=0,right=0)
    elif y.ndim==2:
        out.S=np.vstack([np.interp(new,x,row,left=0,right=0) for row in y])
    else:
        raise NotImplementedError('Only one-dimensional/directional spectra are supported')
    return out


def covinterp(R,t,vari=None):
    """Cubic covariance interpolation, with derivative parity at zero lag."""
    vari=lagtype(R) if vari is None else vari
    if len(vari)!=1:
        raise NotImplementedError('Only one-dimensional covariance interpolation is supported')
    old=np.asarray(R[vari],dtype=float)
    if old.ndim!=1 or len(old)<2 or np.any(np.diff(old)<=0) or not np.all(np.isfinite(old)):
        raise ValueError('Invalid covariance coordinate')
    request=np.asarray(t,dtype=float)
    if request.ndim==0:
        if request<=0:
            raise ValueError('Interpolation time step must be positive')
        new=np.arange(int(np.floor(old[-1]/request))+1)*request
    elif request.shape==(2,):
        step,count=request
        if step<=0 or count<1 or count!=int(count):
            raise ValueError('Expected [positive time step, integer count]')
        new=np.arange(int(count))*step
    elif request.ndim==1:
        new=request
    else:
        raise ValueError('Expected a time grid, time step, or [step,count]')
    if np.any(~np.isfinite(new)) or np.any(new<0) or np.any(new>old[-1]):
        raise ValueError('Interpolation is restricted to the available lag interval')
    out=_copy(R)
    out[vari]=new
    for name,value in R.items():
        if name=='R' or (name.startswith('R') and all(c in 'xyt' for c in name[1:])):
            y=np.asarray(value,dtype=float)
            if y.size==0:
                continue
            if y.shape!=old.shape:
                raise ValueError('Covariance arrays must match the lag grid')
            if old[0]==0:
                sign=(-1)**(len(name)-1)
                xx=np.r_[-old[:0:-1],old]
                yy=np.r_[sign*y[:0:-1],y]
            else:
                xx,yy=old,y
            out[name]=CubicSpline(xx,yy)(new)
    return out


def cov2spec(R,rate=1,nugget=0,trunc=1e-5,fast=True):
    """Even covariance embedding and FFT; negative estimates are clipped to zero."""
    lag=lagtype(R)
    if len(lag)!=1:
        raise NotImplementedError('Only one-dimensional covariances are supported')
    x=np.asarray(R[lag],dtype=float)
    r=np.asarray(R['R'],dtype=float).copy()
    if (x.ndim!=1 or x.shape!=r.shape or len(x)<2 or x[0]!=0 or
            np.any(~np.isfinite(r)) or np.any(~np.isfinite(x)) or
            x[1]<=0 or not np.allclose(np.diff(x),x[1],rtol=1e-8,atol=1e-12)):
        raise ValueError('Covariance must be finite, uniform, and start at zero lag')
    if not np.isfinite(nugget) or nugget<0 or not 0<=trunc<1 or not np.isfinite(rate) or rate<1:
        raise ValueError('Invalid nugget/truncation/interpolation rate')
    nfft=2*len(r)-2
    if fast:
        nfft=2**int(np.ceil(np.log2(nfft)))
    r[0]+=nugget
    embedded=np.r_[r,np.zeros(nfft-2*len(r)+2),r[-2:0:-1]]
    periodogram=np.maximum(np.fft.rfft(embedded).real,0)
    if periodogram.max()>0:
        periodogram[periodogram<periodogram.max()*trunc]=0
    coordinate=np.linspace(0,np.pi/x[1],len(periodogram))
    density=periodogram*x[1]/np.pi
    factor=2**int(np.ceil(np.log2(rate)))
    if factor>1:
        new=np.linspace(0,coordinate[-1],(len(coordinate)-1)*factor+1)
        density=np.interp(new,coordinate,density)
        coordinate=new
    out=createspec('freq' if lag=='t' else 'k1d')
    if lag!='t':
        out.pop('w',None)
        out.k=coordinate
    else:
        out.w=coordinate
    out.S=density
    for key in ('h','tr','norm','note'):
        if key in R:
            out[key]=deepcopy(R[key])
    return out


def spec2cov2(S,nr=0,Nt=None,dt=None):
    """Matrix of covariance derivatives; Nt is the largest lag index."""
    if not isinstance(nr,(int,np.integer)) or not 0<=nr<=4:
        raise ValueError('nr must be an integer from zero to four')
    converted=ttspec(S,'w') if freqtype(S)=='f' else S
    if freqtype(converted)=='k':
        converted=_copy(converted)
        converted.w=converted.pop('k')
    w,_=_validate(converted)
    dt=np.pi/w[-1] if dt is None else float(dt)
    Nt=len(w)-1 if Nt is None else Nt
    if not isinstance(Nt,(int,np.integer)) or Nt<0 or not np.isfinite(dt) or dt<=0:
        raise ValueError('Invalid lag count or spacing')
    out=spec2cov(converted,nr,Nt+1,dt)
    return np.column_stack([out['R'+'t'*n] for n in range(nr+1)])


def spec2spec(S,newtype=None,phi=0,v=0):
    """Convert intrinsic frequency, directional, Cartesian, and encounter spectra.

    One-dimensional frequency/wave-number conversions preserve the native grid.
    Directional projection and encounter folds use conservative bin averages.
    Encounter spectra can be integrated over direction but cannot be inverted
    to an intrinsic spectrum: Doppler folding loses the necessary information.
    ``phi`` adds a coordinate rotation; ``v`` is used only for encounter output.
    """
    from ._directional_transforms import dir2enc,dir2k1d,spa2time,time2spa
    out=_copy(S)
    old=str(S.get('type','freq')).lower()
    target=old if newtype is None else newtype.lower()
    valid={'freq','dir','rotdir','k1d','rotk1d','k2d','rotk2d','enc','encdir'}
    if old not in valid or target not in valid:
        raise ValueError('Unknown spectrum type')
    if not np.isscalar(phi) or not np.isfinite(phi) or not np.isscalar(v) or not np.isfinite(v):
        raise ValueError('phi and v must be finite scalars')
    if out.get('theta_units','radians')=='degrees' and old in ('dir','rotdir','encdir'):
        out=ttspec(out,freqtype(out),'radians')
    if phi:
        out.phi=(float(out.get('phi',0))+phi+np.pi)%(2*np.pi)-np.pi
    if old.startswith('enc') and v:
        raise NotImplementedError('An existing encounter spectrum cannot be re-encountered at another speed')
    if v and not target.startswith('enc'):
        raise ValueError('Ship speed v applies only to encounter output')
    if target==old:
        return out
    if old.startswith('enc'):
        if old=='encdir' and target=='enc':
            values=np.asarray(out.S)
            out.S=(trapezoid(values,x=out.theta,axis=0)
                   if np.size(out.theta)>1 else values.ravel())
            out.pop('theta',None)
            out.pop('theta_units',None)
            out.type='enc'
            return out
        raise NotImplementedError('Encounter-to-intrinsic inversion is ambiguous after Doppler folding')
    if old.removeprefix('rot')==target.removeprefix('rot'):
        out.type=target
        return out
    if target.startswith('enc'):
        if old.endswith(('k1d','k2d')):
            out=spa2time(out)
        return dir2enc(out,2 if target=='encdir' else 1,v)
    if old in ('dir','rotdir') and target=='freq':
        out.S=trapezoid(np.asarray(out.S),x=out.theta,axis=0) if np.size(out.theta)>1 else np.asarray(out.S).ravel()
        out.pop('theta',None)
        out.pop('theta_units',None)
        out.type='freq'
        return out
    if old=='freq' and target in ('k1d','rotk1d'):
        if freqtype(out)=='f':
            out=ttspec(out,'w')
        k=w2k(out.w,h=out.get('h',np.inf),g=float(np.atleast_1d(out.get('g',9.81))[0]))[0]
        result=time2spa(out,k)
        result.type=target
        return result
    if old.endswith('k1d') and target=='freq':
        w=k2w(out.k,h=out.get('h',np.inf),g=float(np.atleast_1d(out.get('g',9.81))[0]))[0]
        return spa2time(out,w)
    if old in ('dir','rotdir') and target.endswith('k2d'):
        result=time2spa(out)
        result.type=target
        return result
    if old in ('dir','rotdir') and target.endswith('k1d'):
        result=dir2k1d(out)
        result.type=target
        return result
    if old.endswith('k2d') and target in ('dir','rotdir','freq','k1d','rotk1d'):
        result=spa2time(out)
        if target in ('dir','rotdir'):
            result.type=target
            return result
        return spec2spec(result,target)
    raise NotImplementedError(f'Spectral conversion {old} -> {target} is not implemented')


def rotspec(S,phi=0,rotateGrid=False):
    """Rotate directional spectra counter-clockwise, metadata or physical grid."""
    if not np.isscalar(phi) or not np.isfinite(phi):
        raise ValueError('Rotation must be a finite scalar in radians')
    out=_copy(S)
    if str(S.get('type','')).lower() not in ('dir','encdir','k2d'):
        if phi:
            raise ValueError('Only two-dimensional spectra can be rotated')
        return out
    out.phi=(float(out.get('phi',0))+phi+np.pi)%(2*np.pi)-np.pi
    if rotateGrid:
        if str(S['type']).lower()=='k2d':
            raise NotImplementedError('Cartesian physical-grid rotation is not implemented')
        if out.get('theta_units','radians')!='radians':
            raise ValueError('Convert directional angles to radians before rotation')
        theta=np.asarray(out.theta)
        density=np.asarray(out.S)
        if theta.shape[0]!=density.shape[0]:
            raise ValueError('Directional grid and density do not match')
        # Drop a duplicate periodic endpoint, rotate and sort, then reclose.
        if np.isclose(theta[-1]-theta[0],2*np.pi):
            theta,density=theta[:-1],density[:-1]
        angles=(theta+out.phi+np.pi)%(2*np.pi)-np.pi
        order=np.argsort(angles)
        angles,density=angles[order],density[order]
        out.theta=np.r_[angles,angles[0]+2*np.pi]
        out.S=np.vstack((density,density[0]))
        out.phi=0.
    return out


def spreading(th=101,type='cos2s',th0=0,data=None,w=None,definition=1):
    """All six WAFO directional kernels, parameterized by cos-2s spread S.

    definition=0: frequency independent; definition=1: Mitsuyasu/Hasselmann
    power law. Legacy Donelan/Banner definitions 2/3 are not implemented.
    """
    if definition not in (0,1):
        raise NotImplementedError('Spreading definitions 0 and 1 are implemented')
    theta=np.linspace(-np.pi,np.pi,int(th)) if np.ndim(th)==0 else np.asarray(th,dtype=float)
    if (theta.ndim!=1 or len(theta)<3 or np.any(~np.isfinite(theta)) or np.any(np.diff(theta)<=0)
            or not np.isclose(theta[-1]-theta[0],2*np.pi)):
        raise ValueError('theta must be increasing and span one full 2*pi period')
    w=np.linspace(0,3,257) if w is None else np.asarray(w,dtype=float)
    if w.ndim!=1 or np.any(~np.isfinite(w)) or np.any(w<0):
        raise ValueError('Frequency grid must be finite and nonnegative')
    par=np.array([15,15,.52,5,-2.5,0,1,np.inf],dtype=float)
    if data is not None:
        values=np.atleast_1d(data).astype(float)
        if len(values)>len(par):
            raise ValueError('At most eight spreading parameters are supported')
        par[:len(values)]=np.where(np.isnan(values),par[:len(values)],values)
    if np.any(~np.isfinite(par[:7])) or par[0]<0 or par[1]<0 or par[2]<=0 or not 0<=par[5]<par[6]<par[7]:
        raise ValueError('Invalid spreading parameters')
    directions=np.atleast_1d(th0).astype(float)
    if not np.all(np.isfinite(directions)) or len(directions) not in (1,len(w)):
        raise ValueError('Mean direction must be scalar or match the frequency grid')
    dependent=definition==1 or len(directions)>1
    if definition==0:
        spread=np.full(len(w) if dependent else 1,par[0])
    else:
        wn=w/par[2]
        spread=np.zeros_like(wn)
        a=(wn>par[5])&(wn<=par[6])
        b=(wn>par[6])&(wn<=par[7])
        spread[a]=par[0]*wn[a]**par[3]
        spread[b]=par[1]*wn[b]**par[4]
    angles=(theta[:,None]-directions[None,:]+np.pi)%(2*np.pi)-np.pi
    if angles.shape[1]==1 and len(spread)>1:
        angles=np.broadcast_to(angles,(len(theta),len(spread)))
    s=spread[None,:]
    r=s/(s+1)
    kind=type.lower()
    if kind.startswith('c'):
        density=np.exp(gammaln(s+1)-gammaln(s+.5))/(2*np.sqrt(np.pi))*np.maximum(np.cos(angles/2),0)**(2*s)
    elif kind.startswith('p'):
        density=(1-r*r)/(1-2*r*np.cos(angles)+r*r)/(2*np.pi)
    elif kind.startswith('w'):
        density=np.full_like(angles,1/(2*np.pi))
        for n in range(1,max(64,len(theta))):
            density+=r**(n*n)*np.cos(n*angles)/np.pi
        density=np.maximum(density,0)
    elif kind.startswith(('m','b','s')):
        parameter=np.zeros_like(spread)
        for j,rr in enumerate(r.ravel()):
            if kind.startswith('m'):
                parameter[j]=0 if rr==0 else brentq(lambda z:i1e(z)/i0e(z)-rr,0,max(10,2/(1-rr)))
            elif kind.startswith('b'):
                parameter[j]=np.pi if rr==0 else brentq(lambda z:np.sinc(z/np.pi)-rr,1e-10,np.pi)
            else:
                parameter[j]=0 if rr==0 else brentq(lambda z:(0 if np.pi/(2*z)>700 else np.pi/(2*z)/np.sinh(np.pi/(2*z)))-rr,1e-4,max(10,2/(1-rr)))
        p=parameter[None,:]
        if kind.startswith('m'):
            density=np.exp(p*(np.cos(angles)-1))/(2*np.pi*i0e(p))
        elif kind.startswith('b'):
            density=(abs(angles)<=p)/(2*p)
        else:
            thv=np.tanh(p*angles)
            density=np.divide(.5*p*(1-thv*thv),np.tanh(np.pi*p),out=np.full_like(angles,1/(2*np.pi)),where=p>0)
    else:
        raise ValueError('Unknown spreading family')
    # Preserve unit mass on the user's actual angle grid, including box edges.
    area=trapezoid(density,x=theta,axis=0)
    if np.any(area<=0):
        raise ValueError('Angle grid is too coarse to resolve the spreading kernel')
    density/=area
    out=Spectrum(S=density,theta=theta,type='dir',phi=0.,th0=directions,data=par,note='Spreading: '+kind,theta_units='radians')
    if dependent:
        out.w=w
    return out


def mkdspec(S=None,D=None):
    """Multiply a frequency spectrum by a normalized directional spreading."""
    S=jonswap() if S is None else S
    key,w,s=_frequency(S)
    if key!='w' or s.ndim!=1:
        raise ValueError('Directional composition requires a 1D angular-frequency spectrum')
    D=spreading(w=w) if D is None else D
    density=np.asarray(D['S'],dtype=float)
    if density.ndim==1:
        density=density[:,None]
    theta=np.asarray(D['theta'],dtype=float)
    if density.shape[0]!=len(theta) or density.shape[1] not in (1,len(w)) or np.any(density<0):
        raise ValueError('Spreading grid and density do not match the spectrum')
    if 'w' in D and not np.allclose(D['w'],w,rtol=0,atol=1e-10):
        raise ValueError('Spreading and spectrum frequency grids must match')
    out=_copy(S)
    out.S=density*s[None,:]
    out.theta=theta.copy()
    out.phi=D.get('phi',0)
    out.type='dir'
    out.theta_units=D.get('theta_units','radians')
    out.note=str(S.get('note',''))+'; '+str(D.get('note',''))
    return out


def demospec(stype='freq'):
    """The source's preconfigured JONSWAP demo, with optional directional spread."""
    S=jonswap(np.linspace(0,1.14,257))
    if stype.lower()=='freq':
        return S
    D=spreading(101,'cos2s',0,15,S.w,0)
    if stype.lower()=='dir':
        return mkdspec(S,D)
    return spec2spec(S,stype)
