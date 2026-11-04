"""Directional spreading, Fourier coefficients and linear sensor responses."""
from copy import deepcopy
import numpy as np
from scipy.integrate import trapezoid
from ._advanced_estimation import emem,hwestimate,memfun
from ..spec import Spectrum,w2k

_SENSORS = ('n','n_t','n_tt','n_x','n_y','n_xx','n_yy','n_xy',
            'p','u','v','w','u_t','v_t','w_t','x_p','y_p','z_p')


def sensortype(sensorid=1):
    """Sensor name for mathematical IDs 1..18; invalid IDs return 'nan'."""
    ids = np.asarray(sensorid)
    def name(value):
        return _SENSORS[int(value)-1] if np.isfinite(value) and int(value)==value and 1<=value<=18 else 'nan'
    result = np.asarray([name(value) for value in ids.ravel()]).reshape(ids.shape)
    return str(result.item()) if result.ndim==0 else result


def sensortypeid(sensor='n'):
    """Sensor mathematical ID 1..18; unknown names return NaN."""
    values = np.asarray(sensor)
    def number(value):
        if isinstance(value,(str,np.str_)):
            value = str(value).strip().lower()
            return _SENSORS.index(value)+1 if value in _SENSORS else np.nan
        return value if np.isfinite(value) and int(value)==value and 1<=value<=18 else np.nan
    result = np.asarray([number(value) for value in values.ravel()]).reshape(values.shape)
    return result.item() if result.ndim==0 else result


def ratio(a,b,sa=1,sb=1):
    """Stable cosh/sinh ratio; sa/sb=1 select cosh, -1 select sinh.

    Identical zero sinh arguments return the source convention 1. Undefined
    unequal infinities are rejected; finite extreme arguments avoid overflow
    unless the mathematical ratio itself exceeds floating-point range.
    """
    a,b,sa,sb = np.broadcast_arrays(np.asarray(a,dtype=float),b,sa,sb)
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(abs(sa)!=1) or np.any(abs(sb)!=1):
        raise ValueError('Arguments must not be NaN and selectors must be +/-1')
    def loghyper(x,selector):
        ax = abs(x)
        with np.errstate(divide='ignore'):
            logarithm = ax+np.where(selector==1,np.log1p(np.exp(-2*ax)),np.log(-np.expm1(-2*ax)))-np.log(2)
        sign = np.where(selector==1,1,np.sign(x))
        return logarithm,sign
    la,signa = loghyper(a,sa); lb,signb = loghyper(b,sb)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        result = signa/signb*np.exp(la-lb)
    result = np.where((a==b)&(sa==sb),1,result)
    # cosh(0)/sinh(0) has a positive one-sided divergent limit.
    result = np.where((b==0)&(sb==-1)&~((a==0)&(sa==-1)),np.copysign(np.inf,signa),result)
    if np.any(np.isnan(result)):
        raise ValueError('Indeterminate hyperbolic ratio')
    return result


def fourier(t,x,T=None,M=None,N=None):
    """Fourier cosine/sine coefficients with trapezoidal period averaging.

    Corrects the hard-coded 1/pi normalization in the MATLAB source so arbitrary
    primitive periods and multiple observed cycles have consistent amplitudes.
    """
    t = np.asarray(t,dtype=float)
    x = np.asarray(x,dtype=float)
    n = len(t) if N is None else int(N)
    if t.ndim!=1 or n!=len(t) or x.shape[0]!=n or n<2 or np.any(np.diff(t)<=0) or not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)):
        raise ValueError('Expected increasing finite times and matching finite signals')
    T = t[-1]-t[0] if T is None else T
    M = n if M is None else M
    if not np.isfinite(T) or T<=0 or int(M)!=M or M<1:
        raise ValueError('Period and coefficient count must be positive')
    x = x[:,None] if x.ndim==1 else x
    phase = 2*np.pi*np.arange(int(M))[:,None]*t[None,:]/T
    scale = 2/(t[-1]-t[0])
    a = scale*trapezoid(x[None,:,:]*np.cos(phase)[:,:,None],x=t,axis=1)
    b = scale*trapezoid(x[None,:,:]*np.sin(phase)[:,:,None],x=t,axis=1)
    return a,b


def dspec2spreading(S):
    """Normalize each directional-spectrum frequency column over theta."""
    result = Spectrum(deepcopy(dict(S)))
    theta = np.asarray(S['theta'],dtype=float)
    density = np.asarray(S['S'],dtype=float)
    if density.ndim!=2 or density.shape[0]!=len(theta) or np.any(np.diff(theta)<=0):
        raise ValueError('Directional density rows must match increasing theta')
    density = np.where(np.isfinite(density)&(density>0),density,0)
    # Scale each column before integration: near-zero spectral tail columns
    # otherwise lose relative precision when their mass is subnormal.
    scale = density.max(axis=0)
    scaled = np.divide(density,scale[None,:],out=np.zeros_like(density),where=scale[None,:]>0)
    mass = trapezoid(scaled,x=theta,axis=0)
    result.S = np.divide(scaled,mass[None,:],out=np.zeros_like(density),where=mass[None,:]>0)
    return result


def tran(w,theta=0,pos=(0,0,0),definition=1,h=np.inf,g=9.81,rho=1028,
         bet=1,igam=1,thx=90,thy=0,kw=None,*,full_output=False):
    """Linear Airy-wave response for the 18 documented sensor types.

    Returns (frequency_factor,direction_phase_factor), or additionally
    (wavenumbers,full_response,location_phase) with full_output=True. Sensor
    IDs stay 1..18. Deep-water vertical profiles use exp(k*z), correcting the
    indeterminate infinity/infinity expressions in the old routine.
    """
    w,theta,pos = np.atleast_1d(w).astype(float),np.atleast_1d(theta).astype(float),np.asarray(pos,dtype=float)
    identifier = sensortypeid(definition)
    if not np.isscalar(identifier) or not np.isfinite(identifier):
        raise ValueError('Unknown sensor definition')
    identifier = int(identifier)
    if w.ndim!=1 or theta.ndim!=1 or pos.shape!=(3,) or np.any(w<0) or not np.all(np.isfinite(w)) or not np.all(np.isfinite(theta)) or not np.all(np.isfinite(pos)) or h<=0 or np.isnan(h) or not np.isfinite(g) or g<=0 or not np.isfinite(rho) or rho<=0 or bet not in (-1,1) or igam not in (1,2,3):
        raise ValueError('Invalid frequency, geometry, depth or physical constants')
    k = w2k(w,h=h,g=g)[0] if kw is None else np.asarray(kw,dtype=float)
    if k.shape!=w.shape or not np.all(np.isfinite(k)) or np.any(k<0):
        raise ValueError('Wavenumbers must match frequencies and be nonnegative')
    cx = bet*np.cos(theta-np.deg2rad(thx)+np.pi/2)
    cy = bet*np.sin(theta-np.deg2rad(thy))
    ee = np.exp(1j*(pos[0]*cx+pos[1]*cy)[:,None]*k[None,:])
    if identifier>8:
        z = pos[2] if igam==1 else -pos[2] if igam==2 else pos[2]-h
        if np.isinf(h):
            if igam==3 or z>0:
                raise ValueError('Deep-water sensors require a mean-level vertical origin and z<=0')
            cc=cs=ss=np.exp(k*z)
        else:
            if not -h<=z<=0:
                raise ValueError('Water-particle sensor must be between bottom and mean surface')
            zk,hk=k*(h+z),k*h
            cc,cs,ss=ratio(zk,hk,1,1),ratio(zk,hk,1,-1),ratio(zk,hk,-1,-1)
    with np.errstate(invalid='ignore'):
        table = {
            1:(np.ones_like(w),ee),2:(w,-1j*ee),3:(w*w,-ee),
            4:(k,1j*cx[:,None]*ee),5:(k,1j*cy[:,None]*ee),
            6:(k*k,-cx[:,None]**2*ee),7:(k*k,-cy[:,None]**2*ee),8:(k*k,-cx[:,None]*cy[:,None]*ee),
        }
        if identifier>8:
            table.update({9:(rho*g*cc,ee),10:(w*cs,cx[:,None]*ee),11:(w*cs,cy[:,None]*ee),12:(w*ss,-1j*ee),
                13:(w*w*cs,-1j*cx[:,None]*ee),14:(w*w*cs,-1j*cy[:,None]*ee),15:(w*w*ss,-ee),
                16:(cs,1j*cx[:,None]*ee),17:(cs,1j*cy[:,None]*ee),18:(ss,ee)})
    amplitude,direction = table[identifier]
    amplitude = np.where(np.isfinite(amplitude),amplitude,0)
    direction = -direction if igam==2 else direction
    return (amplitude,direction,k,direction*amplitude[None,:],ee) if full_output else (amplitude,direction)


from ._estimation import (specoptset,normspfn,getcrossspectra,mlm,imlm,mem,bfsspec,
    dat2dspec,testmeasurements,testsurf,testbuoy)
