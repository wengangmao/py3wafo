"""Myrhaug–Kjeldsen 1987 Weibull/lognormal wave model.

Uses the source's variance parameterization for log steepness. Conditional
densities and CDF-integrands use their mathematically correct Jacobians rather
than the MATLAB helper's common division by both marginal scales.
"""
import numpy as np
from scipy import integrate,stats
from ..spec import Spectrum


def _parameters(h):
    mean=np.where(h<=1.7,.024-1.065*h+.585*h*h,.32*np.arctan(3.14*(h-1.7))-.096)
    variance=np.maximum(-.21*np.arctan(2*(h-1.4))+.325,np.finfo(float).eps)
    return mean,variance


def _scales(Hs,Tz,normalized):
    Hs,Tz=np.broadcast_arrays(np.asarray(Hs,dtype=float),Tz)
    if not np.all(np.isfinite(Hs)) or not np.all(np.isfinite(Tz)) or np.any(Hs<=0) or np.any(Tz<=0):
        raise ValueError('Hs and Tz must be positive finite values')
    return (np.ones_like(Hs),np.ones_like(Hs)) if normalized else (.715*Hs,.0202+.826*Hs/Tz**2)


def mk87pdf(Hd,Scf,Hs=6,Tz=8,condon=0,normalizedInput=False):
    """Joint wave height/steepness density or specified conditional quantity.

    condon:0 joint PDF,1 conditional steepness PDF,3 S*conditional PDF,
    4 S²*conditional PDF,5 f_H*F(S|H),6 F(S|H),7 f_H*SF(S|H).
    """
    Hd,Scf,Hs,Tz=np.broadcast_arrays(Hd,Scf,Hs,Tz)
    Hrms,Erms=_scales(Hs,Tz,normalizedInput)
    h,s=Hd/Hrms,Scf/Erms
    mean,variance=_parameters(h)
    fh=stats.weibull_min.pdf(h,2.39,scale=1.05)/Hrms
    fs=stats.lognorm.pdf(s,np.sqrt(variance),scale=np.exp(mean))/Erms
    F=stats.lognorm.cdf(s,np.sqrt(variance),scale=np.exp(mean))
    if condon==0:return fh*fs
    if condon==1:return fs
    if condon==3:return Scf*fs
    if condon==4:return Scf**2*fs
    if condon==5:return fh*F
    if condon==6:return F
    if condon==7:return fh*stats.lognorm.sf(s,np.sqrt(variance),scale=np.exp(mean))
    raise ValueError('Unknown MK87 conditional definition')


def mk87pdf2(Hd,Scf,Hs=6,Tz=8):
    h,s=np.asarray(Hd,dtype=float),np.asarray(Scf,dtype=float)
    if h.ndim!=1 or s.ndim!=1:raise ValueError('Grid coordinates must be vectors')
    ss,hh=np.meshgrid(s,h,indexing='xy')
    return Spectrum(f=mk87pdf(hh,ss,Hs,Tz),x=[s,h],
        labx=['Crest front steepness','Wave height [m]'],
        title='Myrhaug and Kjeldsen (1987) density')


def mk87cdf(Hd,Scf,Hs=6,Tz=8,tail=0,*,tol=1e-8,full_output=False):
    """Joint lower or joint upper-tail probability via conditional integration.

    Evaluates the source mathematical model for positive sea-state parameters;
    does not silently return zero outside its empirical calibration region.
    """
    if tail not in (0,1) or tol<=0:raise ValueError('tail must be 0/1 and tol positive')
    Hd,Scf,Hs,Tz=np.broadcast_arrays(Hd,Scf,Hs,Tz)
    Hrms,Erms=_scales(Hs,Tz,False)
    out,error=np.zeros(Hd.shape),np.zeros(Hd.shape)
    for idx in np.ndindex(Hd.shape):
        h,s=Hd[idx]/Hrms[idx],Scf[idx]/Erms[idx]
        if np.isnan(h+s):out[idx]=np.nan;continue
        if tail==0 and (h<=0 or s<=0):continue
        if tail==1 and np.isposinf(h):continue
        lower,upper=(0,max(0,h)) if tail==0 else (max(0,h),np.inf)
        def integrand(height):
            mean,variance=_parameters(height)
            distribution=stats.lognorm(np.sqrt(variance),scale=np.exp(mean))
            prob=distribution.cdf(s) if tail==0 else distribution.sf(s)
            return float(stats.weibull_min.pdf(height,2.39,scale=1.05)*prob)
        out[idx],error[idx]=integrate.quad(integrand,lower,upper,epsabs=tol,epsrel=tol,limit=200)
    return (out,error) if full_output else out


def mk87rnd(Hrms=1,Erms=1,N=1,*,seed=None):
    """Return (heights,steepness) using RMS scales, as in original mk87rnd."""
    Hrms,Erms=np.broadcast_arrays(np.asarray(Hrms,dtype=float),Erms)
    if not np.all(np.isfinite([Hrms,Erms])) or np.any(Hrms<=0) or np.any(Erms<=0):
        raise ValueError('RMS scales must be positive finite values')
    rng=np.random.default_rng(seed)
    shape=np.broadcast_shapes(np.empty(N).shape,Hrms.shape)
    h=1.05*rng.weibull(2.39,size=shape)
    mean,variance=_parameters(h)
    return h*Hrms,np.exp(mean+np.sqrt(variance)*rng.normal(size=shape))*Erms
