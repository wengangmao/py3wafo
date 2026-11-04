"""Source rational-polynomial conditional-gamma wave models.

Coefficients are transcribed from b04pdf.m, b04jpdf.m and bmr00pdf.m. These are
empirical model fits, not universal wave laws. Conditional-option Jacobians are
corrected consistently with the mathematical distribution.
"""
import numpy as np
from scipy import integrate,stats
from ..spec import Spectrum,getjonswappeakedness


_COEFFICIENTS={
    'b04':dict(A1=[3.04621335253845,-3.639622295671,2.92552351922864],
        A2=[-.00166846704576,.02393968243269,-.14693767011981,.54351431859392,-.72674041059717,1],
        B1=[-4.17697804696004,2.7040359916694,-1.96751680710343],
        B2=[-.00261584642057,.03498298832599,-.08180403982217,2.05476574371033,-.77207518455457,1],
        h0=4.49999999999989,shape0=23.14319996232766,shape_slope=5.37234232285048,
        scale0=.16439526054778,scale_slope=.00587463478855,rayleigh=.70079568211392),
    'b04j':dict(A1=[2.42767540920461,-5.12101444148408,4.4906025931643],
        A2=[.00572707656955,-.10679331978077,.81081632991506,-3.19389926947809,6.81410759646213,-7.14860241955959,2.4089061946404,1],
        B1=[-.70310486398236,1.74679333441188,-2.47501962948775],
        B2=[.00536662963627,-.100861058951,.77434723641863,-3.10647077544176,6.88647211568827,-7.71847113553527,3.1975517203711,1],
        h0=4.49999999999989,shape0=17.80538369601295,shape_slope=1.53878793498107,
        scale0=.15016765927218,scale_slope=.01549012000025,rayleigh=.67768680795379),
    'bmr00':dict(A1=[4.19193631070257,-12.03484717258579,15.95695603911886,-7.31618437661838,3.03590705490789],
        A2=[-.02838644340635,.29409707273437,-.79138075176131,1.20024905431293,-.58192823600509,1],
        B1=[-5.96970985003487,23.24982082461128,-37.69215199928053,5.18520245368619,-1.87015799594426],
        B2=[-.47236526233572,5.7113041801666,-17.45884089250511,21.6078193490439,.11061694237412,1],
        h0=3.79937912772197,shape0=36.15168757587687,shape_slope=14.03628528937437,
        scale0=.05673842727364,scale_slope=-.37497306844631,log_scale_extrapolation=True,rayleigh=.69233682309018),
}


def _conditional_parameters(model,h):
    p=_COEFFICIENTS[model]
    # Evaluate rational fits only inside their source interval; extrapolate
    # separately to avoid overflow/poles in branches whose values are unused.
    clipped=np.clip(h,0,p['h0'])
    shape=np.polyval(p['A1'],clipped)/np.polyval(p['A2'],clipped)
    scale=np.exp(np.polyval(p['B1'],clipped)/np.polyval(p['B2'],clipped))
    shape=np.where(h>p['h0'],p['shape0']+(h-p['h0'])*p['shape_slope'],shape)
    extrap=(p['scale0']*np.exp((h-p['h0'])*p['scale_slope']) if p.get('log_scale_extrapolation') else p['scale0']+(h-p['h0'])*p['scale_slope'])
    scale=np.where(h>p['h0'],extrap,scale)
    return shape,scale


def _scales(Hm0,Tm02,normalized=False):
    if np.any(Hm0<=0) or np.any(Tm02<=0) or not np.all(np.isfinite([Hm0,Tm02])):
        raise ValueError('Significant wave heights and periods must be positive finite')
    return (np.ones_like(Hm0),np.ones_like(Hm0)) if normalized else (Hm0/np.sqrt(2),1.25*Hm0/Tm02**2)


def _density(model,Hd,Scf,Hm0,Tm02,condon,normalized):
    Hd,Scf,Hm0,Tm02=np.broadcast_arrays(Hd,Scf,Hm0,Tm02)
    Hscale,Sscale=_scales(Hm0,Tm02,normalized)
    h,s=Hd/Hscale,Scf/Sscale
    A,B=_conditional_parameters(model,h)
    marginal=stats.rayleigh.pdf(h,scale=_COEFFICIENTS[model]['rayleigh'])/Hscale
    density=stats.gamma.pdf(s,A,scale=B)/Sscale
    cumulative=stats.gamma.cdf(s,A,scale=B)
    if condon==0:return marginal*density
    if condon==1:return density
    if condon==3:return Scf*density
    if condon==4:return Scf**2*density
    if condon==5:return marginal*cumulative
    if condon==6:return cumulative
    if condon==7:return marginal*stats.gamma.sf(s,A,scale=B)
    raise ValueError('Unknown conditional density option')


def _cdf(model,Hd,Scf,Hm0,Tm02,tail,tol,full_output):
    Hd,Scf,Hm0,Tm02=np.broadcast_arrays(Hd,Scf,Hm0,Tm02)
    Hscale,Sscale=_scales(Hm0,Tm02)
    if tail not in (0,1) or tol<=0:raise ValueError('Require tail=0/1 and positive tolerance')
    out,error=np.zeros(Hd.shape),np.zeros(Hd.shape)
    for idx in np.ndindex(Hd.shape):
        h,s=Hd[idx]/Hscale[idx],Scf[idx]/Sscale[idx]
        if np.isnan(h+s):out[idx]=np.nan;continue
        if tail==0 and (h<=0 or s<=0):continue
        if tail==1 and np.isposinf(h):continue
        lower,upper=(0,max(0,h)) if tail==0 else (max(0,h),np.inf)
        def integrand(height):
            A,B=_conditional_parameters(model,height)
            prob=stats.gamma.cdf(s,A,scale=B) if tail==0 else stats.gamma.sf(s,A,scale=B)
            return float(stats.rayleigh.pdf(height,scale=_COEFFICIENTS[model]['rayleigh'])*prob)
        # Split at the source extrapolation boundary, preserving small tails.
        split=_COEFFICIENTS[model]['h0']
        bounds=[lower]+([split] if lower<split<upper else [])+[upper]
        for a,b in zip(bounds[:-1],bounds[1:]):
            val,err=integrate.quad(integrand,a,b,epsabs=tol/2,epsrel=tol,limit=200)
            out[idx]+=val;error[idx]+=err
    return (out,error) if full_output else out


def _grid(function,Hd,Scf,Hm0,Tm02,normalized):
    h,s=np.asarray(Hd),np.asarray(Scf)
    if h.ndim!=1 or s.ndim!=1:raise ValueError('Density-grid coordinates must be vectors')
    ss,hh=np.meshgrid(s,h,indexing='xy')
    return Spectrum(f=function(hh,ss,Hm0,Tm02,normalizedInput=normalized),x=[s,h],labx=['Scf','Hd'],norm=int(bool(normalized)),title='Source empirical wave-height/steepness density')


def b04pdf(Hd,Scf,Hm0=6,Tm02=8,condon=0,normalizedInput=False):
    return _density('b04',Hd,Scf,Hm0,Tm02,condon,normalizedInput)


def b04jpdf(Hd,Scf,Hm0=6,Tm02=8,condon=0,normalizedInput=False):
    return _density('b04j',Hd,Scf,Hm0,Tm02,condon,normalizedInput)


def bmr00pdf(Hd,Scf,Hm0=6,Tm02=8,condon=0,normalizedInput=False):
    return _density('bmr00',Hd,Scf,Hm0,Tm02,condon,normalizedInput)


def b04cdf(Hd,Scf,Hs=6,Tz=8,tail=0,*,tol=1e-8,full_output=False):
    return _cdf('b04',Hd,Scf,Hs,Tz,tail,tol,full_output)


def b04jcdf(Hd,Scf,Hs=6,Tz=8,tail=0,*,tol=1e-8,full_output=False):
    return _cdf('b04j',Hd,Scf,Hs,Tz,tail,tol,full_output)


def bmr00cdf(Hd,Scf,Hs=6,Tz=8,tail=0,*,tol=1e-8,full_output=False):
    return _cdf('bmr00',Hd,Scf,Hs,Tz,tail,tol,full_output)


def b04pdf2(Hd,Scf,Hs=6,Tz=8,normalizedInput=False):
    return _grid(b04pdf,Hd,Scf,Hs,Tz,normalizedInput)


def b04jpdf2(Hd,Scf,Hs=6,Tz=8,normalizedInput=False):
    return _grid(b04jpdf,Hd,Scf,Hs,Tz,normalizedInput)


def bmr00pdf2(Hd,Scf,Hs=6,Tz=8,normalizedInput=False):
    return _grid(bmr00pdf,Hd,Scf,Hs,Tz,normalizedInput)


def _jonswap_height_parameters(Hm0,Tp,gam,dim,nonlinear):
    Hm0,Tp=np.broadcast_arrays(np.asarray(Hm0,dtype=float),Tp)
    if np.any(Hm0<=0) or np.any(Tp<=0) or not np.all(np.isfinite([Hm0,Tp])):
        raise ValueError('Positive finite sea-state parameters are required')
    gam=getjonswappeakedness(Hm0,Tp) if gam is None else np.asarray(gam,dtype=float)
    if np.any(gam<1) or np.any(gam>7) or not np.all(np.isfinite(gam)):
        raise ValueError('The fitted gamma interval is 1 through 7')
    if not dim.lower().startswith('t'):
        raise NotImplementedError('The source supplies time-domain fits only')
    if nonlinear:
        return (-.01243795213128*gam+1.08025514722235,
                -.03043834819688*gam+2.27161821064622,
                -.01342959276544*gam+.10353423379696)
    return (-.00768035933493*gam+1.12723230522407,
            -.02261969635571*gam+2.31625911055348,
            -.00808703761189*gam+.14438532196018)


def jhwparfun(Hm0,Tp,gam=None,dim='time'):
    """Source linear JONSWAP truncated-Weibull height parameters."""
    return _jonswap_height_parameters(Hm0,Tp,gam,dim,False)


def jhnlwparfun(Hm0,Tp,gam=None,dim='time'):
    """Source second-order-Stokes JONSWAP truncated-Weibull height parameters."""
    return _jonswap_height_parameters(Hm0,Tp,gam,dim,True)
