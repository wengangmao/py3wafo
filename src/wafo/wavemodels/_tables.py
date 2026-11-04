"""Original empirical-fit tables for wave-height and joint wave models.

The coefficients are numeric-only conversions of the supplied WAFO .mat data,
with original SHA-256 hashes in data/PROVENANCE.json. Cubic SciPy interpolation
is an algorithm adaptation, not bitwise MATLAB *cubic parity. Sea states outside
the table rectangle are rejected; no physically unvalidated extrapolation is
silently performed. Height-dependent conditional curves retain the source's
linear extension beyond their fitted height nodes.
"""
from functools import lru_cache
from importlib.resources import files
import json
import numpy as np
from scipy.interpolate import CubicSpline,RectBivariateSpline
from scipy.integrate import quad
from scipy import stats
from ..spec import Spectrum,getjonswappeakedness
from ..wstats import wtweibpdf,wtweibcdf
from ._empirical import jhwparfun,jhnlwparfun


@lru_cache(maxsize=None)
def _data(name):
    with files(__package__).joinpath('data',name+'.npz').open('rb') as stream:
        with np.load(stream,allow_pickle=False) as archive:
            return {key:archive[key].copy() for key in archive.files}


def _dimension(dim):
    if not isinstance(dim,str) or dim.lower() not in ('time','space','t','s'):
        raise ValueError("dim must be time or space")
    return 'space' if dim.lower().startswith('s') else 'time'


def _state(Hm0,Tp):
    H,T=np.broadcast_arrays(np.asarray(Hm0,dtype=float),np.asarray(Tp,dtype=float))
    if not np.all(np.isfinite(H)) or not np.all(np.isfinite(T)) or np.any(H<=0) or np.any(T<=0):
        raise ValueError('Sea-state height and period must be positive and finite')
    return H,T


@lru_cache(maxsize=None)
def _surface(name,variable):
    d=_data(name)
    return RectBivariateSpline(d['Hm0'].ravel(),d['Tp'].ravel(),d[variable],kx=3,ky=3,s=0)


def _table_values(name,H,T,variables=('A00s','B00s','C00s')):
    H,T=_state(H,T)
    d=_data(name); hs,ts=d['Hm0'].ravel(),d['Tp'].ravel()
    if np.any((H<hs[0])|(H>hs[-1])|(T<ts[0])|(T>ts[-1])):
        raise ValueError(f'Sea state is outside the {name} fit table')
    result=tuple(_surface(name,key).ev(H.ravel(),T.ravel()).reshape(H.shape) for key in variables)
    if any(np.any(x<=0) or not np.all(np.isfinite(x)) for x in result):
        raise ValueError('Interpolated fit parameters are nonpositive or nonfinite')
    return result


def thgparfun(Hm0,Tp,dim='time'):
    """Generalized-gamma height-fit parameters for Torsethaugen spectra.

    Fits are indexed by Hm0 in [0.5,12] and Tp in [3.5,20]. Recommended physical
    use also requires Hm0<(Tp-2)*12/11; table coverage alone does not validate it.
    """
    return _table_values('thgpar.mat' if _dimension(dim)=='time' else 'thsspar.mat',Hm0,Tp)


def thwparfun(Hm0,Tp,dim='time'):
    """Truncated-Weibull height-fit parameters from the active source tables."""
    return _table_values('thwnlpar20-Jul-2004.mat' if _dimension(dim)=='time' else 'thsspar27-Jul-2004.mat',Hm0,Tp)


@lru_cache(maxsize=1)
def _oh_coefficients():
    return json.loads(files(__package__).joinpath('data','ohhgparfun_coefficients.json').read_text())


def ohhgparfun(Hm0,definition=1,dim='time'):
    """Generalized-gamma wave-height parameters from source rational polynomials.

    definition=1..11 selects the mathematical source spectral parametrization,
    not a zero-based array index. Values outside 0<Hm0<=12 are rejected.
    """
    H,definition=np.broadcast_arrays(np.asarray(Hm0,dtype=float),definition)
    if not np.all(np.isfinite(H)) or np.any((H<=0)|(H>12)):
        raise ValueError('Ochi-Hubble fitted height range is 0<Hm0<=12')
    if np.any(~np.isfinite(definition)) or np.any(definition!=np.floor(definition)) or np.any((definition<1)|(definition>11)):
        raise ValueError('definition must be an integer in 1..11')
    branch=0 if _dimension(dim)=='space' else 1
    co=_oh_coefficients(); result=[]
    for letter in 'abc':
        value=np.empty(H.shape)
        for index in np.ndindex(H.shape):
            i=int(definition[index])-1
            value[index]=np.exp(np.polyval(co['d'+letter+'1'][branch][i],H[index])/
                                np.polyval(co['d'+letter+'2'][branch][i],H[index]))
        result.append(value)
    return tuple(result)


def thpdf(h,Hm0,Tp,dim='time'):
    """Physical wave-height PDF using the active source truncated-Weibull fit."""
    parameters=thwparfun(Hm0,Tp,dim)
    scale=np.asarray(Hm0)/np.sqrt(2)
    return wtweibpdf(np.asarray(h)/scale,*parameters)/scale


def thcdf(h,Hm0,Tp,dim='time'):
    parameters=thwparfun(Hm0,Tp,dim)
    return wtweibcdf(np.asarray(h)/(np.asarray(Hm0)/np.sqrt(2)),*parameters)


def ohhpdf(h,Hm0,definition=1,dim='time'):
    parameters=ohhgparfun(Hm0,definition,dim)
    scale=np.asarray(Hm0)/np.sqrt(2)
    a,b,c=parameters
    return stats.gengamma.pdf(np.asarray(h)/scale,a,c,scale=b)/scale


def ohhcdf(h,Hm0,definition=1,dim='time'):
    a,b,c=ohhgparfun(Hm0,definition,dim)
    return stats.gengamma.cdf(np.asarray(h)/(np.asarray(Hm0)/np.sqrt(2)),a,c,scale=b)


_JTABLES={'jhs':'jhspar13-Jul-2004.mat','jhsnl':'jhsnlpar21-Jul-2004.mat','jhv':'jhvpar.mat'}


@lru_cache(maxsize=512)
def _curve(family,H,T,gam,key):
    d=_data(_JTABLES[family]); height=d['h2'].ravel()
    raw=d[key]; logarithmic=family!='jhv'
    raw=np.log(raw) if logarithmic else raw
    if 'gam' in d:
        grid=d['gam'].ravel()
        if not grid[0]<=gam<=grid[-1]:
            raise ValueError('Peakedness is outside the conditional fit table')
        values=CubicSpline(grid,raw,axis=1)(gam)
    else:
        hs,ts=d['Hm0'].ravel(),d['Tp'].ravel()
        if not (hs[0]<=H<=hs[-1] and ts[0]<=T<=ts[-1]):
            raise ValueError('Sea state is outside the nonlinear conditional fit table')
        values=np.asarray([RectBivariateSpline(hs,ts,raw[:,:,i],s=0).ev(H,T) for i in range(len(height))])
    return CubicSpline(height,values,bc_type='natural'),height,logarithmic


def _conditional_parameters(family,height,H,T,gam):
    height,H,T,gam=np.broadcast_arrays(height,H,T,gam)
    result=[]
    for key in ('A11s','B11s'):
        value=np.empty(height.shape)
        for index in np.ndindex(height.shape):
            curve,grid,logarithmic=_curve(family,float(H[index]),float(T[index]),float(gam[index]),key)
            point=max(float(height[index]),0)
            clamped=np.clip(point,grid[0],grid[-1])
            fitted=curve(clamped)+min(point-grid[0],0)*curve(grid[0],1)+max(point-grid[-1],0)*curve(grid[-1],1)
            value[index]=np.exp(fitted) if logarithmic else fitted
        if np.any(value<=0) or not np.all(np.isfinite(value)):
            raise ValueError('Conditional extrapolation produced invalid fit parameters')
        result.append(value)
    return tuple(result)


def _sea(H,T,gam,nonlinear=False):
    H,T=_state(H,T)
    fitted_gamma=getjonswappeakedness(H,T)
    if nonlinear and gam is not None and not np.allclose(gam,fitted_gamma,rtol=1e-10,atol=1e-12):
        raise ValueError('The nonlinear table has no independent gamma axis; use gam=None or the fitted getjonswappeakedness value')
    gam=fitted_gamma if gam is None else np.asarray(gam,dtype=float)
    H,T,gam=np.broadcast_arrays(H,T,gam)
    if np.any((gam<1)|(gam>7)) or not np.all(np.isfinite(gam)):
        raise ValueError('Peakedness must lie in the fitted interval 1..7')
    return H,T,gam


def _scales(family,H,T,gam,normalized):
    if normalized:
        return np.ones_like(H),np.ones_like(H)
    Tm02=T*np.polyval([.15659478203944,1.15736959972513,1],gam)/np.polyval([.16183666835624,1.53691936441548,1.55852759524555],gam)
    return H/np.sqrt(2),(2*H/Tm02 if family=='jhv' else 1.25*H/Tm02**2)


def _joint(family,Hd,V,Hm0,Tp,gam,normalized,condon):
    H,T,gam=_sea(Hm0,Tp,gam,family=='jhsnl')
    Hd,V,H,T,gam=np.broadcast_arrays(Hd,V,H,T,gam)
    hs,vs=_scales(family,H,T,gam,normalized)
    h,v=Hd/hs,V/vs
    a,b=_conditional_parameters(family,np.maximum(h,0),H,T,gam)
    conditional=stats.weibull_min(b,scale=a) if family=='jhv' else stats.gamma(a,scale=b)
    density=conditional.pdf(v)/vs
    cumulative=conditional.cdf(v)
    survival=conditional.sf(v)
    marginal_parameters=(jhnlwparfun if family=='jhsnl' else jhwparfun)(H,T,gam)
    marginal=wtweibpdf(h,*marginal_parameters)/hs
    if condon==0:return marginal*density
    if condon==1:return density
    if condon in (3,4):
        with np.errstate(invalid='ignore',over='ignore'):
            moment=V**(condon-2)*density
        return np.where(np.isposinf(V),0,moment)
    if condon==5:return marginal*cumulative
    if condon==6:return cumulative
    if condon==7:return marginal*survival
    raise ValueError('Supported condon values are 0,1,3,4,5,6,7')


def jhspdf(Hd,Scf,Hm0=6,Tp=8,gam=None,normalizedInput=False,condon=0):
    """Linear JONSWAP height/steepness PDF with correct conditional Jacobians."""
    return _joint('jhs',Hd,Scf,Hm0,Tp,gam,normalizedInput,condon)


def jhsnlpdf(Hd,Scf,Hm0=6,Tp=8,gam=None,normalizedInput=False,condon=0):
    """Source nonlinear height/steepness fit; gamma follows the sea-state model.

    The source table has Hm0/Tp axes and no independent gamma dimension. A
    conflicting explicit gamma is rejected instead of silently overwritten.
    """
    return _joint('jhsnl',Hd,Scf,Hm0,Tp,gam,normalizedInput,condon)


def jhvpdf(Hd,Vcf,Hm0=6,Tp=8,gam=None,normalizedInput=False,condon=0):
    """Linear JONSWAP height/front-velocity PDF with Weibull conditionals."""
    return _joint('jhv',Hd,Vcf,Hm0,Tp,gam,normalizedInput,condon)


def _cdf(family,Hd,V,Hm0,Tp,gam,tail,tol,full_output):
    H,T,gam=_sea(Hm0,Tp,gam,family=='jhsnl')
    Hd,V,H,T,gam=np.broadcast_arrays(Hd,V,H,T,gam)
    if tail not in (0,1) or not np.isfinite(tol) or tol<=0:
        raise ValueError('Require tail=0 or 1 and positive finite tolerance')
    out,error=np.zeros(H.shape),np.zeros(H.shape)
    for idx in np.ndindex(H.shape):
        height,velocity,H0,T0,g0=Hd[idx],V[idx],H[idx],T[idx],gam[idx]
        if np.isnan(height+velocity):out[idx]=np.nan;continue
        if (tail==0 and (height<=0 or velocity<=0)) or (tail==1 and (np.isposinf(height) or np.isposinf(velocity))):continue
        hs,vs=_scales(family,np.asarray(H0),np.asarray(T0),np.asarray(g0),False)
        parameters=(jhnlwparfun if family=='jhsnl' else jhwparfun)(H0,T0,g0)
        a0,b0,c0=(float(value) for value in parameters)
        cutoff=float(wtweibcdf(max(0,height)/hs,a0,b0,c0))
        lower,upper=(0,cutoff) if tail==0 else (cutoff,1)
        def function(probability):
            # Integrate with respect to the height marginal probability. This
            # avoids querying linearly extended conditional fits at enormous
            # heights generated by infinite-interval quadrature.
            h=a0*((c0/a0)**b0-np.log1p(-probability))**(1/b0)-c0
            a1,b1=_conditional_parameters(family,h,H0,T0,g0)
            conditional=stats.weibull_min(float(b1),scale=float(a1)) if family=='jhv' else stats.gamma(float(a1),scale=float(b1))
            return float(conditional.cdf(velocity/vs) if tail==0 else conditional.sf(velocity/vs))
        out[idx],error[idx]=quad(function,lower,upper,epsabs=tol,epsrel=tol,limit=150)
    return (out,error) if full_output else out


def jhscdf(Hd,Scf,Hm0=6,Tp=8,gam=None,tail=0,*,tol=1e-8,full_output=False):
    """Lower joint CDF or upper joint survival, integrated without a hard tail cutoff."""
    return _cdf('jhs',Hd,Scf,Hm0,Tp,gam,tail,tol,full_output)


def jhsnlcdf(Hd,Scf,Hm0=6,Tp=8,gam=None,tail=0,*,tol=1e-8,full_output=False):
    return _cdf('jhsnl',Hd,Scf,Hm0,Tp,gam,tail,tol,full_output)


def jhvcdf(Hd,Vcf,Hm0=6,Tp=8,gam=None,tail=0,*,tol=1e-8,full_output=False):
    return _cdf('jhv',Hd,Vcf,Hm0,Tp,gam,tail,tol,full_output)


def _grid(function,Hd,V,H,T,gam,normalized):
    h,v=np.asarray(Hd,dtype=float),np.asarray(V,dtype=float)
    if h.ndim!=1 or v.ndim!=1:
        raise ValueError('Grid coordinates must be vectors')
    vv,hh=np.meshgrid(v,h,indexing='xy')
    return Spectrum(f=function(hh,vv,H,T,gam,normalized),x=[v,h],labx=['Scf' if function!=jhvpdf else 'Vcf','Hd'],norm=int(bool(normalized)),
        title='Source fitted JONSWAP joint wave distribution')


def jhspdf2(Hd,Scf,Hm0=6,Tp=8,gam=None,normalizedInput=False):
    return _grid(jhspdf,Hd,Scf,Hm0,Tp,gam,normalizedInput)


def jhsnlpdf2(Hd,Scf,Hm0=6,Tp=8,gam=None,normalizedInput=False):
    return _grid(jhsnlpdf,Hd,Scf,Hm0,Tp,gam,normalizedInput)


def jhvpdf2(Hd,Vcf,Hm0=6,Tp=8,gam=None,normalizedInput=False):
    return _grid(jhvpdf,Hd,Vcf,Hm0,Tp,gam,normalizedInput)
