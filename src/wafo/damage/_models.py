"""Additional fatigue models and reconstruction of missing sphdam executable."""
import numpy as np
from scipy.special import gamma, ndtr
from scipy.integrate import quad


def cmat2dmat(param,F,beta,K=1):
    """Elementwise Palmgren–Miner amplitude damage on upper-triangular cycles.

    Applies K (accidentally omitted from the original MATLAB implementation).
    """
    from ..cycles import levels
    u = levels(param)
    F = np.asarray(F,dtype=float)
    if F.shape!=(len(u),len(u)) or not np.all(np.isfinite(F)) or np.any(F<0):
        raise ValueError('Expected nonnegative finite cycle counts on the grid')
    if not np.isfinite(beta) or beta<=0 or not np.isfinite(K) or K<0:
        raise ValueError('beta must be positive and K nonnegative')
    amp = abs(u[:,None]-u[None,:])/2
    return K*np.triu(F,1)*amp**beta


def spec2dplus(S,bet):
    """Gaussian spectral amplitude-damage intensity from spec2dplus.m.

    Uses the implemented amplitude convention, not the different full-range
    convention in that file's introductory formula.
    """
    from ..spec import spec2mom
    bet = np.atleast_1d(np.asarray(bet,dtype=float))
    if not np.all(np.isfinite(bet)) or np.any(bet<=0):
        raise ValueError('Damage exponents must be positive')
    m0,m2 = spec2mom(S,2)
    if m0<=0 or m2<0:
        raise ValueError('Spectrum must have positive variance')
    return 2**(bet/2)*m0**((bet-1)/2)*gamma(1+bet/2)*np.sqrt(m2)/(2*np.pi)


def roadspec(sdata=(4/257,4,357),a=2.1,C=50e-8):
    """Truncated power-law road spectrum C/w**a, with density zero below wl."""
    from ..spec import createspec
    wl,wu,n = sdata
    if not 0<wl<wu or n<2 or int(n)!=n or not 2<=a<=3 or C<0 or not np.isfinite(C):
        raise ValueError('Invalid road-spectrum truncation, exponent, or scale')
    w = np.linspace(0,wu,int(n))
    density = np.zeros_like(w)
    inside = w>=wl
    density[inside] = C/w[inside]**a
    result = createspec(w=w,S=density)
    result.note = 'Truncated power-law road spectrum'
    return result


def ftf(e,d,sigma2,sigma_D,number=0,*,time=None):
    """Fatigue failure-time CDF under the Gaussian damage approximation.

    Returns (time,F). Evaluates the intended normal-mixture integral in ftf.m;
    unlike that file, returns the integrated I rather than its unused unnoised
    initial F. sigma_D is the standard deviation parameter used by its formula.
    Negative approximate accumulated damage contributes zero probability.
    """
    if number:
        raise NotImplementedError('Plot the returned time/CDF arrays explicitly')
    if e<=0 or d<=0 or sigma2<=0 or sigma_D<0 or not np.all(np.isfinite([e,d,sigma2,sigma_D])):
        raise ValueError('Require positive e,d,sigma2 and nonnegative sigma_D')
    if time is None:
        center = 1/(d*e)
        time = np.linspace(.5*center,1.5*center,100)
    time = np.atleast_1d(np.asarray(time,dtype=float))
    if not np.all(np.isfinite(time)) or np.any(time<0):
        raise ValueError('Failure times must be finite and nonnegative')
    out = np.zeros_like(time)
    for i,ti in enumerate(time):
        if ti==0:
            continue
        mean,sd = d*e*ti,e*sigma_D*np.sqrt(ti)
        if sd==0:
            out[i] = ndtr(np.log(mean)/np.sqrt(sigma2))
            continue
        lower = -mean/sd
        def integrand(z):
            damage = mean+sd*z
            return ndtr(np.log(damage)/np.sqrt(sigma2))*np.exp(-z*z/2)/np.sqrt(2*np.pi)
        out[i] = quad(integrand,lower,np.inf,epsabs=1e-10,epsrel=1e-8)[0]
    return time,out


def sphdam(L,n,beta):
    """Directional rainflow range damage for a three-component load.

    Reconstructs the capability of the unavailable sphdam.exe: project onto
    an n-by-n polar/azimuth sphere grid and sum positive rainflow ranges**beta.
    Output mapping has directions (n,n,3), damage (n,n,nbeta), theta, phi.
    This explicit grid/output convention is not historical executable parity.
    """
    from ..cycles import rfcfilter,tp2rfc
    L = np.asarray(L,dtype=float)
    beta = np.atleast_1d(np.asarray(beta,dtype=float))
    if L.ndim!=2 or L.shape[1]!=3 or len(L)<2 or not np.all(np.isfinite(L)):
        raise ValueError('L must be a finite N by 3 load array')
    if n<2 or int(n)!=n or not np.all(np.isfinite(beta)) or np.any(beta<=0):
        raise ValueError('Need integer n>=2 and positive damage exponents')
    theta = np.linspace(0,np.pi,int(n))
    phi = np.linspace(0,2*np.pi,int(n),endpoint=False)
    tt,pp = np.meshgrid(theta,phi,indexing='ij')
    directions = np.stack((np.sin(tt)*np.cos(pp),np.sin(tt)*np.sin(pp),np.cos(tt)),axis=-1)
    D = np.zeros((int(n),int(n),len(beta)))
    for i in range(int(n)):
        for j in range(int(n)):
            cycles = tp2rfc(rfcfilter(L@directions[i,j],0,1))
            ranges = cycles[:,1]-cycles[:,0]
            D[i,j] = np.sum(ranges[:,None]**beta,axis=0)
    return dict(theta=theta,phi=phi,directions=directions,damage=D)
