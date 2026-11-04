"""One-dimensional wave spectra; angular frequencies in rad/s."""
import numpy as np
from scipy.integrate import simpson
from ._directional_transforms import dir2enc, dir2k1d, dspec2dcov, spa2time, time2spa


class Spectrum(dict):
    """MATLAB-like spectrum fields, accessible as attributes or mapping keys."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def createspec(type='freq', freqtype='w', *, w=None, S=None, h=np.inf):
    """Construct the source spectrum fields for frequency/directional/k types."""
    type=type.lower()
    if type not in ('freq','dir','k1d','k2d','enc','encdir'):
        raise ValueError('Unknown spectrum type')
    if freqtype not in ('w','f'):
        raise ValueError('Frequency units must be w or f')
    coordinate='k' if type.startswith('k') else freqtype
    out=Spectrum(type=type,S=np.asarray([] if S is None else S,dtype=float),
                 h=h,norm=0,note='',phi=0.,tr=[])
    out[coordinate]=np.asarray([] if w is None else w,dtype=float)
    if type.endswith('dir'):
        out.theta=np.array([])
    if type=='k2d':
        out.k2=np.array([])
    if type.startswith('enc'):
        out.v=0.
    return out


def _grid(w, cutoff):
    if w is not None and np.size(w)==0:
        w=None
    grid=np.linspace(0, cutoff if w is None else float(w),257) if w is None or np.ndim(w)==0 else np.asarray(w,dtype=float)
    if grid.ndim!=1 or len(grid)<2 or not np.all(np.isfinite(grid)) or grid[0]<0 or np.any(np.diff(grid)<=0):
        raise ValueError('Frequency grid must be finite, nonnegative and strictly increasing')
    return grid


def _validate(S):
    w, s = np.asarray(S['w']), np.asarray(S['S'])
    if w.ndim != 1 or w.shape != s.shape or len(w) < 2 or np.any(np.diff(w) <= 0) or w[0] < 0 or np.any(s < 0) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)):
        raise ValueError("Expected increasing nonnegative w and a finite nonnegative 1D density")
    return w, s


def pmspec(w=None, sdata=(7, 11)):
    """Pierson–Moskowitz spectrum, sdata=(Hm0, Tp)."""
    H, T = sdata
    if H < 0 or T <= 0 or not np.isfinite(H+T):
        raise ValueError("Require Hm0 >= 0 and Tp > 0")
    w = _grid(w, 32/T)
    s = np.zeros_like(w)
    wp = 2*np.pi/T
    mask = w > 0
    z = w[mask]/wp
    s[mask] = (H/4)**2/wp*5*z**-5*np.exp(-1.25*z**-4)
    return createspec(w=w, S=s)


def getjonswappeakedness(Hm0, Tp):
    if np.any(np.asarray(Hm0)<=0) or np.any(np.asarray(Tp)<=0) or not np.all(np.isfinite(Hm0)) or not np.all(np.isfinite(Tp)):
        raise ValueError('Hm0 and Tp must be positive and finite')
    x = np.asarray(Tp)/np.sqrt(Hm0)
    D = 0.036-0.0056*x
    return np.where(x <= 5.14285714285714, np.minimum(np.exp(3.484*(1-0.1975*D*x**4)), 7), 1)


def jonswap(w=None, sdata=(7, 11, 0, .07, .09, -1)):
    """JONSWAP; sdata=(Hm0,Tp,gamma,sigma_a,sigma_b,A).

    A=0 uses the legacy approximate normalization, A<0 integrates to Hm0²/16.
    """
    p = [7, 11, 0, .07, .09, -1]
    p[:len(sdata)] = sdata
    H, T, gamma, sa, sb, A = p
    if H < 0 or T <= 0 or sa <= 0 or sb <= 0 or not np.all(np.isfinite(p)):
        raise ValueError("Invalid JONSWAP parameters")
    gamma = float(getjonswappeakedness(H, T)) if gamma < 1 and H else max(gamma, 1)
    w = _grid(w, 33/T)
    s = np.zeros_like(w)
    mask = w > 0
    z = w[mask]/(2*np.pi/T)
    sigma = np.where(z <= 1, sa, sb)
    s[mask] = w[mask]**-5*np.exp(-1.25*z**-4)*gamma**np.exp(-(z-1)**2/(2*sigma**2))
    if A < 0:
        integral = simpson(s, x=w)
        if H and integral <= 0:
            raise ValueError("Frequency grid contains no spectral mass")
        s *= (H/4)**2/integral if integral else 0
    else:
        if A == 0:
            A = 5.061*H**2/T**4*(1-.287*np.log(gamma))
        s *= A*9.81**2
    return createspec(w=w, S=s)


def spec2mom(S, nr=2, vari=None, even=True):
    """Moments m0,m2,... by default; even=False includes every order."""
    if not isinstance(nr,(int,np.integer)) or nr<0:
        raise ValueError('Moment order must be a nonnegative integer')
    if str(S.get('type','freq')).lower()=='dir':
        S=spec2spec(S,'freq')
    if freqtype(S)=='f':
        S=ttspec(S,'w')
    key,w,s=_frequency(S)
    if s.ndim!=1:
        raise NotImplementedError('Mixed moments of multidimensional spectra are not implemented')
    if vari not in (None,'x' if key=='k' else 't'):
        raise NotImplementedError('Only moments along the one-dimensional spectrum coordinate are supported')
    return np.asarray([simpson(s*w**n, x=w) for n in range(0, nr+1, 2 if even else 1)])


def spec2bw(S, fact=(1,)):
    if str(S.get('type','freq')).lower()=='dir':
        S=spec2spec(S,'freq')
    if freqtype(S)=='f':
        S=ttspec(S,'w')
    w, s = _validate(S)
    m0, m1, m2, _, m4 = spec2mom(S, 4, even=False)
    if min(m0,m1,m2,m4)<=0:
        raise ValueError('Bandwidth requires positive spectral moments')
    factors = [m2/np.sqrt(m0*m4), np.sqrt(max(0, m0*m2/m1**2-1)),
               np.sqrt(max(0, 1-m2*m2/(m0*m4))), 2*simpson(w*s*s, x=w)/m0**2]
    ids = np.atleast_1d(fact)
    if np.any((ids < 1) | (ids > 4)) or np.any(ids!=np.floor(ids)):
        raise ValueError("Factors must be 1 through 4")
    return np.asarray(factors)[ids.astype(int)-1]


def spec2dt(S):
    key,w,_=_frequency(S)
    return (.5 if key=='f' else np.pi)/w[-1]


def spec2char(S, fact=(1,)):
    """Spectral characteristics 1–15 (or names), without legacy covariances."""
    from ..misc import gravity
    if str(S.get('type','freq')).lower()=='dir':
        S=spec2spec(S,'freq')
    if freqtype(S)=='f':
        S=ttspec(S,'w')
    w, s = _validate(S)
    m0, m1, m2, _, m4 = spec2mom(S, 4, even=False)
    if min(m0,m1,m2,m4)<=0:
        raise ValueError('Spectral characteristics require positive moments')
    minus = simpson(np.divide(s,w,out=np.zeros_like(s),where=w>0),x=w)
    H = 4*np.sqrt(m0)
    t01, t02, t24 = 2*np.pi*m0/m1, 2*np.pi*np.sqrt(m0/m2), 2*np.pi*np.sqrt(m2/m4)
    peak = w[np.argmax(s)]
    tp = 2*np.pi/peak if peak>0 else np.inf
    bw = spec2bw(S,[1,2,3,4])
    values = [H,t01,t02,t24,2*np.pi*minus/m0,tp,
        2*np.pi*H/(gravity()*t02*t02),2*np.pi*H/(gravity()*tp*tp),
        abs(simpson(s*np.exp(1j*w*t02),x=w))/m0,
        np.interp(np.array([.0146,.0195,.0244])*2*np.pi,w,s).sum()/(3*max(s)),
        2*np.pi*simpson(s**4,x=w)/simpson(w*s**4,x=w),*bw]
    names = ['Hm0','Tm01','Tm02','Tm24','Tm_10','Tp','Ss','Sp','Ka','Rs','Tp1','alpha','eps2','eps4','Qp']
    out = []
    for f in np.atleast_1d(fact):
        if not isinstance(f,str) and f!=int(f):
            raise ValueError('Characteristic indices must be integers')
        idx = [n.lower() for n in names].index(f.lower()) if isinstance(f,str) else int(f)-1
        if not 0 <= idx < 15:
            raise ValueError("Unknown spectral characteristic")
        out.append(values[idx])
    return np.asarray(out)


def spec2cov(S, nr=0, Nt=None, dt=None, *, rate=1, Nx=0, Ny=0, dx=None, dy=None):
    """Covariance and derivatives by spectral quadrature.

    ``Nt`` is a sample count, retaining this Python API's existing convention;
    the private-compatible dspec2dcov instead takes a maximum lag index. The
    fourth positional argument remains explicit spacing ``dt``. Keyword-only
    ``rate`` divides the default spacing; Nx/Ny select symmetric spatial lags
    and all mixed x/y/t derivatives through order four. With no spatial lags,
    a wave-number spectrum returns R(x), while other spectra return R(t).
    Encounter data supports temporal covariance only, since its intrinsic
    spatial dispersion cannot be reconstructed after Doppler folding.
    Uniform integration grids use Simpson quadrature; irregular grids use
    nonnegative trapezoidal weights to preserve covariance positivity.
    """
    if not isinstance(nr,(int,np.integer)) or nr<0:
        raise ValueError('Derivative order must be a nonnegative integer')
    if Nt is not None and (not isinstance(Nt,(int,np.integer)) or Nt<1):
        raise ValueError('Nt must be a positive sample count')
    if dt is not None and (not np.isfinite(dt) or dt<=0):
        raise ValueError('dt must be positive and finite')
    if not isinstance(rate,(int,np.integer)) or rate<1 or rate & (rate-1):
        raise ValueError('rate must be a positive power of two')
    if any(not isinstance(v,(int,np.integer)) or v<0 for v in (Nx,Ny)):
        raise ValueError('Nx and Ny must be nonnegative integers')
    kind=str(S.get('type','freq')).lower()
    if Nx or Ny:
        if kind.startswith('enc'):
            raise NotImplementedError('Encounter spatial covariance needs the intrinsic directional spectrum')
        if kind.endswith(('k1d','k2d')):
            S=spa2time(S)
        count=len(S.get('w',S.get('f'))) if Nt is None else Nt
        spacing=spec2dt(S)/rate if dt is None else dt
        return dspec2dcov(S,nr,count-1,rate,Nx,Ny,dx,dy,dt=spacing)
    if kind.endswith('k2d'):
        S=spec2spec(S,'k1d')
    if kind in ('dir','rotdir','encdir'):
        S=spec2spec(S,'enc' if kind=='encdir' else 'freq')
    if freqtype(S)=='f':
        S=ttspec(S,'w')
    coordinate,w,s=_frequency(S)
    if s.ndim!=1:
        raise ValueError('Expected a one-dimensional marginal spectrum')
    variable='x' if coordinate=='k' else 't'
    t = np.arange(len(w) if Nt is None else Nt)*(spec2dt(S)/rate if dt is None else dt)
    out = Spectrum({variable:t},type='space' if variable=='x' else 'time',h=S.get('h',np.inf))
    for key in ('tr','norm','note','phi','v'):
        if key in S:
            out[key]=S[key]
    from ._directional_transforms import _quadweights
    weights=_quadweights(w)
    for n in range(nr+1):
        out['R' if n == 0 else 'R'+variable*n] = (s[None, :]*w[None, :]**n*np.cos(t[:, None]*w+n*np.pi/2)) @ weights
    return out


def k2w(k1, k2=0, h=np.inf, g=9.81):
    """Dispersion relation; return (angular frequency, propagation angle)."""
    k1, k2 = np.broadcast_arrays(k1, k2)
    k = np.hypot(k1, k2)
    if not np.isscalar(h) or np.isnan(h) or h <= 0 or not np.isscalar(g) or not np.isfinite(g) or g <= 0 or not np.all(np.isfinite(k1)) or not np.all(np.isfinite(k2)):
        raise ValueError("Depth and gravity must be positive")
    return np.sqrt(g*k*(1 if np.isinf(h) else np.tanh(k*h))), np.arctan2(k2, k1)


def w2k(w, theta=0, h=np.inf, g=9.81):
    """Newton solution of w²=g k tanh(kh); return (kx,ky)."""
    w, theta = np.broadcast_arrays(np.asarray(w, dtype=float), theta)
    if not np.isscalar(h) or np.isnan(h) or h <= 0 or not np.isscalar(g) or not np.isfinite(g) or g <= 0 or not np.all(np.isfinite(w)) or not np.all(np.isfinite(theta)):
        raise ValueError("Invalid dispersion parameters")
    target = w*w/g
    if np.isinf(h):
        k = target
    else:
        k = np.maximum(target, abs(w)/np.sqrt(g*h))
        for _ in range(100):
            th = np.tanh(k*h)
            derivative = th+k*h*(1-th*th)
            step = np.divide(k*th-target, derivative, out=np.zeros_like(k), where=derivative>0)
            new = np.maximum(0, k-step)
            if np.all(abs(new-k) < 1e-13*(1+abs(k))):
                k = new
                break
            k = new
        else:
            raise RuntimeError("Dispersion solver did not converge")
    k *= np.sign(w)
    return k*np.cos(theta), k*np.sin(theta)


from ._transforms import (freqtype,lagtype,createcov,phi1,tmaspec,ttspec,scalespec,
    wnormspec,specinterp,covinterp,cov2spec,spec2cov2,spec2spec,rotspec,
    spreading,mkdspec,demospec,_frequency)
from ._models import ohspec,ohspec2,ohspec3,mccormick,wallop,oscspec,torsethaugen
from ._diagnostics import specdiag,dspec2char
from ._plotting import wspecplot
