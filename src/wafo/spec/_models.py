"""Parametric spectra reconstructed from the supplied MATLAB model formulas."""
import numpy as np
from scipy.special import gammaln, gamma
from scipy.optimize import minimize_scalar
from scipy.integrate import simpson
from . import _grid, createspec, w2k


def _parameters(data,defaults):
    p=np.asarray(defaults,dtype=float)
    if data is not None:
        values=np.atleast_1d(data).astype(float)
        if len(values)>len(p):
            raise ValueError('Too many spectral parameters')
        p[:len(values)]=np.where(np.isnan(values),p[:len(values)],values)
    if not np.all(np.isfinite(p)):
        raise ValueError('Spectral parameters must be finite')
    return p


def _height_period(H,T):
    if H<0 or T<=0:
        raise ValueError('Require nonnegative significant height and positive peak period')


def ohspec(w=None,sdata=(7,11,3)):
    """Single-component Ochi-Hubble spectrum: [Hm0,Tp,lambda]."""
    H,T,L=_parameters(sdata,[7,11,3])
    _height_period(H,T)
    if L<=0:
        raise ValueError('Ochi-Hubble lambda must be positive')
    w=_grid(w,33/T)
    s=np.zeros_like(w)
    mask=w>0
    wp=2*np.pi/T
    B=L+.25
    if H:
        s[mask]=np.exp(np.log(.25*H*H)+L*np.log(B*wp**4)-gammaln(L)
                       -(4*L+1)*np.log(w[mask])-B*(wp/w[mask])**4)
    S=createspec(w=w,S=s)
    S.note=f'Ochi-Hubble Hm0={H:g}, Tp={T:g}, lambda={L:g}'
    return S


def _combine(parts,note):
    out=createspec(w=parts[0].w.copy(),S=sum(p.S for p in parts))
    out.note=note
    return out


def ohspec2(w=None,sdata=(7,1),*,components=False):
    """Bimodal Ochi-Hubble confidence family; definitions 1–11.

    components=True returns (total,(component1,component2)).
    """
    H,definition=_parameters(sdata,[7,1])
    if H<0 or int(definition)!=definition or not 1<=definition<=11:
        raise ValueError('Require Hm0>=0 and an integer definition from 1 through 11')
    family={
        1:([.84,.54],[.70,1.15],[.046,.039],[3,1.54,.062]),
        2:([.84,.54],[.93,1.50],[.056,.046],[3,2.77,.112]),
        3:([.84,.54],[.41,.88],[.016,.026],[2.55,1.82,.089]),
        4:([.84,.54],[.74,1.30],[.052,.039],[2.65,3.90,.085]),
        5:([.84,.54],[.62,1.03],[.039,.030],[2.60,.53,.069]),
        6:([.95,.31],[.70,1.50],[.046,.046],[1.35,2.48,.102]),
        7:([.65,.76],[.61,.94],[.039,.036],[4.95,2.48,.102]),
        8:([.90,.44],[.81,1.60],[.052,.033],[1.80,2.95,.105]),
        9:([.77,.64],[.54,.61],[.039,0],[4.50,1.95,.082]),
        10:([.73,.68],[.70,.99],[.046,.039],[6.40,1.78,.069]),
        11:([.92,.39],[.70,1.37],[.046,.039],[.70,1.78,.069])}
    hp,wa,wb,L=family[int(definition)]
    heights=np.array(hp)*H
    periods=2*np.pi*np.exp(np.asarray(wb)*H)/wa
    shapes=[L[0],L[1]*np.exp(-L[2]*H)]
    grid=_grid(w,3)
    parts=tuple(ohspec(grid,[heights[i],periods[i],shapes[i]]) for i in range(2))
    out=_combine(parts,f'Bimodal Ochi-Hubble Hm0={H:g}, definition={int(definition)}')
    return (out,parts) if components else out


def ohspec3(w=None,definition=1,*,components=False):
    """Nine published mixed-sea Ochi-Hubble examples from ohspec3.m."""
    if np.ndim(definition)>0:
        result=[ohspec3(w,d,components=components) for d in definition]
        return result
    if not np.isfinite(definition) or definition!=int(definition) or not 1<=definition<=9:
        raise ValueError('definition must be an integer from 1 through 9')
    data={1:([5.5,3.5],[.07,.11],[3,6.5]),2:([6.5,2],[.07,.15],[3.5,4]),
        3:([5.5,3.5],[.045,.155],[3,6]),4:([2,6.5],[.07,.11],[3,6.5]),
        5:([2,6.5],[.07,.15],[4,3.5]),6:([2,6.5],[.045,.155],[2,7]),
        7:([4.1,5],[.07,.11],[2.1,2.5]),8:([4.1,5],[.07,.15],[2.1,2.5]),
        9:([4.1,5],[.045,.155],[2.1,2.5])}
    H,f,L=data[int(definition)]
    grid=_grid(w,3)
    parts=tuple(ohspec(grid,[H[i],1/f[i],L[i]]) for i in range(2))
    out=_combine(parts,f'Mixed-sea Ochi-Hubble definition={int(definition)}')
    return (out,parts) if components else out


def mccormick(w=None,sdata=(7,11,11/1.228)):
    """McCormick generalized-gamma spectrum, [Hm0,Tp,Tz]."""
    H,T,Tz=_parameters(sdata,[7,11,11/1.228])
    _height_period(H,T)
    if Tz<=0:
        raise ValueError('Tz must be positive')
    result=minimize_scalar(lambda m:((1+1/m)**(1/m)/gamma(1+1/m)-T/Tz)**2,
                           bounds=(1,7),method='bounded')
    M=result.x
    if not result.success:
        raise RuntimeError('McCormick shape optimization failed')
    w=_grid(w,33/T)
    s=np.zeros_like(w)
    mask=w>0
    wp=2*np.pi/T
    s[mask]=(M+1)*(H/4)**2/wp*(wp/w[mask])**(M+1)*np.exp(-(M+1)/M*(wp/w[mask])**M)
    out=createspec(w=w,S=s)
    out.note=f'McCormick Hm0={H:g}, Tp={T:g}, Tz={Tz:g}'
    out.shape=M
    out.shape_residual=float(result.fun)
    return out


def wallop(w=None,sdata=(7,11)):
    """Wallop spectrum, optional explicit high-frequency exponent M."""
    H,T,M=_parameters(sdata,[7,11,0])
    _height_period(H,T)
    if M==0:
        if H==0:
            M=5
        else:
            Lp=2*np.pi/w2k(2*np.pi/T)[0]
            M=abs((np.log(2*np.pi**2)+2*np.log(H/4)-2*np.log(Lp))/np.log(2))
    if M<=1:
        raise ValueError('Wallop exponent M must exceed one for a finite moment m0')
    w=_grid(w,33/T)
    wp=2*np.pi/T
    s=np.zeros_like(w)
    mask=w>0
    if H:
        logB=(M-1)/4*np.log(M)-(M-5)/4*np.log(4)-gammaln((M-1)/4)-np.log(16)
        s[mask]=np.exp(logB+2*np.log(H)-np.log(wp)+M*np.log(wp/w[mask])-M/4*(wp/w[mask])**4)
    out=createspec(w=w,S=s)
    out.note=f'Wallop Hm0={H:g}, Tp={T:g}, M={M:g}'
    return out


def oscspec(sdata=(4/257,4,257),z=.1,w=1,s=1):
    """Harmonic oscillator source spectrum, including its original 1/(2*pi) factor."""
    low,high,n=_parameters(sdata,[4/257,4,257])
    if not 0<z<1 or w<=0 or s<0 or low<0 or high<=low or n<2 or int(n)!=n or not np.isfinite(z+w+s):
        raise ValueError('Invalid oscillator or frequency-grid parameters')
    freq=np.linspace(0,high,int(n))
    density=s*s/((w*w-freq*freq)**2+(2*z*w)**2*freq*freq)/(2*np.pi)
    density[freq<low]=0
    density[0]=0
    out=createspec(w=freq,S=density)
    out.note='Harmonic oscillator spectrum (legacy one-sided convention)'
    return out


def _local_jonswap(grid,H,T,peaked,N,M,A):
    s=np.zeros_like(grid)
    if H<=0:
        return createspec(w=grid,S=s)
    wp=2*np.pi/T
    mask=grid>0
    wn=grid[mask]/wp
    sigma=np.where(wn<=1,.07,.09)
    B=(N-1)/M
    G0=(N/M)**B*M/gamma(B)
    s[mask]=(H/4)**2/wp*G0*wn**(-N)*np.exp(-N/M*wn**(-M))*peaked**np.exp(-(wn-1)**2/(2*sigma*sigma))
    if peaked==1:
        A=1
    if A<0:
        area=simpson(s,x=grid)
        A=(H/4)**2/area if area>0 else 0
    elif A==0:
        if not (3<=N<=50 and 2<=M<=9.5 and 1<=peaked<=20):
            raise ValueError('Empirical Torsethaugen normalization parameters are outside their supported range')
        f1=4.1*(N-2*M**.28+5.3)**(-1.45*M**.1+.96)
        f2=(2.2*M**(-3.3)+.57)*N**(-.58*M**.37+.53)-1.04*M**(-1.9)+.94
        A=(1+f1*np.log(peaked)**f2)/peaked
    return createspec(w=grid,S=s*A)


def torsethaugen(w=None,sdata=(7,11,-1),*,components=False):
    """Torsethaugen wind/swell spectrum from the supplied empirical formulas.

    components=True returns (total,wind,swell), matching the function's actual
    MATLAB declaration rather than its inconsistent example output labels.
    """
    H,T,A=_parameters(sdata,[7,11,-1])
    _height_period(H,T)
    if H==0:
        grid=_grid(w,33/T)
        zero=createspec(w=grid,S=np.zeros_like(grid))
        return (zero,createspec(w=grid,S=zero.S.copy()),createspec(w=grid,S=zero.S.copy())) if components else zero
    if A>0:
        raise ValueError('Torsethaugen normalization accepts A<=0')
    grid=_grid(w,33/T)
    g=9.81
    Tf=6.6*H**(1/3)
    Tl=2*np.sqrt(H)
    El=np.clip((Tf-T)/(Tf-Tl),0,1)
    Eu=np.clip((T-Tf)/(25-Tf),0,1)
    N=.5*np.sqrt(H)+3.2
    if T<Tf:
        Rw=min(.3*np.exp(-(El/.5)**2)+.7,1)
        Hw,Tw=Rw*H,T
        Gw=max(35*(1+3.5*np.exp(-H))*(2*np.pi/g*Rw*H/T**2)**.857,1)
        Hs,Ts,Gs=H*np.sqrt(max(0,1-Rw*Rw)),Tf+2,1
        Mw=Ms=4
    else:
        Rs=min(.4*np.exp(-(Eu/.3)**2)+.6,1)
        Hs,Ts=Rs*H,T
        Gs=max(35*(1+3.5*np.exp(-H))*(2*np.pi/g*H/Tf**2)**.857*(1+6*Eu),1)
        Hw=H*np.sqrt(max(0,1-Rs*Rs))
        Mw=4*(1-.7*np.exp(-H/3))
        Ms=4
        G0=Mw/((N/Mw)**(-(N-1)/Mw)*gamma((N-1)/Mw))
        Tw=(16*.08*(1-np.exp(-H/3))*.4**N/(G0*Hw*Hw))**(-1/(N-1)) if Hw else np.inf
        Gw=1
    wind=_local_jonswap(grid,Hw,Tw,Gw,N,Mw,A)
    swell=_local_jonswap(grid,Hs,Ts,Gs,N,Ms,A)
    wind.note='Torsethaugen wind component'
    swell.note='Torsethaugen swell component'
    out=_combine((wind,swell),f'Torsethaugen Hm0={H:g}, Tp={T:g}')
    return (out,wind,swell) if components else out
