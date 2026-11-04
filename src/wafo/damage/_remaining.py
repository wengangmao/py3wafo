"""Damage integration using explicitly named counting-distribution layouts."""
import numpy as np


def damint(param,beta,M,*,convention='wat',amplitude=False):
    """Exact discrete damage from a counting distribution.

    convention='pj' means cmat2nt's orientation; 'wat' means its source
    flipud(NT.T) layout. Integrates full ranges by default, as damint.m's model
    defines; amplitude=True uses half ranges. This replaces the old mixture of
    finite differences and area quadrature with exact discrete summation by
    parts, and therefore is not a promise of the old quadrature's numeric value.
    """
    from ..cycles import levels,nt2cmat
    u=levels(param); a=np.asarray(M,dtype=float); beta=np.atleast_1d(np.asarray(beta,dtype=float))
    if a.shape!=(len(u),len(u)) or not np.all(np.isfinite(a)) or np.any(a<0) or not np.all(np.isfinite(beta)) or np.any(beta<=0):
        raise ValueError('Need nonnegative finite counting matrix and positive powers')
    if convention=='wat':
        a=np.flipud(a).T
    elif convention!='pj':
        raise ValueError("convention must be 'wat' or 'pj'")
    F=np.triu(nt2cmat(a),1)
    if F.min()<-1e-10*max(1,a.max()):
        raise ValueError('Counting distribution does not correspond to nonnegative cycles')
    ranges=np.maximum(0,u[None,:]-u[:,None])/(2 if amplitude else 1)
    return np.asarray([np.sum(np.maximum(F,0)*ranges**b) for b in beta])


def snplot(S,N,nr=0,*,ax=None,full_output=False):
    """OLS log-life/stress parameters (e,beta,residual variance), optional plot.

    Fits log(N)=-log(e)-beta*log(S). Modes1–4 and11–14 follow the source
    stress/life and log/log orientations; nr=0 fits without creating a plot.
    """
    S,N=np.asarray(S,dtype=float).ravel(),np.asarray(N,dtype=float).ravel()
    if len(S)<3 or S.shape!=N.shape or not np.all(np.isfinite([S,N])) or np.any(S<=0) or np.any(N<=0) or np.ptp(S)==0:
        raise ValueError('Need >=3 positive finite stress/life pairs and varying stress')
    design=np.column_stack((np.ones(len(S)),np.log(S)))
    fit=np.linalg.lstsq(design,np.log(N),rcond=None)[0]
    e,beta=float(np.exp(-fit[0])),float(-fit[1])
    residual=np.log(N)-design@fit
    variance=float(residual@residual/(len(S)-2))
    if nr:
        import matplotlib.pyplot as plt
        if nr not in (1,2,3,4,11,12,13,14):raise ValueError('Plot mode must be 0,1–4,11–14')
        ax=plt.subplots()[1] if ax is None else ax
        swap=nr>10; mode=nr%10
        x,y=(N,S) if swap else (S,N)
        ax.plot(np.log(x) if mode>2 else x,np.log(y) if mode>2 else y,'.')
        if mode in (2,4):
            stress=np.geomspace(S.min(),S.max(),100)
            life=np.exp(variance/2)/e*stress**(-beta)
            xp,yp=(life,stress) if swap else (stress,life)
            ax.plot(np.log(xp) if mode>2 else xp,np.log(yp) if mode>2 else yp)
        ax.set(xlabel='log N' if swap and mode>2 else 'N' if swap else 'log S' if mode>2 else 'S',
               ylabel='log S' if swap and mode>2 else 'S' if swap else 'log N' if mode>2 else 'N')
    return (e,beta,variance,ax) if full_output else (e,beta,variance)
