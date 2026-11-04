"""Finite-chain estimation, matrix distances and Markov-load rainflow analysis.

Source algorithms follow the supplied WAFO MATLAB module; class/state indices
are zero based in Python. Linear systems replace explicit matrix inversion.
"""
import numpy as np
from scipy.special import expit, logit, gammaln, xlogy


def _matrix(F):
    F = np.asarray(F,dtype=float)
    if F.ndim!=2 or F.shape[0]!=F.shape[1] or not np.all(np.isfinite(F)):
        raise ValueError('Expected a finite square matrix')
    return F


def mat2tmat(F, definition=0, K=None):
    """Normalize nonnegative matrix rows; structurally empty rows stay zero."""
    F = _matrix(F)
    if definition not in (-1,0,1):
        raise ValueError('definition must be -1,0,1')
    K = definition if K is None else K
    if not np.isfinite(K) or int(K)!=K:
        raise ValueError('K must be an integer diagonal offset')
    F = np.maximum(F,0)
    if definition==1:
        F = np.triu(F,int(K))
    elif definition==-1:
        F = np.tril(F,int(K))
    total = F.sum(axis=1,keepdims=True)
    return np.divide(F,total,out=np.zeros_like(F),where=total>0)


def estmc(z,r,*,full_output=False):
    """Estimate transition frequencies/probabilities from zero-based states."""
    z = np.asarray(z)
    if z.ndim!=1 or not np.all(np.isfinite(z)) or int(r)!=r or r<1 or np.any(z!=np.floor(z)) or np.any((z<0)|(z>=r)):
        raise ValueError('Expected integer state vector and positive state count')
    counts = np.zeros((int(r),int(r)),dtype=int)
    np.add.at(counts,(z[:-1].astype(int),z[1:].astype(int)),1)
    totals = counts.sum(axis=1)
    P = np.divide(counts,totals[:,None],out=np.full_like(counts,1/r,dtype=float),where=totals[:,None]>0)
    return (P,counts,totals) if full_output else P


def _probabilities(Fobs,F):
    obs,expected = _matrix(Fobs),_matrix(F)
    if obs.shape!=expected.shape or np.any(obs<0) or np.any(expected<0) or obs.sum()<=0 or expected.sum()<=0:
        raise ValueError('Nonnegative matching matrices with positive mass are required')
    return obs,expected/expected.sum()


def chi2cmat(Fobs,F):
    obs,p = _probabilities(Fobs,F)
    expected = obs.sum()*p
    if np.any((expected==0)&(obs>0)):
        return np.inf
    return np.sum(np.divide((obs-expected)**2,expected,out=np.zeros_like(expected),where=expected>0))


def hdcmat(Fobs,F):
    obs,p = _probabilities(Fobs,F)
    return np.arccos(np.clip(np.sum(np.sqrt(obs/obs.sum()*p)),0,1))


def klcmat(Fobs,F):
    """Reverse KL divergence KL(expected || observed proportions), as in WAFO."""
    obs,p = _probabilities(Fobs,F)
    q = obs/obs.sum()
    if np.any((p>0)&(q==0)):
        return np.inf
    keep = p>0
    return np.sum(p[keep]*np.log(p[keep]/q[keep]))


def loglcmat(Fobs,F,definition=0):
    """Multinomial log likelihood; definition1 includes combinatorial constant."""
    obs,p = _probabilities(Fobs,F)
    if definition not in (0,1):
        raise ValueError('definition must be 0 or 1')
    if np.any((p==0)&(obs>0)):
        return -np.inf
    value = np.sum(xlogy(obs,p))
    if definition:
        value += gammaln(obs.sum()+1)-gammaln(obs+1).sum()
    return value


def tr_p2x(P,trans=0,*,return_size=False):
    """Flatten off-diagonal entries row by row, optionally applying log odds."""
    P = _matrix(P)
    if trans not in (0,1):
        raise ValueError('trans must be 0 or 1')
    x = P[~np.eye(len(P),dtype=bool)]
    if trans:
        if np.any((x<0)|(x>1)):
            raise ValueError('Log-odds transformation requires values in [0,1]')
        x = logit(x)
    return (x,len(P)) if return_size else x


def tr_x2p(X,trans=0):
    x = np.asarray(X,dtype=float).reshape(-1)
    r = (1+np.sqrt(1+4*len(x)))/2
    if int(r)!=r or trans not in (0,1):
        raise ValueError('Vector length must equal r*(r-1), and trans must be 0 or 1')
    if trans:
        x = expit(x)
    elif not np.all(np.isfinite(x)):
        raise ValueError('Untransformed entries must be finite')
    r = int(r)
    P = np.zeros((r,r))
    P[~np.eye(r,dtype=bool)] = x
    np.fill_diagonal(P,1-P.sum(axis=1))
    return P


def tr_m2x(M):
    x0,s,lam = np.asarray(M['x0']),float(M['s']),float(M['lam'])
    if x0.shape!=(2,) or not np.all(np.isfinite(x0)) or s<=0 or lam<=0:
        raise ValueError('Model requires finite x0[2] and positive s,lam')
    return np.r_[x0,np.log(s),np.log(lam)]


def tr_x2m(X,known=None):
    x = np.asarray(X,dtype=float)
    if x.shape!=(4,) or not np.all(np.isfinite(x)):
        raise ValueError('Expected a finite length-four model vector')
    return dict(x0=x[:2].copy(),s=np.exp(x[2]),lam=np.exp(x[3]))


def mktestmat(param=(-1,1,32),x0=None,s=None,lam=1,NOsubzero=0,*,both=False):
    from ..cycles import levels
    u = levels(param)
    x0 = np.full(2,(u[0]+u[-1])/2) if x0 is None else np.asarray(x0)
    s = (u[-1]-u[0])/4 if s is None else s
    if s<=0 or lam<=0 or x0.shape!=(2,) or not np.all(np.isfinite(x0)):
        raise ValueError('Invalid elliptical kernel parameters')
    S = .5*s*s*np.array([[lam*lam+1,lam*lam-1],[lam*lam-1,lam*lam+1]])
    coordinates = np.stack(np.meshgrid(u,u,indexing='ij'),axis=-1)-x0
    F = np.exp(-.5*np.einsum('...i,ij,...j->...',coordinates,np.linalg.inv(S),coordinates))
    if not np.isneginf(NOsubzero):
        if not np.isfinite(NOsubzero) or int(NOsubzero)!=NOsubzero:
            raise ValueError('NOsubzero must be an integer or -inf')
        F = np.triu(F,int(NOsubzero)+1)
    return (F,F.T.copy()) if both else F


def _pair_matrices(F):
    if not isinstance(F,(tuple,list)) or len(F)!=2:
        raise ValueError('Expected (min-to-max matrix,max-to-min matrix or None)')
    up = mat2tmat(F[0],1)
    down = mat2tmat(np.asarray(F[0]).T if F[1] is None or np.size(F[1])==0 else F[1],-1)
    if up.shape!=down.shape:
        raise ValueError('Min-to-max/max-to-min matrices must match')
    return up,down


def _stationary_active(P):
    from . import mc2stat
    active = P.sum(axis=1)>0
    if not np.any(active):
        raise ValueError('No recurrent active states')
    sub = P[np.ix_(active,active)]
    if not np.allclose(sub.sum(axis=1),1,atol=1e-10):
        raise ValueError('Transitions leave the active state set')
    result = np.zeros(len(P))
    result[active] = mc2stat(sub)
    return result


def mctp2stat(Q,*,full_output=False):
    up,down = _pair_matrices(Q)
    minimum = _stationary_active(up@down)
    maximum = minimum@up
    if not np.allclose(maximum@down,minimum,atol=1e-9):
        raise ValueError('Alternating transitions have no consistent stationary law')
    return (minimum,maximum,(up,down)) if full_output else (minimum,maximum)


def mctp2reverse(F,*,full_output=False):
    minimum,maximum,(up,down) = mctp2stat(F,full_output=True)
    ru = np.divide(down.T*maximum[None,:],minimum[:,None],out=np.zeros_like(up),where=minimum[:,None]>0)
    rd = np.divide(up.T*minimum[None,:],maximum[:,None],out=np.zeros_like(down),where=maximum[:,None]>0)
    joint = minimum[:,None]*up+maximum[:,None]*down
    reversed_joint = minimum[:,None]*ru+maximum[:,None]*rd
    return ((ru,rd),joint,reversed_joint) if full_output else (ru,rd)


def smctp2rfm(P,F,c_m=1,*,full_output=False):
    """Switching-Markov turning-point rainflow intensity via first-passage systems.

    F is a sequence of (min-max,max-min-or-None) pairs, one per regime.
    This directly implements smctp2rfm.m's counting-distribution calculation.
    """
    from . import _transition
    from ..cycles import nt2cmat,cmat2nt
    P = _transition(P)
    r = len(P)
    if len(F)!=r or not np.isfinite(c_m) or c_m<0:
        raise ValueError('One matrix pair per regime and nonnegative intensity required')
    pairs = [_pair_matrices(pair) for pair in F]
    n = len(pairs[0][0])
    if any(pair[0].shape!=(n,n) for pair in pairs):
        raise ValueError('All regime matrices must have the same size')
    Q,Qh = np.zeros((n*r,n*r)),np.zeros((n*r,n*r))
    rows = np.arange(n)*r
    for z,(up,down) in enumerate(pairs):
        Q[rows+z] = np.kron(up,P)[rows+z]
        Qh[rows+z] = np.kron(down,P)[rows+z]
    ro = _stationary_active(Q@Qh)
    cumulative = np.cumsum(Q[:,::-1],axis=1)[:,::-1]
    mu = np.zeros((n,n))
    for i in range(2,n+1):
        stop = r*(i-1)
        for j in range(i-1,min(i,n-1)+1):
            mu[i-1,j-1] = ro[:stop]@cumulative[:stop,r*j]
        for j in range(i+1,n):
            A = Q[stop:r*(j-1),r*i:r*j]
            Ah = Qh[r*i:r*j,stop:r*(j-1)]
            C = Q[:stop,r*i:r*j]
            d,e = cumulative[:stop,r*j],cumulative[stop:r*(j-1),r*j]
            system = np.eye(len(A))-A@Ah
            if np.all(e==0):
                exit_probability = np.zeros_like(e)
            else:
                try:
                    exit_probability = np.linalg.solve(system,e)
                except np.linalg.LinAlgError:
                    exit_probability = np.linalg.lstsq(system,e,rcond=None)[0]
                    if not np.allclose(system@exit_probability,e,atol=1e-9):
                        raise ValueError('First-passage system has no consistent solution')
            mu[i-1,j-1] = ro[:stop]@(d+C@Ah@exit_probability)
    # The source differentiates only i<j; lower-triangular entries of the
    # extended counting distribution are not cycle frequencies.
    result = np.triu(nt2cmat(mu*c_m),1)
    if np.min(result)<-1e-9*max(1,c_m):
        raise FloatingPointError('Rainflow intensity calculation produced material negative counts')
    result = np.maximum(result,0)
    return (result,cmat2nt(result)) if full_output else result


def mctp2rfm(F,c_m=1,*,full_output=False):
    return smctp2rfm([[1]], [F], c_m, full_output=full_output)


def smctp2joint(P,F,*,full_output=False):
    """Joint level/regime alternating transition matrices for switching loads."""
    from . import _transition
    P = _transition(P)
    r = len(P)
    if len(F)!=r:
        raise ValueError('Expected one min/max matrix pair per regime')
    pairs = [_pair_matrices(pair) for pair in F]
    n = len(pairs[0][0])
    if any(pair[0].shape!=(n,n) for pair in pairs):
        raise ValueError('All regime matrices must have the same size')
    up,down = np.zeros((n*r,n*r)),np.zeros((n*r,n*r))
    rows = np.arange(n)*r
    for z,(a,b) in enumerate(pairs):
        up[rows+z] = np.kron(a,P)[rows+z]
        down[rows+z] = np.kron(b,P)[rows+z]
    return ((up,down),pairs) if full_output else (up,down)


def smctp2stat(P,F,*,full_output=False):
    """Regime, per-regime load and joint stationary laws for a switching MCTP."""
    from . import mc2stat
    (up,down),pairs = smctp2joint(P,F,full_output=True)
    regime = mc2stat(P)
    sub = [mctp2stat(pair) for pair in pairs]
    minimum,maximum = [p[0] for p in sub],[p[1] for p in sub]
    joint_min = _stationary_active(up@down)
    joint_max = joint_min@up
    return ((regime,minimum,maximum,joint_min,joint_max,pairs) if full_output
            else (regime,minimum,maximum))


def f_funm(param,M):
    """Elliptical min/max test kernel from model fields x0,s,lam."""
    return mktestmat(param,M['x0'],M['s'],M['lam'],both=True)
