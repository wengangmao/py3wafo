"""Source-based cycle helpers, discretization and rainflow matrices.

Discrete class indices are zero based. Functions do not invoke legacy plotting
as a side effect. Explicit plotting requests are rejected rather than ignored.
"""
import numpy as np


def _cycle_array(cc):
    a = np.asarray(cc, dtype=float)
    if a.size == 0:
        return np.empty((0,2))
    if a.ndim != 2 or a.shape[1] < 2 or not np.all(np.isfinite(a)):
        raise ValueError('Expected finite cycles with at least two columns')
    return a[:,:2]


def _square(F):
    a = np.asarray(F, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.all(np.isfinite(a)):
        raise ValueError('Expected a finite square matrix')
    return a


def cc2amp(cc):
    """Signed amplitude (second endpoint minus first endpoint)/2, as in WAFO."""
    a = _cycle_array(cc)
    return (a[:,1]-a[:,0])/2


def cc2lc(cc, definition=2, plotflag=0, sa=None):
    """Level crossings from min-to-max cycles, with endpoint definitions 1–4."""
    if plotflag:
        raise NotImplementedError('Plot the returned crossing spectrum explicitly')
    if definition not in (1,2,3,4):
        raise ValueError('definition must be 1,2,3,4')
    c = _cycle_array(cc)
    c = c[c[:,0] <= c[:,1]]
    if not len(c):
        return np.empty((0,2))
    u = np.unique(c)
    # Endpoints: 1 excludes both, 2 includes maxima, 3 includes both,
    # 4 includes minima. This is the source cumulative-extrema construction.
    counts = np.zeros(len(u))
    for lo,hi in c:
        left = u>=lo if definition in (3,4) else u>lo
        right = u<=hi if definition in (2,3) else u<hi
        counts += left & right
    # The original definition4 explicitly repeats the previous final count.
    if definition == 4 and len(u)>1:
        counts[-1] = counts[-2]
    return np.column_stack((u,counts))


def tp2lc(tp, definition=2, plotflag=0, sa=None):
    from . import tp2mm
    return cc2lc(tp2mm(tp),definition,plotflag,sa)


def nt2lc(param, NT):
    from . import levels
    a,u = _square(NT),levels(param)
    if len(u)!=len(a):
        raise ValueError('Grid size does not match counting matrix')
    return np.column_stack((u,np.diag(a)))


def cmat2lc(param, F):
    from . import cmat2nt
    return nt2lc(param,cmat2nt(_square(F)))


def cmat2rmcmat(F, param=None, *, full_output=False):
    """Min/max to range/mean matrix. Coalescing bins are accumulated."""
    a = _square(F)
    n = len(a)
    out = np.zeros_like(a)
    for i in range(n):
        for j in range(i,n):
            out[j-i,(i+j)//2] += a[i,j]
    if not full_output:
        return out
    if param is None:
        raise ValueError('param is required for full_output')
    from . import levels
    u = levels(param)
    if len(u)!=n:
        raise ValueError('Grid size does not match matrix')
    step = (u[1]-u[0])/2
    pr = np.asarray([step,step+u[-1]-u[0],n])
    return out,np.asarray(param),pr,np.r_[pr[:2]/2,pr[2]]


def rmcmat2cmat(Frm, paramM=None, paramR=None, *, full_output=False):
    """Range/mean to min/max matrix; bins outside the grid project to boundaries."""
    a = _square(Frm)
    n = len(a)
    out = np.zeros_like(a)
    for r in range(n):
        for m in range(n):
            i,j = np.clip([int(np.ceil(m-r/2)),int(np.ceil(m+r/2))],0,n-1)
            out[i,j] += a[r,m]
    if full_output:
        if paramM is None:
            raise ValueError('paramM is required for full_output')
        return out,np.asarray(paramM)
    return out


def cmatcombine(F1, F2, limits, *, full_output=False):
    """Combine cycle matrices by minimum/maximum/range class restrictions.

    Class restrictions min/max use zero-based indices in Python.
    """
    a,b = _square(F1),_square(F2)
    if a.shape!=b.shape:
        raise ValueError('Cycle matrices must have matching shapes')
    n = len(a)
    opt = dict(range=0,min=n-1,max=0)
    if isinstance(limits,dict):
        unknown = set(limits)-set(opt)
        if unknown:
            raise TypeError('Unknown cycle matrix limits: '+', '.join(sorted(unknown)))
        opt.update({k:v for k,v in limits.items() if v is not None})
    else:
        opt['range'] = limits
    I,J = np.indices(a.shape)
    use = (J-I>=opt['range'])&(I<=opt['min'])&(J>=opt['max'])
    p,q = np.where(use,a,0),np.where(use,0,b)
    return (p+q,opt,p,q) if full_output else p+q


def cmatresamp(F,Method=3,*,seed=None):
    """Multinomial cycle resampling preserving total count.

    The source's methods 1–3 are alternative implementations of the same
    multinomial draw; Python uses NumPy's multinomial engine for all three.
    """
    F = _square(F)
    if Method not in (1,2,3) or np.any(F<0) or np.any(F!=np.floor(F)):
        raise ValueError('Expected integer nonnegative counts and Method 1,2,3')
    n = int(F.sum())
    if not n:
        return np.zeros_like(F,dtype=int)
    return np.random.default_rng(seed).multinomial(n,F.ravel()/n).reshape(F.shape)


def rfcfilter(x, h=0, definition=0):
    """Rainflow hysteresis filtering, port of cycles/rfcfilter.m.

    definition0 retains ranges equal to h in the interior; definition1 uses
    strict threshold transitions. Constant input is returned as one point,
    correcting the zero-threshold duplicate endpoint in the legacy routine.
    """
    a = np.asarray(x,dtype=float)
    vector = a.ndim==1
    rows = a[:,None] if vector else a
    if rows.ndim!=2 or rows.shape[1] not in (1,2) or not np.all(np.isfinite(rows)):
        raise ValueError('Expected finite values or time/value rows')
    if not np.isfinite(h) or h<0 or definition not in (0,1):
        raise ValueError('h must be nonnegative and definition must be 0 or 1')
    if not len(rows):
        return a.copy()
    out = [rows[0].copy()]
    row0,z0 = rows[0].copy(),0
    for row in rows[1:]:
        y0,xi = row0[-1],row[-1]
        fmi,fpi = y0-h,y0+h
        if z0==0:
            test1 = xi<=fmi if definition==0 else xi<fmi
            test2 = xi>=fpi if definition==0 else xi>fpi
        else:
            test1 = ((z0==1 and xi<=fmi) or (z0==-1 and xi<fpi)) if definition==0 else ((z0==1 and xi<fmi) or (z0==-1 and xi<=fpi))
            test2 = ((z0==1 and xi>fmi) or (z0==-1 and xi>=fpi)) if definition==0 else ((z0==1 and xi>=fmi) or (z0==-1 and xi>fpi))
        z1 = -1 if test1 else (1 if test2 else 0)
        if z0==0:
            row1 = row0 if z1==0 else row
        elif z1!=z0:
            row1 = row
        elif z1==-1:
            row1 = row0 if y0<xi else row
        else:
            row1 = row0 if y0>xi else row
        if abs(z0-z1)==2:
            out.append(row0.copy())
        row0,z0 = row1.copy(),z1
    distance = abs(row0[-1]-out[-1][-1])
    if distance>h or (definition==1 and distance>=h and distance>0):
        out.append(row0)
    result = np.asarray(out)
    return result[:,0] if vector else result


def dat2dtp(param, x, h=None, ddef=1, *, full_output=False):
    """Filter and discretize sampled loads; return zero-based discrete classes."""
    from . import levels
    u = levels(param)
    if h is None:
        h = u[1]-u[0]
    y = rfcfilter(x,max(0,h),int(h<=0))
    a = np.asarray(y)
    vector = a.ndim==1
    rows = a[:,None].copy() if vector else a.copy()
    values = (rows[:,-1]-u[0])/(u[1]-u[0])
    n = len(u)
    if ddef==0:
        d = np.clip(np.floor(values+.5),0,n-1)
    elif ddef in (-1,1):
        first_min = len(values)<2 or values[0]<values[1]
        minima = (np.arange(len(values))%2==0)==first_min
        d = values.copy()
        if ddef==1:
            d[minima] = np.clip(np.floor(d[minima]),0,n-2)
            d[~minima] = np.clip(np.ceil(d[~minima]),1,n-1)
        else:
            d[minima] = np.clip(np.ceil(d[minima]),1,n-1)
            d[~minima] = np.clip(np.floor(d[~minima]),0,n-2)
    else:
        raise ValueError('ddef must be -1,0,1')
    rows[:,-1] = d
    filtered = rfcfilter(rows,0,1)
    if vector:
        dtp = filtered[:,0].astype(int)
        tp = u[dtp]
    else:
        dtp = filtered
        tp = filtered.copy()
        tp[:,-1] = u[filtered[:,-1].astype(int)]
    return (dtp,u,tp) if full_output else dtp


def _discrete_values(x,n):
    a = np.asarray(x,dtype=float)
    v = a if a.ndim==1 else a[:,-1]
    if n<2 or int(n)!=n or not np.all(np.isfinite(v)) or np.any(v!=np.floor(v)) or np.any((v<0)|(v>=n)):
        raise ValueError('Discrete values must be integers in [0,n)')
    return a


def dtp2arfm4p(x,n,res0=None):
    from . import tp2arfc4p,dcc2cmat
    a = _discrete_values(x,n)
    closed,res = tp2arfc4p(a,res0)
    return dcc2cmat(closed,int(n)),res


def dtp2arfm(x,n,definition='up',RFM0=None,res0=None,*,full_output=False):
    from . import res2arfc,dcc2cmat
    closed,res = dtp2arfm4p(x,n,res0)
    if RFM0 is not None:
        prior = _square(RFM0)
        if prior.shape!=closed.shape:
            raise ValueError('Prior cycle matrix shape mismatch')
        closed = closed+prior
    total = closed+dcc2cmat(res2arfc(res,definition),int(n))
    return (total,closed,res) if full_output else total


def dtp2rfm(x,n,definition='up',RFM0=None,res0=None,*,full_output=False):
    total,closed,res = dtp2arfm(x,n,definition,None,res0,full_output=True)
    total,closed = np.triu(total+total.T),np.triu(closed+closed.T)
    if RFM0 is not None:
        prior = _square(RFM0)
        if prior.shape!=closed.shape:
            raise ValueError('Prior cycle matrix shape mismatch')
        total,closed = total+prior,closed+prior
    return (total,closed,res) if full_output else total


def dat2rfm(x,h=0,n=64,*,full_output=False):
    """Rainflow matrix from data, with CS treatment of residuals."""
    from . import levels
    a = np.asarray(x)
    values = a if a.ndim==1 else a[:,-1]
    tp = rfcfilter(values,max(0,h),int(h<=0))
    if np.size(n)==3:
        param = np.asarray(n,dtype=float)
    else:
        lo,hi = np.min(tp),np.max(tp)
        if lo==hi:
            raise ValueError('A nonconstant load is required for automatic discretization')
        param = [lo,hi,int(n)]
    dtp,u,tp = dat2dtp(param,tp,h,0,full_output=True)
    F = dtp2rfm(dtp,len(u),'CS')
    return (F,np.asarray(param),dtp,tp) if full_output else F
