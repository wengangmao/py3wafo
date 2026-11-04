"""Wave events, periods and time-series quality helpers."""
import numpy as np
from scipy.interpolate import PchipInterpolator


def _series(x):
    from . import _values
    a = np.asarray(x,dtype=float)
    y = _values(a)
    t = np.arange(len(y),dtype=float) if a.ndim==1 or a.shape[1]==1 else a[:,0]
    if not np.all(np.isfinite(y)) or len(t)<2 or np.any(np.diff(t)<=0):
        raise ValueError('Expected finite values and increasing sampling times')
    return t,y


def dat2tc(x,v=None,wdef='none',*,full_output=False):
    """Crest/trough samples between consecutive crossings.

    Returns time/value rows; full_output gives (rows,extremum_indices,crossings).
    Unlike the old implementation, one complete excursion is sufficient.
    """
    from . import dat2crossind
    t,y = _series(x)
    v = float(y.mean()) if v is None else v
    cross = dat2crossind(y,v,wdef)
    indices = []
    for left,right in zip(cross[:-1],cross[1:]):
        segment = y[left+1:right+1]
        peak = np.argmax(segment) if y[left+1]>v else np.argmin(segment)
        indices.append(left+1+peak)
    indices = np.asarray(indices,dtype=int)
    rows = np.column_stack((t[indices],y[indices]))
    return (rows,indices,cross) if full_output else rows


def dat2wa(x,h=None,pdef='d2d',wdef='none',rate=1,*,full_output=False):
    """Periods between crossing, crest/trough, or rainflow turning-point events.

    Supported event labels d,u,t,c,m,M retain the original definitions.
    Full output: (periods,all_event_indices,paired_endpoint_indices).
    rate>1 performs shape-preserving interpolation before extraction.
    """
    from . import dat2crossind,dat2tp,ecross
    t,y = _series(x)
    if not np.isfinite(rate) or rate<1:
        raise ValueError('Interpolation rate must be at least one')
    if rate>1:
        times = np.linspace(t[0],t[-1],int(np.ceil((len(t)-1)*rate))+1)
        y,t = PchipInterpolator(t,y)(times),times
    if pdef!='all' and (len(pdef)!=3 or pdef[1]!='2' or pdef[0] not in 'dutcmM' or pdef[2] not in 'dutcmM'):
        raise ValueError('Unknown wave-period definition')
    events = []
    if pdef!='all' and set((pdef[0],pdef[2])).issubset({'m','M'}):
        threshold = 0 if h is None else h
        _,ind = dat2tp(np.column_stack((t,y)),threshold,wdef,return_indices=True)
        for j,index in enumerate(ind):
            if j==0:
                label = 'm' if len(ind)>1 and y[ind[1]]>y[index] else 'M'
            else:
                label = 'M' if y[index]>y[ind[j-1]] else 'm'
            events.append((t[index],index,label))
    else:
        level = float(y.mean()) if h is None else h
        cross = dat2crossind(y,level,wdef)
        times = ecross(t,y,cross,level)
        if pdef=='all' or any(k in pdef for k in ('u','d')):
            events.extend((time,index,'u' if y[index+1]>level else 'd') for time,index in zip(times,cross))
        if pdef=='all' or any(k in pdef for k in ('t','c')):
            rows,ind,_ = dat2tc(np.column_stack((t,y)),level,wdef,full_output=True)
            events.extend((row[0],index,'c' if row[1]>level else 't') for row,index in zip(rows,ind))
    events.sort(key=lambda event:(event[0],event[1]))
    periods,pairs = [],[]
    if pdef=='all':
        selected = list(zip(events[:-1],events[1:]))
    else:
        first,last = pdef[0],pdef[2]
        selected = []
        for i,start in enumerate(events):
            if start[2]!=first:
                continue
            following = next((end for end in events[i+1:] if end[2]==last),None)
            if following is not None:
                selected.append((start,following))
    for start,end in selected:
        periods.append(end[0]-start[0]); pairs.append((start[1],end[1]))
    output = np.asarray(periods)
    return (output,np.asarray([event[1] for event in events],dtype=int),np.asarray(pairs,dtype=int).reshape(-1,2)) if full_output else output


def dat2midind(x,p=.5,ind=None,h=0):
    """Indices preceding interpolated levels between consecutive turning points.

    Selects crossings nearest the intervening extremum; returned indices are
    absolute zero-based input indices (correcting the old relative-index bug).
    """
    from . import dat2tp,findcross
    _,y = _series(x)
    if not 0<p<1:
        raise ValueError('Require 0<p<1')
    if ind is None or isinstance(ind,str):
        _,ind = dat2tp(x,h,'mw' if ind is None else ind,return_indices=True)
    ind = np.asarray(ind,dtype=int)
    if ind.ndim!=1 or np.any(np.diff(ind)<=0) or np.any(ind<0) or np.any(ind>=len(y)):
        raise ValueError('Turning-point indices must increase within the data')
    result = []
    for i in range(0,len(ind)-2,2):
        a,b,c = ind[i:i+3]
        left = findcross(y[a:b+1],(1-p)*y[a]+p*y[b])
        right = findcross(y[b:c+1],p*y[b]+(1-p)*y[c])
        if not len(left) or not len(right):
            raise ValueError('No midpoint crossing between supplied turning points')
        result.extend((a+left[-1],b+right[0]))
    return np.asarray(result,dtype=int)


def dat2lc(x,h=0,definition=2):
    from . import dat2tp
    from ..cycles import cc2lc,tp2mm
    return cc2lc(tp2mm(dat2tp(x,max(h,0))),definition)


def findoutliers(x,zcrit=0,dcrit=None,ddcrit=None,plotflag=0,*,verbose=False):
    """Flag missing samples and large first/second differences.

    Modern interface returns (bad_indices,good_indices); zcrit marks repeated
    adjacent values within a tolerance. Defaults use 1.5 standard deviations.
    """
    from . import _values
    y = _values(x)
    finite = np.isfinite(y)
    sigma = np.std(y[finite],ddof=1) if finite.sum()>1 else 0
    dcrit = 1.5*sigma if dcrit is None else dcrit
    ddcrit = 1.5*sigma if ddcrit is None else ddcrit
    if plotflag:
        raise ValueError('Plot the returned bad/good indices explicitly')
    if np.any(np.isnan([zcrit,dcrit,ddcrit])) or min(dcrit,ddcrit)<0:
        raise ValueError('Jump thresholds must be nonnegative')
    bad = ~finite
    filled = np.where(finite,y,0)
    difference = np.diff(filled)
    bad[np.flatnonzero(difference>dcrit)+1] = True
    bad[np.flatnonzero(difference<-dcrit)] = True
    if zcrit>=0:
        equal = abs(difference)<=zcrit
        for index in np.flatnonzero(equal):
            bad[max(0,index-1):min(len(y),index+3)] = True
    curvature = np.flatnonzero(abs(np.diff(filled,2))>ddcrit)+1
    bad[curvature] = True
    if verbose:
        print(f'{int(bad.sum())} outliers in {len(y)} samples')
    return np.flatnonzero(bad),np.flatnonzero(~bad)


def hs2sign(f,ratio=None):
    """Mean of the largest histogram fraction: [fraction,mean,threshold].

    Explicit ratios use fractional bin weights, correcting the old branch's
    unrelated cumulative-area formula. Without a ratio, return every threshold.
    """
    f = np.asarray(f,dtype=float)
    if f.ndim!=2 or f.shape[1]!=2 or not len(f) or not np.all(np.isfinite(f)) or np.any(f[:,1]<0) or f[:,1].sum()<=0:
        raise ValueError('Expected a finite value/count histogram with positive total weight')
    f = f[np.argsort(f[:,0])]
    weights,values = f[:,1],f[:,0]
    tail_weight = np.cumsum(weights[::-1])[::-1]
    means = np.divide(np.cumsum((weights*values)[::-1])[::-1],tail_weight,
        out=np.zeros_like(tail_weight),where=tail_weight>0)
    if ratio is None:
        positive = tail_weight>0
        return np.column_stack((tail_weight[positive]/weights.sum(),means[positive],values[positive]))
    if not 0<ratio<=1:
        raise ValueError('Require 0<ratio<=1')
    needed = ratio*weights.sum()
    previous = np.r_[0,np.cumsum(weights[::-1])[:-1]]
    selected = np.clip(needed-previous,0,weights[::-1])
    threshold = values[::-1][np.flatnonzero(selected)[-1]]
    return np.asarray([ratio,np.sum(selected*values[::-1])/needed,threshold])


def vc2sign(H,ratio=None):
    """Significant values of a sample via equally weighted histogram heights.

    Returns columns [tail fraction,tail mean,threshold]. Explicit ratio uses
    fractional weighting at its cutoff and supports ratio=1 without index zero.
    """
    H = np.asarray(H,dtype=float).ravel()
    return hs2sign(np.column_stack((H,np.ones(len(H)))),ratio)
