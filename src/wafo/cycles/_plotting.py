"""Headless-compatible cycle/load-spectrum plotting; never calls show()."""
import numpy as np
from ._extended import _cycle_array,_square


def ccplot(cc,ps=12,*,ax=None):
    """Plot cycle endpoints; a list of cycle matrices creates separate axes."""
    import matplotlib.pyplot as plt
    if isinstance(cc,(list,tuple)) and cc and all(np.asarray(a).ndim==2 for a in cc):
        if ax is not None:
            raise ValueError('Supply no single axis for multiple cycle matrices')
        _,axes=plt.subplots(1,len(cc),squeeze=False)
        return [ccplot(c,ps,ax=a) for c,a in zip(cc,axes.ravel())]
    c=_cycle_array(cc)
    if not np.isfinite(ps) or ps<=0:
        raise ValueError('Marker size must be positive')
    ax=plt.subplots()[1] if ax is None else ax
    ax.plot(c[:,0],c[:,1],'.',markersize=ps)
    ax.set(xlabel='Minimum / first endpoint',ylabel='Maximum / second endpoint')
    ax.set_aspect('equal',adjustable='box')
    return ax


def cmatplot(*args,ax=None):
    """Cycle-matrix mesh/surface/color/contour modes 1–4 and 11–14.

    Returns axes; old interactive TechMath mode5 is explicitly unsupported.
    """
    import matplotlib.pyplot as plt
    if len(args) in (1,2):
        F=_square(args[0]); ux=uy=np.arange(len(F)); method=1 if len(args)==1 else args[1]; contours=None
    elif len(args) in (3,4,5):
        ux,uy=np.asarray(args[0]),np.asarray(args[1]); F=_square(args[2]); method=1 if len(args)==3 else args[3]; contours=args[4] if len(args)==5 else None
    else:
        raise TypeError('Use cmatplot(F,method) or cmatplot(ux,uy,F,method,levels)')
    if method not in (1,2,3,4,11,12,13,14):
        raise NotImplementedError('Implemented plot modes are 1–4 and 11–14')
    if len(ux)!=len(F) or len(uy)!=len(F):
        raise ValueError('Matrix and axes must match')
    mode=int(method)%10
    ax=plt.figure().add_subplot(111,projection='3d' if mode<3 else None) if ax is None else ax
    X,Y=np.meshgrid(ux,uy,indexing='xy'); values=F.T
    if mode==1:ax.plot_wireframe(X,Y,values)
    elif mode==2:ax.plot_surface(X,Y,values)
    elif mode==3:ax.pcolormesh(X,Y,values,shading='auto')
    else:ax.contour(X,Y,values,levels=contours if contours is not None else 7)
    ax.set(xlabel='From' if method>10 else 'Minimum',ylabel='To' if method>10 else 'Maximum')
    return ax


def cocc(param,cc,matrix,clevels=None,fs=12,*,ax=None):
    from . import levels
    u=levels(param)
    ax=cmatplot(u,u,matrix,4,*(() if clevels is None else (clevels,)),ax=ax)
    return ccplot(cc,fs,ax=ax)


def lsplot(LS,cum=1,norm=0,beta=0,plottype=None,*,ax=None):
    """Plot amplitude frequencies or cumulative load/damage spectra."""
    import matplotlib.pyplot as plt
    if cum not in (0,1) or norm not in (0,1) or not np.isfinite(beta) or beta<0:
        raise ValueError('cum/norm must be 0/1 and beta nonnegative')
    spectra=list(LS) if isinstance(LS,(list,tuple)) and LS and all(np.asarray(a).ndim==2 for a in LS) else [LS]
    mode=('stairs' if cum else 'bar') if plottype is None else plottype
    allowed=('stairs','plot') if cum else ('bar','stem','stairs','plot')
    if mode not in allowed:
        raise ValueError('Unknown load-spectrum plot type')
    ax=plt.subplots()[1] if ax is None else ax
    first_damage=None
    for spectrum in spectra:
        data=np.asarray(spectrum,dtype=float)
        if data.ndim!=2 or data.shape[1]!=2 or not len(data) or not np.all(np.isfinite(data)) or np.any(data[:,1]<0):
            raise ValueError('Expected finite amplitude/count rows')
        data=data[np.argsort(data[:,0])]
        weight=data[:,1]*data[:,0]**beta
        if beta:
            first_damage=weight.sum() if first_damage is None else first_damage
            if first_damage<=0:raise ValueError('Damage mass must be positive')
            weight=weight/first_damage
        if norm:
            if weight.sum()<=0:raise ValueError('Cannot normalize empty spectrum')
            weight=weight/weight.sum()
        if cum:
            counts=np.cumsum(weight[::-1])[::-1]
            keep=counts>0
            (ax.step if mode=='stairs' else ax.plot)(counts[keep],data[keep,0])
        elif mode=='bar':ax.bar(data[:,0],weight)
        elif mode=='stem':ax.stem(data[:,0],weight)
        else:(ax.step if mode=='stairs' else ax.plot)(data[:,0],weight)
    ax.set(xlabel='Cumulative frequency' if cum else 'Amplitude',ylabel='Amplitude' if cum else 'Frequency')
    if cum:ax.set_xscale('log')
    return ax
