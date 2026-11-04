"""Normalized kernel formulas and numerical helpers from the WAFO KDE module."""
import itertools
import numpy as np
from scipy.special import beta,eval_hermitenorm,gamma


def kernelstats(kernel='gaussian'):
    """Return (second moment, squared-kernel integral, squared K'' integral)."""
    name = str(kernel).lower()[:4]
    values = {'biwe':(1/7,5/7,45/2),'epan':(1/5,3/5,np.inf),
        'epa1':(1/5,3/5,np.inf),'gaus':(1,1/(2*np.sqrt(np.pi)),3/(8*np.sqrt(np.pi))),
        'norm':(1,1/(2*np.sqrt(np.pi)),3/(8*np.sqrt(np.pi))),
        'lapl':(2,1/4,np.inf),'logi':(np.pi**2/3,1/6,1/42),
        'rect':(1/3,1/2,np.inf),'unif':(1/3,1/2,np.inf),
        'tria':(1/6,2/3,np.inf),'triw':(1/9,350/429,np.inf)}
    if name not in values:
        raise ValueError(f"Unknown kernel {kernel!r}")
    return values[name]


def mkernel(x,kernel='gaussian'):
    """Kernel density; rows are points, last axis is dimension.

    A 1D input is a collection of scalar points. epan/biwe/triw are radial
    kernels; epa1/biw1/tri1 are products, matching the original distinctions.
    """
    a = np.asarray(x,dtype=float)
    if a.ndim==0:
        a = a.reshape(1)
    if a.ndim==1:
        a = a[...,None]
    d = a.shape[-1]
    name = str(kernel).lower()[:4]
    radial = {'sphe':0,'epan':1,'biwe':2,'triw':3,'four':4}
    product = {'epa1':1,'biw1':2,'tri1':3,'fou1':4}
    if name in radial:
        r = radial[name]
        radius = np.sum(a*a,axis=-1)
        constant = gamma(d/2+r+1)/(np.pi**(d/2)*gamma(r+1))
        return np.where(radius<=1,constant*np.maximum(0,1-radius)**r,0)
    if name in product:
        r = product[name]
        per_axis = np.where(abs(a)<=1,np.maximum(0,1-a*a)**r/beta(.5,r+1),0)
    elif name in ('gaus','norm'):
        per_axis = np.exp(-a*a/2)/np.sqrt(2*np.pi)
    elif name in ('rect','unif'):
        per_axis = .5*(abs(a)<=1)
    elif name=='tria':
        per_axis = np.maximum(0,1-abs(a))
    elif name=='lapl':
        per_axis = .5*np.exp(-abs(a))
    elif name=='logi':
        z = np.exp(-abs(a))
        per_axis = z/(1+z)**2
    else:
        raise ValueError(f"Unknown kernel {kernel!r}")
    return np.prod(per_axis,axis=-1)


def deriv(t,k='gaussian'):
    """Gaussian derivatives of orders 4,6,8,10 as a tuple."""
    if k.lower()[:4] not in ('gaus','norm'):
        raise ValueError("High-order derivatives are defined here for Gaussian kernels")
    t = np.asarray(t)
    return tuple(eval_hermitenorm(n,t)*np.exp(-t*t/2)/np.sqrt(2*np.pi) for n in (4,6,8,10))


def deriv2(x,y,d):
    """Mixed Gaussian derivatives: d='42' means fourth in x, second in y."""
    if not isinstance(d,str) or len(d)!=2 or not d.isdigit():
        raise ValueError("d must be two derivative-order digits")
    dx,dy = map(int,d)
    x,y = np.broadcast_arrays(x,y)
    return (-1)**(dx+dy)*eval_hermitenorm(dx,x)*eval_hermitenorm(dy,y)*np.exp(-(x*x+y*y)/2)/(2*np.pi)


def bincount(x,f=None):
    """Return (counts,unique_values), optionally also sums of f at each value."""
    x = np.asarray(x).ravel()
    bins,inverse,counts = np.unique(x,return_inverse=True,return_counts=True)
    if f is None:
        return counts,bins
    f = np.asarray(f,dtype=float).ravel()
    if f.shape!=x.shape:
        raise ValueError("f must have one weight per observation")
    return counts,bins,np.bincount(inverse,weights=f,minlength=len(bins))


def gridcount(data,X):
    """Multidimensional linear binning on uniform grids; meshgrid xy ordering."""
    data = np.asarray(data,dtype=float)
    X = np.asarray(X,dtype=float)
    data = data[:,None] if data.ndim==1 else data
    X = X[:,None] if X.ndim==1 else X
    if data.ndim!=2 or X.ndim!=2 or len(X)<2 or data.shape[1]!=X.shape[1] or not np.all(np.isfinite(data)) or not np.all(np.isfinite(X)):
        raise ValueError("Expected finite N by D data and grid arrays")
    n,d = X.shape
    dx = X[1]-X[0]
    if np.any(dx<=0) or not np.allclose(np.diff(X,axis=0),dx):
        raise ValueError("Grid columns must be uniform and strictly increasing")
    if np.any(data<X[0]) or np.any(data>X[-1]):
        raise ValueError("The grid must contain every data point")
    if d>12:
        raise ValueError("Linear binning is limited to 12 dimensions (2**D corners)")
    pos = (data-X[0])/dx
    lower = np.minimum(np.floor(pos).astype(int),n-2)
    fraction = pos-lower
    counts = np.zeros((n,)*d)
    for bits in itertools.product((0,1),repeat=d):
        bits = np.asarray(bits)
        weight = np.prod(np.where(bits,fraction,1-fraction),axis=1)
        np.add.at(counts,tuple((lower+bits).T),weight)
    return counts.swapaxes(0,1) if d>=2 else counts


def qlevels2(r,p=(10,30,50,70,90,95,99,99.9),method=1):
    """Upper quantiles enclosing p percent of observed scalar/column values."""
    from ..wstats import wquantile
    return wquantile(r,1-np.asarray(p)/100,method)


def qlevels(pdf,p=(10,30,50,70,90,95,99,99.9),x1=None,x2=None):
    """Density heights enclosing p% of grid mass, sorted by decreasing density.

    Normalizes the supplied grid mass to one (no legacy artificial deficit).
    Nonuniform 1D/2D grids use trapezoidal point weights.
    """
    f = np.asarray(pdf,dtype=float)
    p = np.asarray(p,dtype=float)
    if np.any(p<0) or np.any(p>100) or not np.all(np.isfinite(p)) or np.any(f<0) or not np.all(np.isfinite(f)) or f.size==0:
        raise ValueError("Expected a nonempty nonnegative finite density and 0<=p<=100")
    def weights(grid):
        grid = np.asarray(grid,dtype=float)
        if grid.ndim!=1 or len(grid)<2 or np.any(np.diff(grid)<=0):
            raise ValueError("Coordinates must be strictly increasing vectors")
        delta = np.diff(grid)/2
        return np.r_[delta,0]+np.r_[0,delta]
    mass = f.copy()
    if x1 is not None:
        wx = weights(x1)
        if f.ndim==1 and len(wx)==len(f):
            mass *= wx
        elif f.ndim==2 and x2 is not None and f.shape==(len(x2),len(wx)):
            mass *= weights(x2)[:,None]*wx[None,:]
        else:
            raise ValueError("Density shape and coordinate grids do not agree")
    order = np.argsort(f.ravel(),kind='stable')[::-1]
    heights = f.ravel()[order]
    total = mass.sum()
    if total<=0:
        raise ValueError("Density has no positive mass")
    cumulative = np.cumsum(mass.ravel()[order])/total
    return np.interp(p/100,cumulative,heights,left=heights[0],right=heights[-1])
