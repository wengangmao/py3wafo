"""Kernel sampling and nearest-grid accumulation from the WAFO KDE module.

The samplers use NumPy Generator streams. Smoothed sampling corrects covariance
through symmetric PSD factors instead of the original noncommuting matrix
square-root quotient. Numeric data, not pickled transformations, are accepted.
"""
import numbers
import numpy as np
from scipy.special import gammaln


def _count(value,name,minimum=0):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Integral) or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _data(A):
    a=np.asarray(A,dtype=float)
    a=a[:,None] if a.ndim==1 else a
    if a.ndim!=2 or not len(a) or not a.shape[1] or not np.all(np.isfinite(a)):
        raise ValueError('Expected a nonempty finite N by D data matrix')
    return a


def sample(A,m,r=1,*,seed=None,return_indices=False):
    """Random data rows, with r=0 without replacement and r=1 with replacement.

    Full-size sampling without replacement retains source row order. Optional
    indices are zero based. One-dimensional input is treated as one column.
    """
    a=_data(A);m=_count(m,'m')
    if r not in (0,1):raise ValueError('r must be 0 or 1')
    if r==0 and m>len(a):raise ValueError('Sample without replacement exceeds the data size')
    rng=np.random.default_rng(seed)
    if r==0 and m==len(a):indices=np.arange(len(a))
    else:indices=rng.choice(len(a),m,replace=bool(r))
    result=a[indices].copy()
    return (result,indices) if return_indices else result


def vsph(d,r0=1):
    """Volume of a d-dimensional ball, using log-gamma for numerical stability."""
    d=np.asarray(d,dtype=float);r0=np.asarray(r0,dtype=float)
    if not np.all(np.isfinite([*np.ravel(d),*np.ravel(r0)])) or np.any(d<0) or np.any(d!=np.floor(d)) or np.any(r0<0):
        raise ValueError('Dimension must be a nonnegative integer and radius nonnegative')
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        volume=np.exp(d/2*np.log(np.pi)+d*np.log(r0)-gammaln(d/2+1))
    return np.where(d==0,1,np.where(r0==0,0,volume))


def mkernel2(X,kernel='gaussian'):
    """Row-wise multivariate kernel evaluation; fixes source product powers."""
    from ._kernels import mkernel
    x=np.asarray(X,dtype=float)
    x=x[:,None] if x.ndim==1 else x
    if x.ndim!=2 or not x.shape[1]:raise ValueError('X must be a points-by-dimensions matrix')
    return mkernel(x,kernel)


def mkernelrnd(kernel='gaussian',N=1,D=1,*,seed=None):
    """Random rows from every normalized kernel supported by mkernel.

    Adds radial beta-radius/product-beta samplers to the source's Gaussian and
    product-Epanechnikov cases; moments match those normalized kernel formulas.
    """
    N,D=_count(N,'N'),_count(D,'D',1)
    rng=np.random.default_rng(seed);name=str(kernel).lower()[:4];shape=(N,D)
    radial={'sphe':0,'epan':1,'biwe':2,'triw':3,'four':4}
    product={'epa1':1,'biw1':2,'tri1':3,'fou1':4}
    if name in radial:
        direction=rng.normal(size=shape)
        length=np.linalg.norm(direction,axis=1)
        unit=np.divide(direction,length[:,None],out=np.zeros_like(direction),where=length[:,None]>0)
        radius=np.sqrt(rng.beta(D/2,radial[name]+1,size=N))
        return unit*radius[:,None]
    if name in product:
        degree=product[name]
        return 2*rng.beta(degree+1,degree+1,size=shape)-1
    if name in ('gaus','norm'):return rng.normal(size=shape)
    if name in ('rect','unif'):return rng.uniform(-1,1,size=shape)
    if name=='tria':return rng.triangular(-1,0,1,size=shape)
    if name=='lapl':return rng.laplace(size=shape)
    if name=='logi':return rng.logistic(size=shape)
    raise ValueError('Unknown kernel: '+str(kernel))


def _kernel_covariance(kernel,d):
    from ._kernels import kernelstats
    name=kernel.lower()[:4]
    if name in ('sphe','epan','biwe','triw','four'):
        degree={'sphe':0,'epan':1,'biwe':2,'triw':3,'four':4}[name]
        variance=1/(d+2*degree+2)
    elif name in ('epa1','biw1','tri1','fou1'):
        degree={'epa1':1,'biw1':2,'tri1':3,'fou1':4}[name]
        variance=1/(2*degree+3)
    else:variance=kernelstats(kernel)[0]
    return np.eye(d)*variance


def _sqrt_psd(matrix,inverse=False):
    values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
    tolerance=np.finfo(float).eps*max(1,float(np.max(abs(values))))*len(values)*10
    if np.min(values)<-tolerance:raise ValueError('Covariance must be positive semidefinite')
    values=np.maximum(values,0)
    if inverse:
        diagonal=np.divide(1,np.sqrt(values),out=np.zeros_like(values),where=values>tolerance)
    else:diagonal=np.sqrt(values)
    return (vectors*diagonal)@vectors.T


def ssample(A,m,options=None,*,seed=None,preserve_covariance=True,boundary='raise',full_output=False):
    """Sample a smoothed empirical distribution in optional power/log space.

    Uses the exact KDE bandwidth/adaptive pilot factors and normalized kernels.
    preserve_covariance applies a PSD whitening/recolouring map in transformed
    space, fixing the source's asymmetric matrix quotient. Log transforms are
    always invertible. Other power transforms have finite-domain boundaries:
    invalid draws raise by default, or boundary='reflect' explicitly reflects
    them before inversion (a reflected-kernel adaptation). Nonparametric table
    transforms are not accepted. Full output adds bandwidth/factor metadata.
    """
    from ._estimation import kdeoptset,kdefun
    a=_data(A);m=_count(m,'m')
    opts=kdeoptset({'kernel':'gaussian'} if options is None else options)
    if boundary not in ('raise','reflect'):raise ValueError("boundary must be 'raise' or 'reflect'")
    n,d=a.shape
    powers=np.broadcast_to(np.asarray(opts['L2'],dtype=float),(d,))
    if not np.all(np.isfinite(powers)):raise ValueError('Power transforms must be finite numbers')
    transformed=a.copy()
    for j,p in enumerate(powers):
        if p==1:continue
        if np.any(a[:,j]<=0):raise ValueError('Nonidentity transformations require positive data')
        transformed[:,j]=np.log(a[:,j]) if p==0 else np.sign(p)*a[:,j]**p
    if not np.all(np.isfinite(transformed)):raise FloatingPointError('Transformation exceeds finite range')
    pilot=dict(opts,L2=1)
    hs=opts['hs']
    if hs is not None:
        hs=np.asarray(hs,dtype=float)
        if hs.ndim<=1 and np.any(hs<=0):
            from . import hns
            vector=np.broadcast_to(hs,(d,)).copy()
            automatic=hns(transformed,opts['kernel'])
            vector[vector<=0]=automatic[vector<=0]
            pilot['hs']=vector
    if np.all(np.std(transformed,axis=0)==0) and preserve_covariance:
        result=np.broadcast_to(a[0],(m,d)).copy()
        return (result,dict(hs=np.zeros((d,d)),lambda_=np.ones(n),boundary=boundary)) if full_output else result
    _,H,factors=kdefun(transformed,pilot,*[np.empty(0) for _ in range(d)],full_output=True)
    rng=np.random.default_rng(seed)
    indices=rng.integers(n,size=m)
    noise=mkernelrnd(opts['kernel'],m,d,seed=rng)*factors[indices,None]
    mean=transformed.mean(axis=0)
    centred=transformed[indices]-mean+noise@H.T
    if preserve_covariance:
        covariance=np.atleast_2d(np.cov(transformed,rowvar=False,bias=True))
        noise_covariance=H@_kernel_covariance(opts['kernel'],d)@H.T*np.mean(factors*factors)
        centred=centred@_sqrt_psd(covariance+noise_covariance,True)@_sqrt_psd(covariance)
    result=centred+mean
    for j,p in enumerate(powers):
        if p==1:continue
        if p==0:result[:,j]=np.exp(result[:,j]);continue
        signed=np.sign(p)*result[:,j]
        if np.any(signed<=0):
            if boundary=='raise':
                raise ValueError('Smoothed draws crossed a power-transform boundary; explicitly choose boundary=reflect or a log transform')
            signed=np.maximum(abs(signed),np.finfo(float).tiny)
        result[:,j]=signed**(1/p)
    if not np.all(np.isfinite(result)):raise FloatingPointError('Inverse transformation exceeds finite range')
    return (result,dict(hs=H,lambda_=factors,indices=indices,boundary=boundary)) if full_output else result


def ffndgrid(x,f,delta=-75,limits=None,aver=1):
    """Nearest-grid binning of scalar values, with source meshgrid axis order.

    delta>0 is spacing; negative delta is a bin count. The imaginary part of
    delta[0] supplies the empty-bin padding value. limits interleave coordinate
    lower/upper bounds, followed by f_min,f_max,min_count. Negative min_count
    is a fraction of retained observations. True zero-valued occupied bins
    remain zero rather than being mistaken for empty cells as in the source.
    """
    a=_data(x);n,d=a.shape
    values=np.broadcast_to(np.asarray(f,dtype=float).ravel(),(n,)).copy()
    if not np.all(np.isfinite(values)):raise ValueError('f must be finite')
    bounds=np.ravel(np.column_stack((a.min(axis=0),a.max(axis=0)))).tolist()+[values.min(),values.max(),0]
    bounds=np.asarray(bounds,dtype=float)
    if limits is not None:
        supplied=np.asarray(limits,dtype=float).ravel()
        if len(supplied)>len(bounds):raise ValueError('Too many gridding limits')
        present=~np.isnan(supplied);bounds[:len(supplied)][present]=supplied[present]
    lower,upper=bounds[:2*d:2],bounds[1:2*d:2]
    if not np.all(np.isfinite(bounds)) or np.any(lower>=upper) or bounds[-3]>bounds[-2]:
        raise ValueError('Grid and value bounds must be finite and ordered')
    requested=np.asarray(delta,dtype=complex).ravel()
    if len(requested)==1:requested=np.repeat(requested,d)
    if len(requested)!=d:raise ValueError('delta must be scalar or have one entry per dimension')
    padding=float(requested[0].imag);steps=requested.real.copy();axes=[]
    for j,step in enumerate(steps):
        if not np.isfinite(step) or step==0:raise ValueError('Grid spacing/count must be finite and nonzero')
        if step<0:
            if int(-step)!=-step or -step<2:raise ValueError('Negative delta entries must encode integer bin counts >=2')
            steps[j]=(upper[j]-lower[j])/(-step-1)
            axes.append(np.linspace(lower[j],upper[j],int(-step)))
        else:
            axes.append(lower[j]+np.arange(int(np.floor((upper[j]-lower[j])/step+1e-12))+1)*step)
    shape=tuple(len(axis) for axis in axes)
    if np.prod(shape)>5_000_000:raise ValueError('Grid exceeds 5,000,000 cells')
    indices=np.floor((a-lower)/steps+.5).astype(int)
    valid=np.all((indices>=0)&(indices<np.asarray(shape)),axis=1)&(values>=bounds[-3])&(values<=bounds[-2])
    selected=indices[valid];values=values[valid]
    sums=np.zeros(shape);counts=np.zeros(shape,dtype=int)
    if len(selected):
        np.add.at(sums,tuple(selected.T),values);np.add.at(counts,tuple(selected.T),1)
    min_count=bounds[-1] if bounds[-1]>=0 else -bounds[-1]*len(selected)
    occupied=counts>min_count
    if aver not in (0,1,False,True):raise ValueError('aver must be 0 or 1')
    result=np.full(shape,padding,dtype=float)
    result[occupied]=sums[occupied]/counts[occupied] if aver else sums[occupied]
    if d in (2,3):result=result.swapaxes(0,1)
    return result,axes
