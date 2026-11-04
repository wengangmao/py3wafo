"""Exact transformed/adaptive kernel estimation and bandwidth selection.

Grid estimates use exact evaluation rather than the original FFT acceleration.
"""
import numpy as np
from scipy.special import eval_hermitenorm,gamma
from scipy.interpolate import RegularGridInterpolator
from ..spec import Spectrum
from ._kernels import mkernel,kernelstats


def kdeoptset(*updates,**kwargs):
    from ..misc import parseoptions
    defaults = dict(kernel='epan',hs=None,hsMethod='hns',alpha=0,L2=1,inc=128)
    if updates and isinstance(updates[0],str) and updates[0] in ('kde','kdebin','kdefun'):
        updates = updates[1:]
    out = parseoptions(defaults,*updates,**kwargs)
    if not 0<=out['alpha']<=1:
        raise ValueError('Adaptive sensitivity alpha must lie between 0 and 1')
    return out


def _sample(A):
    a = np.asarray(A,dtype=float)
    a = a[:,None] if a.ndim==1 else a
    if a.ndim!=2 or len(a)<2 or not np.all(np.isfinite(a)) or np.any(np.std(a,axis=0)==0):
        raise ValueError('Need at least two finite, nonconstant observations per column')
    return a


def _normal_derivative_sum(x,h,order,exclude_diagonal=False):
    total = 0.
    for start in range(0,len(x),128):
        z = (x[start:start+128,None]-x[None,:])/h
        values = eval_hermitenorm(order,z)*np.exp(-z*z/2)/np.sqrt(2*np.pi)
        if exclude_diagonal:
            indices = np.arange(min(128,len(x)-start))
            values[indices,start+indices]=0
        total += values.sum()
    return total/h**(order+1)


def hldpi(A,kernel='gaussian',L=2):
    """L-stage direct plug-in bandwidth using exact Gaussian derivative sums.

    Avoids the source's gridding approximation; memory is bounded by chunks.
    Output is a bandwidth per column. Nonpositive derivative estimates raise.
    """
    from . import hns
    a = _sample(A)
    if int(L)!=L or not 0<=L<=8:
        raise ValueError('L must be an integer from 0 through 8')
    n,d = a.shape
    mu2,R,_ = kernelstats(kernel)
    sigma = hns(a,'gaussian')/(4/(3*n))**.2
    result = np.empty(d)
    for j in range(d):
        r = 2*int(L)+4
        half = r//2
        psi = (-1)**half*gamma(r+1)/(gamma(half+1)*np.sqrt(np.pi)*(2*sigma[j])**(r+1))
        for stage in range(int(L),0,-1):
            order = 2*stage+2
            derivative_zero = eval_hermitenorm(order,0)/np.sqrt(2*np.pi)
            base = -2*derivative_zero/(psi*n)
            if base<=0:
                raise RuntimeError('Nonpositive pilot bandwidth; data need another bandwidth rule')
            pilot = base**(1/(2*stage+5))
            psi = _normal_derivative_sum(a[:,j],pilot,order)/n**2
        if not np.isfinite(psi) or psi<=0:
            raise RuntimeError('Nonpositive density curvature estimate')
        result[j] = (R/(mu2**2*n*psi))**.2
    return result


def _cross_validation(A,kernel,hvec,biased,full_output):
    from . import hos
    a = _sample(A)
    if a.shape[1]!=1:
        raise ValueError('Cross-validation bandwidth selection is univariate')
    x = a[:,0]; n=len(x)
    if hvec is None:
        upper = float(hos(x,kernel)[0]); hvec=np.linspace(.25*upper,upper,100)
    hs = np.asarray(hvec,dtype=float)
    if hs.ndim!=1 or not len(hs) or np.any(hs<=0) or not np.all(np.isfinite(hs)):
        raise ValueError('Candidate bandwidths must be a positive finite vector')
    mu2,R,_ = kernelstats(kernel)
    if not biased and kernel.lower()[:4] not in ('gaus','norm'):
        raise ValueError('Exact convolution LSCV currently requires a Gaussian kernel')
    score = []
    for h in hs:
        if biased:
            curvature = _normal_derivative_sum(x,h,4,True)/n**2
            current = R/(n*h)+mu2**2*h**4*curvature/4
        else:
            integrated = _normal_derivative_sum(x,np.sqrt(2)*h,0)/n**2
            leave_one = _normal_derivative_sum(x,h,0,True)/(n*(n-1))
            current = integrated-2*leave_one
        score.append(current)
    score = np.asarray(score)
    chosen = float(hs[np.argmin(score)])
    return (chosen,hs,score) if full_output else chosen


def hbcv(A,kernel='gaussian',hvec=None,*,full_output=False):
    """Source biased-CV objective with exact pair sums; optional full score grid."""
    return _cross_validation(A,kernel,hvec,True,full_output)


def hlscv(A,kernel='gaussian',hvec=None,*,full_output=False):
    """Gaussian least-squares leave-one-out cross-validation bandwidth."""
    return _cross_validation(A,kernel,hvec,False,full_output)


def kdefun(A,options=None,*coordinates,full_output=False):
    """Exact diagonal/full-matrix, power-transformed, adaptive KDE.

    Coordinates broadcast together; without coordinates, evaluates at data.
    full_output returns (density,bandwidth_matrix,per_sample_factors).
    """
    from . import hns,hos,hmns,hboot,hscv,hste,hstt,hbcv2,hldpi2,hldpi2fft
    a = _sample(A)
    opts = kdeoptset({} if options is None else options)
    n,d = a.shape
    powers = np.broadcast_to(np.asarray(opts['L2'],dtype=float),(d,))
    if not np.all(np.isfinite(powers)):
        raise ValueError('Transformation exponents must be finite')
    def transform(values):
        result = values.copy()
        jacobian = np.ones(len(values))
        valid = np.all(np.isfinite(values),axis=1)
        for j,p in enumerate(powers):
            if p==1:
                continue
            valid &= values[:,j]>0
            safe = np.maximum(values[:,j],np.finfo(float).tiny)
            if p==0:
                result[:,j]=np.log(safe); jacobian/=safe
            else:
                result[:,j]=np.sign(p)*safe**p; jacobian*=abs(p)*safe**(p-1)
        return result,jacobian,valid
    data,_,valid = transform(a)
    if not np.all(valid):
        raise ValueError('Non-identity power/log transforms require positive data')
    hs = opts['hs']
    if hs is None:
        selectors = {'hns':hns,'hos':hos,'hmns':hmns,'hldpi':hldpi,'hlscv':hlscv,'hbcv':hbcv,
                     'hboot':hboot,'hscv':hscv,'hste':hste,'hstt':hstt,
                     'hbcv2':hbcv2,'hldpi2':hldpi2,'hldpi2fft':hldpi2fft}
        method = opts['hsMethod']
        if method not in selectors:
            raise ValueError('Unknown or unimplemented bandwidth selector')
        hs = selectors[method](data,opts['kernel'])
    H = np.asarray(hs,dtype=float)
    H = np.eye(d)*float(H) if H.ndim==0 else np.diag(H) if H.ndim==1 else H
    if H.shape!=(d,d) or not np.all(np.isfinite(H)) or not np.allclose(H,H.T) or np.linalg.eigvalsh(H).min()<=0:
        raise ValueError('Bandwidth must be positive scalar/vector or symmetric positive definite matrix')
    inverse,det = np.linalg.inv(H),np.linalg.det(H)
    def evaluate(transformed,factors):
        result = np.empty(len(transformed))
        for start in range(0,len(transformed),128):
            z = ((transformed[start:start+128,None,:]-data[None,:,:])@inverse.T)/factors[None,:,None]
            result[start:start+128] = np.mean(mkernel(z,opts['kernel'])/factors[None,:]**d,axis=1)/det
        return result
    factors = np.ones(n)
    if opts['alpha']:
        pilot = np.maximum(evaluate(data,factors),np.finfo(float).tiny)
        factors = np.exp(-opts['alpha']*(np.log(pilot)-np.mean(np.log(pilot))))
    if coordinates:
        if len(coordinates)!=d:
            raise ValueError('Provide one coordinate array per data dimension')
        arrays = np.broadcast_arrays(*coordinates)
        points = np.column_stack([v.ravel() for v in arrays]); shape=arrays[0].shape
    else:
        points=a; shape=(n,)
    transformed,jacobian,valid = transform(points)
    f = np.zeros(len(points))
    f[valid] = evaluate(transformed[valid],factors)*jacobian[valid]
    f = f.reshape(shape)
    return (f,H,factors) if full_output else f


def kdebin(A,options=None,xlo=None,xup=None):
    """PDF grid with exact KDE evaluation in place of legacy FFT binning.

    This is more expensive than FFT binning but avoids its discretization error.
    """
    a = _sample(A); opts=kdeoptset({} if options is None else options)
    d=a.shape[1]; inc=opts['inc']
    if int(inc)!=inc or inc<2 or int(inc)**d>2_000_000:
        raise ValueError('Grid must contain 2 through 2,000,000 total points')
    lower = a.min(axis=0)-np.ptp(a,axis=0)/4 if xlo is None else np.broadcast_to(xlo,(d,))
    upper = a.max(axis=0)+np.ptp(a,axis=0)/4 if xup is None else np.broadcast_to(xup,(d,))
    if np.any(lower>=upper):
        raise ValueError('Invalid density grid bounds')
    axes = [np.linspace(lo,up,int(inc)) for lo,up in zip(lower,upper)]
    grid = np.meshgrid(*axes,indexing='xy' if d<=3 else 'ij')
    density,H,factors = kdefun(a,opts,*grid,full_output=True)
    return Spectrum(f=density,x=axes,labx=['x'+str(i+1) for i in range(d)],hs=H,
        lambda_=factors,note='Exact grid evaluation; no FFT binning',title='Kernel density estimate')


def evalpdf(pdf,*coordinates,method='linear'):
    """Interpolate a PDF grid, returning NaN outside its coordinates."""
    if coordinates and isinstance(coordinates[-1],str):
        method=coordinates[-1]; coordinates=coordinates[:-1]
    axes = [np.asarray(x,dtype=float) for x in pdf['x']]
    if len(coordinates)!=len(axes):
        raise ValueError('Provide one evaluation coordinate per PDF dimension')
    values=np.asarray(pdf['f'],dtype=float)
    # PDF structures follow meshgrid xy ordering up to 3 dimensions.
    if len(axes) in (2,3):
        values=values.swapaxes(0,1)
    method = method.lstrip('*')
    method = 'cubic' if method=='spline' else method
    interpolation=RegularGridInterpolator(axes,values,method=method,bounds_error=False,fill_value=np.nan)
    arrays=np.broadcast_arrays(*coordinates)
    points=np.column_stack([x.ravel() for x in arrays])
    return interpolation(points).reshape(arrays[0].shape)


def fftce(x):
    """Circulant embedding [0..n-1,n-2..1] along each axis."""
    a=np.asarray(x)
    indices=[np.r_[np.arange(n),np.arange(n-2,0,-1)] for n in a.shape]
    return a[np.ix_(*indices)]
