"""Portable quadrature, smoothing, filtering and option helpers."""
import ast
from pathlib import Path
import numpy as np
from scipy import integrate,interpolate


def parseoptions(options,*updates,**kwargs):
    """Merge mappings or name/value pairs, with case-insensitive unique prefixes.

    None/empty values retain defaults. Unknown or ambiguous names raise instead
    of silently dropping a numerical option. Function-name dispatch is omitted.
    """
    if not isinstance(options,dict):
        raise TypeError("options must be a mapping of legal names/defaults")
    result = options.copy()
    pairs = []
    index = 0
    while index<len(updates):
        item = updates[index]
        if isinstance(item,dict):
            pairs.extend(item.items()); index+=1
        elif isinstance(item,str) and index+1<len(updates):
            pairs.append((item,updates[index+1])); index+=2
        else:
            raise ValueError("Expected an option mapping or a name/value pair")
    pairs.extend(kwargs.items())
    for name,value in pairs:
        matches = [key for key in result if key.lower()==str(name).lower()]
        if not matches:
            matches = [key for key in result if key.lower().startswith(str(name).lower())]
        if len(matches)!=1:
            raise ValueError(f"Unknown or ambiguous option {name!r}")
        if value is not None and np.size(value):
            result[matches[0]] = value
    return result


def qrule2d(nx,ny):
    from . import qrule
    x,wx = qrule(nx); y,wy = qrule(ny)
    xx,yy = np.meshgrid(x,y,indexing='xy')
    return xx,yy,wy[:,None]*wx[None,:]


def gaussq2d(fun,xlow,xhigh,ylow,yhigh,tol=1e-6,*parameters):
    """Adaptive rectangular 2D integration; returns values and error estimates."""
    arrays = np.broadcast_arrays(xlow,xhigh,ylow,yhigh)
    values,errors = np.empty(arrays[0].shape),np.empty(arrays[0].shape)
    for idx in np.ndindex(values.shape):
        a,b,c,d = (float(v[idx]) for v in arrays)
        values[idx],errors[idx] = integrate.dblquad(lambda y,x:fun(x,y,*parameters),a,b,c,d,epsabs=tol,epsrel=tol)
    return values,errors


def ccquad(fun,a,b,n=10,trace=0):
    """Clenshaw-Curtis polynomial integration with nested-grid error estimate.

    Returns the finer 2*n degree estimate and abs(fine-coarse). The difference
    is an estimate, not a rigorous error bound. Callable functions only.
    """
    if trace:
        raise ValueError("Plot integration nodes explicitly; trace plotting is not supported")
    if int(n)!=n or n<2 or not np.isfinite(a+b):
        raise ValueError("Expected n>=2 and finite integration limits")
    def rule(count):
        nodes = np.cos(np.pi*np.arange(count+1)/count)
        samples = np.asarray(fun((a+b)/2+(b-a)/2*nodes))
        if samples.ndim==0:
            samples = np.full(count+1,samples)
        coefficients = np.polynomial.chebyshev.chebfit(nodes,samples,count)
        integral = np.polynomial.chebyshev.chebint(coefficients)
        return (b-a)/2*(np.polynomial.chebyshev.chebval(1,integral)-np.polynomial.chebyshev.chebval(-1,integral))
    coarse,fine = rule(int(n)),rule(2*int(n))
    return fine,abs(fine-coarse)


def savgol(nl,nr,m,ld=0,length=None):
    """Source wrap-around Savitzky-Golay coefficients, without ld! scaling."""
    length = 2*max(nl,nr)+1 if length is None else length
    if any(int(v)!=v for v in (nl,nr,m,ld,length)) or min(nl,nr,m,ld)<0 or nl+nr<m or ld>m or length<nl+nr+1:
        raise ValueError("Require length>=nl+nr+1>m>=ld and nonnegative integer orders")
    positions = np.arange(-nl,nr+1,dtype=float)
    design = positions[:,None]**np.arange(m+1)
    coefficients = np.linalg.pinv(design)[ld]
    wrapped = np.zeros(int(length))
    wrapped[(-positions.astype(int))%int(length)] = coefficients
    return wrapped


def convlv(x,h,dim=0,flag='nonperiodic'):
    """Apply a wrapped convolution kernel, with linear or periodic boundaries."""
    a = np.moveaxis(np.asarray(x,dtype=float),dim,0)
    h = np.asarray(h,dtype=float)
    if h.ndim!=1 or not len(h) or not len(a):
        raise ValueError("Expected a nonempty signal and vector response")
    n = len(a)
    out = np.zeros_like(a)
    periodic = flag.lower().startswith('p')
    if not periodic and not flag.lower().startswith('n'):
        raise ValueError("Boundary flag must be periodic or nonperiodic")
    for j,coefficient in enumerate(h):
        if coefficient==0:
            continue
        lag = j if j<=(len(h)-1)//2 else j-len(h)
        indices = np.arange(n)-lag
        if periodic:
            out += coefficient*a[indices%n]
        else:
            values = a[np.clip(indices,0,n-1)].copy()
            if n>1:
                left,right = indices<0,indices>=n
                shape = (-1,)+(1,)*(a.ndim-1)
                values[left] = a[0]+indices[left].reshape(shape)*(a[1]-a[0])
                values[right] = a[-1]+(indices[right]-n+1).reshape(shape)*(a[-1]-a[-2])
            out += coefficient*values
    return np.moveaxis(out,0,dim)


def smooth(x,y,p=.99,xx=None,LinExtrap=0,d2=None):
    """Minimize p*weighted_residual + (1-p)*integral(second_derivative**2).

    Returns a callable spline when xx is omitted, otherwise evaluated values.
    p=0 is weighted linear regression, p=1 a natural interpolating spline.
    """
    x,y = np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if x.ndim!=1 or y.shape!=x.shape or len(x)<2 or np.any(np.diff(x)<=0) or not np.all(np.isfinite([x,y])) or not 0<=p<=1:
        raise ValueError("Expected finite increasing x, matching y, and 0<=p<=1")
    variance = np.ones_like(x) if d2 is None else np.broadcast_to(d2,x.shape).astype(float)
    if np.any(variance<=0) or not np.all(np.isfinite(variance)):
        raise ValueError("Observation variances must be positive and finite")
    if p==0 or len(x)==2:
        fit = np.polynomial.polynomial.polyfit(x,y,1,w=1/np.sqrt(variance))
        base = interpolate.CubicSpline(x,fit[0]+fit[1]*x,bc_type='natural')
    elif p==1:
        base = interpolate.CubicSpline(x,y,bc_type='natural')
    elif len(x)>=5:
        base = interpolate.make_smoothing_spline(x,y,w=1/variance,lam=(1-p)/p)
    else:
        basis = interpolate.CubicSpline(x,np.eye(len(x)),axis=0,bc_type='natural')
        nodes,weights = np.polynomial.legendre.leggauss(2)
        t = ((x[1:]+x[:-1])[:,None]+np.diff(x)[:,None]*nodes)/2
        w = (np.diff(x)[:,None]*weights/2).ravel()
        second = basis(t.ravel(),2)
        penalty = second.T@(w[:,None]*second)
        fitted = np.linalg.solve(np.diag(1/variance)+(1-p)/p*penalty,y/variance)
        base = interpolate.CubicSpline(x,fitted,bc_type='natural')
    if LinExtrap:
        def evaluate(points):
            points = np.asarray(points)
            result = base(np.clip(points,x[0],x[-1]))
            return result+np.minimum(points-x[0],0)*base(x[0],1)+np.maximum(points-x[-1],0)*base(x[-1],1)
    else:
        evaluate = base
    return evaluate if xx is None else evaluate(xx)


def clevels(c):
    """Extract unique contour levels and replace MATLAB contour headers by NaN."""
    c = np.asarray(c,dtype=float).copy()
    if c.ndim!=2 or c.shape[0]!=2:
        raise ValueError("Expected a MATLAB two-row contour matrix")
    levels,index = [],0
    while index<c.shape[1]:
        count = c[1,index]
        if not np.isfinite(count) or int(count)!=count or count<0 or index+count>=c.shape[1]:
            raise ValueError("Invalid contour segment header")
        levels.append(c[0,index]); c[:,index]=np.nan
        index += int(count)+1
    return np.unique(levels),c


def geth1line(filename):
    """First MATLAB comment line or first Python module docstring line."""
    path = Path(filename)
    text = path.read_text(encoding='utf-8',errors='replace')
    if path.suffix.lower()=='.py':
        doc = ast.get_docstring(ast.parse(text))
        return doc.splitlines()[0] if doc else ''
    return next((line.strip().lstrip('%').strip() for line in text.splitlines() if line.strip().startswith('%')),'')


def wafoversion():
    """Version of the installed Python package (not the MATLAB source version)."""
    from .. import __version__
    return __version__
