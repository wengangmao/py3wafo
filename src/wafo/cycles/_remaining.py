"""Tagged rainflow counting, smoothing and extreme crossing approximations."""
import numpy as np
from ._extended import _square,_discrete_values


def dtp2arfm_sid(x,y,n,ny,definition=1,*,full_output=False):
    """Four-point asymmetric rainflow with endpoint/regime marks.

    definition1 marks the two cycle endpoints, definition2 the sample at which
    a closed cycle is counted. Uses all input points and y[k] at x[k], fixing
    the source's dropped-last-sample/shifted-label indexing. Residual treatment
    is upward, matching dtp2arfm. full_output returns six source-style arrays.
    """
    x=_discrete_values(x,n)
    if x.ndim!=1:
        raise ValueError('Tagged counting needs vectors of classes and labels')
    y=np.asarray(y)
    if y.shape!=x.shape or not np.all(np.isfinite(y)) or np.any(y!=np.floor(y)) or ny<1 or int(ny)!=ny or np.any((y<0)|(y>=ny)) or definition not in (1,2):
        raise ValueError('Labels must be integer classes in [0,ny); definition is 1 or 2')
    n,ny=int(n),int(ny)
    closed=np.zeros((n,n)); marked=np.zeros((ny,ny if definition==1 else 1,n,n)); stack=[]
    for point,label in zip(x.astype(int),y.astype(int)):
        stack.append((point,label))
        while len(stack)>=4:
            inner=sorted((stack[-3][0],stack[-2][0])); outer=sorted((stack[-4][0],stack[-1][0]))
            if inner[0]<outer[0] or inner[1]>outer[1]:
                break
            a,b=stack[-3],stack[-2]
            closed[a[0],b[0]]+=1
            first,second=(a[1],b[1]) if definition==1 else (label,0)
            marked[first,second,a[0],b[0]]+=1
            del stack[-3:-1]
    total=closed.copy(); all_marked=marked.copy()
    start=0 if len(stack)>1 and stack[1][0]>stack[0][0] else 1
    for k in range(start,len(stack)-1,2):
        a,b=stack[k:k+2]
        total[a[0],b[0]]+=1
        first,second=(a[1],b[1]) if definition==1 else (int(y[-1]),0)
        all_marked[first,second,a[0],b[0]]+=1
    residual=np.asarray([row[0] for row in stack],dtype=int)
    residual_labels=np.asarray([row[1] for row in stack],dtype=int)
    return (total,closed,residual,all_marked,marked,residual_labels) if full_output else total


def dtp2rfm_sid(x,y,n,ny,definition=1,*,full_output=False):
    result=list(dtp2arfm_sid(x,y,n,ny,definition,full_output=True))
    for index in (0,1,3,4):
        a=result[index]
        result[index]=np.triu(a+np.swapaxes(a,-1,-2))
    return tuple(result) if full_output else result[0]


def smoothcmat_hnorm(F,NOsubzero=0):
    """Normal-reference bandwidth from pooled endpoint variance, source rule."""
    F=_square(F)
    if np.any(F<0) or not np.isfinite(NOsubzero) or int(NOsubzero)!=NOsubzero or NOsubzero<0:
        raise ValueError('Nonnegative counts and integer structural offset required')
    N=F.sum()
    if N==0:
        return 0.
    marginal=(F+F.T).sum(axis=0)/(2*N)
    u=np.arange(len(F)); mean=marginal@u
    return float(np.sqrt(marginal@((u-mean)**2))*N**(-1/6))


def smoothcmat(F,method=1,h=None,NOsubzero=0,alpha=.5,*,full_output=False):
    """Source Gaussian mirrored-kernel cycle smoothing, with adaptive stage.

    Each source cycle's mass is normalized on the allowed upper triangle, so
    boundaries and diagonal truncation preserve total counts exactly.
    """
    F=_square(F); n=len(F)
    if np.any(F<0) or np.any(np.tril(F)!=0) or method not in (1,2) or int(NOsubzero)!=NOsubzero or not 0<=NOsubzero<n-1 or not 0<=alpha<=1:
        raise ValueError('Need upper-triangular nonnegative counts and valid kernel options')
    h=.7*smoothcmat_hnorm(F,NOsubzero) if h is None else h
    if F.sum()==0:
        out=np.zeros_like(F)
        return (out,float(h)) if full_output else out
    if not np.isfinite(h) or h<=0:
        raise ValueError('Bandwidth must be positive and finite')
    rows,cols=np.indices(F.shape); allowed=np.triu(np.ones_like(F,dtype=bool),1+int(NOsubzero))
    cells=np.argwhere(F>0)
    if method==2 and np.any((F>0)&~allowed):
        raise ValueError('Adaptive smoothing requires source counts outside the structural zero band')
    def smooth(factors):
        result=np.zeros_like(F)
        for (i,j),factor in zip(cells,factors):
            scale=h*factor
            logweights=-((rows-i)**2+(cols-j)**2)/(2*scale**2)
            # Mirror across the min/max diagonal, using a common scale to
            # prevent underflow when tiny bandwidths meet a structural gap.
            exponent=np.maximum(logweights,logweights.T)
            reference=exponent[allowed].max()
            kernel=np.zeros_like(F)
            kernel[allowed]=np.exp(logweights[allowed]-reference)+np.exp(logweights.T[allowed]-reference)
            result+=F[i,j]*kernel/kernel.sum()
        return result
    result=smooth(np.ones(len(cells)))
    if method==2:
        pilot=result/F.sum(); logpilot=np.log(pilot[tuple(cells.T)])
        geometric=np.exp(np.sum(F[tuple(cells.T)]*logpilot)/F.sum())
        factors=(pilot[tuple(cells.T)]/geometric)**(-alpha)
        result=smooth(factors)
    return (result,float(h)) if full_output else result


def lc2rfmextreme(lc,vect=1,*,full_output=False):
    """Independent-Poisson extreme-cycle approximation from source intensities.

    Returns F, or (F,levels,corrected_counting,raw_counting). Uses the vectorized
    source convention, clips negative differentiated cells as the source does.
    """
    from . import nt2cmat,cmat2nt
    lc=np.asarray(lc,dtype=float)
    if lc.ndim!=2 or lc.shape[1]!=2 or len(lc)<2 or not np.all(np.isfinite(lc)) or np.any(np.diff(lc[:,0])<=0) or np.any(lc[:,1]<0) or vect not in (0,1):
        raise ValueError('Expected increasing crossing levels and nonnegative intensity')
    f=lc[:,1]; denom=f[:,None]+f[None,:]
    raw=np.triu(np.divide(f[:,None]*f[None,:],denom,out=np.zeros_like(denom),where=denom>0))
    F=np.maximum(np.triu(nt2cmat(raw),1),0)
    return (F,lc[:,0].copy(),cmat2nt(F),raw) if full_output else F


def wgpdfit_mld(x,data,*,full_output=False):
    """Weighted discrete GPD profile-likelihood score from the MATLAB helper.

    x parameterizes t=1/max(data)-exp(x); returns score, or (score,k,scale).
    Stable log1p expressions and the t=0 continuous limit avoid 0/0.
    """
    data=np.asarray(data,dtype=float)
    if data.ndim!=2 or data.shape[1]!=2 or not len(data) or not np.all(np.isfinite(data)) or np.any(data<0) or data[:,1].sum()<=0 or np.dot(data[:,0],data[:,1])<=0 or not np.isfinite(x) or abs(x)>700:
        raise ValueError('Expected nonnegative value/count rows with positive mass and maximum')
    values,counts=data.T; N=counts.sum(); M=values.max(); t=1/M-np.exp(x)
    if abs(t)*M<1e-6:
        mean=np.average(values,weights=counts); second=np.average(values**2,weights=counts)
        k=t*mean+t*t*second/2
        scale=mean+t*second/2
        score=N*(second/(2*mean)-mean)
    else:
        base=1-t*values
        top=values==M
        logbase=np.empty_like(values)
        logbase[top]=x+np.log(M)
        logbase[~top]=np.log1p(-t*values[~top])
        k=-np.dot(counts,logbase)/N; scale=k/t
        quotient=np.divide(values,base,out=np.zeros_like(values),where=base>0)
        quotient[values==M]=np.exp(-x)
        score=(1/k-1)*np.dot(counts,quotient)-N/t
    if not np.isfinite(score):
        score=np.copysign(np.finfo(float).max,score)
    return (float(score),float(k),float(scale)) if full_output else float(score)
