"""Covariance, switching-chain, ARMA and second-order wave simulation.

Indices/regime states are zero based. Random streams are reproducible NumPy
streams, not an emulation of MATLAB's historical generator state.
"""
from __future__ import annotations
import numbers
import numpy as np
from scipy.linalg import toeplitz
from scipy.interpolate import PchipInterpolator


def _positive_integer(value,name,minimum=1):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Integral) or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _sizes(np_,default,cases):
    if np_ is None:
        n=default
    elif np.size(np_)==2:
        n,cases=np.asarray(np_).tolist()
        if int(n)==n:n=int(n)
        if int(cases)==cases:cases=int(cases)
    elif np.ndim(np_)==0:
        n=np_
    else:
        raise ValueError('Sample count must be an integer or [sample_count,cases]')
    return _positive_integer(n,'sample count'),_positive_integer(cases,'cases')


def _acf(R):
    if not isinstance(R,dict) or 'R' not in R:
        raise ValueError('R must be a covariance mapping with field R')
    r=np.asarray(R['R'],dtype=float)
    if r.ndim!=1 or len(r)<2 or not np.all(np.isfinite(r)) or r[0]<0 or np.any(abs(r)>r[0]+1e-10*max(1,r[0])):
        raise ValueError('Expected a finite covariance vector with its maximum at zero lag')
    axes=[axis for axis in ('t','x','y') if axis in R and np.size(R[axis])]
    if len(axes)>1:
        raise ValueError('Only one-dimensional covariance simulation is supported')
    if axes:
        grid=np.asarray(R[axes[0]],dtype=float)
        if grid.shape!=r.shape or grid[0]!=0 or grid[1]<=0 or not np.all(np.isfinite(grid)) or not np.allclose(np.diff(grid),grid[1]):
            raise ValueError('Covariance lags must be uniform and start at zero')
        dt=grid[1]
    else:dt=1.
    return r,dt


def _root(S):
    S=(S+S.T)/2
    values,vectors=np.linalg.eigh(S)
    tolerance=1e-10*max(np.max(abs(values),initial=0),np.finfo(float).tiny)
    if np.min(values,initial=0)<-tolerance:
        raise ValueError('The requested covariance is not positive semidefinite')
    return vectors*np.sqrt(np.maximum(values,0))


def cov2sdat(R,np_=None,iseed=None,*,cases=1,seed=None,return_derivative=False):
    """Stationary zero-mean Gaussian samples from an auto-covariance vector.

    Uses exact circulant embedding when nonnegative. If the embedding fails,
    falls back to an exact finite Toeplitz factorization (up to 4096 samples),
    rather than silently clipping material negative eigenvalues. Unspecified
    covariance lags are zero. Derivatives use the successful FFT embedding.
    ``np_`` may be [sample_count,cases]; seed and iseed are aliases.
    """
    r,dt=_acf(R)
    n,cases=_sizes(np_,len(r)-1,cases)
    if seed is not None and iseed is not None:
        raise ValueError('Supply seed or iseed, not both')
    rng=np.random.default_rng(seed if seed is not None else iseed)
    size=2**int(np.ceil(np.log2(max(2*n,2*len(r)-1))))
    embedded=np.zeros(size)
    embedded[:len(r)]=r
    embedded[-len(r)+1:]=r[:0:-1]
    eigenvalues=np.fft.fft(embedded).real
    tolerance=1e-10*max(eigenvalues.max(initial=0),np.finfo(float).tiny)
    if eigenvalues.min(initial=0)>=-tolerance:
        normal=rng.normal(size=(size,cases))
        coefficients=np.fft.fft(normal,axis=0)*np.sqrt(np.maximum(eigenvalues,0))[:,None]
        values=np.fft.ifft(coefficients,axis=0).real[:n]
        derivative=None
        if return_derivative:
            omega=2*np.pi*np.fft.fftfreq(size,dt)
            omega[size//2]=0  # A real Nyquist component has no real grid derivative.
            derivative=np.fft.ifft(1j*omega[:,None]*coefficients,axis=0).real[:n]
    else:
        if n>4096:
            raise ValueError('No nonnegative circulant embedding; finite exact fallback is limited to 4096 samples')
        if return_derivative:
            raise ValueError('Derivative simulation requires a nonnegative circulant embedding')
        column=np.zeros(n);column[:min(n,len(r))]=r[:n]
        values=_root(toeplitz(column))@rng.normal(size=(n,cases))
        derivative=None
    transformation=np.asarray(R.get('tr',[]))
    if transformation.size:
        if transformation.ndim!=2 or transformation.shape[1]!=2 or np.any(np.diff(transformation[:,1])<=0):
            raise ValueError('Transformation must be an increasing two-column [physical,Gaussian] table')
        inverse=PchipInterpolator(transformation[:,1],transformation[:,0],extrapolate=True)
        if derivative is not None:derivative*=inverse(values,1)
        values=inverse(values)
    time=np.arange(n)*dt
    x=np.column_stack((time,values))
    return (x,np.column_stack((time,derivative))) if return_derivative else x


def cov2csdat(xo,R,cases=1,method='exact',inds=None,*,seed=None,full_output=False):
    """Gaussian missing-data samples conditional on all observed values.

    Returns only missing samples by default. full_output=True additionally
    returns conditional mean, conditional standard deviation and missing
    indices. method='pseudo' permits singular observed covariance, provided
    observations lie on its support. The legacy approximate method is not
    silently substituted; select exact or pseudo explicitly.
    """
    r,_=_acf(R)
    x=np.asarray(xo,dtype=float).reshape(-1).copy()
    cases=_positive_integer(cases,'cases')
    if np.any(np.isinf(x)):raise ValueError('Observations must be finite or NaN')
    if inds is not None:
        ind=np.asarray(inds)
        if ind.dtype==bool:
            if ind.shape!=x.shape:raise ValueError('Boolean inds must match the data')
            x[ind]=np.nan
        else:
            if np.any(ind!=np.floor(ind)) or np.any(ind<0) or np.any(ind>=len(x)):raise ValueError('Invalid missing indices')
            x[ind.astype(int)]=np.nan
    missing=np.flatnonzero(np.isnan(x));observed=np.flatnonzero(np.isfinite(x))
    if method not in ('exact','pseudo'):
        raise ValueError('Select exact or pseudo; legacy local/Markov approximations are not implemented')
    column=np.zeros(len(x));column[:min(len(x),len(r))]=r[:len(x)]
    S=toeplitz(column)
    mean=np.zeros(len(missing));cov=S[np.ix_(missing,missing)]
    if len(observed):
        oo=S[np.ix_(observed,observed)];mo=S[np.ix_(missing,observed)]
        if method=='exact':
            try:
                np.linalg.cholesky(oo)
                solve=np.linalg.solve(oo,mo.T)
            except np.linalg.LinAlgError as exc:
                raise ValueError('Observed covariance is singular; use method=pseudo') from exc
        else:
            inverse=np.linalg.pinv(oo,hermitian=True)
            if not np.allclose(oo@inverse@x[observed],x[observed],atol=1e-9,rtol=1e-9):
                raise ValueError('Observed values lie outside the singular Gaussian support')
            solve=inverse@mo.T
        mean=solve.T@x[observed]
        cov=cov-mo@solve
    root=_root(cov)
    sample=mean[:,None]+root@np.random.default_rng(seed).normal(size=(len(missing),cases))
    std=np.sqrt(np.maximum(np.diag(cov),0))
    return (sample,mean,std,missing) if full_output else sample


def _initial(init,n,r):
    init={} if init is None else dict(init)
    unknown=set(init)-{'x0','z0'}
    if unknown:raise ValueError('Unknown initial-state fields: '+', '.join(unknown))
    for key,limit in [('x0',n),('z0',r)]:
        value=init.get(key)
        if value is not None:
            if isinstance(value,(bool,np.bool_)) or int(value)!=value or not 0<=value<limit:raise ValueError(f'Invalid {key}')
            init[key]=int(value)
    return init.get('x0'),init.get('z0')


def _start(joint,n,r,x0,z0,rng):
    probabilities=np.asarray(joint).reshape(n,r).copy()
    if x0 is not None:probabilities[np.arange(n)!=x0,:]=0
    if z0 is not None:probabilities[:,np.arange(r)!=z0]=0
    if probabilities.sum()<=0:raise ValueError('Initial state has zero stationary mass or no valid transitions')
    chosen=rng.choice(n*r,p=probabilities.ravel()/probabilities.sum())
    return chosen//r,chosen%r


def _joint_transition(P,Q):
    """Transition for the actual simulation rule using the destination regime."""
    r,n=len(P),len(Q[0])
    result=np.zeros((n*r,n*r))
    for i in range(n):
        for z in range(r):
            for zp in range(r):result[i*r+z,np.arange(n)*r+zp]=P[z,zp]*Q[zp][i]
    return result


def smcsim(P,Q,T,init=None,*,seed=None):
    """Switching Markov chain; return (zero-based load states,regime states).

    New regime z[k] selects Q[z[k]] for transition x[k-1] -> x[k], matching the
    MATLAB simulation loop. Its stationary initial law is built with this same
    rule, correcting the source's inconsistent old-regime initialization.
    """
    from ..markov import _transition,mc2stat
    P=_transition(np.atleast_2d(P));r=len(P)
    Q=[Q] if r==1 and np.asarray(Q).ndim==2 else list(Q)
    if len(Q)!=r:raise ValueError('One transition matrix is required per regime')
    Q=[_transition(q) for q in Q];n=len(Q[0])
    if any(q.shape!=(n,n) for q in Q):raise ValueError('Load matrices must have the same size')
    T=_positive_integer(T,'T');rng=np.random.default_rng(seed)
    x0,z0=_initial(init,n,r)
    x,z=np.empty(T,dtype=int),np.empty(T,dtype=int)
    if x0 is None or z0 is None:x[0],z[0]=_start(mc2stat(_joint_transition(P,Q)),n,r,x0,z0,rng)
    else:x[0],z[0]=x0,z0
    for k in range(1,T):
        z[k]=rng.choice(r,p=P[z[k-1]])
        x[k]=rng.choice(n,p=Q[z[k]][x[k-1]])
    return x,z


def smctpsim(P,F,T,init=None,whatOut='x',*,seed=None):
    """Switching Markov turning points, initially a minimum; states zero based.

    F is one (upward joint matrix,downward joint matrix or None) pair per
    regime. Returns (x,z), (x,z,RFM), or (RFM,closed_RFM,residual) according to
    whatOut='x','x,RFM','RFM'. Non-Markov occupation-time structures are rejected.
    """
    from ..markov import _transition
    from ..markov._extended import _pair_matrices,_stationary_active
    from ..cycles import dtp2rfm
    if isinstance(P,dict):raise ValueError('Non-Markov occupation-time structures require a separately specified simulator')
    P=_transition(np.atleast_2d(P));r=len(P)
    if len(F)!=r:raise ValueError('Expected one up/down matrix pair per regime')
    pairs=[_pair_matrices(pair) for pair in F];n=len(pairs[0][0])
    if any(pair[0].shape!=(n,n) for pair in pairs):raise ValueError('All regime matrices must have the same size')
    T=_positive_integer(T,'T')
    if whatOut not in ('x','x,RFM','RFM'):raise ValueError('whatOut must be x, x,RFM, or RFM')
    rng=np.random.default_rng(seed);x0,z0=_initial(init,n,r)
    x,z=np.empty(T,dtype=int),np.empty(T,dtype=int)
    if x0 is None or z0 is None:
        up=_joint_transition(P,[a for a,b in pairs]);down=_joint_transition(P,[b for a,b in pairs])
        law=_stationary_active(up@down)
        x[0],z[0]=_start(law,n,r,x0,z0,rng)
    else:x[0],z[0]=x0,z0
    for k in range(1,T):
        z[k]=rng.choice(r,p=P[z[k-1]])
        probabilities=pairs[z[k]][0 if k%2 else 1][x[k-1]]
        if not np.isclose(probabilities.sum(),1):
            raise ValueError('Regime switch reached a level with no valid alternating transition')
        x[k]=rng.choice(n,p=probabilities)
    if whatOut=='RFM':return dtp2rfm(x,n,full_output=True)
    if whatOut=='x,RFM':return x,z,dtp2rfm(x,n)
    return x,z


def mctpsim(F,T,init=None,whatOut='x',*,seed=None):
    """Single-regime Markov turning points; return x by default."""
    result=smctpsim([[1]],[F],T,init,whatOut,seed=seed)
    if whatOut=='x':return result[0]
    if whatOut=='x,RFM':return result[0],result[2]
    return result


def sarmasim(C,A,m,s2,P,T,Tinit=None,x0=None,z0=None,*,seed=None):
    """Switching ARMA recurrence; return (samples,regimes,standard innovations).

    C(z,q)*(m[z]+sqrt(s2[z])*e) uses current-regime mean/variance for all MA
    lags, as in sarmasim.m. All three outputs have exactly T rows, correcting
    the MATLAB routine's untrimmed innovation output. Initial x0 is chronological.
    """
    from ..markov import _transition
    from . import mcsim
    P=_transition(np.atleast_2d(P));r=len(P)
    C,A=np.atleast_2d(np.asarray(C,dtype=float)),np.atleast_2d(np.asarray(A,dtype=float))
    if C.shape[0]!=r or A.shape[0]!=r or not C.shape[1] or not A.shape[1] or not np.all(np.isfinite(C)) or not np.all(np.isfinite(A)) or np.any(A[:,0]==0):
        raise ValueError('Finite coefficient matrices need one row per regime and nonzero A[:,0]')
    mean,variance=np.broadcast_arrays(np.asarray(m,dtype=float).reshape(-1),np.asarray(s2,dtype=float).reshape(-1))
    mean,variance=np.broadcast_to(mean,(r,)),np.broadcast_to(variance,(r,))
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)) or np.any(variance<0):raise ValueError('Invalid means or innovation variances')
    T=_positive_integer(T,'T');na,nc=A.shape[1]-1,C.shape[1]-1
    Tinit=10*max(na,1) if Tinit is None else _positive_integer(Tinit,'Tinit',0)
    rng=np.random.default_rng(seed)
    regimes=mcsim(P,T+Tinit,z0,seed=rng)
    innovations=rng.normal(size=T+Tinit)
    prior=np.zeros(na) if x0 is None else np.asarray(x0,dtype=float).reshape(-1)
    if prior.shape!=(na,) or not np.all(np.isfinite(prior)):raise ValueError('x0 needs exactly the AR order initial values')
    output=np.r_[prior,np.zeros(T+Tinit)]
    for k,z in enumerate(regimes):
        ar=output[k:k+na][::-1]
        noise=np.zeros(nc+1)
        available=min(k+1,nc+1)
        noise[:available]=innovations[max(0,k-nc):k+1][::-1]
        value=(C[z]@(mean[z]+np.sqrt(variance[z])*noise)-A[z,1:]@ar)/A[z,0]
        if not np.isfinite(value):raise FloatingPointError('ARMA recurrence diverged')
        output[k+na]=value
    return output[na+Tinit:],regimes[Tinit:],innovations[Tinit:]


def spec2nlsdat(S,np_=None,dt=None,iseed=None,method='apStochastic',truncationLimit=np.sqrt(2),*,cases=1,seed=None,full_output=False):
    """First-plus-second-order unidirectional waves using the DISUFQ kernels.

    Default returns nonlinear time/data array; full_output gives
    (nonlinear,linear,sum_coefficients,difference_coefficients,linear_coefficients).
    Spectral tails use the source's 1000-wave maximum-amplitude convergence
    criterion. Sum frequencies are explicitly kept inside the FFT buffer.
    """
    from ..spec import _validate,spec2dt,spec2mom,w2k
    from ..misc import gravity
    from . import disufq
    w,density=_validate(S)
    n,cases=_sizes(np_,2*(len(w)-1),cases)
    if n%2:n+=1
    dt=spec2dt(S) if dt is None else dt
    depth=S.get('h',np.inf)
    depth=np.inf if depth is None else float(depth)
    if not np.isfinite(dt) or dt<=0 or np.isnan(depth) or depth<=0 or not np.isfinite(truncationLimit) or truncationLimit<=0:
        raise ValueError('Invalid sample interval, depth or truncation limit')
    if seed is not None and iseed is not None:raise ValueError('Supply seed or iseed, not both')
    rng=np.random.default_rng(seed if seed is not None else iseed)
    omega=2*np.pi*np.fft.rfftfreq(n,dt);dw=2*np.pi/(n*dt)
    interpolated=np.interp(omega,w,density,left=0,right=0)
    interpolated[[0,-1]]=0
    amplitude=np.sqrt(interpolated*dw)/2
    coefficients=np.zeros((cases,n),dtype=complex)
    z=rng.normal(size=(cases,len(omega)))+1j*rng.normal(size=(cases,len(omega)))
    mode=method.lower()
    if mode=='apstochastic':positive=z*amplitude
    elif mode=='adeterministic':positive=np.sqrt(2)*amplitude*np.exp(1j*np.angle(z))
    elif mode=='apdeterministic':positive=np.broadcast_to(np.sqrt(2)*amplitude*np.exp(1j*np.angle(z[0,1])),(cases,len(omega)))
    else:raise ValueError('method must be apStochastic, aDeterministic or apDeterministic')
    coefficients[:,:len(omega)]=positive
    coefficients[:,n//2+1:]=positive[:,1:-1][:,::-1].conj()
    linear_values=np.fft.fft(coefficients,axis=1).real.T
    m0,m2=spec2mom(S,2)
    sums,differences=np.zeros_like(coefficients),np.zeros_like(coefficients)
    if m0>0 and m2>0 and interpolated.max()>0:
        g=float(gravity());amax=np.sqrt(2*np.log(1000)*m0)
        kbar=w2k(np.sqrt(m2/m0),h=depth,g=g)[0]
        finite_factor=1 if np.isinf(depth) else np.tanh(kbar*depth)
        upper=truncationLimit*np.sqrt(g*finite_factor/amax)
        lower=0 if np.isinf(depth) else np.sqrt(g*finite_factor*amax/depth**3)
        active=np.flatnonzero((interpolated>interpolated.max()*1e-3)&(omega>=lower)&(omega<=upper))
        active=active[(active>0)&(2*active<n)]
        if len(active):
            k=w2k(omega,h=depth,g=g)[0]
            sums,differences=disufq(coefficients,omega,k,depth,g,int(active[0]),int(active[-1]),components=True)
    nonlinear_values=linear_values+np.fft.fft(sums+differences,axis=1).real.T
    time=np.arange(n)*dt
    nonlinear=np.column_stack((time,nonlinear_values));linear=np.column_stack((time,linear_values))
    return (nonlinear,linear,sums,differences,coefficients) if full_output else nonlinear


def rfm2dtp(F,res=None,N=None,*,full_output=False):
    """Construct an exact sequence for an admissible integer rainflow matrix.

    Uses nested-cycle insertion, replacing FAT's random combination algorithm.
    With res given, F contains closed cycles only. Without res one maximal
    cycle is reserved as the residual. Inadmissible residual/matrix combinations
    are rejected and the result is verified by the actual four-point counter.
    N-based stationary resampling is not an exact count reconstruction and is
    deliberately unsupported; set the desired integer counts in F instead.
    """
    from ..cycles import dtp2rfm
    F=np.asarray(F,dtype=float)
    if F.ndim!=2 or F.shape[0]!=F.shape[1] or not np.all(np.isfinite(F)) or np.any(F<0) or np.any(F!=np.floor(F)) or np.any(np.tril(F)!=0):
        raise ValueError('Expected a finite nonnegative integer upper-triangular rainflow matrix')
    if N is not None:raise ValueError('N-resampling is not an exact reconstruction; supply integer cycle counts directly')
    remaining=F.astype(int).copy();n=len(F)
    total=int(remaining.sum())
    if total>1000000:raise ValueError('Explicit reconstruction is limited to one million cycles')
    if res is None or np.size(res)==0:
        rows,cols=np.nonzero(remaining)
        if not len(rows):
            x=np.empty(0,dtype=int)
            return (x,x.copy()) if full_output else x
        choice=np.argmax(cols-rows);lo,hi=int(rows[choice]),int(cols[choice])
        residual=np.array([lo,hi],dtype=int);remaining[lo,hi]-=1
        expected_total=F.copy()
    else:
        residual=np.asarray(res).reshape(-1)
        if len(residual)<1 or np.any(residual!=np.floor(residual)) or np.any(residual<0) or np.any(residual>=n):raise ValueError('Residual must contain valid zero-based turning points')
        residual=residual.astype(int)
        expected_total=F+dtp2rfm(residual,n)
    x=residual.tolist()
    entries=[(j-i,int(i),int(j),int(remaining[i,j])) for i,j in zip(*np.nonzero(remaining))]
    for _,lo,hi,count in sorted(entries,reverse=True):
        for _ in range(count):
            location=next((k for k,(a,b) in enumerate(zip(x[:-1],x[1:])) if min(a,b)<=lo<hi<=max(a,b)),None)
            if location is None:raise ValueError('Cycle cannot be nested into the supplied residual; matrix/residual is inadmissible for this reconstruction')
            a,b=x[location:location+2]
            x[location+1:location+1]=[hi,lo] if a<b else [lo,hi]
    result=np.asarray(x,dtype=int)
    total_actual,closed_actual,residual_actual=dtp2rfm(result,n,full_output=True)
    if not np.array_equal(total_actual,expected_total):raise ValueError('Cycle insertion did not preserve the requested rainflow count')
    if res is not None and np.size(res) and not np.array_equal(closed_actual,F):raise ValueError('The supplied residual already contains closed cycles')
    return (result,residual_actual.reshape(-1)) if full_output else result


def lc2sdat(lc,N=10000,alpha=.7,*,seed=None,full_output=False):
    """Simulate a monotone transformed AR(2) from a crossing-intensity shape.

    The source's AR(2) coefficient search targets discrete zero-crossing to
    extrema ratio alpha. Gaussian crossing probabilities are integrated
    directly, then a monotone mapping matches the normalized supplied crossing
    spectrum. Absolute crossing intensity is not identifiable from unit-spaced
    samples; this routine matches its shape. Requires a strictly unimodal
    crossing curve (zero endpoints may be supplied). No plot side effects.
    full_output additionally returns AR coefficients and transformation table.
    """
    from scipy.integrate import quad
    from scipy.special import ndtr
    lc=np.asarray(lc,dtype=float)
    N=_positive_integer(N,'N',2)
    if lc.ndim!=2 or lc.shape[1]!=2 or len(lc)<5 or not np.all(np.isfinite(lc)) or np.any(np.diff(lc[:,0])<=0) or np.any(lc[:,1]<0) or not 0<alpha<1:
        raise ValueError('Expected increasing finite levels, nonnegative intensities and 0<alpha<1')
    peak=int(np.argmax(lc[:,1]))
    if peak==0 or peak==len(lc)-1 or lc[peak,1]<=0 or np.any(np.diff(lc[:peak+1,1])<=0) or np.any(np.diff(lc[peak:,1])>=0):
        raise ValueError('A strictly unimodal crossing spectrum with an interior peak is required')
    f=np.linspace(0,.49999,1000)
    rho=2*np.sin(f*np.pi)**2-1
    temporary=np.sin(alpha*np.arcsin(np.sqrt((1+rho)/2)))**2
    a2=(temporary-rho)/(1-temporary)
    quality=np.minimum(a2+rho,1-a2)
    index=int(np.argmax(quality))
    rho,a2=float(rho[index]),float(a2[index])
    a1=2*rho+a2-1
    r1=-a1/(1+a2)
    r2=(a1*a1-a2-a2*a2)/(1+a2)
    variance=1+a1*r1+a2*r2
    if variance<=0 or abs(r1)>=1 or np.any(abs(np.roots([1,a1,a2]))>=1):
        raise ValueError('AR coefficient search did not produce a stable positive-variance process')
    rng=np.random.default_rng(seed)
    y=np.zeros(N)
    y[:2]=_root(np.array([[1,r1],[r1,1]]))@rng.normal(size=2)
    noise=rng.normal(size=N)*np.sqrt(variance)
    for i in range(2,N):y[i]=-a1*y[i-1]-a2*y[i-2]+noise[i]
    extent=max(4.,float(abs(y).max())*1.02)
    gaussian=np.linspace(-extent,extent,301)
    sd=np.sqrt(1-r1*r1)
    def crossing(level):
        return quad(lambda v:np.exp(-v*v/2)/np.sqrt(2*np.pi)*ndtr((r1*v-level)/sd),
                    -np.inf,level,epsabs=1e-13,epsrel=1e-8)[0]
    probabilities=np.array([crossing(abs(v)) for v in gaussian])
    maximum=np.arccos(r1)/(2*np.pi)
    standardized=np.sign(gaussian)*np.sqrt(np.maximum(0,-2*np.log(np.maximum(probabilities/maximum,np.finfo(float).tiny))))
    positive=lc[:,1]>0
    levels=lc[positive,0]
    intensities=lc[positive,1]/lc[peak,1]
    center=lc[peak,0]
    target=np.sign(levels-center)*np.sqrt(np.maximum(0,-2*np.log(intensities)))
    inverse=PchipInterpolator(target,levels,extrapolate=True)
    mapping=inverse(standardized)
    transform=PchipInterpolator(gaussian,mapping,extrapolate=True)
    process=np.column_stack((np.arange(N),transform(y)))
    information=dict(ar=np.array([1,a1,a2]),innovation_variance=variance,
                     correlation=np.array([1,r1,r2]),transform=np.column_stack((gaussian,mapping)),
                     intensity_note='Normalized crossing-spectrum shape; time spacing is one')
    return (process,information) if full_output else process
