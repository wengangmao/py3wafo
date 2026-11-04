"""Spectral matrix and directional Fourier diagnostics without JLAB binaries."""
import numpy as np
from scipy.integrate import trapezoid
from . import Spectrum
from ._transforms import _frequency,ttspec


def specdiag(S,S22=None,S12=None):
    """Diagonalize a Hermitian 2x2 spectral matrix or broadcast entries.

    Returns (largest eigenvalue, smallest eigenvalue, theta, nu, eigenvectors).
    Eigenvectors diagonalize the matrix but their arbitrary phases can differ
    from the original JLAB rotation-factor convention. Matrix input is shaped
    (2,2,...) and the returned eigenvectors have that same layout.
    """
    if S22 is None and S12 is None:
        matrix=np.asarray(S,dtype=complex)
        if matrix.ndim<2 or matrix.shape[:2]!=(2,2):
            raise ValueError('Expected a 2x2 spectral matrix or three matrix entries')
        matrix=np.moveaxis(matrix,(0,1),(-2,-1))
    elif S22 is not None and S12 is not None:
        a,b,c=np.broadcast_arrays(S,S22,S12)
        matrix=np.empty(a.shape+(2,2),dtype=complex)
        matrix[...,0,0],matrix[...,1,1]=a,b
        matrix[...,0,1],matrix[...,1,0]=c,np.conj(c)
    else:
        raise ValueError('Supply all three spectral entries or one matrix')
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix,np.swapaxes(matrix.conj(),-1,-2),atol=1e-12):
        raise ValueError('Spectral matrices must be finite and Hermitian')
    eigenvalues,eigenvectors=np.linalg.eigh(matrix)
    if np.any(eigenvalues[...,0]<-1e-12*np.maximum(1,abs(eigenvalues[...,1]))):
        raise ValueError('Spectral matrices must be positive semidefinite')
    a,b,c=matrix[...,0,0].real,matrix[...,1,1].real,matrix[...,0,1]
    theta=.5*np.arctan2(2*c.real,a-b)
    nu=.5*np.arctan2(2*c.imag,np.sqrt((a-b)**2+4*c.real*c.real))
    return (eigenvalues[...,1],eigenvalues[...,0],theta,nu,
            np.moveaxis(eigenvectors[...,::-1],(-2,-1),(0,1)))


def dspec2char(S,fact=(1,),*,degrees=False,return_names=False):
    """The 26 directional Fourier characteristics documented in dspec2char.m.

    Returns a list because frequency-dependent entries are arrays while global
    entries are scalars. With return_names=True also returns the chosen names.
    Zero-energy frequency bins have undefined (NaN) direction/spread metrics.
    """
    if str(S.get('type','')).lower()!='dir':
        raise ValueError('Directional spectrum required')
    if S.get('theta_units','radians')=='degrees':
        S=ttspec(S,_frequency(S)[0],'radians')
    _,w,density=_frequency(S)
    theta=np.asarray(S['theta'],dtype=float)+float(S.get('phi',0))
    if density.ndim!=2 or density.shape[0]!=len(theta) or np.any(np.diff(theta)<=0):
        raise ValueError('Invalid directional grid')
    marginal=trapezoid(density,x=theta,axis=0)
    total=trapezoid(marginal,x=w)
    if total<=0:
        raise ValueError('Directional spectrum must have positive total energy')
    angular=trapezoid(density,x=w,axis=1)/total
    def coefficients(values,normalizer):
        return [np.divide(trapezoid(values*np.cos(n*theta)[:,None],x=theta,axis=0),normalizer,
                          out=np.full_like(normalizer,np.nan,dtype=float),where=normalizer>0)
                for n in (1,2)]+[
                np.divide(trapezoid(values*np.sin(n*theta)[:,None],x=theta,axis=0),normalizer,
                          out=np.full_like(normalizer,np.nan,dtype=float),where=normalizer>0)
                for n in (1,2)]
    def metrics(a1,a2,b1,b2):
        c1=np.clip(np.hypot(a1,b1),0,1)
        c2=np.clip(np.hypot(a2,b2),0,1)
        mean=np.arctan2(b1,a1)
        principal=.5*np.arctan2(b2,a2)
        spread=np.sqrt(2*(1-c1))
        with np.errstate(divide='ignore',invalid='ignore'):
            skew=-c2*np.sin(2*(principal-mean))/spread**3
            mean_spread=np.arctan2(np.sqrt(np.maximum(0,.5*b1*b1*(1+a2)-a1*b1*b2+.5*a1*a1*(1-a2))),c1*c1)
            long=np.sqrt((1-c2)/(1+c2))
            s1=c1/(1-c1)
            s2=(1+3*c2+np.sqrt(1+(14+c2)*c2))/(2*(1-c2))
            d1=np.sqrt(-2*np.log(c1))
            d2=np.sqrt(-np.log(c2)/2)
        return [mean,principal,spread,skew,mean_spread,long,s1,s2,d1,d2]
    local=metrics(*coefficients(density,marginal))
    global_metrics=metrics(*coefficients(angular[:,None],np.ones(1)))
    global_metrics=[float(a[0]) for a in global_metrics]
    peak=int(np.argmax(np.max(density,axis=0)))
    main=theta[int(np.argmax(np.max(density,axis=1)))]
    main_weighted=theta[int(np.argmax(angular))]
    peak_indices=np.argmax(density,axis=0)
    weights=np.max(density,axis=0)
    tucker=np.arctan2(np.sum(weights*np.sin(theta[peak_indices])),np.sum(weights*np.cos(theta[peak_indices])))
    values=[*local,local[0][peak],local[2][peak],local[3][peak],main,main_weighted,*global_metrics,tucker]
    names=['FMdir','FPdir','FSpr','FSkew','FMSpr','FLcrst','FS1','FS2','FD1','FD2',
           'TpMdir','TpSpr','TpSkew','Wdir','Wdir2','Mdir','Pdir','Spr','Skew','MSpr','Lcrst','S1','S2','D1','D2','TMdir']
    angle_indices={0,1,2,4,10,11,13,14,15,16,17,19,25}
    for index in angle_indices:
        values[index]=(values[index]+np.pi)%(2*np.pi)-np.pi
        if degrees:
            values[index]=values[index]*180/np.pi
    selected=[]
    for f in np.atleast_1d(fact):
        if isinstance(f,str):
            index=[n.lower() for n in names].index(f.lower())
        else:
            if f!=int(f) or not 1<=f<=26:
                raise ValueError('Directional characteristic index must lie in 1..26')
            index=int(f)-1
        selected.append(index)
    result=[values[i] for i in selected]
    return (result,[names[i] for i in selected]) if return_names else result
