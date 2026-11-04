"""First-passage rainflow formulas for ordinary and switching Markov loads."""
import numpy as np
from ._extended import _matrix,_pair_matrices,_stationary_active,smctp2joint


def _solve(A,b):
    if not np.size(A):
        return np.zeros_like(b)
    try:
        return np.linalg.solve(A,b)
    except np.linalg.LinAlgError:
        x=np.linalg.lstsq(A,b,rcond=None)[0]
        if not np.allclose(A@x,b,atol=1e-10,rtol=1e-8):
            raise ValueError('Inconsistent first-passage system')
        return x


def smc2rfm(P,Qc,definition=1,*,full_output=False):
    """Rainflow frequencies for a switching finite-state Markov chain.

    Implements source first-passage counting intensities with solves instead of
    inverses. Definition 1 is the discrete chain; definition (2,h) reproduces
    the source central-difference intensity grid of size n+1.
    """
    from . import _transition,mc2stat
    from ..cycles import cmat2nt
    P=_transition(P); r=len(P)
    if len(Qc)!=r:
        raise ValueError('One load transition matrix is required per regime')
    matrices=[_transition(q) for q in Qc]; n=len(matrices[0])
    if any(q.shape!=(n,n) for q in matrices):
        raise ValueError('All load state spaces must have the same size')
    definition=np.atleast_1d(definition)
    method=int(definition[0])
    if definition[0]!=method or method not in (1,2) or (method==2 and (len(definition)!=2 or not np.isfinite(definition[1]) or definition[1]<=0)):
        raise ValueError('definition must be 1 or (2,positive_grid_step)')
    joint=np.zeros((n*r,n*r)); rows=np.arange(n)*r
    for z,q in enumerate(matrices):
        joint[rows+z]=np.kron(q,P)[rows+z]
    stationary=mc2stat(joint)
    cumulative=np.cumsum(joint[:,::-1],axis=1)[:,::-1]
    N=n if method==1 else n+1
    mu=np.zeros((N,N))
    if method==1:
        for i in range(1,n):
            start=r*i
            for j in range(i-1,n-1):
                stop=r*(j+1)
                above=stop
                A=joint[start:stop,start:stop]
                C=joint[:start,start:stop]
                d=cumulative[:start,above]
                e=cumulative[start:stop,above]
                mu[i,j]=stationary[:start]@(d+C@_solve(np.eye(len(A))-A,e))
        result=np.zeros((n,n))
        for i in range(n-1):
            for j in range(i+1,n):
                result[i,j]=mu[i+1,j-1]-mu[i,j-1]-mu[i+1,j]+mu[i,j]
        mu=cmat2nt(result)
    else:
        for i in range(1,N-1):
            start=r*i
            for j in range(i,N-1):
                stop=r*j
                A=joint[start:stop,start:stop]
                C=joint[:start,start:stop]
                d=cumulative[:start,stop]
                e=cumulative[start:stop,stop]
                mu[i,j]=stationary[:start]@(d+C@_solve(np.eye(len(A))-A,e))
        wat=np.flipud(mu.T)
        for k in range(1,N):
            for m in range(1,N-k+1):
                i=k+m-1; j=N+k-i-1
                if i<N and 0<j<N:
                    wat[i,j]=wat[i,j-1]+wat[i-1,j]-wat[i-1,j-1]
        intensity=np.zeros_like(wat)
        h=float(definition[1])
        for i in range(1,N-1):
            for j in range(1,N-i-1):
                intensity[i,j]=(wat[i-1,j-1]-wat[i-1,j+1]-wat[i+1,j-1]+wat[i+1,j+1])/(2*h)**2
        result=np.flipud(intensity).T
        mu=np.flipud(wat).T
    if np.min(result)<-1e-9:
        raise FloatingPointError('Materially negative rainflow intensity')
    result=np.maximum(result,0)
    return (result,mu) if full_output else result


def mc2rfm(Q,definition=1,*,full_output=False):
    return smc2rfm([[1]],[Q],definition,full_output=full_output)


def _asymmetric(P,F,c_m,side,single_terminal=False):
    from . import _transition
    P=_transition(P); r=len(P)
    if not np.isfinite(c_m) or c_m<0 or side not in (0,1):
        raise ValueError('Require nonnegative intensity and side information 0 or 1')
    Q,Qh=smctp2joint(P,F); n=len(Q)//r
    ro=_stationary_active(Q@Qh); roh=ro@Q
    Qr=np.divide(Qh.T*roh[None,:],ro[:,None],out=np.zeros_like(Q),where=ro[:,None]>0)
    Qrh=np.divide(Q.T*ro[None,:],roh[:,None],out=np.zeros_like(Q),where=roh[:,None]>0)
    FF=ro[:,None]*Q+roh[:,None]*Qh
    result=np.zeros((n,n)); marked=np.zeros((r,r,n,n)); eye=np.eye(r)
    ones=np.ones(r); zero=np.zeros(r); zeros=np.zeros((r,r))
    block=lambda A,I,J:A[np.ix_(I,J)]
    for i in range(n-1):
        for j in range(i+1,n):
            I=np.arange(r*i,r*(i+1)); J=np.arange(r*j,r*(j+1))
            I1=np.arange(r*(i+1),r*(i+2)); J1=np.arange(r*(j-1),r*j)
            inside=np.arange(r*(i+1),r*j)
            Ro,Roh=ro[I],roh[J]
            if Ro.sum()==0 or Roh.sum()==0:
                continue
            x,y=block(FF,I,J),block(FF,J,I)
            C=np.divide(y.T,Ro[:,None],out=np.zeros_like(y.T),where=Ro[:,None]>0)
            Ch=np.divide(x.T,Roh[:,None],out=np.zeros_like(x.T),where=Roh[:,None]>0)
            E,Eh=1-C.sum(axis=1),1-Ch.sum(axis=1)
            D,Dh,R,Rh,E3,E3h=zero.copy(),zero.copy(),zeros.copy(),zeros.copy(),ones.copy(),ones.copy()
            if j-i>=2:
                qr,qrh=block(Qr,I,I1),block(Qrh,J,J1)
                D=qr@(1-block(Qrh,I1,I).sum(axis=1))
                Dh=qrh@(1-block(Qr,J1,J).sum(axis=1))
                E-=block(Qr,I,inside).sum(axis=1); Eh-=block(Qrh,J,inside).sum(axis=1)
                R=qr@block(Qrh,I1,I); Rh=qrh@block(Qr,J1,J)
                E3=1-block(Q,I,inside).sum(axis=1); E3h=1-block(Qh,J,inside).sum(axis=1)
            if j-i>=3:
                II=np.arange(r*(i+1),r*(j-1)); JJ=np.arange(r*(i+2),r*j)
                A,Ah=block(Qr,II,JJ),block(Qrh,JJ,II)
                inv=_solve(np.eye(len(A))-A@Ah,np.eye(len(A)))
                a,ah,b,bh=block(Qr,I,JJ),block(Qrh,J,II),block(Qr,II,J),block(Qrh,JJ,I)
                e=1-Qr[np.ix_(II,np.arange(r*(i+2),r*(j+1)))].sum(axis=1)
                eh=1-Qrh[np.ix_(JJ,np.arange(r*i,r*(j-1)))].sum(axis=1)
                C+=a@Ah@inv@b; Ch+=ah@inv@A@bh
                D+=a@eh+a@Ah@inv@A@eh; Dh+=ah@inv@e
                E+=a@Ah@inv@e; Eh+=ah@inv@A@eh
                R+=a@bh+a@Ah@inv@A@bh; Rh+=ah@inv@b
                A3,A3h=block(Q,II,JJ),block(Qh,JJ,II)
                inv3=_solve(np.eye(len(A3))-A3@A3h,np.eye(len(A3)))
                c3,c3h=block(Q,I,JJ),block(Qh,J,II)
                e3=1-Q[np.ix_(II,np.arange(r*(i+2),r*j))].sum(axis=1)
                e3h=1-Qh[np.ix_(JJ,np.arange(r*(i+1),r*(j-1)))].sum(axis=1)
                E3+=c3@A3h@inv3@e3; E3h+=c3h@inv3@A3@e3h
            if i==0 and j==n-1:
                # The standalone source assigns its outermost cycle standing;
                # the switching source splits this ambiguous cycle equally.
                if single_terminal:
                    result[i,j]=Ro@E3
                else:
                    result[i,j]=(Roh@E3h)*Roh.sum()/(Ro.sum()+Roh.sum())
                    result[j,i]=(Ro@E3)*Ro.sum()/(Ro.sum()+Roh.sum())
                    EE=block(Q,I,J)
                    if j-i>=3:
                        EE=EE+c3@A3h@inv3@block(Q,II,J)
                    marked[:,:,i,j]=.5*Ro[:,None]*EE
                    marked[:,:,j,i]=marked[:,:,i,j]
            else:
                if j!=n-1:
                    pS2=_solve(eye-Rh,Ch)
                    pS1=_solve(eye-R-C@pS2,E+C@_solve(eye-Rh,Dh))
                    result[i,j]=(Roh*E3h)@pS2@pS1
                    marked[:,:,i,j]=pS1[:,None]*pS2.T*(Roh*E3h)[None,:]
                if i!=0:
                    pH2=_solve(eye-R,C)
                    pH1=_solve(eye-Rh-Ch@pH2,Eh+Ch@_solve(eye-R,D))
                    result[j,i]=(Ro*E3)@pH2@pH1
                    marked[:,:,j,i]=pH1[:,None]*pH2.T*(Ro*E3)[None,:]
    if result.min()<-1e-8:
        raise FloatingPointError('Asymmetric rainflow calculation produced negative mass')
    return np.maximum(result,0)*c_m,np.maximum(marked,0)*c_m,(Q,Qh,Qr,Qrh,ro,roh)


def smctp2arfm(P,F,c_m=1,SideInfo=0,*,full_output=False):
    """Asymmetric switching-MCTP rainflow via the source elementary-event blocks.

    SideInfo=1 returns min/max-regime marks in full_output. The unimplemented
    source SideInfo=2 counting-time definition is explicitly rejected.
    """
    result,marked,info=_asymmetric(P,F,c_m,SideInfo)
    return (result,marked,info) if full_output else result


def mctp2arfm(F,c_m=1,*,full_output=False):
    """Single-regime asymmetric cycles, including the source terminal convention."""
    result,_,info=_asymmetric([[1]],[F],c_m,0,True)
    Q,Qh,Qr,Qrh,ro,roh=info
    forward=(ro[:,None]*Q,roh[:,None]*Qh)
    backward=(roh[:,None]*Qrh,ro[:,None]*Qr)
    return (result,forward,backward) if full_output else result


def scalemat(param,F,m,s,param1,*,full_output=False):
    """Source matrix interpolation/translation in its documented WAT layout.

    Interpolates densities (does not renormalize to conserve discrete counts).
    """
    from ..cycles import levels
    from scipy.interpolate import RegularGridInterpolator
    u,v=levels(param),levels(param1); F=_matrix(F)
    if len(F)!=len(u) or not np.isfinite(m) or not np.isfinite(s) or s<=0:
        raise ValueError('Matching matrix/grid and finite translation/positive scale required')
    pj=np.flipud(F).T
    interp=RegularGridInterpolator((u,u),pj,bounds_error=False,fill_value=0)
    minimum,maximum=np.meshgrid(v,v,indexing='ij')
    values=interp(np.stack(((minimum-m)/s,(maximum-m)/s),axis=-1))
    values=np.triu(values,1)
    result=np.flipud(values.T)
    return (result,np.asarray(param1)) if full_output else result
