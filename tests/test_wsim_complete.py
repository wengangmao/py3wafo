"""Covariance and process-simulation identities, not historical RNG parity."""
import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from scipy.signal import lfilter
from scipy.linalg import toeplitz
from wafo import wsim,cycles,markov,spec


TESTED_FUNCTIONS=['cov2sdat','cov2csdat','smcsim','smctpsim','mctpsim','sarmasim','spec2nlsdat','rfm2dtp','lc2sdat']


def test_covariance_simulation_empirical_lags_and_reproducibility():
    r=.6**np.arange(32)
    R=dict(R=r,t=np.arange(len(r))*.2)
    x=wsim.cov2sdat(R,[12,40000],seed=14)
    assert x.shape==(12,40001)
    assert_allclose(x[:,0],np.arange(12)*.2)
    assert_allclose(x[0,1:].mean(),0,atol=.02)
    assert_allclose(np.cov(x[:,1:]),toeplitz(r[:12]),atol=.025)
    assert_array_equal(wsim.cov2sdat(R,12,seed=1),wsim.cov2sdat(R,12,seed=1))


def test_covariance_fft_derivative_and_transform_chain_rule():
    # Covariance is a finite cosine, so the interpolated FFT frequencies and
    # derivative share exact spectral differentiation on the embedding grid.
    r=np.zeros(17);r[0]=1
    R=dict(R=r,t=np.arange(17)*.5)
    x,dx=wsim.cov2sdat(R,16,cases=2,seed=7,return_derivative=True)
    transformed=dict(R,tr=[[-20,-10],[20,10]])
    y,dy=wsim.cov2sdat(transformed,16,cases=2,seed=7,return_derivative=True)
    assert_allclose(y[:,1:],2*x[:,1:])
    assert_allclose(dy[:,1:],2*dx[:,1:])
    assert np.all(np.isfinite(dx))
    assert_allclose(x[:,0],dx[:,0])


def test_failed_circulant_uses_exact_finite_covariance():
    # R=[1,.8] does not form a nonnegative zero-tail circulant; its 2x2
    # Toeplitz covariance is positive definite and the exact fallback applies.
    R=dict(R=[1,.8],t=[0,1])
    x=wsim.cov2sdat(R,[2,50000],seed=7)
    assert_allclose(np.cov(x[:,1:]),[[1,.8],[.8,1]],atol=.02)
    with pytest.raises(ValueError,match='Derivative'):
        wsim.cov2sdat(R,2,return_derivative=True)
    with pytest.raises(ValueError,match='semidefinite'):
        wsim.cov2sdat(R,3)


def test_conditional_simulation_closed_form_and_no_missing_case():
    # X1 conditional on X0=1,X2=-1 for a unit variance AR(1) covariance.
    rho=.5
    R=dict(R=rho**np.arange(3),t=np.arange(3))
    sample,mean,std,inds=wsim.cov2csdat([1,np.nan,-1],R,60000,seed=8,full_output=True)
    expected_var=(1-rho*rho)/(1+rho*rho)
    assert_array_equal(inds,[1])
    assert_allclose(mean,[0],atol=1e-14)
    assert_allclose(std,[np.sqrt(expected_var)])
    assert_allclose(sample.mean(),0,atol=.01)
    assert_allclose(sample.var(),expected_var,atol=.01)
    empty=wsim.cov2csdat([1,2,3],R,full_output=True)
    assert empty[0].shape==(0,1) and not len(empty[1])
    allmissing=wsim.cov2csdat([np.nan]*3,R,50000,seed=7)
    assert_allclose(np.cov(allmissing),toeplitz(rho**np.arange(3)),atol=.02)


def test_pseudoinverse_support_and_missing_indices():
    R=dict(R=[1,1,1],t=np.arange(3))
    samples,mean,std,inds=wsim.cov2csdat([2,99,2],R,10,method='pseudo',inds=[1],seed=7,full_output=True)
    assert_allclose(samples,2,atol=1e-7)
    assert_allclose(mean,2)
    assert_allclose(std,0,atol=1e-7)
    with pytest.raises(ValueError,match='support'):
        wsim.cov2csdat([1,np.nan,2],R,method='pseudo')
    with pytest.raises(ValueError,match='singular'):
        wsim.cov2csdat([2,np.nan,2],R,method='exact')


def test_switching_markov_empirical_transition_and_states():
    P=np.array([[.95,.05],[.1,.9]])
    Q=[np.array([[.8,.2],[.3,.7]]),np.array([[.2,.8],[.1,.9]])]
    x,z=wsim.smcsim(P,Q,35000,seed=7)
    assert set(np.unique(x))==set(np.unique(z))=={0,1}
    for regime in range(2):
        for previous in range(2):
            event=(z[1:]==regime)&(x[:-1]==previous)
            assert_allclose((x[1:][event]==1).mean(),Q[regime][previous,1],atol=.035)
    a,b=wsim.smcsim([[1]],[[0,1],[1,0]],5,dict(x0=0,z0=0),seed=2)
    assert_array_equal(a,[0,1,0,1,0]);assert_array_equal(b,[0]*5)


def test_markov_turning_points_and_rainflow_output_modes():
    F=np.triu(np.ones((5,5)),1)
    x=wsim.mctpsim((F,None),5000,seed=7)
    assert np.all(np.diff(x)[::2]>0)
    assert np.all(np.diff(x)[1::2]<0)
    direct,expected=wsim.mctpsim((F,None),100,seed=8,whatOut='x,RFM')
    assert_allclose(expected,cycles.dtp2rfm(direct,5))
    total,closed,res=wsim.mctpsim((F,None),100,seed=8,whatOut='RFM')
    assert_allclose(total,expected)
    out,z=wsim.smctpsim([[.9,.1],[.2,.8]],[(F,None),(2*F,None)],400,seed=7)
    assert np.all(np.diff(out)[::2]>0) and np.all(np.diff(out)[1::2]<0)
    assert set(np.unique(z))=={0,1}


def test_arma_source_recurrence_and_stationary_statistics():
    x,z,e=wsim.sarmasim([[1,.2]],[[1,-.7]],[0],[1],[[1]],500,Tinit=0,seed=7)
    assert_allclose(x,lfilter([1,.2],[1,-.7],e),atol=1e-12)
    assert_array_equal(z,np.zeros(500,dtype=int))
    x,z,e=wsim.sarmasim([[1]],[[1,-.7]],[.6],[.5],[[1]],50000,Tinit=300,seed=4)
    assert_allclose(x.mean(),2,atol=.04)
    assert_allclose(x.var(),.5/(1-.7**2),atol=.03)
    assert x.shape==z.shape==e.shape==(50000,)


@pytest.mark.parametrize('method',['apStochastic','aDeterministic','apDeterministic'])
def test_second_order_simulation_is_sum_of_reported_components(method):
    S=spec.jonswap(sdata=[.5,4]);S.h=np.inf
    nonlinear,linear,s,d,A=wsim.spec2nlsdat(S,[64,2],seed=7,method=method,full_output=True)
    assert nonlinear.shape==linear.shape==(64,3)
    assert s.shape==d.shape==A.shape==(2,64)
    assert_allclose(nonlinear[:,1:]-linear[:,1:],np.fft.fft(s+d,axis=1).real.T,atol=1e-12)
    assert_allclose(linear[:,1:],np.fft.fft(A,axis=1).real.T)
    assert np.all(np.isfinite(nonlinear))
    assert_array_equal(nonlinear,wsim.spec2nlsdat(S,[64,2],seed=7,method=method))


def test_rainflow_reconstruction_roundtrip_with_and_without_residual():
    x=np.array([0,5,1,4,2,3,0,5,1,4,0,5])
    total,closed,res=cycles.dtp2rfm(x,6,full_output=True)
    reconstructed,newres=wsim.rfm2dtp(closed,res,full_output=True)
    rt,rc,rr=cycles.dtp2rfm(reconstructed,6,full_output=True)
    assert_allclose(rt,total);assert_allclose(rc,closed)
    assert_array_equal(newres,res.ravel())
    assert_allclose(cycles.dtp2rfm(wsim.rfm2dtp(total),6),total)
    assert wsim.rfm2dtp(np.zeros((3,3))).shape==(0,)
    assert_array_equal(wsim.rfm2dtp(np.zeros((3,3)),[1]),[1])


def test_rainflow_reconstruction_for_empirical_integer_matrices():
    rng=np.random.default_rng(18)
    for _ in range(20):
        turning=cycles.rfcfilter(rng.integers(0,8,size=80),0,1).astype(int)
        total,closed,residual=cycles.dtp2rfm(turning,8,full_output=True)
        out=wsim.rfm2dtp(closed,residual)
        assert_array_equal(cycles.dtp2rfm(out,8),total)


def test_second_order_amplitude_scaling_and_deterministic_energy():
    S=spec.jonswap(sdata=[.5,4])
    larger=dict(S,S=4*S.S)
    small=wsim.spec2nlsdat(S,64,seed=3,truncationLimit=100,method='aDeterministic',full_output=True)
    large=wsim.spec2nlsdat(larger,64,seed=3,truncationLimit=100,method='aDeterministic',full_output=True)
    assert_allclose(large[1][:,1:],2*small[1][:,1:])
    assert_allclose(large[0][:,1:]-large[1][:,1:],4*(small[0][:,1:]-small[1][:,1:]),atol=1e-14)
    A=small[-1]
    assert_allclose(np.mean(small[1][:,1:]**2,axis=0),np.sum(abs(A)**2,axis=1))


def test_simulation_input_errors_are_explicit():
    with pytest.raises(ValueError):wsim.cov2sdat(dict(R=[1,2],t=[0,1]),10)
    with pytest.raises(ValueError):wsim.cov2csdat([np.nan,1],dict(R=[1,0]),method='approximate')
    with pytest.raises(ValueError):wsim.smctpsim({'P':[[1]]},[],10)
    with pytest.raises(ValueError):wsim.rfm2dtp([[0,.5],[0,0]])
    with pytest.raises(ValueError):wsim.rfm2dtp([[0,1],[0,0]],N=10)


def test_crossing_spectrum_simulation_matches_normalized_shape():
    from wafo.onedim import findcross,findextrema
    u=np.linspace(-5,5,301)
    curve=np.column_stack((u,np.exp(-u*u/2)))
    process,info=wsim.lc2sdat(curve,70000,.75,seed=8,full_output=True)
    y=process[:,1]
    levels=np.array([-1.5,-1,-.5,0,.5,1,1.5])
    counts=np.array([len(findcross(y,v)) for v in levels])
    assert_allclose(counts/counts[3],np.exp(-levels*levels/2),atol=.035)
    assert_allclose(len(findcross(y,0))/len(findextrema(y)),.75,atol=.025)
    assert info['innovation_variance']>0
    assert np.all(np.isfinite(process))
    assert_array_equal(process,wsim.lc2sdat(curve,70000,.75,seed=8))
