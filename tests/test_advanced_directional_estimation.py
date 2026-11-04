"""Analytical Fourier/entropy references and sensor-identity calibration."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import simpson, trapezoid
from wafo import multidim as m


def fourier_density(theta, coefficients):
    coefficients=np.asarray(coefficients)
    n=len(coefficients)//2
    phase=theta[:,None]*np.arange(1,n+1)
    logdensity=np.cos(phase)@coefficients[:n]+np.sin(phase)@coefficients[n:]
    D=np.exp(logdensity-np.max(logdensity))
    return D/trapezoid(D,x=theta)


def harmonic_sensors(theta, nf=1):
    return np.repeat(np.exp(1j*np.arange(3)[:,None,None]*theta[None,:,None]),nf,axis=2)


def test_emem_recovers_fourier_parameters_and_aic_order():
    theta=np.linspace(-np.pi,np.pi,257)
    truths=[np.array([1.2,-.4]),np.array([1.1,.35,-.4,.2])]
    D=np.column_stack([fourier_density(theta,c) for c in truths])
    G=harmonic_sensors(theta,2)
    R=m.getcrossspectra(theta,G,D)
    estimated,info=m.emem(R,G,theta,[.5,1.],opt=dict(maxmodelorder=3,
        errortol=1e-9,coefabstol=1e-12,relax=1,maxiter=80),full_output=True)
    assert_allclose(estimated,D,atol=2e-8,rtol=2e-7)
    assert_array_equal(info['model_order'],[1,2])
    assert np.all(info['converged']) and max(info['residual'])<1e-9
    for actual,truth in zip(info['coefficients'],truths):
        assert_allclose(actual,truth,atol=2e-7)
    assert len(info['trials'][1])==3
    # The selected Fourier order has a lower source AIC than both neighbors.
    scores=[trial['aic'] for trial in info['trials'][1]]
    assert scores[1]<scores[0] and scores[1]<scores[2]
    assert_allclose(m.getcrossspectra(theta,G,estimated),R,atol=1e-9)


def test_emem_entropy_reference_and_active_zero_based_frequencies():
    theta=np.linspace(-np.pi,np.pi,257)
    density=fourier_density(theta,[.8,.3])
    G=harmonic_sensors(theta,3)[:2]
    truth=np.column_stack([density,density,density])
    R=m.getcrossspectra(theta,G,truth)
    D,info=m.emem(R,G,theta,k=[1],opt={'relax':1,'errortol':1e-9},full_output=True)
    assert_allclose(D[:,1],density,atol=1e-8)
    assert_allclose(D[:,[0,2]],1/(2*np.pi),atol=1e-15)
    assert_array_equal(info['model_order'],[0,1,0])
    # An additive second harmonic preserves mass and the first two moments.
    # The von Mises reference maximizes entropy among those feasible densities.
    competitor=density+.005*np.cos(2*theta)
    assert np.all(competitor>0)
    for power in (0,1):
        assert_allclose(trapezoid(competitor*np.exp(1j*power*theta),x=theta),
                        trapezoid(density*np.exp(1j*power*theta),x=theta),atol=1e-14)
    entropy=lambda d:-trapezoid(d*np.log(d),x=theta)
    assert entropy(D[:,1])>=entropy(competitor)-1e-10


def test_emem_limited_fourier_model_is_not_mem_alias_and_reports_nonconvergence():
    theta=np.linspace(-np.pi,np.pi,181)
    G=np.stack([m.tran([1.],theta,pos,1)[1] for pos in ((0,0,0),(8,0,0),(0,12,0))])
    truth=fourier_density(theta,[.2,1.6,.1,-.3])[:,None]
    R=m.getcrossspectra(theta,G,truth)
    opts=dict(maxmodelorder=1,maxiter=80,errortol=1e-7,relax=1)
    D,info=m.emem(R,G,theta,opt=opts,full_output=True)
    other,meminfo=m.mem(R,G,theta,opt={'errortol':1e-7},full_output=True)
    assert info['method']=='EMEM' and meminfo['method']=='MEM'
    assert not info['converged'][0] and info['residual'][0]>1e-3
    assert np.max(abs(D-other))>1e-3
    assert_allclose(trapezoid(D,x=theta,axis=0),1,atol=1e-14)
    with pytest.warns(RuntimeWarning,match='did not converge'):
        m.emem(R,G,theta,opt=opts)
    _,short=m.emem(R,G,theta,opt={'maxiter':1,'maxcoef':.01,'maxmodelorder':1},full_output=True)
    assert not short['converged'][0]
    assert np.max(abs(short['coefficients'][0]))<=.01
    _,constant=m.emem(np.array([[[2.]]]),np.ones((1,len(theta),1)),theta,full_output=True)
    assert not constant['converged'][0] and constant['residual'][0]==1


@pytest.mark.parametrize('option',[{'maxiter':0},{'errortol':0},{'coefabstol':-1},
    {'relax':0},{'maxcoef':0},{'maxmodelorder':0},{'minmodelorder':9},
    {'aictol':-1}])
def test_emem_rejects_invalid_options(option):
    theta=np.linspace(-np.pi,np.pi,17)
    G=harmonic_sensors(theta)
    R=m.getcrossspectra(theta,G,np.ones((len(theta),1))/(2*np.pi))
    with pytest.raises(ValueError):m.emem(R,G,theta,opt=option)


def test_memfun_exact_reference_complex_moments_and_exponential_scaling():
    theta=np.linspace(-np.pi,np.pi,1001)
    basis=np.column_stack([np.cos(theta),np.sin(theta)])
    coef=np.array([.2,-.4])
    moments=np.column_stack([np.ones_like(theta),np.cos(theta)+1j*np.sin(theta),theta**2])
    expected=np.sum(abs(simpson(moments*np.exp(basis@coef)[:,None],x=theta,axis=0)))
    assert_allclose(m.memfun(coef,theta,moments,basis,3),expected,rtol=1e-14)
    # Log-domain rescaling loses a few dozen ulps near the overflow limit.
    assert_allclose(m.memfun([700],theta,np.ones_like(theta),np.ones((len(theta),1))),2*np.pi*np.exp(700),rtol=1e-13)
    assert np.isinf(m.memfun([1000],theta,np.ones_like(theta),np.ones((len(theta),1))))
    assert m.memfun([100000],theta,np.zeros_like(theta),np.ones((len(theta),1)))==0
    with pytest.raises(ValueError):m.memfun(coef,theta,moments,basis,2)
    with pytest.raises(ValueError):m.memfun([np.nan,0],theta,moments,basis)


def test_emem_irregular_angles_preserve_transfer_moments_and_normalization():
    increments=np.random.default_rng(7).uniform(.3,1.7,128)
    theta=-np.pi+2*np.pi*np.r_[0,np.cumsum(increments)]/sum(increments)
    truth=fourier_density(theta,[.8,-.25])[:,None]
    G=harmonic_sensors(theta)[:2]
    R=m.getcrossspectra(theta,G,truth)
    D,info=m.emem(R,G,theta,opt={'maxmodelorder':1,'errortol':1e-10,'coefabstol':1e-12,'relax':1},full_output=True)
    assert info['converged'][0]
    assert_allclose(D,truth,rtol=1e-8,atol=1e-10)
    assert_allclose(trapezoid(D,x=theta,axis=0),1,atol=1e-14)
    assert_allclose(info['coefficients'][0],[.8,-.25],atol=1e-8)


def test_hwestimate_all_sensor_families_recover_transfer_amplitudes():
    theta=np.linspace(-np.pi,np.pi,361)
    omega=np.array([.4,.8,1.2,1.6])
    best=np.array([0.,2.,1.,.5])
    positions=np.zeros((18,5));positions[:,2]=-3;positions[:,3]=np.arange(1,19)
    density=fourier_density(theta,[.7,.2,.1,-.3])
    response=np.zeros((18,len(omega)))
    auto=np.zeros_like(response)
    for row,identifier in enumerate(range(1,19)):
        response[row],G=m.tran(omega,theta,positions[row,:3],identifier,h=20.)
        auto[row]=best*response[row]**2*trapezoid(abs(G)**2*density[:,None],x=theta,axis=0)
    initial=np.full_like(response,.123)
    actual,info=m.hwestimate(auto,best,initial,positions,full_output=True)
    assert_allclose(actual[1:,1:],response[1:,1:],rtol=2e-14,atol=1e-14)
    assert_allclose(actual[0],.123)
    assert_allclose(actual[:,0],.123)
    assert info['estimated'].sum()==17*3
    assert len(info['families'])==11
    assert_array_equal(initial,np.full_like(initial,.123))


def test_hwestimate_source_pair_order_and_missing_family_policy():
    pos=np.zeros((3,4));pos[:,3]=[4,4,5]
    auto=np.array([[2.,8.],[100.,200.],[6.,24.]])
    result=m.hwestimate(auto,[2.,8.],np.ones_like(auto),pos)
    assert_allclose(result,2.)  # Uses first 4/5 pair, then broadcasts to extras.
    missing=pos[:1]
    with pytest.raises(ValueError,match='complete'):
        m.hwestimate(auto[:1],[2.,8.],np.ones((1,2)),missing)
    with pytest.warns(RuntimeWarning,match='Incomplete'):
        unchanged,info=m.hwestimate(auto[:1],[2.,8.],np.ones((1,2)),missing,strict=False,full_output=True)
    assert_array_equal(unchanged,1.)
    assert info['missing_families']==[(4,5)]
    assert_array_equal(m.hwestimate(auto[:1],[0.,0.],np.ones((1,2)),missing),1.)


def test_dat2dspec_emem_dispatch_and_explicit_transfer_calibration():
    data=m.testmeasurements(np.zeros((3,3)),[1,4,5],thet0=45.,f0=.125,
                            N=512,dt=.5,amp=2.,noise=0)
    pos=np.zeros((3,5));pos[:,3]=[1,4,5];pos[0,4]=1
    output=m.dat2dspec(data,pos,nfft=128,nt=73,method='EMEM',
        options={'maxmodelorder':2,'maxiter':50,'errortol':1e-3,'estimate_transfer':True},full_output=True)
    S,D,W,F,G,C,info=output
    assert info['method']=='EMEM' and info['transfer_estimation'] is not None
    assert_allclose(trapezoid(S.S,x=S.theta,axis=0),W.S,atol=1e-13)
    peak=int(np.argmax(W.S))
    mean=trapezoid(D.S[:,peak]*np.exp(1j*D.theta),x=D.theta)
    assert abs(np.angle(mean)-np.pi/4)<.08
    assert info['model_order'][peak]>=1
    assert len(info['trials'][peak])>=1
    assert np.all(info['transfer_amplitudes']>=0)
    assert_allclose(trapezoid(W.S,x=W.w),2.,rtol=1e-10)
    with pytest.raises(ValueError,match='estimate_transfer'):
        m.dat2dspec(data,pos,options={'estimate_transfer':'yes'})
