"""Source steering identities, directional recovery, and measured spectral units."""
import numpy as np
from scipy.integrate import trapezoid
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from scipy.special import i0,i1
from wafo import multidim as m,spec


def test_options_and_stable_directional_normalization():
    opts=m.specoptset('dat2dspec','maxIter',31,{'ftype':'f'})
    assert opts['maxiter']==31 and opts['ftype']=='f'
    with pytest.raises(ValueError):m.specoptset('not-an-option',2)
    theta=np.linspace(-np.pi,np.pi,101)
    d=np.column_stack((np.exp(np.cos(theta)),np.zeros(len(theta)),np.full(len(theta),1e-320)))
    normalized=m.normspfn(d,theta)
    assert_allclose(trapezoid(normalized,theta,axis=0),1,atol=1e-14)
    assert_allclose(normalized[:,1],1/(2*np.pi))


def test_cross_spectrum_reconstruction_exact_fourier_moments():
    theta=np.linspace(-np.pi,np.pi,721)
    G=np.stack((np.ones(len(theta)),np.exp(1j*theta)))[:,:,None]
    D=(1+.4*np.cos(theta))[:,None]/(2*np.pi)
    C=m.getcrossspectra(theta,G,D)
    assert_allclose(C[:,:,0],[[1,.2],[.2,1]],atol=1e-14)
    estimates,info=m.mlm(C,G,theta,full_output=True)
    assert estimates.shape==D.shape
    assert np.argmax(estimates[:,0])==len(theta)//2
    assert info['rank'][0]==2
    assert_allclose(trapezoid(estimates[:,0],theta),1)


def test_mem_solves_von_mises_moment_constraint():
    theta=np.linspace(-np.pi,np.pi,361)
    G=np.stack((np.ones(len(theta)),np.exp(1j*theta)))[:,:,None]
    concentration=1.25
    expected=np.exp(concentration*np.cos(theta))/(2*np.pi*i0(concentration))
    R=np.array([[1,i1(concentration)/i0(concentration)],
                [i1(concentration)/i0(concentration),1]],dtype=complex)[:,:,None]
    D,info=m.mem(R,G,theta,opt={'maxiter':100,'errortol':1e-7},full_output=True)
    assert info['converged'][0] and info['residual'][0]<1e-7
    assert_allclose(D[:,0],expected,rtol=2e-6,atol=1e-8)
    assert_allclose(m.getcrossspectra(theta,G,D),R,atol=1e-7)
    # A constant steering moment of2 is impossible; it is reported, not hidden.
    _,bad=m.mem(np.array([[[2.]]]),np.ones((1,len(theta),1)),theta,full_output=True)
    assert not bad['converged'][0] and bad['residual'][0]==1


def test_imlm_source_iteration_preserves_positive_unit_mass():
    theta=np.linspace(-np.pi,np.pi,121)
    G=np.stack((np.ones(len(theta)),np.exp(1j*theta)))[:,:,None]
    C=np.array([[1,.5],[.5,1]],complex)[:,:,None]
    D,info=m.imlm(C,G,theta,opt={'maxiter':20,'errortol':1e-4},full_output=True)
    assert np.all(D>=0) and np.all(np.isfinite(D))
    assert_allclose(trapezoid(D,theta,axis=0),[1],atol=1e-14)
    assert 1<=info['iterations']<=20 and np.isfinite(info['last_change'])


def test_bfs_frequency_spectra_independent_pairs_and_curvatures():
    # All normalized directional-independent combinations equal S_eta=2.
    P=np.zeros((6,5));P[:,3]=[1,4,5,6,7,8];P[:,4]=1
    theta=.3
    power=2*np.array([1,np.cos(theta)**2,np.sin(theta)**2,
        np.cos(theta)**4,np.sin(theta)**4,np.cos(theta)**2*np.sin(theta)**2])
    response=np.arange(1,7.)[:,None]*np.ones((1,3))
    autos=power[:,None]*response**2
    assert_allclose(m.bfsspec(autos,response,P),[2,2,2])
    with pytest.raises(ValueError):m.bfsspec(autos[:2],response[:2],P[:2])


@pytest.fixture
def plane_measurements():
    positions=np.array([[0,0,0],[10,0,0],[0,10,0.]])
    data=m.testmeasurements(positions,[1,1,1],30,.125,2048,.5,2,noise=0)
    return data,np.column_stack((positions,[1,1,1],[1,1,1]))


def test_welch_mlm_recovers_direction_and_variance(plane_measurements):
    data,positions=plane_measurements
    S,D,W,F,G,C,info=m.dat2dspec(data,positions,nfft=128,nt=181,full_output=True)
    peak=np.argmax(W.S)
    assert_allclose(S.w[peak],2*np.pi*.125)
    assert_allclose(np.rad2deg(S.theta[np.argmax(S.S[:,peak])]),30,atol=2)
    assert_allclose(trapezoid(S.S,S.theta,axis=0),W.S,atol=1e-12)
    assert_allclose(trapezoid(W.S,W.w),2,rtol=1e-10)
    assert_allclose(F[0],1,atol=1e-12)
    assert G.shape==(3,181,65) and C.shape==(3,3,65)
    assert info['normalization']=='physical Welch density'


def test_dat2dspec_frequency_and_angle_unit_conservation(plane_measurements):
    data,positions=plane_measurements
    S=m.dat2dspec(data,positions,nfft=128,nt=91,options={'ftype':'f','thtype':'degrees'})
    assert_allclose(trapezoid(trapezoid(S.S,S.theta,axis=0),S.f),2,rtol=1e-10)
    assert S.theta_units=='degrees'


def test_dat2dspec_all_implemented_methods(plane_measurements):
    data,positions=plane_measurements
    for method in ('MEM','IMLM','EMEM'):
        S,D,W,F,G,C,info=m.dat2dspec(data,positions,nfft=128,nt=61,method=method,
            options={'maxiter':20,'errortol':1e-3},full_output=True)
        assert np.all(np.isfinite(S.S)) and np.all(S.S>=0)
        assert_allclose(trapezoid(S.S,S.theta,axis=0),W.S,atol=1e-12)
        assert info['method']==method
    with pytest.raises(NotImplementedError):m.dat2dspec(data,positions,method='BDM')
    malformed=positions.copy();malformed[0,3]=1.5
    with pytest.raises(ValueError):m.dat2dspec(data,malformed)


def test_fixture_sensor_derivatives_bearings_and_random_reproducibility():
    positions=np.zeros((3,3))
    data=m.testmeasurements(positions,[1,4,5],thet0=0,f0=.125,N=128,dt=.5,amp=2,noise=0)
    t=data[:,0];omega=2*np.pi*.125;k=spec.w2k([omega])[0][0]
    assert_allclose(data[:,1],2*np.cos(omega*t))
    assert_allclose(data[:,2],2*k*np.sin(omega*t),atol=1e-14)
    assert_allclose(data[:,3],0,atol=1e-14)
    surface=m.testsurf(128,.5,2,.125,90,[0,10],[0,0],noise=0)
    assert_allclose(surface[:,1],data[:,1],atol=1e-14)
    buoy=m.testbuoy(128,.5,2,.125,90,noise=0)
    assert_allclose(buoy[:,1:],data[:,1:],atol=1e-14)
    assert_array_equal(m.testbuoy(128,.5,2,.125,0,seed=9),m.testbuoy(128,.5,2,.125,0,seed=9))
    assert_array_equal(m.testsurf(128,.5,2,.125,0,[0,1],[0,0],seed=9),m.testsurf(128,.5,2,.125,0,[0,1],[0,0],seed=9))


def test_invalid_sampling_and_unavailable_sensor_spectrum_fail(plane_measurements):
    data,pos=plane_measurements
    uneven=data.copy();uneven[20,0]+=.01
    with pytest.raises(ValueError):m.dat2dspec(uneven,pos)
    missing=pos.copy();missing[:,4]=0
    with pytest.raises(ValueError):m.dat2dspec(data,missing)
    with pytest.raises(ValueError):m.dat2dspec(data,pos,options={'noverlap':128},nfft=128)
    with pytest.raises(ValueError):m.dat2dspec(data,pos,options={'calibrate':'off'})
    theta=np.linspace(-np.pi,np.pi,21)
    with pytest.raises(ValueError):
        m.mlm(np.ones((2,2,1)),np.ones((2,len(theta),1)),theta,opt={'diagonal_loading':0})
