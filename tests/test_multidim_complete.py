import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from wafo import multidim as m,spec


def test_sensor_name_id_roundtrip():
    names = m.sensortype(np.arange(1,19))
    assert_array_equal(m.sensortypeid(names),np.arange(1,19))
    assert m.sensortype(1)=='n' and m.sensortypeid('W_t')==15
    assert m.sensortype(0)=='nan' and np.isnan(m.sensortypeid('unknown'))


@pytest.mark.parametrize('sa,sb',[(1,1),(1,-1),(-1,1),(-1,-1)])
def test_stable_hyperbolic_ratio(sa,sb):
    a,b = np.array([-2,1,3]),np.array([1,2,-1])
    f = np.cosh if sa==1 else np.sinh
    g = np.cosh if sb==1 else np.sinh
    assert_allclose(m.ratio(a,b,sa,sb),f(a)/g(b))
    assert_allclose(m.ratio(1000,1001,sa,sb),np.exp(-1))
    assert_allclose(m.ratio(0,0,-1,-1),1)


def test_fourier_coefficients_arbitrary_period_multiple_cycles():
    t = np.linspace(0,12,12001)
    x = 2+3*np.cos(2*np.pi*t/3)+4*np.sin(4*np.pi*t/3)
    a,b = m.fourier(t,x,T=3,M=4)
    assert_allclose(a[:,0],[4,3,0,0],atol=1e-12)
    assert_allclose(b[:,0],[0,0,4,0],atol=1e-12)


def test_spreading_normalization():
    S = spec.demospec('dir')
    D = m.dspec2spreading(S)
    mass = np.trapezoid(D.S,D.theta,axis=0)
    assert_allclose(mass[S.S.sum(axis=0)>0],1)
    bad = dict(S,S=S.S.copy())
    bad['S'][0,0]=np.nan
    assert np.all(np.isfinite(m.dspec2spreading(bad).S))


@pytest.mark.parametrize('identifier',range(1,19))
def test_every_linear_sensor_transfer_finite(identifier):
    amplitude,direction,k,response,phase = m.tran([0,.5,1,2],[0,.5,1],(1,2,-3),identifier,h=20,full_output=True)
    assert amplitude.shape==(4,) and response.shape==(3,4)
    assert np.all(np.isfinite(response))
    assert_allclose(response,direction*amplitude)
    assert_allclose(abs(phase),1)


def test_transfer_derivative_and_depth_physics():
    w = np.array([.5,1,2.])
    def response(identifier,h=np.inf,z=0):
        return m.tran(w,[0,np.pi/2],(0,0,z),identifier,h=h,full_output=True)[3]
    eta = response('n')
    assert_allclose(response('n_t'),-1j*w*eta)
    assert_allclose(response('n_tt'),-w*w*eta)
    k = spec.w2k(w)[0]
    assert_allclose(response('n_x')[0],1j*k)
    assert_allclose(response('n_y')[1],1j*k)
    assert_allclose(response('p',z=-5),1028*9.81*np.exp(-5*k)[None,:]*np.ones((2,1)))
    assert_allclose(response('w',h=20,z=-20),0,atol=1e-12)
