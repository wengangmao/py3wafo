"""Rice identities, covariance construction and native-driver integration tests.

These do not assert complete continuous-path or historical binary parity.
"""
import numpy as np
import pytest
from scipy.integrate import trapezoid
from numpy.testing import assert_allclose
from wafo.spec import createspec
from wafo.trgauss._wave_integrals import (
    _covariance, _normalization, _rice, _spectrum,
    spec2Acdf, spec2mmtpdf, spec2thpdf, spec2tccpdf, wminmax,
)


@pytest.fixture
def gaussian_spectrum():
    # R(t)=exp(-t²/2), m0=m2=1, m4=3 to quadrature accuracy.
    w = np.linspace(0,10,501)
    return createspec(w=w,S=np.sqrt(2/np.pi)*np.exp(-w*w/2))


OPTIONS = dict(maxpts=2048,replications=8,seed=1729,releps=.05,abseps=1e-7)


def test_derivative_covariance_and_rice_rates(gaussian_spectrum):
    w,s,m = _spectrum(gaussian_spectrum)
    assert_allclose(m,[1,1,3],atol=1e-12)
    C = _covariance(w,s,[0,1,0,0],[0,0,1,2])
    assert_allclose(C[0,1],np.exp(-.5),atol=1e-12)
    assert_allclose(C[0,3],-1,atol=1e-12)
    assert_allclose(C[1,2],np.exp(-.5),atol=1e-12)
    assert_allclose(np.diag(C),[1,1,1,3],atol=1e-12)
    assert_allclose(C,C.T)
    assert_allclose(_normalization(m),2*np.pi)
    assert_allclose(_normalization(m,extrema=True),2*np.pi/np.sqrt(3))


def test_local_maximum_rice_intensity(gaussian_spectrum):
    # E[|X''| I(X''<0) | X'=0] f_X'(0) = sqrt(m4/m2)/(2*pi).
    w,s,m = _spectrum(gaussian_spectrum)
    value,error,_ = _rice(w,s,[],[(0,2)],[(0,1)],[0],[-np.inf],[0],OPTIONS)
    expected = np.sqrt(m[2]/m[1])/(2*np.pi)
    assert_allclose(value[0],expected,rtol=.015)
    assert error[0]>=0


def test_two_crossing_no_interior_exact_identity(gaussian_spectrum):
    # First positive point on a three-point grid has no interior path nodes,
    # so the exact bivariate positive-part moment checks the driver layout.
    T = 1.
    w,s,m = _spectrum(gaussian_spectrum)
    C = _covariance(w,s,[0,T,0,T],[1,1,0,0])
    K = C[2:,2:]
    V = C[:2,:2]-C[:2,2:]@np.linalg.solve(K,C[2:,:2])
    sigma = np.sqrt(np.prod(np.diag(V)))
    rho = -V[0,1]/sigma
    moment = sigma*(np.sqrt(1-rho*rho)+rho*(np.pi/2+np.arcsin(rho)))/(2*np.pi)
    endpoint_density = 1/(2*np.pi*np.sqrt(np.linalg.det(K)))
    expected = moment*endpoint_density*_normalization(m)
    out = spec2thpdf(gaussian_spectrum,definition='Tc',paramt=[0,2*T,3],options=OPTIONS)
    assert_allclose(out.f[1],expected,rtol=.025)
    assert out.f[0]==0
    assert out.f.shape==out.err.shape==(3,)


def test_crest_trough_symmetry_and_finite_horizon_cdf(gaussian_spectrum):
    crest = spec2Acdf(gaussian_spectrum,definition='Tc',paramt=[0,3,4],h=[0,.5,2],options=OPTIONS)
    trough = spec2Acdf(gaussian_spectrum,definition='Tt',paramt=[0,3,4],h=[0,.5,2],options=OPTIONS)
    assert crest.f[0]==trough.f[0]==0
    assert_allclose(crest.f,trough.f,atol=.04)
    assert np.all(crest.f>=0)
    assert crest.f[-1]>=crest.f[1]-.02
    assert crest.period_subdensity.shape==(4,3)
    assert crest.horizon==3


def test_mmtpdf_integrates_time_and_minmax_sign_symmetry(gaussian_spectrum):
    full = spec2mmtpdf(gaussian_spectrum,definition='MmTMm',paramt=[0,3,4],paramu=[-1,1,3],options=OPTIONS)
    marginal = spec2mmtpdf(gaussian_spectrum,definition='Mm',paramt=[0,3,4],paramu=[-1,1,3],options=OPTIONS)
    assert full.f.shape==(3,3,4)
    assert np.all(full.f>=0)
    assert_allclose(marginal.f,trapezoid(full.f,full.x[-1],axis=-1))
    assert np.all(np.tril(marginal.f)==0)
    other = wminmax(gaussian_spectrum,paramu=[-1,1,3],t=np.linspace(0,3,4),options=OPTIONS)
    assert_allclose(other.f,marginal.f[::-1,::-1].T)


@pytest.mark.parametrize('definition',['TcAc','TtAt','TcfAc','TtbAt'])
def test_joint_period_height_driver_branches(gaussian_spectrum,definition):
    out = spec2thpdf(gaussian_spectrum,definition=definition,paramt=[0,2,3],h=[0,1],options=OPTIONS)
    assert out.f.shape==(3,2)
    assert np.all(np.isfinite(out.f))
    assert np.all(out.f>=0)
    assert np.all(out.f[:,0]==0)


def test_full_period_driver(gaussian_spectrum):
    out = spec2tccpdf(gaussian_spectrum,paramt=[0,3,4],h1=2,h2=2,options=OPTIONS)
    assert out.f.shape==(4,)
    assert out.f[0]==out.f[1]==0
    assert np.all(np.isfinite(out.f))
    assert np.all(out.f>=0)
    assert np.all(out.err>=0)
    zero = spec2tccpdf(gaussian_spectrum,paramt=[0,3,4],h1=0,h2=2,options=OPTIONS)
    assert np.all(zero.f==0)
    unbounded = spec2tccpdf(gaussian_spectrum,paramt=[0,3,4],options=OPTIONS)
    positive = spec2tccpdf(gaussian_spectrum,definition='t>',paramt=[0,3,4],h1=0,h2=0,options=OPTIONS)
    assert_allclose(positive.f,unbounded.f)


def test_unsupported_and_invalid_inputs_are_explicit(gaussian_spectrum):
    with pytest.raises(NotImplementedError):
        spec2mmtpdf(gaussian_spectrum,definition='rfc')
    with pytest.raises(NotImplementedError):
        spec2thpdf(gaussian_spectrum,definition='TAc')
    with pytest.raises(NotImplementedError):
        spec2Acdf(gaussian_spectrum,definition='Lc')
    with pytest.raises(ValueError):
        spec2Acdf(gaussian_spectrum,h=[-1])
    transformed = dict(gaussian_spectrum,tr=np.array([[0,0],[1,1]]))
    with pytest.raises(NotImplementedError):
        spec2thpdf(transformed)
