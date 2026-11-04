"""Capability tests using analytic identities, independent of native binaries."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm, nct, t
from wafo.trgauss._rind import rind, rindoptset
from wafo.trgauss._probability import mvnormprb_full, mvnormpcprb_full, mvtprb, mvnortpcprb


def test_normal_probability_and_error_status():
    v,e,_,info = rind(np.eye(5),np.zeros(5),-np.inf,0,[0,5],full_output=True,seed=1)
    assert_allclose(v,[1/32],atol=1e-14)
    assert_allclose(e,[0],atol=1e-14)
    assert info['inform'][0]==0
    assert info['evaluations'][0]<=info['maxpts']
    assert_allclose(rind(np.triu([[1,.5],[.5,1]]),[0,0],[-np.inf]*2,[0]*2,seed=2,releps=1e-5),[1/3],atol=2e-5)


@pytest.mark.parametrize('rho',[-.5,0,.7])
def test_rind_positive_derivative_product(rho):
    expected = (rho*(np.pi/2+np.arcsin(rho))+np.sqrt(1-rho*rho))/(2*np.pi)
    v,e,_,info = rind([[1,rho],[rho,1]],[0,0],0,np.inf,[0,2],Nt=0,
                       seed=7,maxpts=131072,abseps=2e-5,releps=1e-4,full_output=True)
    assert_allclose(v,[expected],atol=4e-5)
    assert e[0]>=0
    assert info['evaluations'][0]<=131072


def test_absolute_derivative_with_mean_and_finite_bounds():
    # E[|N(2,9)|] has a closed form.
    expected = 3*np.sqrt(2/np.pi)*np.exp(-2**2/(2*9))+2*(2*norm.cdf(2/3)-1)
    v,e,_,_ = rind([[9]],[2],[-np.inf],[np.inf],Nt=0,full_output=True,seed=1)
    assert_allclose(v,[expected],rtol=1e-14)
    assert e[0]<1e-14
    v = rind([[1]],[0],[-1],[2],Nt=0,seed=1)
    assert_allclose(v,[2*norm.pdf(0)-norm.pdf(1)-norm.pdf(2)],rtol=1e-14)


def test_conditioning_density_and_affine_barriers():
    # X|C=c ~ N(.5*c,.75), lower=c, upper=inf.
    cs = np.array([[-1,0,2.]])
    S = [[1,.5],[.5,1]]
    out,err,_,info = rind(S,[0,0],[[0],[1]],[[np.inf],[0]],[0,1],cs,Nt=1,
                         seed=1,full_output=True)
    expected = norm.sf(.5*cs[0]/np.sqrt(.75))*norm.pdf(cs[0])
    assert_allclose(out,expected,atol=1e-13)
    assert np.all(err<1e-13)
    assert np.all(info['inform']==0)
    assert_allclose(rind(S,[0,0],[-np.inf],[np.inf],xc=cs,xcscale=2),norm.pdf(cs[0])*np.exp(2),atol=1e-13)


def test_singular_integrated_covariance_and_deterministic_variable():
    v,e,_,info = rind([[1,1],[1,1]],[0,0],[-np.inf,-np.inf],[0,0],seed=8,
                     abseps=1e-5,maxpts=65536,full_output=True)
    assert_allclose(v,[.5],atol=1e-14)
    assert info['rank']==1
    # A deterministic point at the barrier is excluded by strict inequalities.
    assert_allclose(rind([[0]],[0],[-np.inf],[0]),[0])
    # Singular conditioners have no ordinary conditioning density.
    with pytest.raises(ValueError,match='conditioning'):
        rind([[1,0],[0,0]],[0,0],[-np.inf],[0],xc=[[0]])


def test_budget_and_rare_singular_zero_are_not_false_success():
    _,_,_,info = rind([[1,.5],[.5,1]],[0,0],[-np.inf]*2,[0]*2,
        seed=7,maxpts=64,minpts=64,abseps=0,releps=0,full_output=True)
    assert info['evaluations'][0]==64
    assert info['inform'][0]==1
    # Opposite deterministic constraints can lead to all sampled zeros; without
    # an analytic feasibility proof the backend must not report success/error0.
    v,e,_,info = rind([[1,1],[1,1]],[0,0],[-np.inf,2],[0,np.inf],
        seed=2,maxpts=64,minpts=64,full_output=True)
    assert v[0]==0 and np.isinf(e[0]) and info['inform'][0]==1


def test_rind_input_checks_and_legacy_option_rejection():
    with pytest.raises(ValueError,match='semidefinite'):
        rind([[1,2],[2,1]],[0,0],0,1,[0,2])
    with pytest.raises(ValueError,match='indI'):
        rind(np.eye(2),[0,0],0,1,[1,2])
    with pytest.raises(NotImplementedError):
        rindoptset(method=3)
    with pytest.raises(TypeError):
        rindoptset(nc1c2=2)


def test_product_normal_quadrature_including_unit_correlations():
    rho = [.5,-.8]
    expected = .25+np.arcsin(np.prod(rho))/(2*np.pi)
    value,error,flag = mvnormpcprb_full(rho,[0,0],[np.inf,np.inf],tol=1e-9,releps=0)
    assert_allclose(value,expected,atol=1e-10)
    assert error<1e-9 and flag==0
    assert_allclose(mvnormpcprb_full([1,-1],[-1,-2],[2,1])[0],norm.cdf(2)-norm.cdf(-1),atol=1e-14)
    assert_allclose(mvnortpcprb([0,0],[-np.inf]*2,[1,2],D=[1,2]),.25,atol=1e-14)


def test_normal_full_output_and_mvt_normal_constraints():
    assert_allclose(mvnormprb_full(np.eye(2),[-np.inf]*2,[0]*2)[0],.25,atol=1e-14)
    # One linear constraint Z1+Z2 has variance2.
    value = mvtprb(np.eye(2),[-np.inf],[1],0,constr=[[1,1]],delta=[.5])
    assert_allclose(value,norm.cdf(.5/np.sqrt(2)),atol=1e-14)


@pytest.mark.parametrize('nu',[1,5,100])
def test_noncentral_t_univariate(nu):
    expected = nct.cdf(2,nu,1.5)-nct.cdf(-1,nu,1.5)
    actual = mvtprb([[1]],[-1],[2],nu,delta=[1.5])
    assert_allclose(actual,expected,rtol=1e-12)
    assert_allclose(mvnortpcprb([.7],[-1],[2],D=[1.5],NDF=nu),expected,rtol=1e-12)


def test_noncentral_t_shared_scale_and_sign_probability():
    # With zero numerator means the orthant is independent of chi-square scale.
    rho = .35
    expected = .25+np.arcsin(rho)/(2*np.pi)
    value,error,flag = mvtprb([[1,rho],[rho,1]],[-np.inf]*2,[0]*2,5,
        seed=12,maxpts=65536,abseps=1e-5,releps=1e-5,full_output=True)
    assert_allclose(value,expected,atol=3e-5)
    assert error>=0
    # For a rank-one vector both numerator components equal Z, so their common
    # bounded probability reduces to a univariate t probability.
    value = mvtprb([[1,1],[1,1]],[-1,-1],[2,2],5,seed=12,maxpts=262144,abseps=1e-5,releps=1e-5)
    assert_allclose(value,t.cdf(2,5)-t.cdf(-1,5),atol=5e-5)


def test_t_scale_is_shared_even_when_gaussian_coordinates_are_independent():
    # For nu=2, P(|T1|<a,|T2|<a) can be independently computed by a
    # one-dimensional chi-square mixture of normal rectangle probabilities.
    from scipy.integrate import quad
    from scipy.stats import chi2
    a = 1.2
    expected = quad(lambda v: (2*norm.cdf(a*np.sqrt(v/2))-1)**2*chi2.pdf(v,2),0,np.inf,epsabs=1e-10)[0]
    value,error,flag = mvtprb(np.eye(2),[-a]*2,[a]*2,2,seed=46,abseps=1e-5,releps=1e-5,
        maxpts=131072,full_output=True)
    assert_allclose(value,expected,atol=3e-5)
    assert abs(value-(2*t.cdf(a,2)-1)**2)>.01


def test_large_conditioned_psd_covariance_is_finite():
    tgrid = np.linspace(0,1,20)
    S = np.exp(-.5*(tgrid[:,None]-tgrid[None,:])**2)
    v,e,_,info = rind(S,np.zeros(20),[-np.inf]*18,[0]*18,
        xc=np.zeros((2,1)),Nt=18,seed=18,maxpts=2048,abseps=1e-4,full_output=True)
    assert np.isfinite(v[0]) and v[0]>=0
    assert info['evaluations'][0]<=2048
