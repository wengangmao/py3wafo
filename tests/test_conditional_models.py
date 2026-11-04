"""Independent-factor and affine-conditional identities for DIST2D APIs."""
import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from scipy import stats,integrate
from wafo import wstats as w


def rayleigh_model():
    return dict(x=[np.array([[0,2],[5,2],[10,2.]]),[3]],dist=['rayleigh','rayleigh'])


def test_independent_factor_pdf_cdf_and_rectangle():
    phat=rayleigh_model()
    v,h=np.array([.5,1,3]),np.array([1,2,4])
    jointpdf=stats.rayleigh.pdf(v,scale=2)*stats.rayleigh.pdf(h,scale=3)
    jointcdf=stats.rayleigh.cdf(v,scale=2)*stats.rayleigh.cdf(h,scale=3)
    assert_allclose(w.dist2dpdf(v,h,phat),jointpdf,rtol=1e-12)
    value,error=w.dist2dcdf(v,h,phat,full_output=True)
    assert_allclose(value,jointcdf,atol=1e-9)
    assert np.all(error>=0)
    assert_allclose(w.dist2dcdf(v,h,phat,2),stats.rayleigh.cdf(v,scale=2))
    assert_allclose(w.dist2dcdf(v,h,phat,1),stats.rayleigh.cdf(h,scale=3),atol=1e-8)
    assert_allclose(w.dist2dfun(h,v,phat,0),stats.rayleigh.pdf(h,scale=3)*stats.rayleigh.cdf(v,scale=2))
    assert_allclose(w.dist2dfun(h,v,phat,3),stats.rayleigh.pdf(v,scale=2))
    rect,error=w.dist2dprb(phat,1,4,2,5,full_output=True)
    expected=(stats.rayleigh.cdf(4,scale=2)-stats.rayleigh.cdf(1,scale=2))*(stats.rayleigh.cdf(5,scale=3)-stats.rayleigh.cdf(2,scale=3))
    assert_allclose(rect,expected,atol=1e-10)
    assert error>=0
    grid=w.dist2dpdf2([1,2,3],[2,4],phat)
    assert grid.f.shape==(2,3)
    assert_allclose(grid.f[0],w.dist2dpdf([1,2,3],2,phat))


def test_parameter_smoothing_and_optional_location_column():
    phat=dict(x=[np.array([[0,1,2],[1,2,3],[2,3,4.]]),[2]],dist=['rayl','rayl'])
    a,b,c=w.dist2dsmfun(phat,[.5,1.5])
    assert_allclose(a,[1.5,2.5]);assert_allclose(b,[0,0]);assert_allclose(c,[2.5,3.5])
    assert_allclose(w.dist2dcdf([2.4,3.4],[.5,1.5],phat,2),[0,0])
    smooth=w.dist2dsmfun2(phat,[.25,.75,1.5],csm=0)
    assert_allclose(smooth['x'][0],[[.25,1.25,2.25],[.75,1.75,2.75],[1.5,2.5,3.5]])
    assert phat['x'][0].shape==(3,3)
    with pytest.raises(NotImplementedError):w.dist2dsmfun2(phat,visual=1)


def test_marginal_conditional_moments_and_random_pairs():
    phat=rayleigh_model()
    mean,cov=w.dist2dstat(phat)
    expectedmean=np.sqrt(np.pi/2)*np.array([2,3])
    expectedvar=(4-np.pi)/2*np.array([4,9])
    assert_allclose(mean,expectedmean,atol=1e-7)
    assert_allclose(cov,np.diag(expectedvar),atol=1e-7)
    mean1,var1=w.dist2dstat(phat,2,[1,3])
    assert_allclose(mean1,[expectedmean[0]]*2)
    assert_allclose(var1,[expectedvar[0]]*2)
    mean2,var2=w.dist2dstat(phat,1,[1,3])
    assert_allclose(mean2,[expectedmean[1]]*2,atol=1e-7)
    assert_allclose(var2,[expectedvar[1]]*2,atol=1e-7)
    v,h=w.dist2drnd(40000,phat,seed=42)
    assert_allclose([v.mean(),h.mean()],expectedmean,atol=.025)
    assert_allclose([v.var(),h.var()],expectedvar,atol=.08)
    v2,h2=w.dist2drnd(40000,phat,seed=42)
    assert_array_equal(v,v2);assert_array_equal(h,h2)


def test_affine_normal_conditional_covariance_and_negative_support():
    # H~N(1,4), V|H=h~N(2+3h,5): joint covariance is analytic.
    levels=np.array([-10,0,10.])
    phat=dict(x=[np.column_stack((levels,2+3*levels,np.full(3,5.))),[1,4]],dist=['normal','normal'])
    mean,cov=w.dist2dstat(phat)
    assert_allclose(mean,[5,1],atol=1e-7)
    assert_allclose(cov,[[41,12],[12,4]],atol=1e-6)
    value=w.dist2dcdf(-2,-1,phat)
    expected=stats.multivariate_normal(mean=[5,1],cov=[[41,12],[12,4]],seed=np.random.default_rng(7)).cdf([-2,-1])
    assert_allclose(value,expected,atol=1e-5)


@pytest.mark.parametrize('family,params',[
    ('weibull',[2,3]),('gamma',[2,3]),('ggamma',[2,3,2]),
    ('lognormal',[-.2,.7]),('gumbel',[2,-1]),('tgumbel',[2,-1]),
    ('tweibull',[2,3,1]),('trayleigh',[2,-1])])
def test_all_documented_conditional_families_normalize_and_sample(family,params):
    table=np.array([[0,*params],[5,*params]],dtype=float)
    phat=dict(x=[table,[2]],dist=[family,'rayleigh'])
    lo=-np.inf if family=='gumbel' else 0
    density=lambda value:float(w.dist2dfun(1,value,phat,3))
    integral=integrate.quad(density,lo,np.inf,epsabs=1e-7)[0]
    assert_allclose(integral,1,atol=1e-6)
    v,h=w.dist2drnd(50,phat,seed=3)
    assert np.all(np.isfinite(v)) and np.all(np.isfinite(h))
    cdf=w.dist2dcdf(v,h,phat,2)
    assert np.all((cdf>=0)&(cdf<=1))


def test_binned_mle_fitting_and_rejecting_unimplemented_modes():
    rng=np.random.default_rng(3)
    h=rng.rayleigh(2,4000)
    v=rng.rayleigh(1+.1*h)
    fit=w.dist2dfit(v,h,['rayl','rayl'],res=[.6,1])
    assert fit['x'][0].shape[1]==2
    assert_allclose(fit['x'][1],[2],atol=.05)
    a,_,_=w.dist2dsmfun(fit,[1,2,3])
    assert_allclose(a,[1.1,1.2,1.3],atol=.08)
    with pytest.raises(NotImplementedError):w.dist2dfit(v,h,['rayl','rayl'],method='MOM')
    with pytest.raises(ValueError):w.dist2dcdf(1,1,fit,condon=8)
    with pytest.raises(ValueError):w.dist2dprb(fit,2,1,0,1)


def test_joint_infinite_boundaries_and_model_validation():
    phat=rayleigh_model()
    assert_allclose(w.dist2dcdf(np.inf,np.inf,phat),1,atol=1e-8)
    assert_allclose(w.dist2dcdf(-np.inf,np.inf,phat),0)
    assert_allclose(w.dist2dcdf(np.inf,-np.inf,phat),0)
    assert_allclose(w.dist2dpdf([1,1],[np.inf,-np.inf],phat),[0,0])
    with pytest.raises(ValueError):w.dist2dfun(np.inf,1,phat,2)
    invalid=dict(phat,x=[phat['x'][0],[-1]])
    with pytest.raises(ValueError):w.dist2dpdf(1,1,invalid)


def test_plackett_marginal_and_conditional_moments():
    independent=dict(x=[[2],[3],1],dist=['rayl','rayl'])
    mean,cov=w.mdist2dstat(independent)
    expectedmean=np.sqrt(np.pi/2)*np.array([2,3])
    expectedvar=(4-np.pi)/2*np.array([4,9])
    assert_allclose(mean,expectedmean)
    assert_allclose(cov,np.diag(expectedvar))
    for selector,number in [(1,1),(2,0)]:
        conditionalmean,conditionalvariance=w.mdist2dstat(independent,selector,[1,3])
        assert_allclose(conditionalmean,[expectedmean[number]]*2,atol=1e-7)
        assert_allclose(conditionalvariance,[expectedvar[number]]*2,atol=1e-7)
    dependent=dict(independent,x=[[2],[3],4])
    _,cov=w.mdist2dstat(dependent,quadrature=192)
    refined=w.mdist2dstat(dependent,quadrature=256)[1]
    assert_allclose(cov,refined,atol=.002)
    x,y=w.mdist2drnd(dependent,size=12000,seed=17)
    assert_allclose(cov,np.cov(np.stack((x,y))),atol=.15)


def test_plackett_marginal_and_simultaneous_mle():
    from scipy.stats import qmc
    model=dict(x=[[2],[3],3],dist=['rayl','rayl'])
    uniforms=qmc.Sobol(2,scramble=True,seed=7).random_base2(10)
    x=stats.rayleigh.ppf(uniforms[:,0],scale=2)
    y=w.mdist2dcinv(x,uniforms[:,1],model)
    mml=w.mdist2dfit(x,y,['rayl','rayl'],method='MML')
    assert_allclose(np.r_[mml['x'][0],mml['x'][1],mml['x'][2]],[2,3,3],rtol=.04)
    sml=w.mdist2dfit(x,y,['rayl','rayl'],method='SML')
    assert w.mdist2dlike(sml,x,y)<=w.mdist2dlike(mml,x,y)+1e-7
    assert_allclose(np.r_[sml['x'][0],sml['x'][1],sml['x'][2]],[2,3,3],rtol=.05)
    with pytest.raises(NotImplementedError):w.mdist2dfit(x,y,['rayl','rayl'],alpha=.05)
