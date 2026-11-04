"""Independent identities and source conventions for additional TRGAUSS APIs."""
import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from scipy.integrate import trapezoid,quad
from scipy.stats import norm,gamma
from wafo import trgauss as t,spec,markov


def test_legacy_markov_turning_point_recursion_matches_first_passage_engine():
    F=np.triu(np.arange(16,dtype=float).reshape(4,4),1)
    expected=markov.mctp2rfm((F,None),F.sum())
    actual=t.mctp2rfc(F[:,::-1])[:,::-1]
    assert_allclose(actual,expected,atol=1e-12)
    assert_allclose(actual.sum(),F.sum())
    assert_allclose(t.mctp2rfc(3*F[:,::-1]),3*t.mctp2rfc(F[:,::-1]))
    # General MC diagonal probability contributes directly, without old 0/0.
    diagonal=np.fliplr(np.diag([.1,.2,.3,.4]))
    assert_allclose(t.mc2rfc(diagonal),diagonal)
    general=t.mc2rfc(F[:,::-1])
    assert np.all(np.isfinite(general)) and np.all(general>=0)
    assert_allclose(t.mc2rfc(F[:,::-1]*2),general*2)
    # At N=3 the recursion has no interior system: the two off-diagonal
    # entries are exactly the closed-form Max_rfc/Min_rfc expressions.
    F3=np.array([[.2,.3,.1],[.4,.2,0],[.5,0,0]])
    dd=np.diag(np.rot90(F3));plus=F3.sum(axis=1);minus=F3.sum(axis=0)[::-1]
    div=plus+minus-2*dd
    maxrf=np.divide((plus-dd)**2,div,out=np.zeros(3),where=div>0)
    minrf=np.divide((minus-dd)**2,div,out=np.zeros(3),where=div>0)
    out=t.mc2rfc(F3)
    assert_allclose(out[1,0],maxrf[1])
    assert_allclose(out[0,1],minrf[1])


def test_fixed_point_inversion_reports_residual_and_preserves_fixed_points():
    F=np.array([[0,1.],[0,0]])
    for function,forward in ((t.iter,t.mctp2rfc),(t.iter_mc,t.mc2rfc)):
        target=forward(F[:,::-1])[:,::-1]
        fitted,reconstructed,info=function(target,F,10,1e-10,full_output=True)
        assert_allclose(fitted,F)
        assert_allclose(reconstructed,target)
        assert info['converged'] and info['residual']==0
    with pytest.raises(ValueError):t.iter(np.eye(2),k=-1)


def test_counting_distribution_mixed_difference_conventions():
    counting=np.outer(np.arange(4.),np.arange(4.))
    low=t.nt2fr(counting,0);high=t.nt2fr(counting,1)
    assert_array_equal(low[:3,:3],1)
    assert_array_equal(high[1:,1:],1)
    assert_array_equal(low[3],0)
    assert_array_equal(high[0],0)


def test_period_height_to_velocity_height_jacobian_and_layout():
    time=np.linspace(.001,15,3001);height=np.linspace(.05,10,301)
    density=np.exp(-height[:,None])*time[None,:]*np.exp(-time[None,:])
    old=spec.Spectrum(f=density,x=[time,height],labx=['T','H'],err=np.zeros_like(density))
    velocity=np.geomspace(.001,100,3001)
    converted=t.th2vhpdf(old,velocity)
    hindex=20;vindex=np.argmin(abs(velocity-1))
    H,V=height[hindex],velocity[vindex]
    expected=np.exp(-H)*H/V*np.exp(-H/V)*H/V**2
    assert_allclose(converted.f[hindex,vindex],expected,rtol=1e-5)
    # Finite V/H/T coordinates omit known probability mass. Compare the exact
    # finite-domain independent Gamma(2,1)/Exponential(1) probability, not one.
    exact=quad(lambda h:np.exp(-h)*(gamma.cdf(min(time[-1],h/velocity[0]),2)-
                  gamma.cdf(max(time[0],h/velocity[-1]),2)),height[0],height[-1])[0]
    assert_allclose(trapezoid(trapezoid(converted.f,x=velocity),x=height),exact,atol=.0004)
    python_layout=dict(old,f=density.T,err=np.zeros_like(density.T),labx=['period','height'])
    assert_allclose(t.th2vhpdf(python_layout,velocity).f,converted.f)


@pytest.fixture
def gaussian_spectrum():
    w=np.linspace(0,8,81)
    return spec.createspec(w=w,S=np.sqrt(2/np.pi)*np.exp(-w*w/2))


def test_half_period_wrappers_and_amplitude_selection(gaussian_spectrum):
    options=dict(seed=7,maxpts=1024)
    base=t.spec2thpdf(gaussian_spectrum,definition='Tc',paramt=[0,3,4],options=options)
    actual=t.spec2tpdf2(gaussian_spectrum,definition='Tc',paramt=[0,3,4],options=options)
    assert_allclose(actual.f,base.f)
    assert_allclose(t.spec2tpdf(gaussian_spectrum,paramt=[0,3,4],options=options).f,base.f)
    selected=t.spec2tpdf(gaussian_spectrum,paramt=[0,3,4],h=.5,options=options)
    assert np.all(selected.f<=base.f+base.err+selected.err)
    assert selected.f.shape==base.f.shape
    with pytest.raises(NotImplementedError):t.spec2tpdf2(gaussian_spectrum,definition='Lc')
    with pytest.raises(NotImplementedError):t.spec2tpdf(gaussian_spectrum,nit=2)


def test_cycle_probability_wrapper_and_joint_survival(gaussian_spectrum):
    options=dict(seed=7,maxpts=1024)
    rfc,mm=t.spec2cmat(gaussian_spectrum,paramt=[0,3,4],paramu=[-1,1,3],options=options,full_output=True)
    assert rfc.f.shape==mm.f.shape==(3,3)
    assert_allclose(rfc.f,t.mctp2rfc(mm.f[:,::-1])[:,::-1])
    assert np.all(mm.f>=0)
    survival=t.spec2AcAt(gaussian_spectrum,paramt=[0,3,4],h1=[0],h2=[0],options=options)
    period=t.spec2tccpdf(gaussian_spectrum,paramt=[0,3,4],h1=0,h2=0,definition='t>',options=options)
    assert_allclose(survival.f[0,0],trapezoid(period.f,x=period.x[0]))
    with pytest.raises(NotImplementedError):t.spec2cmat(gaussian_spectrum,definition='AcAt')


def test_speed_mapping_is_usable_python_rind_options():
    slow=t.initoptions(1);fast=t.initoptions(10)
    assert slow['maxpts']>fast['maxpts'] and slow['abseps']<fast['abseps']
    assert t.rindoptset(slow)==slow
    assert_allclose(t.rind(np.eye(2),[0,0],[-np.inf]*2,[0]*2,options=fast),[.25])


def test_cumulative_crossing_transform_recovers_gaussian_shape():
    x=np.linspace(-5,5,1001)
    crossing=np.column_stack((x,1e7*np.exp(-x*x/2)))
    fitted,departure,raw=t.lc2tr2(crossing,options=dict(param=[-2,2,101],ne=0,ntr=500,csm=1,gsm=1),full_output=True)
    assert_allclose(fitted[:,1],fitted[:,0],atol=.003)
    assert departure<.01 and raw.shape[1]==2
    assert 'csm' in t.lc2tr2('defaults')


def test_gaussian_monte_carlo_returns_exact_run_count_and_seed(gaussian_spectrum):
    options=dict(ne=0,param=[-2,2,51],gsm=.01)
    result,info=t.mctrtest(gaussian_spectrum,[256,3],test0=1,definition='mnonlinear',options=options,seed=7,full_output=True)
    assert result.shape==(3,) and np.all(np.isfinite(result))
    assert 0<info['pvalue']<=1
    assert_array_equal(result,t.mctrtest(gaussian_spectrum,[256,3],definition='mnonlinear',options=options,seed=7))


def test_second_order_deep_water_source_formula_and_amplitude_scaling(gaussian_spectrum):
    result=t.spec2skew(gaussian_spectrum,full_output=True)
    skew,kurt,mean,sigma,Hs,Hd=result
    w=gaussian_spectrum.w;k=spec.w2k(w)[0]
    assert_allclose(Hs,.25*(k[:,None]+k[None,:]))
    assert_allclose(Hd,-.25*abs(k[:,None]-k[None,:]))
    assert mean==0 and sigma>0 and skew>0
    assert_allclose(kurt,3+(4*skew/3)**2)
    larger=dict(gaussian_spectrum,S=4*gaussian_spectrum.S)
    new=t.spec2skew(larger)
    assert_allclose(new[0],2*skew)
    assert_allclose(new[3],2*sigma)
    for method in ('quasi','eigenvalue'):
        actual=t.spec2skew(gaussian_spectrum,method=method)
        assert np.all(np.isfinite(actual)) and actual[1]>3 and actual[3]>=sigma
    finite=t.spec2skew(gaussian_spectrum,h=20)
    assert np.all(np.isfinite(finite)) and finite[3]>0
