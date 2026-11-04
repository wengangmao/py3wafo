"""Wave-model density normalization, identities and repaired-source regressions."""
import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from scipy import integrate,optimize,stats
from scipy.special import jv
from wafo import wavemodels as w


@pytest.mark.parametrize('a,b',[(2.5,2),(2,2),(0,2),(2,0)])
def test_ochi_repaired_density_matches_cdf_and_normalizes(a,b):
    x=np.linspace(.01,12,1001)
    pdf=w.ochi98pdf(x,a,b)
    derivative=(w.ochi98cdf(x+1e-5,a,b)-w.ochi98cdf(x-1e-5,a,b))/2e-5
    assert_allclose(pdf,derivative,rtol=2e-6,atol=1e-10)
    integral=integrate.quad(lambda x:float(w.ochi98pdf(x,a,b)),0,np.inf)[0]
    assert_allclose(integral,1,atol=2e-9)
    assert_allclose(w.ochi98cdf([-np.inf,0,np.inf],a,b),[0,0,1])


def test_ochi_equal_scale_is_continuous_and_matches_independent_amplitudes():
    assert_allclose(w.ochi98cdf([1,2,5],2,2),w.ochi98cdf([1,2,5],2.00000001,2),atol=1e-8)
    a,b=2.5,1.
    expected=integrate.quad(lambda z:stats.rayleigh.pdf(z,scale=a/np.sqrt(2))*stats.rayleigh.cdf(3-z,scale=b/np.sqrt(2)),0,3)[0]
    assert_allclose(w.ochi98cdf(3,a,b),expected,atol=1e-12)
    p=(np.arange(300)+.5)/300
    data=np.array([optimize.brentq(lambda x:float(w.ochi98cdf(x,a,b))-q,0,15) for q in p])
    fitted=w.ochi98fit(data)
    assert_allclose(w.ochi98cdf(data,*fitted),p,atol=.005)


def test_beta_rayleigh_fixed_and_free_endpoint_fits():
    p=(np.arange(1000)+.5)/1000
    data=3*np.sqrt(stats.beta.ppf(p,2,4))
    assert_allclose(w.braylfit(data,c=3),[2,4,3],rtol=.002)
    fitted=w.braylfit(data)
    assert fitted[2]>data.max() and fitted[1]>1
    assert_allclose(w.braylcdf(data,*fitted),p,atol=.015)


def test_transformed_rayleigh_identity_and_linear_scaling():
    x=np.linspace(0,8,1001)
    identity=w.trraylpdf(x)
    assert_allclose(identity.f,stats.rayleigh.pdf(x))
    assert_allclose(w.trraylpdf(x,'At').f,identity.f)
    assert_allclose(w.trraylpdf(x,'AcAt').f,stats.rayleigh.pdf(x,scale=2))
    table=np.column_stack((np.linspace(-20,20,101),np.linspace(-20,20,101)/2))
    assert_allclose(w.trraylpdf(x,'Ac',table).f,stats.rayleigh.pdf(x,scale=2),atol=1e-12)
    assert_allclose(w.trraylpdf(x,'AcAt',table).f,stats.rayleigh.pdf(x,scale=4),atol=1e-12)


def test_joint_approximate_models_are_densities_and_linear_transform_equivalent():
    t=np.r_[0,np.geomspace(.005,1000,2500)]
    h=np.linspace(0,8,801)
    for function,mom in [(w.cav76pdf,[1,1,3]),(w.lh83pdf,[1,.8,1])]:
        result=function(t,h,mom)
        assert result.f.shape==(len(h),len(t)) and np.all(result.f>=0)
        mass=integrate.simpson(integrate.simpson(result.f,x=t,axis=1),x=h)
        assert_allclose(mass,1,atol=.004)
        table=np.column_stack((np.linspace(-20,20,101),np.linspace(-20,20,101)))
        assert_allclose(function([0,1,3],[0,1,2],mom,table).f,function([0,1,3],[0,1,2],mom).f,atol=1e-12)


def test_tayfun_1990_density_cdf_normalization_and_rayleigh_limit():
    for correlation in (0,.5,.95):
        integral=integrate.quad(lambda x:float(w.tay90pdf(x,2,correlation)),0,30,epsabs=1e-7)[0]
        assert_allclose(integral,1,atol=2e-6)
        x=np.array([1.,3.,6.])
        derivative=(w.tay90cdf(x+1e-4,2,correlation)-w.tay90cdf(x-1e-4,2,correlation))/2e-4
        assert_allclose(derivative,w.tay90pdf(x,2,correlation),rtol=2e-6,atol=1e-8)
    assert_allclose(w.tay90pdf([0,2,4],2,1),stats.rayleigh.pdf([0,2,4],scale=4))
    assert_allclose(w.tay90cdf([0,2,4],2,-1),stats.rayleigh.cdf([0,2,4],scale=4))
    values,error=w.tay90pdf([1,2],2,.3,full_output=True)
    assert np.all(error>=0) and np.all(np.isfinite(values))
    # Source weighted form and endpoint substitution must produce the same PDF.
    x,b=2,.3
    weighted=integrate.quad(lambda u:float(w.tay90fun(u,x,b))*np.sqrt(u/(1-u)),0,1)[0]
    assert_allclose(weighted,w.tay90pdf(x,1,b),rtol=1e-9)


def test_tayfun_1981_integrand_matches_documented_bessel_form():
    u=np.linspace(0,20,101)
    assert_allclose(w.tay81fun(u,1.5,4),jv(0,1.5*u)*u*jv(0,u/2)**4,atol=1e-14)
    assert_allclose(w.tay81fun(u,1.5,4,'cdf'),jv(1,1.5*u)*u*1.5*jv(0,u/2)**4,atol=1e-14)


def test_mk87_marginals_conditional_jacobians_and_probabilities():
    Hs,Tz=6,8
    Hrms,Erms=.715*Hs,.0202+.826*Hs/Tz**2
    h=3.
    density=integrate.quad(lambda s:float(w.mk87pdf(h,s,Hs,Tz)),0,np.inf)[0]
    assert_allclose(density,stats.weibull_min.pdf(h,2.39,scale=1.05*Hrms),rtol=1e-8)
    conditional=integrate.quad(lambda s:float(w.mk87pdf(h,s,Hs,Tz,1)),0,np.inf)[0]
    assert_allclose(conditional,1,rtol=1e-8)
    s=.12
    assert_allclose(w.mk87pdf(h,s,Hs,Tz,3),s*w.mk87pdf(h,s,Hs,Tz,1))
    assert_allclose(w.mk87pdf(h,s,Hs,Tz,4),s*s*w.mk87pdf(h,s,Hs,Tz,1))
    assert_allclose(w.mk87pdf(h,s,Hs,Tz,5)+w.mk87pdf(h,s,Hs,Tz,7),density)
    assert 0<w.mk87pdf(h,s,Hs,Tz,6)<1
    assert_allclose(w.mk87cdf(np.inf,np.inf,Hs,Tz),1,atol=1e-9)
    assert_allclose(w.mk87cdf(np.inf,np.inf,Hs,Tz,1),0)
    assert_allclose(w.mk87cdf(np.inf,s,Hs,Tz)+w.mk87cdf(0,s,Hs,Tz,1),1,atol=1e-8)
    assert_allclose(w.mk87cdf(h,np.inf,Hs,Tz),stats.weibull_min.cdf(h,2.39,scale=1.05*Hrms),atol=1e-9)
    grid=w.mk87pdf2([1,2,3],[.02,.04],Hs,Tz)
    assert grid.f.shape==(3,2)
    rng_h,rng_s=w.mk87rnd(Hrms,Erms,40000,seed=7)
    expected=float(w.mk87cdf(h,s,Hs,Tz))
    assert_allclose(np.mean((rng_h<=h)&(rng_s<=s)),expected,atol=.012)
    h2,s2=w.mk87rnd(Hrms,Erms,40000,seed=7)
    assert_array_equal(rng_h,h2);assert_array_equal(rng_s,s2)


def test_invalid_model_parameters_are_explicit():
    assert np.isnan(w.ochi98pdf(1,-1,2))
    with pytest.raises(ValueError):w.cav76pdf(mom=[1,1,1])
    with pytest.raises(ValueError):w.lh83pdf(mom=[1,1,1])
    with pytest.raises(ValueError):w.mk87pdf(1,1,Hs=0)
    with pytest.raises(ValueError):w.trraylpdf([1],definition='unknown')


@pytest.mark.parametrize('pdf,cdf,grid,rayleigh,A0,B0',[
    (w.b04pdf,w.b04cdf,w.b04pdf2,.70079568211392,2.92552351922864,np.exp(-1.96751680710343)),
    (w.b04jpdf,w.b04jcdf,w.b04jpdf2,.67768680795379,4.4906025931643,np.exp(-2.47501962948775)),
    (w.bmr00pdf,w.bmr00cdf,w.bmr00pdf2,.69233682309018,3.03590705490789,np.exp(-1.87015799594426)),
])
def test_source_empirical_gamma_models_and_dimensionally_correct_conditionals(pdf,cdf,grid,rayleigh,A0,B0):
    assert_allclose(pdf(0,.7,condon=1,normalizedInput=True),stats.gamma.pdf(.7,A0,scale=B0),rtol=1e-12)
    Hs,Tz=6,8
    Hscale,Sscale=Hs/np.sqrt(2),1.25*Hs/Tz**2
    for height in (1,5,25):
        marginal=stats.rayleigh.pdf(height,scale=Hscale*rayleigh)
        total=integrate.quad(lambda s:float(pdf(height,s,Hs,Tz)),0,np.inf)[0]
        assert_allclose(total,marginal,atol=1e-9,rtol=2e-6)
    assert_allclose(integrate.quad(lambda s:float(pdf(3,s,Hs,Tz,1)),0,np.inf)[0],1,atol=1e-8)
    assert_allclose(cdf(np.inf,np.inf,Hs,Tz),1,atol=1e-9)
    assert_allclose(cdf(np.inf,np.inf,Hs,Tz,tail=1),0)
    assert_allclose(cdf(3,np.inf,Hs,Tz),stats.rayleigh.cdf(3,scale=Hscale*rayleigh),atol=1e-9)
    assert_allclose(cdf(np.inf,.1,Hs,Tz)+cdf(0,.1,Hs,Tz,tail=1),1,atol=1e-8)
    assert grid([1,2],[.01,.02,.03],Hs,Tz).f.shape==(2,3)
    assert_allclose(pdf(2,.1,Hs,Tz,3),.1*pdf(2,.1,Hs,Tz,1))
    assert_allclose(pdf(2,.1,Hs,Tz,4),.01*pdf(2,.1,Hs,Tz,1))
    assert_allclose(pdf(2,.1,Hs,Tz,5)+pdf(2,.1,Hs,Tz,7),stats.rayleigh.pdf(2,scale=Hscale*rayleigh))
    assert 0<=pdf(2,.1,Hs,Tz,6)<=1


def test_source_jonswap_height_parameter_models():
    gamma=np.array([1,3,7])
    A,B,C=w.jhwparfun(6,10,gamma)
    assert_allclose(A,-.00768035933493*gamma+1.12723230522407)
    assert_allclose(B,-.02261969635571*gamma+2.31625911055348)
    assert_allclose(C,-.00808703761189*gamma+.14438532196018)
    A,B,C=w.jhnlwparfun(6,10,gamma)
    assert_allclose(A,-.01243795213128*gamma+1.08025514722235)
    assert_allclose(B,-.03043834819688*gamma+2.27161821064622)
    assert_allclose(C,-.01342959276544*gamma+.10353423379696)
    assert np.all(A>0) and np.all(B>0) and np.all(C>0)
    assert all(np.isfinite(w.jhwparfun(6,10)))
    assert all(np.isfinite(w.jhnlwparfun(6,10)))
    with pytest.raises(ValueError):w.jhwparfun(6,10,9)
    with pytest.raises(NotImplementedError):w.jhnlwparfun(6,10,3,'space')


@pytest.mark.parametrize('function,name',[
    (w.thgparfun,'thgpar.mat'),(w.thwparfun,'thwnlpar20-Jul-2004.mat')])
def test_source_height_table_knots_and_broadcasting(function,name):
    from wafo.wavemodels._tables import _data
    d=_data(name)
    i,j=10,15
    expected=[d[key][i,j] for key in ('A00s','B00s','C00s')]
    assert_allclose(function(d['Hm0'].ravel()[i],d['Tp'].ravel()[j]),expected,rtol=1e-12)
    assert_allclose(np.asarray(function([6,7],[10,12])).shape,(3,2))
    with pytest.raises(ValueError):function(20,10)
    with pytest.raises(ValueError):function(6,25)
    assert all(np.all(value>0) for value in function([6,7],[10,12],'space'))


def test_ochi_hubble_rational_coefficient_branches_and_domains():
    from wafo.wavemodels._tables import _oh_coefficients
    co=_oh_coefficients()
    for dim,branch in [('space',0),('time',1)]:
        actual=w.ohhgparfun(6,1,dim)
        expected=[np.exp(np.polyval(co['d'+letter+'1'][branch][0],6)/np.polyval(co['d'+letter+'2'][branch][0],6)) for letter in 'abc']
        assert_allclose(actual,expected,rtol=1e-14)
        assert np.all(np.asarray(w.ohhgparfun(6,np.arange(1,12),dim))>0)
    with pytest.raises(ValueError):w.ohhgparfun(6,0)
    with pytest.raises(ValueError):w.ohhgparfun(13)


@pytest.mark.parametrize('pdf,cdf,parameters',[
    (w.thpdf,w.thcdf,(6,10)),(w.ohhpdf,w.ohhcdf,(6,1)),
    (w.thpdf,w.thcdf,(6,10,'space')),(w.ohhpdf,w.ohhcdf,(6,1,'space'))])
def test_table_marginal_pdf_cdf_consistency_and_normalization(pdf,cdf,parameters):
    # Generalized-gamma fits can have an integrable singularity at zero.
    mass=integrate.quad(lambda value:float(pdf(value,*parameters)),0,np.inf,epsabs=1e-8,limit=200)[0]
    assert_allclose(mass,1,atol=3e-6)
    assert_allclose(cdf([0,np.inf],*parameters),[0,1],atol=1e-12)
    points=np.array([2,5,8.])
    derivative=(cdf(points+1e-4,*parameters)-cdf(points-1e-4,*parameters))/2e-4
    assert_allclose(derivative,pdf(points,*parameters),rtol=2e-6,atol=1e-9)


@pytest.mark.parametrize('pdf,cdf,grid,nonlinear',[
    (w.jhspdf,w.jhscdf,w.jhspdf2,False),
    (w.jhsnlpdf,w.jhsnlcdf,w.jhsnlpdf2,True),
    (w.jhvpdf,w.jhvcdf,w.jhvpdf2,False)])
def test_jonswap_table_joint_density_scaling_and_tail_identities(pdf,cdf,grid,nonlinear):
    H,T,gam=6,10,3
    if nonlinear:
        from wafo.spec import getjonswappeakedness
        gam=float(getjonswappeakedness(H,T))
    height,velocity=4,.15
    Hrms=H/np.sqrt(2)
    p=(w.jhnlwparfun if nonlinear else w.jhwparfun)(H,T,gam)
    from wafo.wstats import wtweibpdf,wtweibcdf
    marginal=float(wtweibpdf(height/Hrms,*p)/Hrms)
    value=integrate.quad(lambda v:float(pdf(height,v,H,T,gam)),0,np.inf,epsabs=1e-10)[0]
    assert_allclose(value,marginal,atol=2e-9)
    assert_allclose(integrate.quad(lambda v:float(pdf(height,v,H,T,gam,False,1)),0,np.inf)[0],1,atol=1e-8)
    assert_allclose(pdf(height,velocity,H,T,gam,False,3),velocity*pdf(height,velocity,H,T,gam,False,1))
    assert_allclose(pdf(height,velocity,H,T,gam,False,4),velocity**2*pdf(height,velocity,H,T,gam,False,1))
    assert_allclose(pdf(height,velocity,H,T,gam,False,5)+pdf(height,velocity,H,T,gam,False,7),marginal)
    assert 0<=pdf(height,velocity,H,T,gam,False,6)<=1
    assert_allclose(cdf(height,np.inf,H,T,gam),wtweibcdf(height/Hrms,*p),atol=2e-8)
    assert_allclose(cdf(np.inf,np.inf,H,T,gam),1,atol=1e-12)
    assert_allclose(cdf(np.inf,np.inf,H,T,gam,tail=1),0,atol=1e-12)
    assert_allclose(cdf(np.inf,velocity,H,T,gam)+cdf(0,velocity,H,T,gam,tail=1),1,atol=3e-7)
    assert grid([1,2],[.02,.1,.2],H,T,gam).f.shape==(2,3)
    assert np.asarray(pdf([1,2],[.02,.1],H,T,gam)).shape==(2,)
    with pytest.raises(ValueError):pdf(1,.1,H,T,9)


def test_conditional_joint_table_knot_fixtures():
    from wafo.wavemodels._tables import _data,_conditional_parameters
    d=_data('jhspar13-Jul-2004.mat')
    h,g=d['h2'].ravel()[7],d['gam'].ravel()[4]
    a,b=_conditional_parameters('jhs',h,6,10,g)
    assert_allclose([a,b],[d['A11s'][7,4],d['B11s'][7,4]],rtol=1e-12)
    d=_data('jhsnlpar21-Jul-2004.mat')
    i,j,k=8,12,9
    a,b=_conditional_parameters('jhsnl',d['h2'].ravel()[k],d['Hm0'].ravel()[i],d['Tp'].ravel()[j],3)
    assert_allclose([a,b],[d['A11s'][i,j,k],d['B11s'][i,j,k]],rtol=1e-12)
