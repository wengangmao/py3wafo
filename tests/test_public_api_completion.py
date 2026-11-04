"""Numerical checks for public APIs absent from the previous test suite."""
import numpy as np
from scipy.integrate import trapezoid
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from scipy.special import gamma
from wafo import cycles,kdetools,misc,onedim,trgauss,wstats


def test_cycle_helpers():
    param = [0,4,5]
    d = [[0,2],[1,4]]
    assert_array_equal(cycles.dcc2cc(param,d),d)
    F = cycles.dcc2cmat(d,5)
    assert_allclose(cycles.cmat2amp(param,F),[[0,0],[.5,0],[1,1],[1.5,1],[2,0]])
    up,down = cycles.tp2mm([0,3,1,2,-1],both=True)
    assert_allclose(up,[[0,3],[1,2]])
    assert_allclose(down,[[3,1],[2,-1]])


def test_kde_bandwidth_and_kernel():
    data = np.linspace(-2,2,101)
    h = kdetools.hns(data)
    expected = min(data.std(ddof=1),2/1.349)*(4/(3*101))**.2
    assert_allclose(h,[expected])
    assert_allclose(kdetools.mkernel([0,1]),np.exp(-np.array([0,1])/2)/np.sqrt(2*np.pi))
    assert_allclose(kdetools.mkernel([[0,0],[1,1]]),[1/(2*np.pi),np.exp(-1)/(2*np.pi)])


def test_misc_numerical_utilities(tmp_path):
    assert_allclose(misc.binom([4,5],2),[6,10])
    assert_allclose(misc.sinc([0,1,.5]),[1,0,2/np.pi],atol=1e-15)
    assert_allclose(misc.hypgf(1,1,2,.5),2*np.log(2))
    x = np.linspace(0,2,101)
    assert_allclose(misc.simpson(x,x**3),4)
    val,error = misc.gaussq(lambda x:x**3,0,2)
    assert_allclose(val,4)
    assert error<1e-10
    a,b = misc.comnsize([[1],[2]],[3,4,5])
    assert a.shape==b.shape==(2,3)
    path = tmp_path/'data.txt'
    np.savetxt(path,np.eye(2))
    assert_array_equal(misc.loaddata(path),np.eye(2))


def test_crossing_types_and_spectral_estimation():
    y = [-1,1,-1,1]
    assert_array_equal(onedim.dat2crossind(y,wdef='u'),[0,2])
    assert_array_equal(onedim.dat2crossind(y,wdef='d'),[1])
    t = np.arange(1024)/128
    S = onedim.dat2spec(np.column_stack((t,np.sin(2*np.pi*8*t))),L=256)
    assert_allclose(trapezoid(S.S,S.w),.5,rtol=1e-8)
    assert_allclose(S.w[np.argmax(S.S)],2*np.pi*8)


def test_probability_and_multivariate_statistics():
    assert_allclose(trgauss.bvnormprb([-np.inf]*2,[0]*2,.5),1/3,atol=1e-5)
    assert_allclose(trgauss.mvnormpcprb([.5,.8],[0,0],[np.inf,np.inf]),.25+np.arcsin(.4)/(2*np.pi),atol=1e-5)
    assert_allclose(wstats.iqr([1,2,3,4]),1.5)
    assert_allclose(wstats.wskewness([1,2,3]),0)
    assert_allclose(wstats.wkurtosis([1,2,3]),1.5)
    assert_allclose(wstats.mvnormpdf([0,0],[0,0],np.eye(2)),1/(2*np.pi))
    assert_allclose(wstats.wnormndpdf([0,0],[0,0],np.eye(2)),1/(2*np.pi))
    samples = wstats.wmnormrnd([2,3],[[1,.2],[.2,2]],size=40000,seed=7)
    assert_allclose(samples.mean(axis=0),[2,3],atol=.025)
    assert_allclose(np.cov(samples.T),[[1,.2],[.2,2]],atol=.025)


MOMENTS = [
    ('norm',(2,4),2,4),
    ('logn',(.2,.7),np.exp(.55),np.exp(1.1)*np.expm1(.7)),
    ('weib',(2,3),2*gamma(4/3),4*(gamma(5/3)-gamma(4/3)**2)),
    ('gam',(2,3),6,18),('beta',(2,4),1/3,2*4/(6**2*7)),
    ('chi2',(3,),3,6),('exp',(2,),2,4),
    ('rayl',(3,),3*np.sqrt(np.pi/2),9*(4-np.pi)/2),
    ('f',(5,8),8/6,2*8**2*11/(5*6**2*4)),('t',(6,),0,1.5),
    ('gev',(.2,2,1),1+2*(1-gamma(1.2))/.2,4*(gamma(1.4)-gamma(1.2)**2)/.2**2),
    ('gpd',(.2,2,1),1+2/1.2,4/(1.2**2*1.4)),
    ('gumb',(2,1),1+2*np.euler_gamma,4*np.pi**2/6),
    ('frech',(2,3),2*gamma(2/3),4*(gamma(1/3)-gamma(2/3)**2)),
    ('ggam',(2,3,4),3*gamma(2.25),9*(gamma(2.5)-gamma(2.25)**2)),
    ('invg',(2,3),2,8/3),
]


@pytest.mark.parametrize('family,params,mean,variance',MOMENTS)
def test_all_distribution_moments(family,params,mean,variance):
    actual = getattr(wstats,'w'+family+'stat')(*params)
    assert_allclose(actual,[mean,variance],rtol=1e-11,atol=1e-12)


@pytest.mark.parametrize('family,params',[
    ('norm',(2,4)),('logn',(.2,.7)),('weib',(2,3)),('gam',(2,3)),
    ('beta',(2,4)),('chi2',(3,)),('exp',(2,)),('rayl',(3,)),
    ('gev',(.2,2,1)),('gpd',(.2,2,0)),('gumb',(2,1)),('t',(6,)),
])
def test_all_existing_fits_recover_quantile_parameters(family,params):
    data = getattr(wstats,'w'+family+'inv')((np.arange(4000)+.5)/4000,*params)
    fitted = np.atleast_1d(getattr(wstats,'w'+family+'fit')(data))
    assert_allclose(fitted,params,rtol=.025,atol=.012)
