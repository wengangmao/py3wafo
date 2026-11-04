import numpy as np
import pytest
from scipy.stats import norm
from scipy.integrate import simpson
from numpy.testing import assert_allclose,assert_array_equal
from wafo import kdetools as kd


def test_exact_kde_matches_independent_normal_mixture():
    data=np.array([-2,-1,.3,.7,2])
    x=np.linspace(-5,5,101)
    f,H,factors=kd.kdefun(data,{'kernel':'gaussian','hs':.4},x,full_output=True)
    expected=norm.pdf((x[:,None]-data[None,:])/.4).mean(axis=1)/.4
    assert_allclose(f,expected)
    assert_allclose(H,[[.4]]); assert_array_equal(factors,np.ones(5))
    assert kd.kdeoptset('kdefun','hs',.4)['hs']==.4


@pytest.mark.parametrize('alpha',[0,.5,1])
def test_log_transformed_adaptive_kde_is_normalized(alpha):
    data=np.exp(norm.ppf((np.arange(100)+.5)/100))
    x=np.geomspace(.00001,1000,10001)
    f,H,factors=kd.kdefun(data,{'kernel':'gaussian','hs':.3,'L2':0,'alpha':alpha},x,full_output=True)
    # Adaptive tail kernels can have appreciable mass beyond a fixed [1e-5,1e3]
    # interval. Compare that finite-domain mass to its exact mixture CDF first.
    exact_mass=np.mean(norm.cdf((np.log(x[-1])-np.log(data))/(.3*factors))-
                       norm.cdf((np.log(x[0])-np.log(data))/(.3*factors)))
    assert_allclose(simpson(f,x=x),exact_mass,atol=1e-7)
    log_bound=max(abs(np.log(data)))+12*.3*max(factors)
    wide=np.exp(np.linspace(-log_bound,log_bound,20001))
    whole=kd.kdefun(data,{'kernel':'gaussian','hs':.3,'L2':0,'alpha':alpha},wide)
    assert_allclose(simpson(whole,x=wide),1,atol=1e-6)
    assert np.all(factors>0)
    assert_allclose(np.exp(np.mean(np.log(factors))),1)


def test_full_matrix_kde_and_grid_interpolation():
    data=np.array([[-1,-1],[0,.3],[1,1],[.5,-.5]])
    H=np.array([[.5,.1],[.1,.8]])
    x=np.linspace(-6,6,201); xx,yy=np.meshgrid(x,x)
    f=kd.kdefun(data,{'kernel':'gaussian','hs':H},xx,yy)
    assert_allclose(simpson(simpson(f,x=x),x=x),1,atol=1e-6)
    pdf=kd.kdebin(data,{'kernel':'gaussian','hs':H,'inc':21},[-3,-3],[3,3])
    assert_allclose(kd.evalpdf(pdf,pdf.x[0][5],pdf.x[1][7]),pdf.f[7,5])
    assert np.isnan(kd.evalpdf(pdf,100,100))
    curve=kd.kdebin(data[:,0],{'kernel':'gaussian','hs':.5,'inc':31},-3,3)
    assert_allclose(kd.evalpdf(curve,curve.x[0][9],'nearest'),curve.f[9])


def test_fft_embedding_axes_and_bandwidth_scaling():
    assert_array_equal(kd.fftce([1,2,3,4]),[1,2,3,4,3,2])
    matrix=np.array([[1,2,3],[4,5,6],[7,8,9]])
    result=kd.fftce(matrix)
    assert result.shape==(4,4)
    assert_array_equal(result[-1,:],[4,5,6,5])
    data=norm.ppf((np.arange(150)+.5)/150)
    for L in (0,1,2,3):
        h=kd.hldpi(data,L=L)
        assert np.all(h>0)
        assert_allclose(kd.hldpi(3*data+2,L=L),3*h,rtol=1e-10)


def test_cross_validation_scores_have_source_objective_and_valid_minima():
    data=norm.ppf((np.arange(60)+.5)/60)
    grid=np.linspace(.15,.8,20)
    for function in (kd.hbcv,kd.hlscv):
        h,candidates,score=function(data,hvec=grid,full_output=True)
        assert h==candidates[np.argmin(score)] and np.all(np.isfinite(score))
    h=.5; z=(data[:,None]-data[None,:])/h
    fourth=(z**4-6*z*z+3)*norm.pdf(z)
    np.fill_diagonal(fourth,0)
    expected=1/(2*np.sqrt(np.pi)*len(data)*h)+h**4*fourth.sum()/(4*len(data)**2*h**5)
    assert_allclose(kd.hbcv(data,hvec=[h],full_output=True)[2],[expected])


def test_invalid_kde_options_are_explicit():
    with pytest.raises(ValueError):
        kd.kdefun([1,2,3],{'hs':-.5},[1])
    with pytest.raises(ValueError):
        kd.kdefun([-1,1,2],{'L2':0,'hs':.5},[1])
    with pytest.raises(ValueError):
        kd.kdeoptset(alpha=2)
