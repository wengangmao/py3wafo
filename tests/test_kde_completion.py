import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from scipy.integrate import simpson
from scipy.stats import norm
from wafo import kdetools as kd


@pytest.mark.parametrize('kernel',['epanechnikov','biweight','triweight','triangular',
    'gaussian','rectangular','laplace','logistic'])
def test_kernel_normalization_and_source_moments(kernel):
    compact = kernel not in ('gaussian','laplace','logistic')
    x = np.linspace(-1 if compact else -40,1 if compact else 40,40001)
    k = kd.mkernel(x,kernel)
    mu2,R,Rdd = kd.kernelstats(kernel)
    assert np.all(k>=0) and Rdd>0
    assert_allclose(simpson(k,x=x),1,atol=1e-6)
    assert_allclose(simpson(x*x*k,x=x),mu2,rtol=1e-5)
    assert_allclose(simpson(k*k,x=x),R,rtol=1e-5)
    h = kd.hns(np.linspace(-1,1,51),kernel)
    assert np.all(h>0)
    assert_allclose(kd.hos(np.linspace(-1,1,51),kernel),h/.93)


@pytest.mark.parametrize('kernel',['epan','biwe','triw','epa1','biw1','tri1','sphe','four','fou1'])
def test_radial_and_product_kernel_normalization(kernel):
    x = np.linspace(-1,1,601)
    xx,yy = np.meshgrid(x,x,indexing='xy')
    density = kd.mkernel(np.stack((xx,yy),axis=-1),kernel)
    mass = simpson(simpson(density,x=x,axis=1),x=x)
    assert_allclose(mass,1,atol=3e-4)


def test_derivatives_and_multivariate_bandwidth():
    fourth,sixth,eighth,tenth = kd.deriv(0)
    assert_allclose([fourth,sixth,eighth,tenth],np.array([3,-15,105,-945])/np.sqrt(2*np.pi))
    assert_allclose(kd.deriv2(0,0,'42'),-3/(2*np.pi))
    rng = np.random.default_rng(7)
    a = rng.normal(size=(1000,2))@np.array([[1,.3],[0,2]])
    H = kd.hmns(a)
    expected = np.cov(a,rowvar=False)*1000**(-2/6)
    assert_allclose(H@H,expected)
    assert_allclose(kd.hmns(a[:,0]),kd.hns(a[:,0]))


def test_counts_preserve_mass_and_first_moments():
    count,unique,sums = kd.bincount([2,1,2,3],[1,2,3,4])
    assert_array_equal(unique,[1,2,3])
    assert_array_equal(count,[1,2,1])
    assert_allclose(sums,[2,4,4])
    assert_array_equal(kd.bincount([])[0],[])
    x = np.arange(5.)
    data = [.25,1.5,4]
    count = kd.gridcount(data,x)
    assert_allclose(count.sum(),3)
    assert_allclose(count@x,sum(data))
    xy = np.column_stack((x,x))
    data = np.array([[.2,1.3],[2.5,.6],[4,4]])
    count = kd.gridcount(data,xy)
    xx,yy = np.meshgrid(x,x,indexing='xy')
    assert_allclose(count.sum(),3)
    assert_allclose((count*xx).sum(),data[:,0].sum())
    assert_allclose((count*yy).sum(),data[:,1].sum())


def test_high_density_quantile_levels_and_empirical_quantiles():
    x = np.linspace(-8,8,16001)
    p = np.array([10,50,90,99])
    result = kd.qlevels(norm.pdf(x),p,x)
    expected = norm.pdf(norm.ppf((1-p/100)/2))
    assert_allclose(result,expected,atol=5e-4)
    assert_allclose(kd.qlevels2([1,2,3,4],[25,50,75]),[3.5,2.5,1.5])
    with pytest.raises(ValueError):
        kd.gridcount([5],np.arange(5))
