import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from wafo import misc,__version__


def test_option_merging_prefixes_and_errors():
    initial = {'alpha':0,'alpine':1,'beta':2}
    result = misc.parseoptions(initial,{'beta':None},'ALPH',3,beta=4)
    assert result==dict(alpha=3,alpine=1,beta=4) and initial['alpha']==0
    with pytest.raises(ValueError):
        misc.parseoptions(initial,'al',4)


def test_two_dimensional_quadrature_and_rules():
    x,y,w = misc.qrule2d(4,3)
    assert_allclose(np.sum(w*x**4*y**2),4/15)
    value,error = misc.gaussq2d(lambda x,y:x*x*y,0,2,0,3)
    assert_allclose(value,12)
    assert error<1e-8
    value,error = misc.ccquad(np.exp,0,1,10)
    assert_allclose(value,np.expm1(1),atol=1e-13)
    assert error<1e-10


def test_source_savgol_coefficient_moments_and_convolution():
    c = misc.savgol(2,2,2)
    assert_allclose(c.sum(),1)
    t = np.arange(30.)
    assert_allclose(misc.convlv(t,c),t,atol=1e-12)
    smoothed = misc.convlv(t*t,c)
    assert_allclose(smoothed[2:-2],t[2:-2]**2,atol=1e-11)
    d = misc.savgol(2,2,4,1)
    assert_allclose(misc.convlv(t*t,d)[2:-2],2*t[2:-2],atol=1e-10)
    wave = np.sin(np.arange(100)*2*np.pi/100)
    expected = np.cos(np.arange(100)*2*np.pi/100)
    assert_allclose(misc.convlv(wave,d,flag='periodic')/(2*np.pi/100),expected,atol=1e-6)


@pytest.mark.parametrize('n',[2,3,4,8])
def test_smoothing_endpoints_and_variational_penalty(n):
    x = np.linspace(0,1,n)
    y = x*x+.1*np.sin(20*x)
    assert_allclose(misc.smooth(x,y,1,x),y,atol=1e-12)
    linear = misc.smooth(x,y,0,x)
    assert_allclose(np.diff(linear,2),0,atol=1e-12)
    smooth = misc.smooth(x,y,.5)
    assert np.all(np.isfinite(smooth(x)))
    extended = misc.smooth(x,y,.5,LinExtrap=1)
    assert_allclose(np.diff(extended([-2,-1,0]),2),0,atol=1e-12)


def test_contour_headers_and_documentation_helpers(tmp_path):
    levels,coords = misc.clevels([[2,0,1,3,2],[2,1,2,1,4]])
    assert_array_equal(levels,[2,3])
    assert np.all(np.isnan(coords[:,[0,3]]))
    path = tmp_path/'sample.m'
    path.write_text('function y=f(x)\n% F First comment\n')
    assert misc.geth1line(path)=='F First comment'
    assert misc.wafoversion()==__version__
