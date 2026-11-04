import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from wafo import onedim as o


@pytest.fixture
def sine_series():
    t = np.arange(0,10,.025)
    return np.column_stack((t,np.sin(2*np.pi*t)))


@pytest.mark.parametrize('definition,first_down,parity',[
    ('dw',True,1),('uw',False,1),('tw',True,0),('cw',False,0)])
def test_complete_crossing_definitions(sine_series,definition,first_down,parity):
    x = sine_series
    indices = o.dat2crossind(x,0,definition)
    assert len(indices)%2==parity
    assert bool(x[indices[0]+1,1]<0)==first_down


def test_crests_troughs_and_midpoint_absolute_indices(sine_series):
    x = sine_series
    tc,indices,cross = o.dat2tc(x,0,full_output=True)
    assert_allclose(abs(tc[:,1]),1,atol=1e-12)
    assert_array_equal(tc[:,0],x[indices,0])
    assert_allclose(np.diff(tc[:,0]),.5,atol=1e-12)
    assert len(indices)==len(cross)-1
    mid = o.dat2midind(x,p=.5)
    assert np.all(np.diff(mid)>0) and mid[-1]>len(x)//2
    tp,turns = o.dat2tp(x,0,'mw',return_indices=True)
    for j,index in enumerate(mid):
        a,b = turns[j:j+2]
        level = (x[a,1]+x[b,1])/2
        assert a<=index<b
        assert (x[index,1]-level)*(x[index+1,1]-level)<=1e-12


@pytest.mark.parametrize('definition,expected',[
    ('d2d',1),('u2u',1),('d2u',.5),('u2d',.5),
    ('t2t',1),('c2c',1),('t2c',.5),('c2t',.5),
    ('d2t',.25),('t2u',.25),('u2c',.25),('c2d',.25),
    ('m2M',.5),('M2m',.5),('m2m',1),('M2M',1),
])
def test_wave_periods(sine_series,definition,expected):
    values,events,pairs = o.dat2wa(sine_series,0,definition,full_output=True)
    # Endpoints in the legacy dat2tp convention may represent partial waves.
    values = values[1:-1]
    assert len(values)>5
    assert_allclose(values,expected,atol=.026)
    assert events.ndim==1 and pairs.shape[1]==2


def test_interpolated_event_timeline_and_level_crossing_spectrum(sine_series):
    values = o.dat2wa(sine_series[::3],0,'all',rate=3)
    assert np.all(values>0)
    assert_allclose(np.median(values),.25,atol=.005)
    spectrum = o.dat2lc(sine_series,0,1)
    assert spectrum.shape[1]==2 and np.all(spectrum[:,1]>=0)


def test_significant_values_are_tail_means():
    result = o.vc2sign([1,2,3,4])
    assert_allclose(result,[[1,2.5,1],[.75,3,2],[.5,3.5,3],[.25,4,4]])
    assert_allclose(o.vc2sign([1,2,3,4],.5),[.5,3.5,3])
    assert_allclose(o.vc2sign([1,2,3,4],1),[1,2.5,1])
    assert_allclose(o.hs2sign([[1,2],[3,2]],.75),[.75,7/3,1])


def test_spurious_points_missing_and_repeated_runs():
    bad,good = o.findoutliers([0,1,2,2,2,3,4,20,5,np.nan],zcrit=0,dcrit=5,ddcrit=10)
    assert set([1,2,3,4,5,7,9]).issubset(set(bad))
    assert not set(bad)&set(good)
    assert set(bad)|set(good)==set(range(10))
    with pytest.raises(ValueError):
        o.dat2wa([0,1,0],pdef='bad')
