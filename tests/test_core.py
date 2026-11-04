import importlib
import json
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import simpson
from wafo import cycles, damage, kdetools, misc, onedim, spec, trgauss, wavemodels, wsim, wstats


@pytest.mark.parametrize('x,expected', [
    ([], []), ([0], []), ([0,0,0], []), ([-1,1,-1], [0,1]),
    ([0,0,1,0,0,-1,0,1], [1,4,6]), ([1,0,0,1], []),
    ([-1,0,0,-1], []), ([0,-1,0,1], [0,2]),
])
def test_crossing_plateaus(x, expected):
    assert_array_equal(onedim.findcross(x), expected)


def test_crossing_nonzero_level_and_interpolation():
    assert_array_equal(onedim.findcross([1,3,1], 2), [0,1])
    assert_allclose(onedim.ecross([0,2,4], [1,3,1], [0,1], 2), [1,3])
    with pytest.raises(ValueError):
        onedim.findcross([0,np.nan])


def test_turning_points_endpoint_conventions():
    y = [0,1,0,2,0]
    assert_array_equal(onedim.findextrema(y), [1,2,3])
    tp, ind = onedim.dat2tp(y, return_indices=True)
    assert_array_equal(ind, [0,1,2,3,4])
    assert_array_equal(tp, y)
    assert len(onedim.dat2tp([0,1,2])) == 0
    assert_array_equal(onedim.dat2tp(y, -1), y)


def test_covariance_and_legacy_moving_mean():
    x = np.array([1,2,3,4,5.])
    C = onedim.dat2cov(x,2)
    assert_allclose(C.R,[2,.8,-.2])
    R,v = onedim.dat2cor(x,2,return_variance=True)
    assert_allclose(R.R,C.R/2)
    assert_allclose(v,2)
    y,trend = onedim.detrendma(x,1,return_trend=True)
    assert_allclose(trend,[2,8/3,10/3,4,4])
    assert_allclose(y+trend,x)
    assert_allclose(onedim.detrendma(x,3),x-x.mean())


def test_findrfc_literal_c_case():
    # First peak 2 closes to following min 1; peak 3 closes to min 0.
    assert_array_equal(onedim.findrfc([0,2,1,3,0], .5), [0,1,2,3])
    assert_array_equal(onedim.findrfc([0,2,1,3,0], 2), [0,3])
    assert_array_equal(onedim.findrfc([0,2,1,3,0], 4), [])
    assert_array_equal(onedim.findrfc([0], 1), [])


def test_rainflow_nested_and_residual():
    x = [0,3,1,2,-1,4]
    closed, res = cycles.tp2arfc4p(x)
    assert_allclose(closed, [[1,2]])
    assert_allclose(res[:, 0], [0,3,-1,4])
    assert_allclose(cycles.tp2rfc(x), [[1,2],[0,3],[-1,4]])
    assert_allclose(cycles.tp2rfc([3,0,2,-1], 'down'), [[0,2],[-1,3]])


def test_rainflow_streaming():
    x = np.array([0,3,1,2,-1,4,0,5,-2,3])
    all_one, closed, res = cycles.tp2arfc(x, full_output=True)
    _, c0, r0 = cycles.tp2arfc(x[:5], full_output=True)
    all_split, c1, r1 = cycles.tp2arfc(x[5:], ARFC0=c0, res0=r0, full_output=True)
    assert_allclose(all_split, all_one)
    assert_allclose(c1, closed)
    assert_allclose(r1, res)


def test_timed_rainflow():
    tp = np.column_stack((np.arange(6)*10, [0,3,1,2,-1,4]))
    rf = cycles.tp2rfc(tp, {'time':1})
    assert_allclose(rf, [[1,2,20,30],[0,3,0,10],[-1,4,40,50]])


def test_cycle_discretization_matrix_and_damage():
    param = [0,4,5]
    cc = np.array([[0,2],[1,4],[3,4]])
    F = cycles.cc2cmat(param, cc)
    assert F.sum() == 3
    assert_allclose(cycles.nt2cmat(cycles.cmat2nt(F)), F)
    assert_allclose(damage.cc2dam(cc, [2,3]), damage.cmat2dam(param, F, [2,3]))
    assert_allclose(damage.cc2dam([[0,2],[0,4]], [2,3]), [5,9])


def test_pm_and_jonswap_moments():
    S = spec.pmspec(np.linspace(0,20,10001), [4,8])
    assert_allclose(spec.spec2mom(S,0), [1], rtol=1e-5)
    J = spec.jonswap(sdata=[4,8])
    assert_allclose(spec.spec2mom(J,0), [1], rtol=1e-12)
    assert spec.spec2mom(J,4).shape == (3,)
    bw = spec.spec2bw(J,[1,2,3,4])
    assert 0 < bw[0] < 1
    assert_allclose(bw[0]**2+bw[2]**2, 1)
    C = spec.spec2cov(J, nr=2, Nt=5)
    assert_allclose(C.R[0], 1)
    assert_allclose(C.Rtt[0], -spec.spec2mom(J,2)[1])
    assert_allclose(spec.spec2char(J,'Hm0'),[4])
    assert_allclose(spec.spec2char(J,'alpha'),[bw[0]])


@pytest.mark.parametrize('h',[.5,10,100,np.inf])
def test_dispersion_round_trip(h):
    w = np.array([0,.01,.5,2,10])
    kx, ky = spec.w2k(w,.3,h)
    back, angle = spec.k2w(kx,ky,h)
    assert_allclose(back,w,rtol=1e-10,atol=1e-12)
    assert_allclose(angle[1:], .3)


def test_second_order_deep_water_exact_coefficients():
    a = np.zeros(16, complex)
    a[1], a[2] = 1+2j, 3-1j
    w = np.arange(5, dtype=float)
    k = w*w/9.81
    s,d = wsim.disufq(a,w,k,np.inf,9.81,1,2,components=True)
    assert_allclose(s[2],2*a[1]**2*k[1])
    assert_allclose(s[3],2*a[1]*a[2]*(k[1]+k[2]))
    assert_allclose(d[1],-2*a[1]*np.conj(a[2])*(k[2]-k[1]))
    assert_allclose(d[0],0)
    assert_allclose(s+d, wsim.disufq(a,w,k,np.inf,9.81,1,2))
    finite = wsim.disufq(a,w,k,10000,9.81,1,2)
    assert_allclose(finite,s+d,rtol=2e-3,atol=2e-3)


def test_spectral_simulation_variance_and_reproducibility():
    S = spec.jonswap(sdata=[4,8])
    a = wsim.spec2sdat(S, 4096, cases=80, seed=7)
    assert a.shape == (4096,81)
    assert_allclose(a[:,1:].var(),1,rtol=.06)
    assert_array_equal(a,wsim.spec2sdat(S,4096,cases=80,seed=7))


def test_markov_stationarity():
    P = [[.8,.2],[.1,.9]]
    from wafo.markov import mc2stat
    assert_allclose(mc2stat(P),[1/3,2/3])
    x = wsim.mcsim(P,40000,seed=123)
    assert abs((x==0).mean()-1/3)<.025
    with pytest.raises(ValueError):
        mc2stat(np.eye(2))


@pytest.mark.parametrize('alpha',[2,1.5])
def test_duffing_gaussian_and_stable_noise(alpha):
    x = wsim.duffsim(1,.1,.1,alf=alpha,seed=17)
    assert x.shape==(11,3)
    assert np.all(np.isfinite(x))
    assert_allclose(np.diff(x[:,0]),.1)
    assert_array_equal(x,wsim.duffsim(1,.1,.1,alf=alpha,seed=17))


@pytest.mark.parametrize('name,params', [
    ('norm',(2,4)),('logn',(.2,.7)),('weib',(2,3)),('gam',(2,3)),
    ('beta',(2,4)),('chi2',(3,)),('exp',(2,)),('rayl',(3,)),
    ('f',(5,8)),('t',(6,)),('gev',(.2,2,1)),('gpd',(.2,2,1)),
    ('gumb',(2,1)),('frech',(2,3)),('ggam',(2,3,4)),('invg',(2,3))])
def test_distribution_cdf_inverse_and_pdf(name,params):
    p = np.array([.01,.2,.5,.8,.99])
    x = getattr(wstats,'w'+name+'inv')(p,*params)
    assert_allclose(getattr(wstats,'w'+name+'cdf')(x,*params),p,atol=1e-10)
    assert np.all(getattr(wstats,'w'+name+'pdf')(x,*params)>=0)
    rng = getattr(wstats,'w'+name+'rnd')
    assert_array_equal(rng(*params,size=10,seed=13),rng(*params,size=10,seed=13))


def test_legacy_distribution_conventions():
    assert_allclose(wstats.wnormpdf(0,0,4),1/np.sqrt(8*np.pi))
    assert_allclose(wstats.wgpdinv(1,.5,2),4)
    assert_allclose(wstats.wgevinv(1,.5,2,1),5)
    assert_allclose(wstats.empdistr([3,1,2])[:,1],[1/6,.5,5/6])
    assert_allclose(wstats.cempdistr([1,2,3],2)[:,0],[2,3])
    assert_allclose(wstats.wquantile([1,2,3,4],.25),1.5)


def test_gaussian_probabilities_and_transforms():
    assert_allclose(trgauss.bvnormcdf(0,0,0),.25,atol=1e-10)
    assert_allclose(trgauss.bvnormcdf(0,0,1),.5,atol=1e-10)
    assert_allclose(trgauss.mvnormprb(np.eye(3),[-np.inf]*3,[0]*3),.125,atol=1e-10)
    g = np.column_stack((np.linspace(-10,10,41),np.linspace(-10,10,41)*2))
    x = np.column_stack((np.arange(5),[-2,-1,0,1,2]))
    assert_allclose(trgauss.gaus2dat(trgauss.dat2gaus(x,g),g),x)
    assert_allclose(trgauss.lomaxcdf([-1,0,1],1,1),[0,0,1-np.exp(-.5)])


def test_beta_rayleigh_normalization():
    x = np.linspace(0,3,10001)
    assert_allclose(simpson(wavemodels.braylpdf(x,2,3,3),x=x),1,atol=1e-8)
    assert_allclose(wavemodels.braylcdf([-1,0,3,4],2,3,3),[0,0,1,1])


def test_utilities_and_density():
    assert_allclose(misc.wdensity(0,0,0),999.842594)
    assert_allclose(misc.wdensity(35,0,0),1028.1063314,atol=1e-6)
    assert misc.wdensity(35,10,1000)>misc.wdensity(35,10,0)
    a = [3,-1,3,2]
    sorted_,i = misc.csort(a)
    assert_array_equal(i,[1,3,0,2])
    assert_array_equal(sorted_,np.asarray(a)[i])
    A = np.array([[1,2],[2,4.]])
    L,p,rank = misc.genchol(A)
    assert rank==1
    assert_allclose(L@L.T,A[np.ix_(p,p)])
    x,w = misc.qrule(4)
    assert_allclose(np.dot(w,x**6),2/7)
    assert_allclose(misc.qrule(4,2)[1].sum(),np.pi)


def test_kde_and_real_dataset():
    from wafo.data import load
    data = load('sea.dat')
    assert data.ndim==2 and data.shape[1]==2
    tp = onedim.dat2tp(data)
    rfc = cycles.tp2rfc(tp)
    assert len(rfc)>100
    assert damage.cc2dam(rfc,[3])[0]>0
    x = np.random.default_rng(2).normal(size=1000)
    estimate = kdetools.kde(x,np.linspace(-6,6,1001))
    assert_allclose(simpson(estimate,x=np.linspace(-6,6,1001)),1,atol=1e-4)


def test_coverage_records_are_honest_and_exhaustive():
    import wafo
    records = wafo.coverage()
    assert len(records)==762
    assert wafo.source_info('rind')[0]['status']=='adapted'
    for row in records:
        if row['status'] in ('implemented','adapted'):
            module,name = row['python'].rsplit('.',1)
            assert callable(getattr(importlib.import_module(module),name))
