"""Source-identity fixtures for the additional fatigue/Markov APIs.

These tests identify specific numerical cases; they do not certify all original
GUI, extrapolation, switching-asymmetric or parameter-estimation branches.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from scipy.stats import norm
from scipy.special import gammaln
from wafo import cycles as c,markov as m,damage as d


TESTED_FUNCTIONS = [
    'cycles.cc2amp','cycles.cc2lc','cycles.tp2lc','cycles.nt2lc','cycles.cmat2lc',
    'cycles.cmat2rmcmat','cycles.rmcmat2cmat','cycles.cmatcombine',
    'cycles.rfcfilter','cycles.dat2dtp','cycles.dtp2arfm4p','cycles.dtp2arfm',
    'cycles.dtp2rfm','cycles.dat2rfm','markov.mat2tmat','markov.estmc',
    'markov.chi2cmat','markov.hdcmat','markov.klcmat','markov.loglcmat',
    'markov.tr_p2x','markov.tr_x2p','markov.tr_m2x','markov.tr_x2m',
    'markov.mktestmat','markov.mctp2stat','markov.mctp2reverse',
    'markov.smctp2rfm','markov.mctp2rfm','damage.cmat2dmat',
    'damage.spec2dplus','damage.roadspec','damage.ftf','damage.sphdam',
    'markov.smctp2joint','markov.smctp2stat','markov.f_funm','cycles.cmatresamp',
]


def test_cycle_amplitudes_and_crossing_endpoint_conventions():
    cc = [[0,2],[1,3]]
    assert_allclose(c.cc2amp(cc),[1,1])
    assert_allclose(c.cc2amp([[3,1]]),[-1])
    u = [0,1,2,3]
    expected = {1:[0,1,1,0],2:[0,1,2,1],3:[1,2,2,1],4:[1,2,1,1]}
    for definition,counts in expected.items():
        assert_allclose(c.cc2lc(cc,definition),np.column_stack((u,counts)))
    assert_allclose(c.tp2lc([0,2,1,3]),c.cc2lc(cc))
    F = c.cc2cmat([0,3,4],cc,0)
    assert_allclose(c.cmat2lc([0,3,4],F),c.nt2lc([0,3,4],c.cmat2nt(F)))


def test_range_mean_conversions_preserve_cycle_mass_and_metadata():
    F = np.triu(np.arange(25).reshape(5,5),1)
    rm,pm,pr,pa = c.cmat2rmcmat(F,[0,4,5],full_output=True)
    assert_allclose(rm.sum(),F.sum())
    assert_allclose(pr,[.5,4.5,5])
    back,param = c.rmcmat2cmat(rm,pm,pr,full_output=True)
    assert_allclose(back.sum(),F.sum())
    assert_allclose(param,[0,4,5])
    assert_allclose(pa,[.25,2.25,5])


def test_cmatcombine_masks_and_parts():
    a,b = np.ones((4,4)),np.full((4,4),2.)
    total,opt,p,q = c.cmatcombine(a,b,dict(range=2,min=1,max=2),full_output=True)
    assert_allclose(total,p+q)
    assert total[0,2]==1 and total[1,3]==1 and total[2,3]==2
    assert opt['range']==2


def test_hysteresis_filter_plateau_threshold_and_time_rows():
    x = np.array([0,1,1,.8,2,0,.2,-1,0.])
    out = c.rfcfilter(x,.5)
    assert_allclose(out,[0,2,-1,0])
    xt = np.column_stack((np.arange(len(x)),x))
    timed = c.rfcfilter(xt,.5)
    assert_allclose(timed[:,1],out)
    assert_array_equal(timed[:,0],[0,4,7,8])
    assert_allclose(c.rfcfilter([1,1,1],0,1),[1])
    assert_allclose(c.rfcfilter([0,1,0,2,0],0,1),[0,1,0,2,0])
    with pytest.raises(ValueError):
        c.rfcfilter(x,-1)


def test_discrete_turning_points_and_rainflow_matrices_streaming():
    x = np.array([0,3,1,2,-1,4,0,5,-2,3.])
    dtp,u,tp = c.dat2dtp([-2,5,8],x,0,0,full_output=True)
    assert_allclose(u[dtp],tp)
    full,closed,res = c.dtp2arfm(dtp,8,full_output=True)
    cycles = c.tp2arfc(dtp)
    assert_allclose(full,c.dcc2cmat(cycles,8))
    part,r0 = c.dtp2arfm4p(dtp[:5],8)
    streamed,closed1,res1 = c.dtp2arfm(dtp[5:],8,RFM0=part,res0=r0,full_output=True)
    assert_allclose(full,streamed)
    assert_allclose(closed,closed1)
    assert_allclose(res,res1)
    symmetric = c.dtp2rfm(dtp,8)
    assert_allclose(symmetric,np.triu(full+full.T))
    assert_allclose(c.dat2rfm(x,0,[-2,5,8]),c.dtp2rfm(dtp,8,'CS'))


def test_transition_normalization_estimation_and_zero_based_model_transforms():
    P = m.mat2tmat([[2,2],[1,3]])
    assert_allclose(P,[[.5,.5],[.25,.75]])
    assert_allclose(m.mat2tmat(np.ones((3,3)),1),[[0,.5,.5],[0,0,1],[0,0,0]])
    estimate,counts,totals = m.estmc([0,1,1,0,1],2,full_output=True)
    assert_array_equal(counts,[[0,2],[1,1]])
    assert_array_equal(totals,[2,2])
    assert_allclose(estimate,[[0,1],[.5,.5]])
    for transform in (0,1):
        vector,r = m.tr_p2x(P,transform,return_size=True)
        assert r==2
        assert_allclose(m.tr_x2p(vector,transform),P)
    model = dict(x0=[-.1,.2],s=.3,lam=2.)
    recovered = m.tr_x2m(m.tr_m2x(model))
    assert_allclose(recovered['x0'],model['x0'])
    assert_allclose([recovered['s'],recovered['lam']],[.3,2])


def test_cycle_matrix_distances_likelihood_and_zero_mass_semantics():
    counts = np.array([[1,2],[3,4]])
    assert_allclose(m.chi2cmat(counts,counts),0)
    assert_allclose(m.hdcmat(counts,counts),0,atol=2e-8)
    assert_allclose(m.klcmat(counts,counts),0)
    expected = np.sum(counts*np.log(counts/10))
    assert_allclose(m.loglcmat(counts,counts),expected)
    assert_allclose(m.loglcmat(counts,counts,1),expected+gammaln(11)-gammaln(counts+1).sum())
    assert np.isinf(m.klcmat([[1,0],[0,1]],np.ones((2,2))))
    assert m.loglcmat(np.ones((2,2)),[[1,0],[0,1]])==-np.inf
    assert np.isinf(m.chi2cmat(np.ones((2,2)),[[1,0],[0,1]]))


def test_markov_turning_point_stationarity_reverse_and_rainflow():
    F = np.array([[0,.5,.5],[0,0,1],[0,0,0]])
    minimum,maximum,(up,down) = m.mctp2stat((F,None),full_output=True)
    assert_allclose(minimum@up,maximum)
    assert_allclose(maximum@down,minimum)
    reversed_,joint,reversed_joint = m.mctp2reverse((F,None),full_output=True)
    assert_allclose(joint,reversed_joint.T)
    rfc,mu = m.mctp2rfm((F,None),full_output=True)
    assert_allclose(rfc,[[0,.25,.25],[0,0,.5],[0,0,0]])
    assert_allclose(mu,c.cmat2nt(rfc))
    assert_allclose(rfc.sum(),1)
    switched = m.smctp2rfm([[.8,.2],[.1,.9]],[(F,None),(F,None)])
    assert_allclose(switched,rfc,atol=1e-12)
    simple = np.array([[0,1],[0,0]])
    assert_allclose(m.mctp2rfm((simple,None),3),[[0,3],[0,0]])
    test,testh = m.mktestmat([-1,1,5],[-.2,.2],.25,.5,both=True)
    assert_allclose(testh,test.T)
    assert np.all(np.tril(test)==0)


def test_switching_joint_stationary_laws_and_model_kernel():
    F = np.array([[0,.5,.5],[0,0,1],[0,0,0]])
    P = np.array([[.8,.2],[.1,.9]])
    loads = [(F,None),(F,None)]
    up,down = m.smctp2joint(P,loads)
    regime,submin,submax,jointmin,jointmax,pairs = m.smctp2stat(P,loads,full_output=True)
    assert_allclose(regime,[1/3,2/3])
    assert_allclose(jointmin@up,jointmax)
    assert_allclose(jointmax@down,jointmin)
    assert_allclose(jointmin.reshape(3,2).sum(axis=0),regime)
    assert_allclose(jointmin.reshape(3,2).sum(axis=1),submin[0])
    kernel,transpose = m.f_funm([-1,1,5],dict(x0=[-.2,.2],s=.25,lam=.5))
    assert_allclose(kernel,m.mktestmat([-1,1,5],[-.2,.2],.25,.5))
    assert_allclose(transpose,kernel.T)


def test_multinomial_cycle_resampling_mass_and_support():
    counts = np.array([[0,10,2],[0,0,4],[0,0,0]])
    for method in (1,2,3):
        draw = c.cmatresamp(counts,method,seed=3)
        assert draw.sum()==counts.sum()
        assert np.all(draw[counts==0]==0)
        assert_array_equal(draw,c.cmatresamp(counts,method,seed=3))
    assert_array_equal(c.cmatresamp(np.zeros((3,3))),np.zeros((3,3)))


def test_damage_matrix_spectral_road_and_failure_models():
    cc = [[0,2],[1,4]]
    F = c.cc2cmat([0,4,5],cc,0)
    assert_allclose(d.cmat2dmat([0,4,5],F,3,2).sum(),d.cmat2dam([0,4,5],F,[3],2)[0])
    road = d.roadspec([.5,4,1001],2.1,2.)
    assert_allclose(road.S[road.w<.5],0)
    assert_allclose(road.S[-1],2/4**2.1)
    from wafo.spec import createspec
    w = np.linspace(0,10,1001)
    S = createspec(w=w,S=np.sqrt(2/np.pi)*np.exp(-w*w/2))
    assert_allclose(d.spec2dplus(S,[2]),[1/np.pi],atol=1e-12)
    times,Ft = d.ftf(1,1,.25,0,time=[0,.5,1,2])
    assert_allclose(Ft[1:],norm.cdf(np.log(times[1:])/.5))
    assert Ft[0]==0
    _,noisy = d.ftf(1,1,.25,.1,time=[.5,1,2])
    assert np.all((noisy>=0)&(noisy<=1))


def test_spherical_damage_source_missing_reconstruction_rotation_identity():
    load = np.column_stack((np.zeros(5),np.zeros(5),[0,1,0,2,0]))
    out = d.sphdam(load,3,[1,2])
    assert out['damage'].shape==(3,3,2)
    assert_allclose(out['directions'][0,:,2],1)
    assert_allclose(out['damage'][0,:,0],3)
    assert_allclose(out['damage'][0,:,1],5)
    assert_allclose(out['damage'][1],0,atol=1e-14)
