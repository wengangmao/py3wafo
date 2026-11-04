"""First-passage identities, tagged counting and actual plot-render tests."""
import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from wafo import cycles as c,markov as m,damage as d


def test_ordinary_markov_rainflow_exact_two_state_and_iid_three_state():
    F,mu=m.mc2rfm([[.8,.2],[.1,.9]],full_output=True)
    assert_allclose(F,[[0,1/15],[0,0]])
    assert_allclose(mu,c.cmat2nt(F))
    expected=np.array([[0,1/18,1/6],[0,0,1/18],[0,0,0]])
    assert_allclose(m.mc2rfm(np.ones((3,3))/3),expected)
    states=np.random.default_rng(7).integers(0,3,60000)
    tp=c.rfcfilter(states,0,1)
    empirical=c.dtp2rfm(tp,3)/len(states)
    assert_allclose(empirical,expected,atol=.003)


def test_switching_chain_identical_regimes_and_discretized_scaling():
    Q=np.array([[.5,.3,.2],[.2,.4,.4],[.1,.2,.7]])
    P=[[.9,.1],[.2,.8]]
    assert_allclose(m.smc2rfm(P,[Q,Q]),m.mc2rfm(Q))
    f,mu=m.smc2rfm(P,[Q,Q],[2,.2],full_output=True)
    assert f.shape==mu.shape==(4,4)
    assert np.all(np.isfinite(f)) and np.all(f>=0)
    assert_allclose(m.smc2rfm(P,[Q,Q],[2,.4]),f/4)


def test_asymmetric_turning_chain_aggregates_and_source_terminal_rule():
    up=np.triu(np.ones((5,5)),1)
    down=np.tril(np.arange(1,26).reshape(5,5),-1)
    asymmetric,forward,backward=m.mctp2arfm((up,down),full_output=True)
    assert_allclose(np.triu(asymmetric+asymmetric.T,1),m.mctp2rfm((up,down)),atol=1e-12)
    assert_allclose(sum(matrix.sum() for matrix in forward),2)
    assert_allclose(sum(matrix.sum() for matrix in backward),2)
    two=np.array([[0,1],[0,0.]])
    assert_allclose(m.mctp2arfm((two,None)),two)
    assert_allclose(m.smctp2arfm([[1]],[(two,None)]),[[0,.5],[.5,0]])


def test_switching_asymmetric_side_marks_conserve_total():
    up=np.triu(np.ones((5,5)),1)
    P=[[.9,.1],[.2,.8]]
    full,marks,_=m.smctp2arfm(P,[(up,None),(up,None)],c_m=3,SideInfo=1,full_output=True)
    assert_allclose(marks.sum(axis=(0,1)),full,atol=1e-12)
    assert_allclose(np.triu(full+full.T,1),m.smctp2rfm(P,[(up,None),(up,None)],3),atol=1e-12)
    with pytest.raises(ValueError):m.smctp2arfm(P,[(up,None),(up,None)],SideInfo=2)


@pytest.mark.parametrize('definition',[1,2])
def test_discrete_tagged_rainflow_matches_untagged_and_closed_marks(definition):
    x=np.array([0,3,1,2,0,4,1,3,0,2])
    labels=np.arange(len(x))%2
    total,closed,res,marks,closed_marks,res_labels=c.dtp2arfm_sid(x,labels,5,2,definition,full_output=True)
    assert_allclose(total,c.dtp2arfm(x,5))
    assert_allclose(marks.sum(axis=(0,1)),total)
    assert_allclose(closed_marks.sum(axis=(0,1)),closed)
    assert len(res)==len(res_labels)
    symmetric=c.dtp2rfm_sid(x,labels,5,2,definition,full_output=True)
    assert_allclose(symmetric[0],c.dtp2rfm(x,5))
    assert_allclose(symmetric[3].sum(axis=(0,1)),symmetric[0])


def test_smoothing_normal_bandwidth_and_adaptive_mass():
    F=np.zeros((7,7));F[0,3]=10;F[1,5]=4;F[2,6]=3
    u=np.arange(7); marginal=(F+F.T).sum(axis=0)/(2*F.sum())
    expected=np.sqrt(marginal@((u-marginal@u)**2))*F.sum()**(-1/6)
    assert_allclose(c.smoothcmat_hnorm(F),expected)
    for method in (1,2):
        smoothed,h=c.smoothcmat(F,method,NOsubzero=1,full_output=True)
        assert_allclose(smoothed.sum(),F.sum())
        assert np.all(np.tril(smoothed,1)==0)
        assert_allclose(h,.7*expected)
    assert_allclose(c.smoothcmat(F,2,h=.5,alpha=0),c.smoothcmat(F,1,h=.5))


def test_extreme_rfm_counting_and_weighted_gpd_score():
    x=np.linspace(-3,3,31)
    F,levels,corrected,raw=c.lc2rfmextreme(np.column_stack((x,np.exp(-x*x/2))),full_output=True)
    assert_array_equal(levels,x)
    assert np.all(F>=0)
    assert_allclose(corrected,c.cmat2nt(F))
    assert_allclose(raw[4,25],np.exp(-x[4]**2/2)*np.exp(-x[25]**2/2)/(np.exp(-x[4]**2/2)+np.exp(-x[25]**2/2)))
    data=np.array([[.2,3],[.8,4],[1.5,2],[3,1.]])
    transformed=.3
    score,k,scale=c.wgpdfit_mld(transformed,data,full_output=True)
    t=1/3-np.exp(transformed); N=data[:,1].sum()
    expected_k=-np.dot(data[:,1],np.log1p(-t*data[:,0]))/N
    expected=(1/expected_k-1)*np.dot(data[:,1],data[:,0]/(1-t*data[:,0]))-N/t
    assert_allclose([score,k,scale],[expected,expected_k,expected_k/t])
    assert np.all(np.isfinite(c.wgpdfit_mld(np.log(1/3),data,full_output=True)))


def test_counting_distribution_damage_and_matrix_scaling():
    param=[0,4,5]; F=np.triu(np.arange(25).reshape(5,5),1)
    NT=c.cmat2nt(F); beta=[2,3,4]
    u=c.levels(param); ranges=np.maximum(0,u[None,:]-u[:,None])
    expected=[np.sum(F*ranges**power) for power in beta]
    assert_allclose(d.damint(param,beta,NT,convention='pj'),expected)
    assert_allclose(d.damint(param,beta,np.flipud(NT.T)),expected)
    assert_allclose(d.damint(param,beta,NT,convention='pj',amplitude=True),d.cmat2dam(param,F,beta))
    wat=np.flipud(F.T)
    assert_allclose(m.scalemat(param,wat,0,1,param),wat)


def test_cycle_fatigue_and_hidden_regime_plots_render(tmp_path):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    F=np.triu(np.arange(25).reshape(5,5),1); cc=np.array([[0,2],[1,3],[2,4]])
    axes=[c.ccplot(cc)]
    axes.extend(c.ccplot([cc,cc]))
    for mode in (1,2,3,4,11,12,13,14):axes.append(c.cmatplot(F,mode))
    axes.append(c.cocc([0,4,5],cc,F))
    for cumulative,mode in [(1,'stairs'),(1,'plot'),(0,'bar'),(0,'stem'),(0,'stairs'),(0,'plot')]:
        axes.append(c.lsplot([[.5,3],[1,2],[2,1]],cumulative,1,2,mode))
    # Closed figures can still be rendered below; release GUI registrations
    # between groups to avoid Matplotlib's many-open-figures warning.
    plt.close('all')
    S=np.geomspace(1,10,20); N=1e6/S**3
    assert_allclose(d.snplot(S,N),[1e-6,3,0],atol=1e-12)
    for mode in (1,2,3,4,11,12,13,14):
        e,beta,var,ax=d.snplot(S,N,mode,full_output=True); axes.append(ax)
        assert_allclose([e,beta],[1e-6,3],atol=1e-12)
    axes.extend(m.hmmplot([0,1,2,1,0],[0,0,1,1,0],colType=1))
    for index,ax in enumerate(axes):
        path=tmp_path/f'fatigue-{index}.png'
        ax.figure.savefig(path)
        assert path.stat().st_size>1000
        plt.close(ax.figure)
