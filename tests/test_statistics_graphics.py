"""Render real diagnostic figures and verify their underlying numerical data."""
from types import SimpleNamespace
import numpy as np
from scipy.integrate import trapezoid
import pytest
import matplotlib
matplotlib.use('Agg',force=True)
import matplotlib.pyplot as plt
from scipy import stats
from numpy.testing import assert_allclose
from wafo import wstats as w


def _save(ax,path):
    ax.figure.savefig(path)
    assert path.stat().st_size>1000
    plt.close(ax.figure)


def test_probability_graphics_numerical_coordinates_and_png(tmp_path):
    p=(np.arange(200)+.5)/200
    x=stats.norm.ppf(p,loc=2,scale=3)
    y=4+2*x
    ax=w.wqqplot(x,y)
    assert_allclose(ax.lines[0].get_ydata(),y)
    _save(ax,tmp_path/'qq.png')
    ax=w.wnormplot(np.column_stack((x,y)))
    assert_allclose(ax.lines[0].get_ydata(),stats.norm.ppf(p))
    _save(ax,tmp_path/'normal.png')
    ax=w.distplot(x,'wnormpdf')
    assert len(ax.lines)==1
    _save(ax,tmp_path/'distribution.png')
    for family,params in [('exp',[2]),('rayl',[3]),('weib',[2,3]),('gumb',[2,1])]:
        data=getattr(w,'w'+family+'inv')(p,*params)
        estimate,ax=getattr(w,'w'+family+'plot')(data,full_output=True)
        assert_allclose(estimate,params if len(params)>1 else params[0],rtol=.025,atol=.01)
        _save(ax,tmp_path/(family+'.png'))


def test_histogram_scatter_and_kde_render(tmp_path):
    x=stats.norm.ppf((np.arange(200)+.5)/200)
    width,count,edges,ax=w.whisto(x,15,1,1,full_output=True)
    assert_allclose(np.sum(count*np.diff(edges)),1)
    assert_allclose(np.diff(edges),width)
    _save(ax,tmp_path/'histogram.png')
    axes=w.pairs(np.column_stack((x,x*x)),labels=['x','square'])
    _save(axes[0,0],tmp_path/'pairs.png')
    ax=plt.subplots()[1]
    density=w.kdeplot(x,h=.3,ax=ax)
    assert_allclose(trapezoid(density[:,1],density[:,0]),1,atol=1e-4)
    _save(ax,tmp_path/'kde.png')


def test_factorial_model_fit_selection_and_graphics(tmp_path):
    D=w.ffd(3); signed=2*D-1
    y=3+2*signed[:,0]-signed[:,1]+4*signed[:,0]*signed[:,2]
    fitted,residual,sd,dof=w.fitmodel(y,['A','B','AC'])
    assert_allclose(fitted,y,atol=1e-12)
    assert_allclose(residual,0,atol=1e-12)
    assert sd<1e-12 and dof==4
    assert_allclose(w.fitmodel(y,[])[0],np.full(8,3))
    effects,labels=w.yates(y,return_labels=True)
    chosen,indices=w.getmodel(effects,labels,-1.5)
    assert set(chosen)=={'A','B','AC'}
    assert len(indices)==3
    _save(w.nplot(effects,labels),tmp_path/'effects.png')
    _save(w.cplot(y,D),tmp_path/'cube.png')


def test_gumbel_pwm_test_detects_shape_and_affine_invariance():
    p=(np.arange(2000)+.5)/2000
    gumbel=stats.gumbel_r.ppf(p)
    gev=stats.genextreme.ppf(p,.3)
    result=w.wgumbtest(gumbel,full_output=True)
    assert result['pvalue']>.9 and not result['reject']
    assert w.wgumbtest(gev)<1e-10
    assert_allclose(w.wgumbtest(4+3*gev),w.wgumbtest(gev),rtol=1e-10)
    with pytest.raises(ValueError):w.wgumbtest(np.ones(10))


def test_identify_pick_callback_and_disconnect(tmp_path):
    picker=w.identify([0,1,2],[2,1,3],text=['first','second','third'])
    picker._pick(SimpleNamespace(artist=picker.artist,ind=[1]))
    picker.select(2)
    assert picker.indices==[1,2]
    assert [item.get_text() for item in picker.ax.texts]==['second','third']
    picker.disconnect()
    _save(picker.ax,tmp_path/'identified.png')


def test_conditional_cdf_moment_and_parameter_graphics(tmp_path):
    rng=np.random.default_rng(7)
    v,h=rng.rayleigh(2,size=2000),rng.rayleigh(3,size=2000)
    models={'dist2d':dict(x=[np.array([[0,2],[12,2]]),[3]],dist=['rayleigh','rayleigh']),
        'mdist2d':dict(x=[[2],[3],1.],dist=['rayl','rayl']),
        'weib2d':[2*np.sqrt(2),2,3*np.sqrt(2),2,0]}
    expected_mean=2*np.sqrt(np.pi/2)
    for family,model in models.items():
        cdfplot=getattr(w,family+'cdfplot'); statplot=getattr(w,family+'statplot')
        curves=cdfplot(v,h,model,res=3,plotflag=0)
        for curve in curves:
            assert_allclose(curve['model'],stats.rayleigh.cdf(curve['x'],scale=2),atol=1e-10)
        axes=cdfplot(v,h,model,res=3)
        _save(axes[0],tmp_path/(family+'-cdf.png'))
        ax,empirical,theoretical=statplot(v,h,model,res=3,full_output=True)
        assert_allclose(theoretical[:,1],expected_mean,atol=1e-9)
        assert empirical.shape[1]==4
        _save(ax,tmp_path/(family+'-moments.png'))
    model=models['dist2d']
    assert_allclose(w.dist2dparamplot(model,plotflag=0),model['x'][0])
    axes=w.dist2dparamplot(model,model)
    _save(axes[0],tmp_path/'conditional-parameters.png')
