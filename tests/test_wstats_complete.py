"""Explicit source/identity tests for additional WAFO statistical APIs."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats, special, integrate
from wafo import wstats as w


def test_stable_characteristic_function_and_repeatability():
    for alpha in (1,1.5,2):
        x=w.walpharnd(alpha,size=50000,seed=81)
        assert_allclose(np.mean(np.cos(.5*x)),np.exp(-.5**alpha),atol=.009)
    assert_array_equal(w.walpharnd(1,size=10,seed=4),w.walpharnd(1,size=10,seed=4))
    with pytest.raises(ValueError):
        w.walpharnd(3)


def test_truncated_weibull_and_rayleigh_integrate_to_one():
    x=np.linspace(0,20,5001)
    assert_allclose(integrate.simpson(w.wtweibpdf(x,2,1.5,1),x=x),1,atol=1e-7)
    assert_allclose(w.wtweibcdf([0,20],2,1.5,1),[0,1],atol=1e-12)
    assert_allclose(w.wtraylpdf(x,2,-1),w.wtweibpdf(x,2*np.sqrt(2),2,1))
    assert_allclose(w.wtraylcdf(x,2,-1),w.wtweibcdf(x,2*np.sqrt(2),2,1))
    with pytest.raises(ValueError):
        w.wtraylpdf([1],2,1)
    with pytest.raises(ValueError):
        w.wtraylcdf([1],2,1)


def test_truncated_fit_and_objectives():
    p=(np.arange(500)+.5)/500
    x=2*((1/2)**1.5-np.log1p(-p))**(1/1.5)-1
    fitted=w.wtweibfit(x)
    assert_allclose(w.wtweibcdf(x,*fitted),p,atol=.015)
    ray=2*np.sqrt(2*((1/(2*np.sqrt(2)))**2-np.log1p(-p)))-1
    rfit=w.wtraylfit(ray)
    assert_allclose(w.wtraylcdf(ray,*rfit),p,atol=.01)
    for definition in (1,2,3,4):
        assert w.wtweibfun([2,1.5,1],x,p,definition)<1e-20


def test_additional_mle_and_score_functions():
    p=(np.arange(1000)+.5)/1000
    inv=w.winvgfit(stats.invgauss.ppf(p,2/3,scale=3))
    assert_allclose(inv,[2,3],rtol=.01)
    gg=w.wggamfit(stats.gengamma.ppf(p,2,2,scale=3),exponent=2)
    assert_allclose(gg,[2,3,2],rtol=.01)
    gpd=w.wgpdfit_ml(stats.genpareto.ppf(p,.2,scale=2))
    assert_allclose(gpd,[-.2,2,0],rtol=.05,atol=.01)
    shape=3
    x=stats.gamma.ppf(p,shape,scale=2)
    assert abs(w.wgamafit(shape,np.mean(x),np.exp(np.mean(np.log(x)))))<.002
    x=stats.gumbel_r.ppf(p,loc=1,scale=2)
    assert abs(w.wgumbafit(2,x))<.005
    x=stats.weibull_min.ppf(p,3,scale=2)
    assert abs(w.wweibcfit(3,x))<.005
    gx=stats.gengamma.ppf(p,2,2,scale=3)
    assert abs(w.wggambfit(2,gx,np.log(gx)))<.01
    assert w.wggambfit([2,3,2],stats.gengamma.ppf(p,2,2,scale=3),p,2)<1e-25
    g=lambda k:special.gamma(2+k/2)
    ratio12=g(0)*g(2)/g(1)**2
    ratio23=np.sqrt(g(0))*g(3)/g(2)**1.5
    ratio24=g(0)*g(4)/g(2)**2
    assert w.wggambfit([2,2],ratio12,ratio23,3)<1e-25
    assert w.wggambfit([2,2],ratio23,ratio24,4)<1e-25


def test_log_likelihood_and_covariance():
    x=stats.norm.ppf((np.arange(100)+.5)/100,loc=2,scale=3)
    expected=-stats.norm.logpdf(x,loc=2,scale=3).sum()
    assert_allclose(w.loglike([2,9],x,'wnormpdf'),expected)
    ll,cov=w.loglike([2,9],x,'wnormpdf',return_covariance=True)
    assert_allclose(ll,expected)
    assert np.all(np.linalg.eigvalsh(cov)>0)
    assert_allclose(w.wgevlike([0,2,1],x),-stats.genextreme.logpdf(x,0,loc=1,scale=2).sum())
    with pytest.raises(ValueError):
        w.loglike([2],x,'notafunction')


def test_numerical_helpers():
    x=[[1,3],[2,7],[3,11]]
    assert_allclose(w.range(x),[2,8])
    assert_allclose(w.var(x),[1,16])
    assert_allclose(w.var(x,1),[2/3,32/3])
    a,b=w.comnsize(2,[1,2,3])
    assert_array_equal(a,[2,2,2]);assert_array_equal(b,[1,2,3])


def test_column_fits_and_truncated_gumbel_parameter_arrays():
    x=np.column_stack(([1,2,3],[10,20,30]))
    assert_allclose(w.wnormfit(x),[[2,1],[20,100]])
    assert_allclose(w.wlognfit(np.exp(x)),[[2,1],[20,100]])
    assert_allclose(w.wnormfit([[1,2,3]]),[2,1])
    assert np.all(np.isnan(w.wgumbinv([-.1,1.1],2,1,1)))
    assert w.wgumbinv(0,2,1,1)==0
    mean,variance=w.wgumbstat(np.array([1,2]),np.array([-1,1]),1)
    assert np.all(mean>0) and np.all(variance>0)
    for j in (0,1):
        m,v=w.wgumbstat([1,2][j],[-1,1][j],1)
        assert_allclose([mean[j],variance[j]],[m,v],rtol=1e-6)


def test_weibull_independence_and_conditional_inverse():
    p=[2,3,4,2,0]
    x,y=np.meshgrid(np.linspace(.1,4,7),np.linspace(.1,6,9))
    assert_allclose(w.weib2dpdf(x,y,p),w.wweibpdf(x,2,3)*w.wweibpdf(y,4,2))
    assert_allclose(w.weib2dcdf(x,y,p),w.wweibcdf(x,2,3)*w.wweibcdf(y,4,2))
    p[-1]=.7
    probabilities=np.array([.01,.2,.5,.9,.99])
    given=np.array([0,.2,1,3,8])
    quantile=w.weib2dcinv(given,probabilities,p)
    assert_allclose(w.weib2dcdf(given,quantile,p,condon=1),probabilities,atol=1e-12)
    assert_allclose(w.weib2dcdf(quantile,given,[4,2,2,3,.7],condon=2),probabilities,atol=1e-12)
    assert_allclose(w.weib2dpdf2([1,2],[1,2],p).f,w.weib2dpdf(*np.meshgrid([1,2],[1,2]),p))


def test_weibull_moments_simulation_rectangles_and_likelihood():
    p=[2,3,4,2,.7]
    x,y=w.weib2drnd(p,size=50000,seed=7)
    mean,cov=w.weib2dstat(p)
    assert_allclose([x.mean(),y.mean()],mean,rtol=.01)
    assert_allclose(np.cov(x,y),cov,rtol=.03)
    m,v=w.weib2dstat(p,condon=1,cvar=[0,1,3])
    assert np.all(m>0) and np.all(v>0)
    assert_allclose(w.weib2dprb(p,0,2,0,4),w.weib2dcdf(2,4,p))
    assert_allclose(w.weib2dlike(p,x[:100],y[:100]),-np.log(w.weib2dpdf(x[:100],y[:100],p)).sum())
    fit=w.weib2dfit(x[:1500],y[:1500])
    assert_allclose(fit[:4],p[:4],rtol=.12)
    assert abs(fit[4]-.7)<.08


def test_plackett_marginals_and_conditional_inverse():
    phat={'x':[[2,3],[4],1],'dist':['weibull','rayleigh']}
    x,y=np.meshgrid([.2,1,2],[.3,1,3])
    assert_allclose(w.mdist2dcdf(x,y,phat),w.wweibcdf(x,2,3)*w.wraylcdf(y,4))
    assert_allclose(w.mdist2dpdf(x,y,phat),w.wweibpdf(x,2,3)*w.wraylpdf(y,4))
    phat['x'][2]=10
    given=np.array([.1,1,2,4])
    p=np.array([.1,.3,.7,.9])
    q=w.mdist2dcinv(given,p,phat)
    assert_allclose(w.mdist2dcdf(given,q,phat,1),p,atol=1e-11)
    assert_allclose(w.mdist2dpdf2([1,2],[1,2],phat).f,w.mdist2dpdf(*np.meshgrid([1,2],[1,2]),phat))
    r1,r2=w.mdist2drnd(phat,size=200,seed=7)
    assert np.all(r1>0) and np.all(r2>0)
    assert_allclose(w.mdist2dlike(phat,r1,r2),-np.log(w.mdist2dpdf(r1,r2,phat)).sum())


def test_factorial_design_yates_and_inverse():
    D=w.ffd(3)
    assert_array_equal(D[:4],[[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    X=2*D-1
    y=4+3*X[:,0]-2*X[:,1]+.5*X[:,0]*X[:,2]
    effects,labels=w.yates(y,return_labels=True)
    assert_allclose(effects,[4,6,-4,0,0,1,0,0])
    assert labels==['A','B','AB','C','AC','BC','ABC']
    assert_allclose(w.ryates(effects),y)
    assert_allclose(w.ryates(w.yates(np.column_stack((y,2*y)))),np.column_stack((y,2*y)))


def test_generator_tables_resolution_aliases_and_labels():
    assert w.cnr2cl(w.cl2cnr(['ABC','-DE','aZ']))==['ABC','-DE','Za']
    for n,p,expected in [(3,1,3),(7,4,3),(8,4,4),(8,3,4),(8,2,5),(9,2,6)]:
        generators,resolution=w.sudg(n,p,return_resolution=True)
        relation,r=w.cdr(generators,return_resolution=True)
        assert resolution==r==expected
        D=w.ffd(n,generators)
        assert D.shape==(2**(n-p),n)
        X=2*D-1
        for row in relation:
            nz=row[row!=0]
            assert_array_equal(np.prod(X[:,abs(nz)-1],axis=1),np.full(len(X),np.prod(np.sign(nz))))
    groups=w.alias('ABC',2)
    assert ['A','BC'] in groups
    assert ['B','AC'] in groups
    assert ['AB','C'] in groups
    with pytest.raises(ValueError):
        w.sudg(20,2)
