"""Source/conservation/statistical checks for sampling and gridding helpers."""
import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from wafo import kdetools as k


def test_random_row_sampling_replacement_and_indices():
    data=np.arange(24).reshape(8,3)
    rows,indices=k.sample(data,5,0,seed=7,return_indices=True)
    assert len(np.unique(indices))==5
    assert_array_equal(rows,data[indices])
    assert_array_equal(k.sample(data,8,0,seed=9),data)
    rows,indices=k.sample(data,100,1,seed=8,return_indices=True)
    assert_array_equal(rows,data[indices])
    assert_array_equal(rows,k.sample(data,100,1,seed=8))
    assert k.sample(data,0).shape==(0,3)
    with pytest.raises(ValueError):k.sample(data,9,0)


def test_ball_volumes_and_matrix_kernel_semantics():
    assert_allclose(k.vsph([0,1,2,3],[0,2,2,2]),[1,4,4*np.pi,32*np.pi/3])
    x=np.array([[0,0],[.3,.2],[2,0]])
    for family in ('gaussian','epan','biwe','triw','epa1','biw1','tri1','rect','logistic'):
        assert_allclose(k.mkernel2(x,family),k.mkernel(x,family))
    assert_allclose(k.mkernel2([[0,0]],'gaussian'),[1/(2*np.pi)])
    with pytest.raises(ValueError):k.vsph(1.5)


@pytest.mark.parametrize('kernel,variance',[
    ('gauss',1),('epa1',1/5),('biw1',1/7),('tri1',1/9),('fou1',1/11),
    ('rect',1/3),('tria',1/6),('laplace',2),('logistic',np.pi**2/3),
    ('sphe',1/4),('epan',1/6),('biwe',1/8),('triw',1/10),('four',1/12)])
def test_every_kernel_sampler_has_its_density_moments(kernel,variance):
    rows=k.mkernelrnd(kernel,100000,2,seed=128)
    assert rows.shape==(100000,2)
    assert_allclose(rows.mean(axis=0),0,atol=.025)
    assert_allclose(np.var(rows,axis=0),variance,rtol=.025,atol=.003)
    assert_allclose(np.cov(rows.T)[0,1],0,atol=.035)
    if kernel in ('sphe','epan','biwe','triw','four'):
        assert np.max(np.linalg.norm(rows,axis=1))<=1
    assert_array_equal(k.mkernelrnd(kernel,5,2,seed=9),k.mkernelrnd(kernel,5,2,seed=9))


def test_smoothed_sampling_preserves_mean_covariance_and_reproducibility():
    rng=np.random.default_rng(7)
    data=rng.normal(size=(1000,2))@np.array([[1,.2],[.3,2]])+[2,4]
    options=dict(kernel='gaussian',hs=[[.6,.1],[.1,.8]])
    rows,metadata=k.ssample(data,100000,options,seed=8,full_output=True)
    assert_allclose(rows.mean(axis=0),data.mean(axis=0),atol=.02)
    assert_allclose(np.cov(rows.T,bias=True),np.cov(data.T,bias=True),rtol=.025,atol=.02)
    assert_allclose(metadata['hs'],options['hs'])
    assert_array_equal(rows[:10],k.ssample(data,100000,options,seed=8)[:10])
    constant=np.tile([2,3],(5,1))
    assert_array_equal(k.ssample(constant,4,seed=2),np.tile([2,3],(4,1)))


def test_log_smoothed_adaptive_sampling_and_power_boundary():
    data=np.exp(np.linspace(-2,2,200))
    options=dict(kernel='gaussian',hs=.3,L2=0,alpha=.5)
    rows=k.ssample(data,50000,options,seed=11)
    assert np.all(rows>0)
    assert_allclose(np.log(rows).mean(),np.log(data).mean(),atol=.02)
    assert_allclose(np.log(rows).var(),np.log(data).var(),rtol=.025)
    options=dict(kernel='gaussian',hs=10,L2=.5)
    with pytest.raises(ValueError,match='boundary'):
        k.ssample(data,100,options,seed=3,preserve_covariance=False)
    reflected=k.ssample(data,100,options,seed=3,preserve_covariance=False,boundary='reflect')
    assert np.all(reflected>0)


def test_nearest_grid_accumulation_padding_mean_and_zero_occupancy():
    grid,axes=k.ffndgrid([0,1,2,2],[0,2,4,6],-3+7j,limits=[0,2,-100,100,0])
    assert_array_equal(axes[0],[0,1,2]);assert_allclose(grid,[0,2,5])
    grid,_=k.ffndgrid([0,2,2],[0,4,6],-3+7j,limits=[0,2,-100,100,0],aver=0)
    assert_allclose(grid,[0,7,10])
    threshold,_=k.ffndgrid([0,2,2],[0,4,6],-3+7j,limits=[0,2,-100,100,1])
    assert_allclose(threshold,[7,7,5])
    positions=[[0,0],[1,0],[1,1],[1,1]]
    grid,axes=k.ffndgrid(positions,[1,2,3,4],-2,aver=0)
    assert_allclose(grid,[[1,2],[0,7]])
    averaged,_=k.ffndgrid(positions,1,[1,1],aver=1)
    assert_allclose(averaged,[[1,1],[0,1]])
    with pytest.raises(ValueError):k.ffndgrid(positions,1,[-1,-2])
