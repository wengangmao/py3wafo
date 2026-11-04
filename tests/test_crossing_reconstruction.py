import numpy as np
import pytest
from numpy.testing import assert_allclose
from wafo.damage import down2cc, lc2dplus


def test_triangle_layer_cake_identity():
    # mu(u)=1-|u|. At slice y the cycle amplitude is 1-y.
    cc,delta = down2cc([[-1,0],[0,1],[1,0]],1000)
    assert cc.shape == (1000,2)
    assert_allclose(cc[:,0],-cc[:,1])
    assert_allclose(delta, .001)
    assert_allclose(lc2dplus([[-1,0],[0,1],[1,0]],[1,2,3],1000),[1/2,1/3,1/4],rtol=3e-6)


def test_multiple_lobes_stay_disconnected():
    cc,delta = down2cc([[-3,0],[-2,1],[-1,0],[0,0],[1,1],[2,0]],20)
    assert cc.shape == (40,2)
    assert np.all((cc[:,1] <= -1) | (cc[:,0] >= 0))
    assert_allclose(delta, .05)


def test_refinement_and_empty_intensity():
    curve = [[-2,0],[0,2],[2,0]]
    coarse = lc2dplus(curve,[4],10)[0]
    fine = lc2dplus(curve,[4],1000)[0]
    exact = 2*2**4/5
    assert abs(fine-exact) < abs(coarse-exact)
    c,d = down2cc([[-1,0],[0,0],[1,0]])
    assert c.shape == (0,2) and d==0


@pytest.mark.parametrize('mu,n', [([[0,1],[1,2],[2,0]],10),
    ([[0,0],[1,-1],[2,0]],10),([[0,0],[0,1],[2,0]],10),
    ([[0,0],[1,1],[2,0]],0),([[0,0],[1,1],[2,0]],1.5)])
def test_invalid_curves(mu,n):
    with pytest.raises(ValueError):
        down2cc(mu,n)
