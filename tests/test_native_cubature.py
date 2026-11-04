import numpy as np
import pytest
from numpy.testing import assert_allclose
from wafo.source._integration import integrate_box


def test_scalar_polynomial_integral_and_budget():
    result = integrate_box(lambda x:x[:,0]*x[:,1]**2,[0,0],[2,3],maxpts=10000,seed=5)
    assert_allclose(result.value,18,atol=.002)
    assert result.evaluations<=10000
    assert result.error>=0


def test_vector_integral_and_reproducibility():
    f = lambda x:np.column_stack((np.ones(len(x)),x[:,0],x[:,0]**2))
    a = integrate_box(f,[0],[1],seed=7)
    b = integrate_box(f,[0],[1],seed=7)
    assert_allclose(a.value,[1,.5,1/3],atol=1e-6)
    assert_allclose(a.value,b.value,atol=0,rtol=0)
    assert a.converged


@pytest.mark.parametrize('lower,upper', [([1],[0]),([0],[np.inf]),([],[])])
def test_invalid_rectangle(lower,upper):
    with pytest.raises(ValueError):
        integrate_box(lambda x:x[:,0],lower,upper)
