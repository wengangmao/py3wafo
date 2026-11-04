"""Source-coefficient transforms and real packed-correlation gateway tests."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

from wafo import trgauss as tr
from wafo.trgauss._compatibility import _unpack_correlation


def test_hermite_coefficients_softening_round_trip_and_exact_departure():
    c3, c4 = .05, .03
    y = np.linspace(-4, 4, 301)
    K = 1/np.sqrt(1+2*c3*c3+6*c4*c4)
    x = 2+3*K*(y+c3*(y*y-1)+c4*(y**3-3*y))
    table, departure = tr.hermitefun([c3, c4], x, 2, 3, full_output=True)
    assert_array_equal(table[:, 0], x)
    assert_allclose(table[:, 1], y, atol=2e-13)
    z = (x-2)/3
    assert_allclose(departure, np.trapezoid((z-y)**2, z), atol=1e-13)
    # This private helper takes coefficients, rather than moments like hermitetr.
    assert_allclose(tr.hermitefun([0, 0], x, 2, 3)[:, 1], z, atol=2e-13)
    cubic = tr.hermitefun([0, 1/3], np.linspace(-2, 2, 101))
    assert_allclose(cubic[:, 1], np.cbrt(cubic[:, 0]*3*np.sqrt(1+6/9)), atol=3e-13)


def test_hermite_hardening_and_valid_quadratic_branch():
    x = np.linspace(-2, 2, 101)
    c3, c4 = .03, -.02
    direct = tr.hermitefun([c3, c4], x, gdef=-1)
    assert_allclose(direct[:, 1], x-c3*(x*x-1)-c4*(x**3-3*x))
    quadratic = tr.hermitefun([.1, 0], x)
    K = 1/np.sqrt(1.02)
    assert_allclose(K*(quadratic[:, 1]+.1*(quadratic[:, 1]**2-1)), x, atol=2e-13)
    with pytest.raises(ValueError, match='central'):
        tr.hermitefun([.3, 0], np.linspace(-5, 5, 20))
    with pytest.raises(ValueError, match='increasing'):
        tr.hermitefun([1, .2], np.linspace(-2, 2, 20), gdef=-1)


def test_ochi_direct_parameters_zero_limit_nan_defaults_and_normalization():
    x = np.linspace(-3, 5, 101)
    parameters = [.1, .2, 2, 1, .8, -.3]
    table, departure = tr.ochifun(x, parameters, full_output=True)
    z = (x-1)/2
    gamma = np.where(z >= 0, .1, .2)
    expected = ((1-np.exp(-gamma*z))/gamma+.3)/.8
    assert_allclose(table[:, 1], expected, atol=3e-15)
    assert_allclose(departure, np.trapezoid((z-expected)**2, z), atol=1e-14)
    assert_allclose(tr.ochifun(x, [0, 0, 2, 1, .8, -.3])[:, 1], (z+.3)/.8)
    assert_allclose(tr.ochifun(x, [np.nan]*6), tr.ochifun(x))
    tiny = tr.ochifun(x, [1e-15, 1e-15, 2, 1])
    assert_allclose(tiny[:, 1], z, atol=3e-15)
    assert tr.ochifun().shape == (513, 2)
    with pytest.raises(ValueError):
        tr.ochifun(x, [1, 1, 0])


@pytest.mark.parametrize('gateway', [tr.mexmvnprb, tr.mexmvnprb2, tr.mexGenzMvnPrb])
def test_packed_gateways_nontrivial_three_dimensional_probability(gateway):
    rho = np.array([.3, -.5, .7])
    R = np.outer(rho, rho); np.fill_diagonal(R, 1.)
    a = np.array([-1.5, -.2, -.7]); b = np.array([.6, 1.8, .3])
    expected = tr.mvnormpcprb_full(rho, a, b, tol=1e-12)[0]
    result = gateway(R[np.tril_indices(3, -1)], a, b, 2e-6, 1e-5, 131072, seed=71)
    value, error, inform = result
    assert_allclose(value, expected, atol=3e-6)
    assert error >= 0 and inform in (0, 1)
    assert_array_equal(gateway(R[np.tril_indices(3, -1)], a, b, 2e-6, 1e-5, 131072, seed=71), result)


def test_four_dimensional_row_packing_distinguishes_misleading_matlab_note():
    rho = np.array([.2, .4, -.5, .7])
    R = np.outer(rho, rho); np.fill_diagonal(R, 1.)
    row_packed = R[np.tril_indices(4, -1)]
    column_packed = R.T[np.triu_indices(4, 1)]
    assert not np.array_equal(row_packed, column_packed)
    assert_array_equal(_unpack_correlation(row_packed, 4), R)
    a = [-.5, -1., .3, -1.8]; b = [1., .1, 1.5, .2]
    expected = tr.mvnormpcprb_full(rho, a, b, tol=1e-12)[0]
    value, error, flag = tr.mexmvnprb(row_packed, a, b, 1e-6, 1e-5, 262144, seed=18)
    assert_allclose(value, expected, atol=3e-6)
    wrong = tr.mexmvnprb(column_packed, a, b, 1e-6, 1e-5, 262144, seed=18)[0]
    assert abs(wrong-expected) > 1e-4


def test_packed_limits_infinity_sentinels_and_singular_inputs():
    value, error, flag = tr.mexmvnprb([], [-37], [0.])
    assert_allclose(value, .5); assert error == 0 and flag == 0
    # A singular covariance yields a discontinuous conditional integrand;
    # verify reported numerical accuracy rather than pretend it is analytic.
    value, error, flag = tr.mexmvnprb([1.], [-100, -.4], [.7, 37], 1e-5, 1e-5, 262144, seed=4)
    assert abs(value-(stats.norm.cdf(.7)-stats.norm.cdf(-.4))) < 3*error
    assert error < 3e-5 and flag in (0, 1)
    # The intended 500D mexmvnprb2 gateway repairs its stale native arrays.
    n = 101
    assert tr.mexmvnprb(np.zeros(n*(n-1)//2), np.full(n, -100.), np.full(n, 100.)) == (0., 1., 2)
    assert tr.mexmvnprb2(np.zeros(n*(n-1)//2), np.full(n, -100.), np.full(n, 100.)) == (1., 0., 0)
    assert tr.mexmvnprb2([], [], []) == (0., 1., 2)
    with pytest.raises(ValueError):
        tr.mexmvnprb([.1], [-1, -1, -1], [1, 1, 1])
    with pytest.raises(ValueError):
        tr.mexmvnprb([-.9, -.9, -.9], [-1]*3, [1]*3)
    with pytest.raises(NotImplementedError, match='Native method'):
        tr.mexGenzMvnPrb([.2], [-1, -1], [1, 1], method=2)
