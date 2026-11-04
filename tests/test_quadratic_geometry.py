import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize
from wafo import trgauss as tr


def test_closest_point_linear_one_dimensional_and_repeated_hard_cases():
    b = np.array([1., 2, -1])
    assert_allclose(tr.mindist([0, 0, 0], b, 3), 3*b/(b@b))
    assert_allclose(tr.mindist3([.5], [2], 3), [-2+np.sqrt(10)])
    assert_allclose(tr.mindist3([.5], [2], -1), [-2+np.sqrt(2)])
    point, mask = tr.mindist3([2, 2, -1], [0, 0, 0], 8, full_output=True)
    assert_allclose(point, [2, 0, 0])
    assert_array_equal(mask, [1, 1, 0])
    assert_allclose(tr.mindist3([-2, -2, 1], [0, 0, 0], -8), point)
    assert_allclose(tr.mindist3([-1, -2], [2, 4], 3), [1, 1])
    assert_allclose(tr.mindist3([0, 1], [0, 0], 0), [0, 0])
    # Resolving lambda-edge directly retains the tiny nonzero linear term.
    assert_allclose(tr.mindist3([2, -1], [1e-30, .1], 1), tr.mindist3([2, -1], [0, .1], 1), rtol=1e-10)
    with pytest.raises(ValueError, match='outside'):
        tr.mindist3([-1, -2], [2, 4], 3.1)


def test_closest_points_satisfy_surface_and_global_optimality_against_multistart():
    rng = np.random.default_rng(40)
    for g, b, level in [([.2, .5, -.3], [1, -.4, .6], 2),
                         ([.2, .5, -.3], [1, -.4, .6], -1),
                         ([-.2, -.5, -.3], [1, -.4, .6], .2),
                         ([.2, .5, .3], [1, -.4, .6], -.1)]:
        g, b = np.asarray(g), np.asarray(b)
        point = tr.mindist(g, b, level, n0=1)
        assert_allclose(np.sum(g*point**2+b*point), level, rtol=1e-10)
        solutions = []
        for initial in rng.normal(size=(12, 3)):
            result = minimize(lambda x: x@x, initial, jac=lambda x: 2*x,
                              constraints=[{'type': 'eq', 'fun': lambda x: g@(x*x)+b@x-level,
                                            'jac': lambda x: 2*g*x+b}],
                              method='SLSQP', options={'ftol': 1e-12, 'maxiter': 500})
            if result.success: solutions.append(result.fun)
        assert solutions
        assert point@point <= min(solutions)+1e-8
        assert_allclose(point@point, min(solutions), atol=1e-8)


def test_form_sorm_gaussian_limit_is_exact_for_both_signs_and_zero():
    b = np.array([1., 2])
    S12 = np.array([[0, .2], [-.2, 0]])
    S22 = np.array([[2., .1], [.1, 1.]])
    u = np.array([-3., 0, 2])
    expected = np.sqrt((b@S22@b)/(b@b))*np.exp(-u*u/(2*(b@b)))/(2*np.pi)
    for mode in (-1, 0):
        result, details = tr.chitwo2lc_sorm(u, [0, 0], b, S22, S12, mode, full_output=True)
        assert_allclose(result[:, 1], expected)
        assert_allclose(details['points'], b[:, None]*u[None, :]/(b@b))


def test_sorm_single_branch_one_dimension_and_sign_reversal():
    gamma, beta, u = .05, 1., np.array([.5, 1., 2.])
    result = tr.chitwo2lc_sorm(u, [gamma], [beta], [[2]], [[0]])
    closest = (-beta+np.sqrt(beta*beta+4*gamma*u))/(2*gamma)
    expected = np.sqrt(2)*np.exp(-closest**2/2)/(2*np.pi)
    assert_allclose(result[:, 1], expected)
    # A sign change of the process leaves each corresponding crossing rate.
    S22 = [[2, .1], [.1, 1]]; S12 = [[0, .1], [-.1, 0]]
    positive = tr.chitwo2lc_sorm(u, [.04, -.02], [1, .4], S22, S12)
    negative = tr.chitwo2lc_sorm(-u, [-.04, .02], [-1, -.4], S22, S12)
    assert_allclose(positive[:, 1], negative[:, 1], rtol=1e-12)


def test_geometry_invalid_approximation_domains_are_not_silent():
    with pytest.raises(ValueError, match='nonunique'):
        tr.chitwo2lc_sorm([1], [1], [0], [[1]], [[0]])
    with pytest.raises(ValueError, match='Zero levels'):
        tr.chitwo2lc_sorm([0], [.1], [1], [[1]], [[0]])
    with pytest.raises(ValueError, match='skew'):
        tr.chitwo2lc_sorm([1], [.1], [1], [[1]], [[.1]])
