import itertools
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import ndtr, roots_hermitenorm
from wafo import trgauss as tr


def _normal_nodes(S, order):
    nodes, weights = roots_hermitenorm(order)
    index = np.array(list(itertools.product(range(order), repeat=len(S))))
    eigenvalues, vectors = np.linalg.eigh(S)
    root = vectors*np.sqrt(np.maximum(eigenvalues, 0))
    x = nodes[index]@root.T
    mass = np.prod(weights[index]/np.sqrt(2*np.pi), axis=1)
    return x, mass


def test_holmquist_moments_match_exact_gaussian_polynomial_quadrature():
    S = np.array([[1, .3], [.3, 2.]])
    forms = [np.array([[1, .2], [.4, 2]]), np.array([[.5, -.3], [-.3, 1.1]]), np.array([[-1, .1], [.1, .6]])]
    x, mass = _normal_nodes(S, 5)
    product = np.ones(len(x))
    for j, fn in enumerate((tr.holmquist1, tr.holmquist2, tr.holmquist3)):
        product *= np.einsum('ni,ij,nj->n', x, forms[j], x)
        assert_allclose(fn(S, *forms[:j+1]), mass@product, rtol=1e-13)
    singular = np.array([[1, 1], [1, 1.]])
    assert_allclose(tr.holmquist3(singular, np.eye(2), np.eye(2), np.eye(2)), 8*15)


def test_holmberg_moments_match_independent_normal_cdf_quadrature():
    S = np.array([[1., .25], [.25, .7]])
    a, b = np.array([.4, -.2]), np.array([.3, .9])
    forms = [np.array([[1, .2], [.3, .7]]), np.array([[.5, -.3], [-.3, 1.1]]), np.array([[-1, .1], [.1, .6]])]
    x, mass = _normal_nodes(S, 40)
    product = np.sqrt(np.pi/2)*(x@b)*(2*ndtr(x@a)-1)
    for j, fn in enumerate((tr.holmberg1, tr.holmberg2, tr.holmberg3)):
        product *= np.einsum('ni,ij,nj->n', x, forms[j], x)
        assert_allclose(fn(S, a, b, *forms[:j+1]), mass@product, rtol=2e-12, atol=1e-13)
        assert_allclose(fn(S, -a, b, *forms[:j+1]), -mass@product, rtol=2e-12, atol=1e-13)
        assert fn(S, np.zeros(2), b, *forms[:j+1]) == 0
    # Symmetry in the quadratic forms follows from their actual product.
    assert_allclose(tr.holmberg3(S, a, b, *forms), tr.holmberg3(S, a, b, *forms[::-1]))


def test_first_holmberg_moment_matches_original_closed_formula():
    S = np.array([[2., .1], [.1, 1]])
    Q = np.array([[.5, .2], [.2, 1.5]])
    a, b = np.array([.2, .3]), np.array([1., -1])
    c = a@S@a
    expected = ((a@S@b)*np.trace(S@Q)+2*a@S@Q@S@b-(a@S@b)*(a@S@Q@S@a)/(1+c))/np.sqrt(1+c)
    assert_allclose(tr.holmberg1(S, a, b, Q), expected, rtol=1e-14)


def test_quadratic_cumulant_generator_matches_noncentral_chi_square_and_affine_normal():
    s, gamma, beta = .12, .3, 1.4
    expected = -.5*np.log1p(-2*s*gamma)+.5*(s*beta)**2/(1-2*s*gamma)
    assert_allclose(tr.kchitwo(s, 0, [gamma], [beta], [[0]], [[1]]), expected)
    b = np.array([1., 2]); C = np.array([[.1, .2], [-.1, .3]])
    D = C.T@C+np.array([[1., .1], [.1, 2.]])
    t = .15
    expected = (s*s*(b@b)+t*t*(b@D@b)+2*s*t*(b@C@b))/2
    assert_allclose(tr.kchitwo(s, t, [0, 0], b, C, D), expected, rtol=1e-13)


def test_joint_quadratic_cumulant_generator_matches_direct_gaussian_quadrature():
    # Z'=c Z+sqrt(v)E, with Z,E independent standard normals.
    gamma, beta, c, v, s, t = .2, .7, .15, .8, .08, -.06
    points, weights = _normal_nodes(np.eye(2), 35)
    z, zp = points[:, 0], c*points[:, 0]+np.sqrt(v)*points[:, 1]
    value = np.log(weights@np.exp(s*(gamma*z*z+beta*z)+t*(2*gamma*z+beta)*zp))
    assert_allclose(tr.kchitwo(s, t, [gamma], [beta], [[c]], [[c*c+v]]), value, rtol=2e-12)
    with pytest.raises(ValueError, match='domain'):
        tr.kchitwo(2, 0, [1], [1], [[0]], [[1]])
    with pytest.raises(ValueError, match='semidefinite'):
        tr.kchitwo(0, 0, [1], [1], [[2]], [[1]])


def test_rational_addition_preserves_values_and_cancels_initial_common_zeros():
    A = np.array([[1., 2], [3, 1]])
    B = np.array([[2., -1, .5], [1, .2, 1]])
    result = tr.rfadd(A, B)
    x = np.linspace(.1, 2, 50)
    def value(R):
        return np.polynomial.polynomial.polyval(x, R[0])/np.polynomial.polynomial.polyval(x, R[1])
    assert_allclose(value(result), value(A)+value(B), rtol=1e-14)
    assert_allclose(tr.rfadd(np.pad(A, ((0, 0), (1, 0))), B), result)
    with pytest.raises(ValueError):
        tr.rfadd([[1, 2], [0, 0]], B)
    with pytest.raises(ValueError):
        tr.holmquist1([[1, 2], [2, 1]], np.eye(2))
