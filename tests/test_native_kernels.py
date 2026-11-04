"""Source-derived algorithm and physical-invariant tests for native replacements."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from wafo.wsim._legacy_rng import ParkMiller, simduff
from wafo.wsim._second_order import disufq
from wafo.spec import w2k
from wafo.onedim import findcross, findrfc
from wafo.misc import csort


def test_park_miller_known_integer_sequence_and_resume():
    # Standard published minimal-standard recurrence checkpoint, independent of
    # floating-point output and the original Fortran split-word arithmetic.
    rng = ParkMiller(1)
    for _ in range(10000):
        u = rng.random()
        assert 0 < u < 1
    assert rng.state == 1043618065
    restored = ParkMiller(rng.state)
    assert rng.random() == restored.random()


def test_gaussian_source_consumes_two_uniforms_without_cache():
    rng = ParkMiller(12345)
    first = 207482415/2147483647
    second = 1790989824/2147483647
    expected = np.sqrt(-2*np.log(first))*np.cos(2*np.pi*second)
    assert_allclose(rng.normal(), expected, atol=2e-15)
    assert rng.state == 1790989824


def test_stable_cauchy_special_case():
    # For alpha=1 the source expression simplifies exactly to tan(pi*(u-.5)).
    state = ParkMiller(12345)
    first = 207482415/2147483647
    assert_allclose(state.stable(1), np.tan(np.pi*(first-.5)), atol=2e-14)
    assert state.state == 1790989824


def test_simduff_deterministic_euler_matrix_identity_and_output_indices():
    # With z=a=0,b=1 the independent exact discrete solution is a matrix power.
    seed, dt, n = 123456789, .002, 250
    initial_rng = ParkMiller(seed)
    initial = np.array([initial_rng.normal(), initial_rng.normal()])
    output, final = simduff(n, dt, 0, 0, 1, 2, seed, return_seed=True)
    assert output.shape == (2,3)
    update = np.array([[1,dt],[-dt,1]])
    assert_allclose(output[:,0], [.2,.4])
    for row, index in zip(output, (100,200)):
        assert_allclose(row[1:], np.linalg.matrix_power(update,index-1)@initial, rtol=1e-13)
    assert final == seed*pow(16807,2*(n+1),2147483647)%2147483647
    assert simduff(99,dt,0,seed=seed).shape == (0,3)


@pytest.mark.parametrize('alpha',[.75,1,1.5,2])
def test_simduff_stochastic_paths_are_reproducible(alpha):
    a, state = simduff(300,.001,.15,.1,-1,alpha,123456789,return_seed=True)
    b, state_b = simduff(300,.001,.15,.1,-1,alpha,123456789,return_seed=True)
    assert_array_equal(a,b)
    assert state == state_b
    assert np.all(np.isfinite(a))


@pytest.mark.parametrize('seed',[0,-1,2147483647,1.5,True])
def test_park_miller_rejects_invalid_state(seed):
    with pytest.raises(ValueError):
        ParkMiller(seed)


@pytest.mark.parametrize('args',[(2.5,.01,.1),(1,.01,.1),(100,0,.1),(100,.01,-.1),(100,.01,np.nan)])
def test_simduff_invalid_inputs(args):
    with pytest.raises(ValueError):
        simduff(*args)


@pytest.mark.parametrize('depth',[.5,2,10,100])
def test_disufq_finite_depth_diagonal_matches_stokes_coefficient(depth):
    # Independent simplification using w²=g*k*tanh(kh): the second-order
    # sum-frequency coefficient is k*(3-tanh(kh)²)/(2*tanh(kh)³).
    w = np.arange(5,dtype=float)*1.1
    k,_ = w2k(w,h=depth)
    a = np.zeros(12,complex)
    a[2] = .2-.3j
    s,d = disufq(a,w,k,depth,9.81,2,2,components=True)
    sigma = np.tanh(k[2]*depth)
    coefficient = k[2]*(3-sigma**2)/(2*sigma**3)
    assert_allclose(s[4], 2*a[2]**2*coefficient, rtol=2e-13)
    assert np.count_nonzero(s) == 1
    assert np.count_nonzero(d[1:]) == 0


@pytest.mark.parametrize('depth',[2,100,np.inf])
def test_disufq_phase_and_simulation_row_invariants(depth):
    w = np.arange(6,dtype=float)
    k,_ = w2k(w,h=depth)
    rng = np.random.default_rng(1)
    a = np.zeros((3,16),complex)
    a[:,1:5] = rng.normal(size=(3,4))+1j*rng.normal(size=(3,4))
    s,d = disufq(a,w,k,depth,9.81,1,4,components=True)
    rotation = np.exp(.7j)
    rotated_s, rotated_d = disufq(a*rotation,w,k,depth,9.81,1,4,components=True)
    assert_allclose(rotated_s,rotation**2*s,atol=2e-14)
    assert_allclose(rotated_d,d,atol=2e-14)
    for i in range(3):
        assert_allclose(disufq(a[i],w,k,depth,9.81,1,4),s[i]+d[i])


@pytest.mark.parametrize('change',[
    {'A':np.zeros((2,2,2))}, {'w':[[0,1,2]]}, {'nmin':1.5}, {'nmax':True},
    {'h':np.nan}, {'g':np.inf}, {'w':[0,np.nan,2]}, {'kw':[0,1,np.inf]},
    {'w':[0,2,1]}, {'kw':[0,2,1]}, {'A':np.zeros(4)},
])
def test_disufq_invalid_inputs(change):
    args = dict(A=np.zeros(8),w=np.arange(3),kw=np.arange(3),h=20,g=9.81,nmin=1,nmax=2)
    args.update(change)
    with pytest.raises(ValueError):
        disufq(**args)


def test_disufq_rejects_resonant_diagonal_instead_of_nan():
    # Chosen without enforcing the physical dispersion relation so the source
    # diagonal denominator 1-(.5*g*k/w²)*tanh(2kh) is exactly zero.
    k = np.array([0,2.])
    g,h = 9.81,1.
    w = np.array([0,np.sqrt(.5*g*k[1]*np.tanh(2*k[1]*h))])
    with pytest.raises(FloatingPointError,match='denominator'):
        disufq(np.ones(4),w,k,h,g,1,1)


def test_crossing_sort_and_rainflow_native_examples():
    assert_array_equal(findcross([0,0,-1,0,0,2,0,3,-1]),[1,4,7])
    values, index = csort([4,-2,4,-2,0])
    assert_array_equal(values,[-2,-2,0,4,4])
    assert_array_equal(index,[1,3,4,0,2])
    assert_array_equal(findrfc([5,0,2,1,3,0],.5),[1,2,3,4])
