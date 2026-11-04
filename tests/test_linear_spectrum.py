import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from wafo import spec, wsim


def test_linear_nonlinear_spectra_decompose_target_and_converge_reproducibly():
    S = spec.jonswap(np.linspace(0, 3, 65), [2, 8])
    original = S.S.copy()
    L, N, info = wsim.spec2linspec(S, [256, 8], seed=2, tolerance=1e-5, max_iter=60, full_output=True)
    assert info['converged'] and info['iterations'] < 60
    assert_array_equal(S.S, original)
    assert np.all(L.S >= 0) and np.all(N.S >= 0) and np.max(N.S) > 0
    assert_allclose(L.S+N.S, S.S, rtol=1e-14)
    assert info['cases'] == 8 and info['samples'] == 256
    L2, N2 = wsim.spec2linspec(S, [256, 8], seed=2, tolerance=1e-5, max_iter=60)
    assert_array_equal(L.S, L2.S)
    assert_array_equal(N.S, N2.S)


def test_quadratic_power_and_separation_obey_weak_wave_amplitude_order():
    S = spec.jonswap(np.linspace(0, 3, 65), [1, 8])
    _, _, full = wsim.spec2linspec(S, [256, 5], seed=10, tolerance=1, damping=1, full_output=True)
    small = dict(S, S=S.S/4)
    L, N, weak = wsim.spec2linspec(small, [256, 5], seed=10, tolerance=1, damping=1, full_output=True)
    assert_allclose(weak['linear_power'], full['linear_power']/4, rtol=1e-11, atol=1e-25)
    assert_allclose(weak['quadratic_power'], full['quadratic_power']/16, rtol=1e-10, atol=1e-25)
    assert_allclose(L.S+N.S, small['S'])


def test_zero_spectrum_and_budget_or_invalid_options_are_explicit():
    S = spec.jonswap(np.linspace(0, 3, 33), [2, 8])
    with pytest.warns(RuntimeWarning, match='max_iter'):
        _, _, info = wsim.spec2linspec(S, [128, 2], max_iter=1, tolerance=1e-15, seed=3, full_output=True)
    assert not info['converged']
    zero = dict(S, S=np.zeros_like(S.S))
    L, N, info = wsim.spec2linspec(zero, full_output=True)
    assert info['converged'] and info['iterations'] == 0
    assert_allclose(L.S+N.S, 0)
    for kwargs in ({'max_iter': 0}, {'damping': 0}, {'fnLimit': np.inf}, {'seed': 1, 'iseed': 2}):
        with pytest.raises(ValueError): wsim.spec2linspec(S, **kwargs)
