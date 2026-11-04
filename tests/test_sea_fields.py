import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from wafo import wsim, spec


def _single_mode():
    S = spec.createspec('dir', w=[0, 1, 2], S=np.zeros((3, 3)))
    S.theta = np.array([-np.pi/4, 0, np.pi/4])
    S.S[1, 1] = 4/np.pi  # trapezoid direction/frequency mass exactly one
    return S


def test_direct_sea_is_gaussian_and_has_exact_traveling_phase_and_covariance():
    S = _single_mode()
    a = wsim.seasim(S, 2, 0, 2, dx=9.81*.3, dt=.3, fftdim=1, seed=13)
    b = wsim.seasim(S, 2, 0, 2, dx=9.81*.3, dt=.3, fftdim=1, seed=13)
    assert_array_equal(a.Z, b.Z)
    assert a.Z.shape == (2, 2) and a.shape == (1, 2, 2)
    assert_allclose(a.Z[1, 0], a.Z[0, 1])
    assert_allclose(a.discrete_variance, 1)
    samples = np.array([wsim.seasim(S, 1, 0, 2, dt=.3, fftdim=1, seed=seed).Z for seed in range(1200)])
    assert_allclose(np.mean(samples, axis=0), 0, atol=.08)
    assert_allclose(np.cov(samples.T), [[1, np.cos(.3)], [np.cos(.3), 1]], atol=.08)
    S2 = dict(S, tr=np.column_stack((2*np.linspace(-10, 10, 100)+3, np.linspace(-10, 10, 100))))
    transformed = wsim.seasim(S2, 2, 0, 2, dx=9.81*.3, dt=.3, fftdim=1, seed=13)
    assert_allclose(transformed.Z, 2*a.Z+3)


def _broadband():
    S = spec.createspec('dir', w=np.linspace(0, 3, 65))
    S.theta = np.linspace(-np.pi, np.pi, 33)
    S.S = np.tile(S.w**4*np.exp(-S.w**2)/ (2*np.pi), (33, 1))
    return S


def test_fft_field_variance_is_reported_without_rescaling_and_depth_changes_dispersion():
    S = _broadband()
    a = wsim.seasim(S, 12, 10, 3, dx=4, dy=4, dt=.5, fftdim=2, seed=2)
    assert a.Z.shape == (10, 12, 3)
    assert .9 < a.discrete_variance/a.target_variance < 1.1
    # The Gaussian law at any point has the reported Cartesian quadrature variance.
    samples = [wsim.seasim(S, 1, 1, 1, dx=4, dy=4, dt=.5, fftdim=2, seed=seed).Z.item() for seed in range(400)]
    assert abs(np.var(samples)/a.discrete_variance-1) < .2
    shallow = wsim.seasim(dict(S, h=15), 12, 10, 3, dx=4, dy=4, dt=.5, seed=2)
    assert not np.allclose(shallow.Z, a.Z)
    with pytest.warns(RuntimeWarning, match='variance differs'):
        lost = wsim.seasim(S, 1, 1, 2, dx=100, dy=100, seed=2)
    assert abs(lost.discrete_variance/lost.target_variance-1) > .1


def test_movie_frames_and_optional_seasim_graphics_are_portable(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    S = _broadband()
    field, movie = wsim.seasim(S, 8, 0, 3, fftdim=1, seed=3, plotflag=1, full_output=True)
    assert movie['frames'].dtype == np.uint8 and movie['frames'].shape[0] == 1
    assert movie['frames'].shape[-1] == 4
    plt.close(movie['figure'])
    field = wsim.seasim(S, 7, 6, 3, fftdim=1, seed=3)
    for style in (1, 2, 3):
        movie = wsim.seamovie(field, style, frames=[0, 2])
        assert_array_equal(movie['times'], field.t[[0, 2]])
        assert movie['frames'].shape[0] == 2
        assert not np.array_equal(movie['frames'][0], movie['frames'][1])
        target = tmp_path/f'frame-{style}.png'
        movie['figure'].savefig(target)
        assert target.stat().st_size > 1000
        plt.close(movie['figure'])
    field, movie = wsim.seasim(S, 3, 0, 2, fftdim=1, seed=3, plotflag=2, full_output=True)
    assert len(movie['frames']) == 2
    plt.close(movie['figure'])
    assert wsim.seasim(S, 3, 0, 2, fftdim=1, full_output=True)[1] is None


def test_sea_field_input_validation_is_explicit():
    S = _single_mode()
    for kwargs in ({'Nx': 0}, {'Ny': -1}, {'Nt': 1.5}, {'dx': 0}, {'fftdim': 3}, {'plotflag': 3}):
        with pytest.raises(ValueError): wsim.seasim(S, **kwargs)
    with pytest.raises(ValueError, match='directional'):
        wsim.seasim(spec.jonswap())
    with pytest.raises(ValueError, match='match'):
        wsim.seamovie({'Z': [1, 2], 'x': [0, 1], 't': [0, 1]})


def test_irregular_spectral_grid_uses_nonnegative_mode_variances():
    S = spec.createspec('dir', w=[0, .01, 1, 2], S=np.ones((3, 4)))
    S.theta = np.array([-1., -.99, 1.])
    field = wsim.seasim(S, 2, 0, 3, fftdim=1, seed=1)
    assert np.all(np.isfinite(field.Z))
    assert_allclose(field.discrete_variance, 4)
