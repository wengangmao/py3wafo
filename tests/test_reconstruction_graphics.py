import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from wafo import onedim as o


def test_reconstruct_known_ar1_conditional_law_and_preserves_observations():
    rho = .7
    cov = {'R': 4*rho**np.arange(5), 't': .2*np.arange(5)}
    x = np.column_stack((.2*np.arange(5), [99., 101., np.nan, 103., 100.]))
    result, info = o.reconstruct(x, covariance=cov, mean=100, definition='linear', full_output=True)
    expected = 100+rho/(1+rho**2)*4
    sigma = 2*np.sqrt((1-rho**2)/(1+rho**2))
    assert_allclose(result[2, 1], expected)
    assert_array_equal(result[:, 0], x[:, 0])
    assert_array_equal(result[[0, 1, 3, 4]], x[[0, 1, 3, 4]])
    assert_allclose(info['latent_mean'], [(expected-100)/2])
    assert_allclose(info['latent_std'], [sigma/2])
    assert info['method'] == 'exact' and info['iterations'] == 1
    draws = [o.reconstruct(x, covariance=cov, mean=100, definition='linear', sample=True, seed=seed)[2, 1] for seed in range(500)]
    assert abs(np.mean(draws)-expected) < .16
    assert abs(np.std(draws)-sigma) < .15
    assert_allclose(o.reconstruct(x, covariance=cov, mean=100, definition='linear', sample=True, seed=5),
                    o.reconstruct(x, covariance=cov, mean=100, definition='linear', sample=True, seed=5))


@pytest.mark.parametrize('definition', ['linear', 'nonlinear', 'mnonlinear'])
def test_estimated_reconstruction_iterates_and_keeps_known_values(definition):
    rng = np.random.default_rng(21)
    t = np.arange(300)*.2
    original = np.sin(2*t)+.1*rng.normal(size=len(t))
    original = np.sinh(original/2)
    missing = np.array([20, 21, 155])
    x = original.copy()
    x[missing] = np.nan
    filled, info = o.reconstruct(x, Nsim=3, L=12, definition=definition, full_output=True)
    assert np.all(np.isfinite(filled))
    assert_array_equal(filled[np.isfinite(x)], original[np.isfinite(x)])
    assert_array_equal(info['inds'], missing)
    assert 1 <= info['iterations'] <= 3 and np.all(info['latent_std'] >= 0)
    assert np.all(np.diff(info['g'][:, 0]) > 0) and np.all(np.diff(info['g'][:, 1]) > 0)
    assert info['covariance']['R'][0] > 0
    # The same spurious indices can be explicit, or supplied as a mask.
    explicit = o.reconstruct(original[:, None], missing, Nsim=3, L=12, definition=definition)
    mask = np.zeros(len(x), dtype=bool); mask[missing] = True
    assert_allclose(explicit[:, 0], filled)
    assert_allclose(o.reconstruct(original, mask, Nsim=3, L=12, definition=definition), filled)


def test_local_conditioning_matches_ar1_markov_prediction_for_long_records():
    x = np.sin(np.arange(1100)*.05)
    x[[40, 800]] = np.nan
    rho = .6
    cov = {'R': rho**np.arange(len(x))}
    filled, info = o.reconstruct(x, definition='linear', mean=0, covariance=cov, context=10, max_points=64, full_output=True)
    assert info['method'] == 'local'
    assert_allclose(filled[[40, 800]], rho/(1+rho**2)*(x[[39, 799]]+x[[41, 801]]))
    assert_allclose(info['latent_std'], np.sqrt((1-rho**2)/(1+rho**2)))
    with pytest.raises(ValueError, match='max_points'):
        o.reconstruct(x, definition='linear', mean=0, covariance=cov, conditioning='exact', max_points=64)
    with pytest.raises(ValueError, match='window'):
        o.reconstruct(x, definition='linear', mean=0, covariance=cov, context=40, max_points=64)


def test_reconstruction_validation_and_no_missing_copy():
    x = np.arange(8.)
    filled, info = o.reconstruct(x, definition='linear', full_output=True)
    assert_array_equal(filled, x)
    assert filled is not x and not len(info['inds']) and info['iterations'] == 0
    for kwargs in ({'inds': [8]}, {'inds': [1.5]}, {'inds': [True]}, {'Nsim': 0}, {'definition': 'bad'}, {'L': 8}, {'tolerance': 0}):
        with pytest.raises(ValueError):
            o.reconstruct(x, **kwargs)
    with pytest.raises(ValueError, match='three known'):
        o.reconstruct([1, np.nan, np.nan, 2])
    with pytest.raises(ValueError, match='uniformly'):
        o.reconstruct([[0, 1], [1, 2], [3, 3]])


def _save(axes, path):
    import matplotlib.pyplot as plt
    figures = {ax.figure for ax in np.asarray(axes, dtype=object).reshape(-1)}
    for index, figure in enumerate(figures):
        target = path.with_name(path.stem+f'-{index}.png')
        figure.savefig(target)
        assert target.stat().st_size > 1000
        plt.close(figure)


def test_crossing_plot_gaussian_normalization_and_short_spectra(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    x = np.linspace(-3, 3, 51)
    counts = 40*np.exp(-x*x/(2*.7**2))
    for mode in (1, 2, 11, 12):
        ax = o.lcplot(np.column_stack((x, counts)), mode)
        assert_allclose(ax.lines[0].get_ydata(), counts/(40 if mode > 2 else 1))
        if mode % 10 == 2:
            assert_allclose(ax.lines[1].get_ydata(), ax.lines[0].get_ydata(), rtol=1e-12)
        _save(ax, tmp_path/f'crossing-{mode}.png')
    _save(o.lcplot([[-1, 0], [0, 4], [1, 0]], sa=1), tmp_path/'short.png')


def test_covariance_graphics_have_correct_coordinates_sections_and_bands(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    t = np.arange(12)*.2
    R = {'R': np.exp(-t), 't': t, 'stdev': np.full(12, .1)}
    for mode in (1, 2, 3):
        axes = o.covplot(R, 6, mode)
        for ax, physical in zip(np.atleast_1d(axes), [False, True] if mode == 3 else [mode == 1]):
            assert_allclose(ax.lines[0].get_xdata(), t[:7] if physical else np.arange(7))
            assert_allclose(ax.lines[2].get_ydata(), .2)
        _save(axes, tmp_path/f'covariance-{mode}.png')
    x = np.linspace(0, 2, 8)
    R2 = {'R': np.exp(-x[:, None]**2-t[None, :]**2), 'x': x, 't': t}
    for mode in (1, 2, 3):
        axes = o.covplot(R2, [5, 6], mode)
        if mode == 3:
            assert_allclose(axes[0].lines[0].get_ydata(), R2['R'][:6, 0])
            assert_allclose(axes[1].lines[0].get_ydata(), R2['R'][0, :7])
        _save(axes, tmp_path/f'covariance2d-{mode}.png')


def test_wave_plots_include_last_sample_and_group_selected_waves(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    t = np.linspace(0, 20*np.pi, 1001)
    x = np.column_stack((t, np.sin(t)))
    axes = o.waveplot(x, Nsub=3, Nf=2)
    points = np.concatenate([ax.lines[0].get_xdata() for ax in axes])
    assert_array_equal(points, t)
    _save(axes, tmp_path/'waves.png')
    gap = x.copy(); gap[12:14, 1] = np.nan
    _save(o.waveplot(gap, Nsub=1), tmp_path/'gap.png')
    _, _, bounds = o.dat2tc(x, 0, 'tw', full_output=True)
    axes = o.spwaveplot(x, [0, 1, 3], x2=x[::10])
    assert len(axes) == 2
    assert_allclose(axes[0].lines[0].get_xdata()[[0, -1]], t[bounds[[0, 4]]])
    assert_allclose(axes[1].lines[0].get_xdata()[[0, -1]], t[bounds[[6, 8]]])
    _save(axes, tmp_path/'selected.png')
    _save(o.spwaveplot(x, [0], bounds), tmp_path/'explicit.png')
    with pytest.raises(ValueError, match='complete'):
        o.spwaveplot(x, [50])
