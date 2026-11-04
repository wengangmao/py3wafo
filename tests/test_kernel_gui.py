import numpy as np
import pytest
from numpy.testing import assert_allclose
from wafo import kdetools as k


def test_1d_explorer_controls_recompute_actual_estimate_and_render(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    data = np.random.default_rng(45).normal(size=40)
    gui = k.kde1dgui(data, grid=np.linspace(-4, 4, 80))
    initial = gui.estimate.copy()
    gui.widgets['bandwidth'].set_val(1.8)
    assert not np.allclose(initial, gui.estimate)
    assert_allclose(gui.estimate, k.kdefun(data, gui.options, gui.grid[0]))
    gui.widgets['kernel'].set_active(1)
    gui.widgets['adaptive'].set_val(.4)
    assert_allclose(gui.estimate, k.kdefun(data, gui.options, gui.grid[0]))
    assert gui.options['kernel'] == 'epanechnikov' and gui.options['alpha'] == .4
    gui.widgets['transform'].set_active(2)
    assert 'positive' in gui.status.get_text()
    gui.widgets['transform'].set_active(0)
    target = tmp_path/'kde1d.png'
    gui.figure.savefig(target)
    assert target.stat().st_size > 1000
    gui.close()


def test_2d_explorer_transformed_estimates_follow_coordinates(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    data = np.exp(np.random.default_rng(20).normal(size=(35, 2))*.4)
    axes = [np.linspace(.1, 3, 16), np.linspace(.1, 3, 18)]
    gui = k.kde2dgui(data, grid=axes, bandwidth=[.2, .3])
    gui.widgets['transform'].set_active(1)
    assert gui.estimate.shape == (18, 16) and gui.options['L2'] == .5
    x, y = np.meshgrid(*axes)
    assert_allclose(gui.estimate, k.kdefun(data, gui.options, x, y))
    gui.widgets['transform'].set_active(2)
    assert_allclose(gui.estimate, k.kdefun(data, gui.options, x, y))
    gui.figure.savefig(tmp_path/'kde2d.png')
    gui.close()
    default = k.kde1dgui(seed=5)
    default.close()
    default = k.kde2dgui(seed=5)
    default.close()
    with pytest.raises(ValueError, match='columns'):
        k.kde2dgui([1, 2, 3])
    with pytest.raises(ValueError, match='positive'):
        k.kde1dgui([-1, 2, 3], transform=0)


def test_kde_demos_reproduce_original_comparisons_and_mixtures(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from scipy.stats import norm, rayleigh
    first = k.kdedemo1(seed=13)
    assert len(first['data']) == 7
    for estimate, bandwidth in zip(first['estimates'], first['bandwidths']):
        direct = norm.pdf((first['x'][:, None]-first['data'])/bandwidth).mean(axis=1)/bandwidth
        assert_allclose(estimate, direct)
    second = k.kdedemo2(seed=13)
    assert_allclose(second['truth'], rayleigh.pdf(second['x']))
    assert not np.allclose(second['estimates'][0], second['estimates'][1])
    for j, result in enumerate((first, second)):
        path = tmp_path/f'demo-{j}.png'
        result['axes'][0].figure.savefig(path)
        assert path.stat().st_size > 1000
        plt.close(result['axes'][0].figure)
