import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import cumulative_trapezoid
from wafo import trgauss as t


def _save(ax, path):
    import matplotlib.pyplot as plt
    ax.figure.savefig(path)
    assert path.stat().st_size > 1000
    plt.close(ax.figure)


def test_transform_plot_matches_standardization_and_accepts_mapping(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    x = np.linspace(-4, 6, 101)
    g = np.column_stack((x, (x-1)/2))
    for table in (g, t.trmak(g, 1, 2)):
        ax = t.trplot(table, g, 1, 2)
        assert_allclose(ax.lines[0].get_ydata(), (x-1)/2)
        assert_allclose(ax.lines[1].get_ydata(), (x-1)/2)
        assert len(ax.lines) == 3
        _save(ax, tmp_path/('mapping.png' if isinstance(table, dict) else 'table.png'))
    _save(t.trplot(g), tmp_path/'defaults.png')
    with pytest.raises(ValueError):
        t.trplot(t.trmak(lambda y: y))


def test_pdf_1d_modes_have_correct_integrated_probabilities_and_log_scales(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    x = np.linspace(.01, 8, 300)
    density = np.exp(-x)
    CDF = cumulative_trapezoid(density, x, initial=0)
    f = {'x': [x], 'f': density, 'labx': ['height'], 'title': 'Exponential'}
    for mode in (1, 2, 3, 4, 5, 11, 12, 13):
        ax = t.pdfplot(f, mode)
        y = CDF if mode >= 11 else density.copy()
        if mode in (2, 3, 12, 13): y = 1-y
        if mode in (3, 5, 13): y = -np.log(y)
        if mode in (2, 3, 4, 5, 12, 13): y = np.where(y > 0, y, np.nan)
        assert_allclose(ax.lines[0].get_ydata(), y)
        assert ax.get_xscale() == ('log' if mode in (3, 5, 13) else 'linear')
        assert ax.get_yscale() == ('log' if mode in (2, 3, 4, 5, 12, 13) else 'linear')
        _save(ax, tmp_path/f'exponential-{mode}.png')
    _save(t.pdfplot(np.column_stack((x, density))), tmp_path/'raw.png')
    assert t.pdfplot(f, 0) is None


def test_pdf_2d_graphics_support_source_xy_and_explicit_ij(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    x, y = np.linspace(-3, 3, 20), np.linspace(-2, 2, 15)
    xx, yy = np.meshgrid(x, y)
    density = np.exp(-(xx*xx+yy*yy)/2)/(2*np.pi)
    f = {'x': [x, y], 'f': density, 'labx': ['X', 'Y'], 'cl': [.01, .05, .1]}
    for mode in (1, 2, 3, 4, 5, 6, 10):
        ax = t.pdfplot(f, mode, shading='interp')
        assert ax.get_xlabel() == 'X' and ax.get_ylabel() == 'Y'
        assert len(ax.collections)+len(ax.lines) > 0
        _save(ax, tmp_path/f'pdf2d-{mode}.png')
    ij = dict(f, f=density.T, indexing='ij')
    _save(t.pdfplot(ij), tmp_path/'ij.png')
    with pytest.raises(ValueError, match='shape'):
        t.pdfplot(dict(f, f=density.T))


def test_pdf_3d_static_slices_and_indexed_contours(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    axes = [np.linspace(-2, 2, n) for n in (7, 8, 9)]
    xx, yy, zz = np.meshgrid(*axes, indexing='ij')
    density = np.exp(-(xx*xx+yy*yy+zz*zz)/2)
    f = {'x': axes, 'f': density, 'indexing': 'ij', 'labx': ['a', 'b', 'c']}
    for mode in (1, 2, 3, 4, 5, 6):
        ax = t.pdfplot(f, mode)
        assert len(ax.collections) >= (3 if mode < 4 else 1)
        _save(ax, tmp_path/f'pdf3d-{mode}.png')
    xy = dict(f, f=density.swapaxes(0, 1), indexing='xy')
    _save(t.pdfplot(xy, 3, x1=[-.5, .5], x2=0, x3=1, method='nearest'), tmp_path/'coordinates.png')
    _save(t.pdfplot(f, 4, x1=1), tmp_path/'index.png')
    with pytest.raises(ValueError, match='coordinates'):
        t.pdfplot(f, 3, x1=10)
    with pytest.raises(ValueError, match='index'):
        t.pdfplot(f, 4, x1=.5)
