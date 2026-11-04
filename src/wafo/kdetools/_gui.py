"""Nonblocking KDE exploration, replacing the MATLAB GUIDE applications."""
from __future__ import annotations
import numpy as np


class _KDEExplorer:
    def __init__(self, data, dimension, grid, kernel, bandwidth, transform, adaptive):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, RadioButtons
        from . import hns, kdeoptset
        from ._estimation import _sample
        self.data = _sample(data)
        if self.data.shape[1] != dimension:
            raise ValueError(f'Expected {dimension} data columns')
        self.dimension = dimension
        if transform != 1 and np.any(self.data <= 0):
            raise ValueError('Power/log transformations require positive observations')
        if transform not in (0, .5, 1) or not 0 <= adaptive <= 1:
            raise ValueError('transform must be 0, 0.5, or 1 and adaptive between 0 and 1')
        self.options = kdeoptset(kernel=kernel, L2=transform, alpha=adaptive)
        self._bandwidth = bandwidth
        self._hns = hns
        if grid is None:
            self.grid = []
            for column in self.data.T:
                span = np.ptp(column)
                lo, hi = column.min()-.2*span, column.max()+.2*span
                if transform != 1:
                    lo = max(lo, max(np.finfo(float).tiny, column.min()/4))
                self.grid.append(np.linspace(lo, hi, 160 if dimension == 1 else 48))
        else:
            self.grid = [np.asarray(grid)] if dimension == 1 else [np.asarray(v) for v in grid]
            if len(self.grid) != dimension or any(v.ndim != 1 or len(v) < 2 or not np.all(np.isfinite(v)) or np.any(np.diff(v) <= 0) for v in self.grid):
                raise ValueError('Provide one increasing finite grid per dimension')
        self.figure, self.axes = plt.subplots(figsize=(9, 6))
        self.figure.subplots_adjust(left=.1, right=.73, bottom=.25)
        self.status = self.figure.text(.1, .02, '', fontsize=9)
        self.widgets = {}
        self.widgets['bandwidth'] = Slider(self.figure.add_axes([.15, .14, .5, .035]), 'Bandwidth scale', .15, 3, valinit=1)
        self.widgets['adaptive'] = Slider(self.figure.add_axes([.15, .08, .5, .035]), 'Adaptive alpha', 0, 1, valinit=adaptive)
        names = ['gaussian', 'epanechnikov', 'triangular']
        if kernel not in names:
            names.append(kernel)
        self.widgets['kernel'] = RadioButtons(self.figure.add_axes([.77, .55, .21, .3]), names, active=names.index(kernel))
        self.widgets['transform'] = RadioButtons(self.figure.add_axes([.77, .25, .21, .22]), ['identity', 'sqrt', 'log'], active={1: 0, .5: 1, 0: 2}[transform])
        self.widgets['bandwidth'].on_changed(lambda value: self._event())
        self.widgets['adaptive'].on_changed(lambda value: self._event())
        self.widgets['kernel'].on_clicked(lambda value: self._event())
        self.widgets['transform'].on_clicked(lambda value: self._event())
        self.update()
        # Matplotlib widgets use weak callback references. Keep the complete
        # explorer alive while a caller retains its figure.
        self.figure._wafo_kde_explorer = self

    def _event(self):
        try:
            self.update()
        except (ValueError, RuntimeError) as error:
            self.status.set_text(str(error))
            self.figure.canvas.draw_idle()

    def update(self):
        from . import kdefun
        transform = {'identity': 1, 'sqrt': .5, 'log': 0}[self.widgets['transform'].value_selected]
        if transform != 1 and np.any(self.data <= 0):
            raise ValueError('This transform requires positive observations; select identity')
        transformed = self.data if transform == 1 else np.log(self.data) if transform == 0 else np.sqrt(self.data)
        base = self._hns(transformed, self.widgets['kernel'].value_selected) if self._bandwidth is None else np.asarray(self._bandwidth)
        self.options.update(kernel=self.widgets['kernel'].value_selected, L2=transform,
                            alpha=self.widgets['adaptive'].val,
                            hs=base*self.widgets['bandwidth'].val)
        if self.dimension == 1:
            self.estimate = kdefun(self.data, self.options, self.grid[0])
        else:
            xx, yy = np.meshgrid(*self.grid)
            self.estimate = kdefun(self.data, self.options, xx, yy)
        self.axes.clear()
        if self.dimension == 1:
            self.axes.plot(self.grid[0], self.estimate, label='Kernel density estimate')
            self.axes.plot(self.data[:, 0], np.zeros(len(self.data)), '|', alpha=.4)
            self.axes.set(xlabel='Value', ylabel='Density')
        else:
            self.axes.contourf(*self.grid, self.estimate, levels=12)
            self.axes.plot(self.data[:, 0], self.data[:, 1], '.', color='k', alpha=.2)
            self.axes.set(xlabel='First variable', ylabel='Second variable')
        self.axes.set_title(f'{self.dimension}D KDE: {self.options["kernel"]}')
        self.status.set_text('Bandwidths: '+np.array2string(np.asarray(self.options['hs']), precision=3))
        self.figure.canvas.draw_idle()
        return self.estimate

    def close(self):
        """Close this explorer's figure without affecting other figures."""
        import matplotlib.pyplot as plt
        plt.close(self.figure)


def kde1dgui(data=None, *, grid=None, kernel='gaussian', bandwidth=None, transform=1, adaptive=0, seed=0):
    """Create an interactive univariate KDE explorer; return its controller.

    Replaces GUIDE callback strings and workspace data loading with explicit
    data/grid inputs and live bandwidth, kernel, power-transform and adaptive
    controls. Default data are a reproducible normal sample. No show() or GUI
    event loop is started; call pyplot.show() in an interactive application.
    controller exposes figure, axes, widgets, estimate, options, update/close.
    A supplied bandwidth is measured in transformed coordinates.
    """
    data = np.random.default_rng(seed).normal(size=100) if data is None else data
    return _KDEExplorer(data, 1, grid, kernel, bandwidth, transform, adaptive)


def kde2dgui(data=None, *, grid=None, kernel='gaussian', bandwidth=None, transform=1, adaptive=0, seed=0):
    """Create a nonblocking bivariate KDE explorer with the same live controls.

    Data have two columns. Default data are a reproducible correlated normal
    sample. The displayed array uses meshgrid/xy row/column orientation.
    This Python UI replaces the source GUIDE layouts and callback interface.
    """
    data = np.random.default_rng(seed).multivariate_normal([0, 0], [[1, .6], [.6, 1]], size=150) if data is None else data
    return _KDEExplorer(data, 2, grid, kernel, bandwidth, transform, adaptive)


def kdedemo1(*, seed=0):
    """Show seven normal observations, individual kernels and three bandwidths.

    Reproducible, nonblocking adaptation of the original script. Returns the
    three axes, sample, evaluation grid, bandwidths and exact mixture curves.
    """
    import matplotlib.pyplot as plt
    from . import hns, kdefun
    from scipy.stats import norm
    data = np.random.default_rng(seed).normal(size=7)
    x = np.linspace(-4, 4, 200)
    bandwidths = hns(data, 'gaussian')[0]*np.array([.5, 1, 2])
    _, axes = plt.subplots(3, 1, figsize=(8, 8))
    estimates = []
    for ax, h in zip(axes, bandwidths):
        estimate = kdefun(data, {'kernel': 'gaussian', 'hs': h}, x)
        estimates.append(estimate)
        ax.plot(x, estimate, 'k-')
        ax.plot(x, norm.pdf(x), 'k:')
        ax.plot(data, np.zeros(7), 'bx')
        ax.plot(x, norm.pdf((x[:, None]-data)/h)/(7*h), 'b--', alpha=.5)
        ax.set(xlabel='x', ylabel='Density', title=f'Bandwidth {h:.3g}')
    axes[0].figure.tight_layout()
    return dict(axes=axes, data=data, x=x, bandwidths=bandwidths, estimates=np.asarray(estimates))


def kdedemo2(*, seed=0):
    """Compare ordinary and square-root-transformed KDE on Rayleigh data.

    Returns axes, data, grid, true density and both estimates; no show().
    """
    import matplotlib.pyplot as plt
    from scipy.stats import rayleigh
    from . import kdefun
    data = np.random.default_rng(seed).rayleigh(size=300)
    x = np.linspace(np.sqrt(np.finfo(float).eps), 5, 160)
    truth = rayleigh.pdf(x)
    _, axes = plt.subplots(2, 1, figsize=(8, 7))
    estimates = []
    for ax, exponent in zip(axes, (1, .5)):
        estimate = kdefun(data, {'kernel': 'gaussian', 'L2': exponent}, x)
        estimates.append(estimate)
        ax.plot(x, estimate)
        ax.plot(x, truth, ':')
        ax.set(xlabel='x', ylabel='Density', title='Ordinary KDE' if exponent == 1 else 'Square-root transformed KDE')
    axes[0].figure.tight_layout()
    return dict(axes=axes, data=data, x=x, truth=truth, estimates=np.asarray(estimates))
