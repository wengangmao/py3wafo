"""Plot tabulated transformations and probability structures without show()."""
from __future__ import annotations
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator


def trplot(g, g2=None, ma=None, sa=None, *, ax=None):
    """Plot smoothed/empirical transformations and a linear Gaussian model.

    Accepts a table or a table-form trmak mapping. Returns Matplotlib axes.
    Default mean/scale follow the source table's center and one tenth of span.
    Spline-form objects require evaluation to a table before plotting.
    """
    from ._transformations import _table
    import matplotlib.pyplot as plt
    if isinstance(g, dict):
        if g.get('form') != 'table':
            raise ValueError('Evaluate spline-form transformations on a grid before plotting')
        ma = g.get('mean') if ma is None else ma
        sa = g.get('std') if sa is None else sa
    table = _table(g)
    ma = np.mean(table[:, 0]) if ma is None else ma
    sa = np.ptp(table[:, 0])/10 if sa is None else sa
    if not np.isfinite(ma) or not np.isfinite(sa) or sa <= 0:
        raise ValueError('Mean must be finite and standard deviation positive')
    ax = plt.subplots()[1] if ax is None else ax
    ax.plot(table[:, 0], table[:, 1], 'r-', label='Estimated transform')
    ax.plot(table[:, 0], (table[:, 0]-ma)/sa, 'g--', label='Linear transform')
    if g2 is not None:
        empirical = _table(g2)
        ax.step(empirical[:, 0], empirical[:, 1], where='post', label='Empirical transform')
    ax.set(title='Estimated transform', xlabel='u', ylabel='g(u)')
    return ax


def pdfplot(f, plotflag=1, x1=None, x2=None, x3=None, sym='k-', method='linear',
            shading='faceted', *, ax=None, indexing=None):
    """Plot a 1-D/2-D/3-D WAFO PDF/CDF mapping, or an N by 2 table.

    One-dimensional modes 1/2/3 plot f, 1-f, or -log(1-f); 4/5 plot f or
    -log(f); 11/12/13 integrate the supplied density before plotting the CDF,
    survival, or cumulative hazard. Modes 2/4/12 have logarithmic y axes;
    3/5/13 use log-log axes, following the source implementation.

    Two-dimensional modes are contour (1), wireframe (2), surface (3), row
    waterfall (4), pcolor (5), 3-D contour (6), and filled contour (10).
    Three-dimensional modes 1/2/3 show static orthogonal interpolated slices
    at supplied physical coordinates x1/x2/x3 (default through the maximum).
    Mode 2 uses index coordinates. Modes 4/5/6 plot a 2-D contour slice at a
    zero-based third/first/second-axis index supplied as x1. The unavailable
    MATLAB sliceomatic widget is replaced by explicit reproducible slices.

    Original grid mappings use meshgrid/xy storage, i.e. rows belong to x[1].
    indexing='ij' supports arrays whose axes follow the x-list order. Explicit
    mapping field indexing (including Python Gaussian driver output) takes
    precedence over the default 'xy'. Returns axes and never calls show().
    """
    import matplotlib.pyplot as plt
    if plotflag == 0:
        return None
    if not isinstance(f, dict):
        a = np.asarray(f, dtype=float)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError('A raw PDF/CDF table must have two columns')
        f = {'x': [a[:, 0]], 'f': a[:, 1]}
    coordinates = [np.asarray(v, dtype=float) for v in f['x']]
    dim = len(coordinates)
    values = np.asarray(f['f'], dtype=float)
    if dim not in (1, 2, 3) or not np.all(np.isfinite(values)) or any(v.ndim != 1 or len(v) < 2 or not np.all(np.isfinite(v)) or np.any(np.diff(v) <= 0) for v in coordinates):
        raise ValueError('Expected one through three increasing finite coordinate grids and finite values')
    order = f.get('indexing', 'xy') if indexing is None else indexing
    if order not in ('xy', 'ij'):
        raise ValueError('indexing must be xy or ij')
    shape = tuple(map(len, coordinates))
    if dim > 1 and order == 'xy':
        shape = (shape[1], shape[0])+shape[2:]
    if values.shape != shape:
        raise ValueError('PDF shape does not match coordinate grids and indexing; specify indexing explicitly')
    if dim > 1 and order == 'xy':
        values = values.swapaxes(0, 1)
    labels = list(f.get('labx', []))+['']*3
    labels = [labels[j] or f'x{j+1}' for j in range(dim)]
    title = str(f.get('title', ''))
    if dim == 1:
        if plotflag not in (1, 2, 3, 4, 5, 11, 12, 13):
            raise ValueError('Unknown one-dimensional PDF plot mode')
        x = coordinates[0]
        y = cumulative_trapezoid(values, x, initial=0) if plotflag >= 11 else values.copy()
        if plotflag in (2, 3, 12, 13):
            y = 1-y
        if plotflag in (3, 5, 13):
            with np.errstate(divide='ignore', invalid='ignore'):
                y = -np.log(y)
        # Logarithmic curves omit nonpositive ordinates/abscissas explicitly.
        logy = plotflag in (2, 3, 4, 5, 12, 13)
        logx = plotflag in (3, 5, 13)
        y = np.where((np.isfinite(y)) & ((not logy) | (y > 0)) & ((not logx) | (x > 0)), y, np.nan)
        ax = plt.subplots()[1] if ax is None else ax
        ax.plot(x, y, sym)
        if logy:
            ax.set_yscale('log')
        if logx:
            ax.set_xscale('log')
        ax.set(xlabel=labels[0], ylabel='Cumulative probability' if plotflag == 11 else 'Value', title=title)
        return ax
    if shading not in ('faceted', 'flat', 'interp'):
        raise ValueError('shading must be faceted, flat, or interp')
    if dim == 2 or plotflag in (4, 5, 6):
        if dim == 3:
            fixed = {4: 2, 5: 0, 6: 1}[plotflag]
            value = int(np.unravel_index(np.argmax(values), values.shape)[fixed]) if x1 is None else x1
            if np.ndim(value) or not np.isfinite(value) or int(value) != value or not 0 <= value < len(coordinates[fixed]):
                raise ValueError('Slice index must be a valid zero-based scalar')
            values = np.take(values, int(value), axis=fixed)
            title += f' {labels[fixed]} = {coordinates[fixed][int(value)]:g}'
            coordinates = [v for j, v in enumerate(coordinates) if j != fixed]
            labels = [v for j, v in enumerate(labels) if j != fixed]
            mode = 1
        else:
            mode = plotflag
        if mode not in (1, 2, 3, 4, 5, 6, 10):
            raise ValueError('Unknown two-dimensional PDF plot mode')
        three = mode in (2, 3, 4, 6)
        if ax is None:
            ax = plt.figure().add_subplot(111, projection='3d' if three else None)
        if three and not hasattr(ax, 'plot_surface'):
            raise ValueError('This mode requires a 3-D axes')
        xx, yy = np.meshgrid(*coordinates)
        z = values.T
        levels = np.unique(np.asarray(f['cl'], dtype=float)) if 'cl' in f and np.size(f['cl']) else 8
        if mode == 1:
            ax.contour(xx, yy, z, levels=levels)
        elif mode == 2:
            ax.plot_wireframe(xx, yy, z)
        elif mode == 3:
            ax.plot_surface(xx, yy, z, cmap='viridis', linewidth=.3 if shading == 'faceted' else 0, shade=shading != 'flat')
        elif mode == 4:
            for row in range(len(yy)):
                ax.plot(xx[row], yy[row], z[row])
        elif mode == 5:
            ax.pcolormesh(xx, yy, z, shading='gouraud' if shading == 'interp' else 'auto')
        elif mode == 6:
            ax.contour(xx, yy, z, levels=levels)
        else:
            ax.contourf(xx, yy, z, levels=levels)
        ax.set(xlabel=labels[0], ylabel=labels[1], title=title)
        return ax
    if plotflag not in (1, 2, 3) or method not in ('linear', 'nearest', 'cubic'):
        raise ValueError('Unknown three-dimensional plot or interpolation mode')
    if plotflag == 2:
        coordinates = [np.arange(len(v), dtype=float) for v in coordinates]
    maximum = np.unravel_index(np.argmax(values), values.shape)
    positions = [np.atleast_1d(coordinates[j][maximum[j]] if value is None else value).astype(float) for j, value in enumerate((x1, x2, x3))]
    if any(not np.all(np.isfinite(v)) or np.any(v < coordinates[j][0]) or np.any(v > coordinates[j][-1]) for j, v in enumerate(positions)):
        raise ValueError('Slice coordinates must lie within the PDF grids')
    interpolate = RegularGridInterpolator(coordinates, values, method=method)
    if ax is None:
        ax = plt.figure().add_subplot(111, projection='3d')
    if not hasattr(ax, 'plot_surface'):
        raise ValueError('Orthogonal slices require a 3-D axes')
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    norm = Normalize(values.min(), values.max())
    for fixed in range(3):
        varying = [j for j in range(3) if j != fixed]
        u, v = np.meshgrid(coordinates[varying[0]], coordinates[varying[1]])
        for position in positions[fixed]:
            grids = [None]*3
            grids[fixed] = np.full_like(u, position)
            grids[varying[0]], grids[varying[1]] = u, v
            density = interpolate(np.stack(grids, axis=-1))
            ax.plot_surface(*grids, facecolors=colormaps['viridis'](norm(density)), shade=False, linewidth=0)
    ax.set(xlabel=labels[0], ylabel=labels[1], zlabel=labels[2], title=title)
    return ax
