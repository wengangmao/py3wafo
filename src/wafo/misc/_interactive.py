"""Modern, nonblocking equivalents of WAFO's figure and demo utilities.

Matplotlib is imported only by plotting calls.  MATLAB callback strings are
replaced by Python callables; none of these functions calls ``show`` or starts
a GUI event loop.  They work with the Agg backend for batch figure production.
"""
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
import re
import shutil

import numpy as np


def _pyplot():
    import matplotlib.pyplot as plt
    return plt


def findpeaks(S, Np=2, min_h=None, min_p=0, *, return_subscripts=False):
    """Highest rainflow-filtered peaks of a vector or a row-wise 2-D field.

    This uses ``dat2tp`` and the original strict comparison with neighboring
    rows, rather than substituting a prominence filter.  Indices are zero
    based.  Matrix linear indices use column-major order, as in MATLAB;
    ``return_subscripts=True`` returns ``(rows, columns)`` instead.  If a row
    has no retained turning points, its global maximum is considered.
    """
    from ..onedim import dat2tp
    a = np.asarray(S, dtype=float)
    if a.ndim not in (1, 2) or not np.all(np.isfinite(a)):
        raise ValueError('S must be a finite vector or matrix')
    if Np is None:
        Np = 2
    if not (np.isscalar(Np) and (Np == np.inf or (np.isfinite(Np) and Np >= 0 and int(Np) == Np))):
        raise ValueError('Np must be a nonnegative integer or infinity')
    if not np.isscalar(min_p) or not np.isfinite(min_p) or not 0 <= min_p <= 1:
        raise ValueError('min_p must lie between zero and one')
    empty = np.empty(0, dtype=int)
    if not a.size:
        return (empty.copy(), empty.copy()) if return_subscripts else empty
    h = .05 * np.ptp(a) if min_h is None else min_h
    if not np.isscalar(h) or not np.isfinite(h) or h < 0:
        raise ValueError('min_h must be finite and nonnegative')
    vector = a.ndim == 1 or min(a.shape) == 1
    rows = a.reshape(1, -1) if vector else a
    candidates = []
    for row, values in enumerate(rows):
        _, turns = dat2tp(values, h, return_indices=True)
        peaks = turns[1::2] if len(turns) else np.array([int(np.argmax(values))])
        for col in peaks:
            if not vector and ((row > 0 and values[col] <= rows[row-1, col]) or
                               (row+1 < len(rows) and values[col] <= rows[row+1, col])):
                continue
            candidates.append((row, int(col)))
    if not candidates or Np == 0:
        return (empty.copy(), empty.copy()) if return_subscripts else empty
    r, c = np.asarray(candidates, dtype=int).T
    height = rows[r, c]
    order = np.argsort(-height, kind='stable')
    if np.isfinite(Np):
        order = order[:int(Np)]
    if min_p > 0 and len(order):
        order = order[height[order] > min_p * height[order[0]]]
    r, c = r[order], c[order]
    if vector:
        flat = c
        if a.ndim == 1:
            r = c
            c = np.zeros_like(c)
        elif a.shape[1] == 1:
            r, c = c, np.zeros_like(c)
        else:
            r = np.zeros_like(c)
    else:
        flat = np.ravel_multi_index((r, c), a.shape, order='F')
    return (r, c) if return_subscripts else flat


@dataclass
class ChoiceMenu:
    """A registered menu, retaining its widget callbacks for interactive use."""
    name: str
    figure: object
    labels: tuple
    callbacks: tuple
    buttons: tuple = ()
    interruptible: bool = False
    _running: bool = field(default=False, repr=False)

    def select(self, index):
        """Invoke a zero-based choice, returning its callable's result."""
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.callbacks):
            raise IndexError('choice index outside menu')
        if self._running and not self.interruptible:
            return None
        running = self._running
        self._running = True
        try:
            return self.callbacks[index]()
        finally:
            self._running = running

    def close(self):
        return choicex(self.name)


_CHOICES = {}


def choices(name, header, labels, callbacks, inter=0):
    """Create registered Matplotlib buttons for callable demo actions.

    Returns a :class:`ChoiceMenu`; duplicate live names return the same menu.
    Callbacks take no arguments.  ``inter`` permits callbacks to select another
    choice recursively, replacing MATLAB's interruptible callback property.
    The last button closes the menu.  GUI display is left to the caller.
    """
    if not isinstance(name, str) or not name.strip() or not isinstance(header, str):
        raise ValueError('name must be nonblank and header must be a string')
    name = name.strip()
    labels = tuple(labels.splitlines() if isinstance(labels, str) else labels)
    callbacks = (callbacks,) if callable(callbacks) else tuple(callbacks)
    if not labels or not all(isinstance(label, str) for label in labels):
        raise ValueError('labels must be a nonempty sequence of strings')
    if len(labels) != len(callbacks) or not all(callable(callback) for callback in callbacks):
        raise ValueError('provide one Python callable per label')
    plt = _pyplot()
    old = _CHOICES.get(name)
    if (old is not None and plt.fignum_exists(old.figure.number)
            and plt.figure(old.figure.number) is old.figure):
        return old
    from matplotlib.widgets import Button
    fig = plt.figure(figsize=(4, max(2.0, .55 * (len(labels)+2))))
    fig.canvas.manager.set_window_title(name)
    fig.text(.5, .94, header, ha='center', va='top')
    menu = ChoiceMenu(name, fig, labels, callbacks, interruptible=bool(inter))
    widgets = []
    step = .78 / (len(labels)+1)
    for index, label in enumerate((*labels, 'Close')):
        ax = fig.add_axes([.1, .09+(len(labels)-index)*step, .8, .8*step])
        button = Button(ax, label)
        if index < len(labels):
            button.on_clicked(lambda event, i=index: menu.select(i))
        else:
            button.on_clicked(lambda event: menu.close())
        widgets.append(button)
    menu.buttons = tuple(widgets)
    _CHOICES[name] = menu
    def unregister(event):
        if _CHOICES.get(name) is menu:
            _CHOICES.pop(name)
    fig.canvas.mpl_connect('close_event', unregister)
    return menu


def choicex(name):
    """Close and unregister a menu by name; return whether it was registered."""
    if not isinstance(name, str):
        raise TypeError('name must be a string')
    menu = _CHOICES.pop(name.strip(), None)
    if menu is None:
        return False
    _pyplot().close(menu.figure)
    return True


def figtext(x, y, T, x_unit='data', y_unit=None, x_justify='left',
            y_justify='middle', *, ax=None, **kwargs):
    """Place axes text with independent data/normalized x and y coordinates.

    ``norm`` and ``normalized`` refer to axes coordinates, as in the source,
    rather than to the whole figure.  Returns the editable Text artist.
    """
    from matplotlib.transforms import blended_transform_factory
    ax = _pyplot().gca() if ax is None else ax
    x_unit = 'data' if x_unit is None else x_unit
    y_unit = x_unit if y_unit is None else y_unit
    def transform(unit):
        if unit == 'data':
            return ax.transData
        if unit in ('norm', 'normalized', 'axes'):
            return ax.transAxes
        raise ValueError('coordinates must use data or normalized units')
    kwargs.setdefault('horizontalalignment', x_justify or 'left')
    kwargs.setdefault('verticalalignment', {'middle': 'center'}.get(y_justify, y_justify) or 'center')
    kwargs.setdefault('transform', blended_transform_factory(transform(x_unit), transform(y_unit)))
    text = '\n'.join(map(str, T)) if isinstance(T, (list, tuple)) else str(T)
    return ax.text(x, y, text, **kwargs)


def cltext(clevels=None, PL=0, *, ax=None):
    """Add a replaceable contour-level annotation; return ``(texts, axes)``."""
    if clevels is None or not np.size(clevels):
        return [], None
    levels = np.asarray(clevels, dtype=float).ravel()
    if not np.all(np.isfinite(levels)) or not np.isscalar(PL) or PL not in (0, 1):
        raise ValueError('levels must be finite and PL must be zero or one')
    ax = _pyplot().gca() if ax is None else ax
    fig = ax.figure
    for old in tuple(fig.axes):
        if old.get_gid() == 'cltxt':
            old.remove()
    x, y, w, h = ax.get_position().bounds
    overlay = fig.add_axes([x, y+.725*h, .5*w, .25*h], label='wafo contour levels')
    overlay.set_gid('cltxt')
    overlay.set_axis_off()
    header = 'Level curves enclosing:' if PL else 'Level curves at:'
    texts = [figtext(.036, .94, header, 'norm', y_justify='top', ax=overlay,
                     fontweight='bold', gid='cltxt'),
             figtext(.036, .72, '\n'.join(f'{v:.4g}' for v in levels), 'norm',
                     y_justify='top', ax=overlay, gid='cltxt')]
    _pyplot().sca(ax)
    return texts, overlay


def fcolorbar(D, L='r', label='', W=1/20, hand=None, *, ax=None, cmap=None):
    """Discrete contour colorbar with equal spacing between contour levels.

    Accepts a level vector, MATLAB two-row contour matrix, or Matplotlib
    ContourSet.  ``L`` starts with r/l/t/b; prefix o places it outside without
    resizing the graph and prefix a hides the data axes.  The source's half
    mean-step extension of the color limits is retained.  Returns the data
    and colorbar axes.  ``fcolorbar('delete')`` restores the data axes size.
    """
    ax = _pyplot().gca() if ax is None else ax
    previous = getattr(ax, '_wafo_colorbar', None)
    if isinstance(D, str):
        if D.lower() != 'delete':
            raise ValueError('the only string operation is delete')
        if previous is not None:
            previous[0].remove()
            ax.set_position(previous[1])
            ax.set_visible(previous[2])
            del ax._wafo_colorbar
        _pyplot().sca(ax)
        return ax, None
    if hasattr(D, 'levels'):
        values = np.asarray(D.levels, dtype=float)
        cmap = D.cmap if cmap is None else cmap
    else:
        data = np.asarray(D, dtype=float)
        if data.ndim <= 1 or (data.ndim == 2 and min(data.shape) == 1):
            values = data.ravel()
        elif data.ndim == 2 and data.shape[0] == 2:
            from ._extended import clevels
            values = clevels(data)[0]
        else:
            raise ValueError('D must be levels or a contour matrix/set')
    values = np.unique(values)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError('contour levels must be nonempty and finite')
    if not np.isfinite(W) or not 0 < W < .25:
        raise ValueError('W must lie strictly between zero and one quarter')
    loc = (L or 'r').lower()
    outside, alone = loc.startswith('o'), loc.startswith('a')
    if outside or alone:
        loc = loc[1:] or 'r'
    loc = loc[0]
    if loc not in 'rltb':
        raise ValueError('location must be right, left, top or bottom')
    if previous is not None:
        original, visible = previous[1], previous[2]
        previous[0].remove()
        ax.set_position(original)
        ax.set_visible(visible)
    else:
        original, visible = ax.get_position().frozen(), ax.get_visible()
    x, y, w, h = original.bounds
    graph = [x, y, w, h]
    vertical = loc in 'rl'
    width = W * (w if vertical else h)
    ticks_same_side = ((loc == 'l' and ax.yaxis.get_ticks_position() == 'left') or
                       (loc == 'r' and ax.yaxis.get_ticks_position() == 'right') or
                       (loc == 'b' and ax.xaxis.get_ticks_position() == 'bottom') or
                       (loc == 't' and ax.xaxis.get_ticks_position() == 'top'))
    gap = width * (2 if ticks_same_side else 1)
    if loc == 'r':
        position = [x+w+gap if outside else x+w-width, y, width, h]
        if not outside:
            graph[2] -= width+gap
    elif loc == 'l':
        position = [x-gap-width if outside else x, y, width, h]
        if not outside:
            graph[0] += width+gap
            graph[2] -= width+gap
    elif loc == 't':
        position = [x, y+h+gap if outside else y+h-width, w, width]
        if not outside:
            graph[3] -= width+gap
    else:
        position = [x, y-gap-width if outside else y, w, width]
        if not outside:
            graph[1] += width+gap
            graph[3] -= width+gap
    ax.set_position(graph)
    if alone:
        ax.set_visible(False)
    cax = ax.figure.add_axes(position)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    extension = np.mean(np.diff(values))/2 if len(values) > 1 else max(abs(values[0]), 1)
    limits = values[0]-extension, values[-1]+extension
    targets = [] if hand is None else (list(np.asarray(hand, dtype=object).ravel())
                                      if isinstance(hand, (tuple, list, np.ndarray)) else [hand])
    for target in (ax, *targets):
        for artist in (*target.collections, *target.images):
            if hasattr(artist, 'set_clim'):
                artist.set_clim(*limits)
                if cmap is None:
                    cmap = artist.cmap
    mappable = ScalarMappable(norm=Normalize(*limits), cmap=cmap)
    bar = ax.figure.colorbar(mappable, cax=cax, boundaries=np.r_[limits[0], values, limits[1]],
                             ticks=values, spacing='uniform',
                             orientation='vertical' if vertical else 'horizontal')
    bar.set_label(label)
    if loc == 'l':
        cax.yaxis.set_ticks_position('left')
        cax.yaxis.set_label_position('left')
    elif loc == 't':
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
    cax.set_gid('Colorbar')
    ax._wafo_colorbar = (bar, original, visible)
    _pyplot().sca(ax)
    return ax, cax


def fwaitbar(x, name=None, msg=None):
    """Create/update a nonblocking progress figure, returning its Figure.

    Pass a title string to create a new figure or its returned Figure to
    update it.  Fractions are clamped to [0,1]; an omitted message preserves
    the previous message.  The complete bar is redrawn so decreasing progress
    also works, unlike the historical incremental-patch implementation.
    """
    if not np.isscalar(x) or not np.isfinite(x):
        raise ValueError('progress must be a finite scalar')
    fraction = float(np.clip(x, 0, 1))
    plt = _pyplot()
    if name is None or isinstance(name, str):
        from matplotlib.patches import Rectangle
        fig = plt.figure(figsize=(4.5, 1.2))
        title = name or 'Please wait...'
        fig.canvas.manager.set_window_title(title)
        ax = fig.add_axes([.06, .25, .88, .3], xlim=(0, 1), ylim=(0, 1), yticks=[], xticks=[])
        patch = Rectangle((0, 0), fraction, 1, facecolor='r', edgecolor='r')
        ax.add_patch(patch)
        fig._wafo_waitbar = (patch, ax.set_title(msg or ''))
    else:
        fig = name
        if not hasattr(fig, '_wafo_waitbar'):
            raise ValueError('name must be a title or a figure returned by fwaitbar')
    patch, message = fig._wafo_waitbar
    patch.set_width(fraction)
    if msg is not None:
        message.set_text(str(msg))
    fig._wafo_progress = fraction
    fig.canvas.draw_idle()
    return fig


def maximizefigs(*args, screen_size=None):
    """Resize specified/all figures using the source's taskbar pixel margins.

    Arguments may be Figure objects, existing figure numbers, their sequences,
    ``'gcf'``, ``'all'``, ``'bottom'`` or ``'left'`` in any order.  For batch
    backends ``screen_size`` defaults to 1920 by 1080 pixels; supply it for a
    reproducible export size.  A GUI screen size is queried when available.
    Returns the resized figures and retains their original inch sizes.
    """
    plt = _pyplot()
    selection, position = [], 'bottom'
    def parse(value):
        nonlocal position
        if isinstance(value, str):
            text = value.lower()
            if text in ('bottom', 'left'):
                position = text
            elif text == 'gcf':
                selection.append(plt.gcf())
            elif text == 'all':
                selection.extend(plt.figure(i) for i in plt.get_fignums())
            elif text.isdigit() and plt.fignum_exists(int(text)):
                selection.append(plt.figure(int(text)))
            else:
                raise ValueError('unknown figure or taskbar position')
        elif hasattr(value, 'set_size_inches'):
            selection.append(value)
        elif np.isscalar(value) and np.isfinite(value) and int(value) == value and plt.fignum_exists(int(value)):
            selection.append(plt.figure(int(value)))
        elif isinstance(value, (list, tuple, np.ndarray)):
            for child in value:
                parse(child)
        else:
            raise ValueError('specify existing figures or their numbers')
    for arg in args:
        parse(arg)
    if not selection:
        selection = [plt.figure(i) for i in plt.get_fignums()]
    selection = list(dict.fromkeys(selection))
    if screen_size is None:
        screen_size = (1920, 1080)
        if selection:
            window = getattr(selection[0].canvas.manager, 'window', None)
            if window is not None and hasattr(window, 'winfo_screenwidth'):
                screen_size = (window.winfo_screenwidth(), window.winfo_screenheight())
            elif window is not None and hasattr(window, 'screen'):
                rect = window.screen().availableGeometry()
                screen_size = (rect.width(), rect.height())
    screen = np.asarray(screen_size, dtype=float)
    if screen.shape != (2,) or not np.all(np.isfinite(screen)) or np.any(screen <= [84, 111]):
        raise ValueError('screen_size must contain two sufficiently large pixel dimensions')
    pixels = screen - ([84, 78] if position == 'left' else [0, 111])
    for fig in selection:
        if not hasattr(fig, '_wafo_original_size'):
            fig._wafo_original_size = fig.get_size_inches().copy()
        fig.set_size_inches(pixels / fig.dpi, forward=True)
    return selection


def mkcontents(h1line=None, wver=None, srt=0, cfile='Contents.py', *, directory='.', date=None):
    """Create a documented contents index for local Python/MATLAB source files.

    Sorting selectors are unchanged: 0 discovery order, 1 H1 text, 2 filename,
    3 reversed filename.  Python module docstrings replace MATLAB H1 comments.
    A nonstring/None ``cfile`` returns the index without writing.  Existing
    output is backed up to ``<cfile>old`` and supplies header/version defaults;
    Python output is an importable module docstring, MATLAB output uses ``%``.
    The returned string contains the index entries, as in the original.
    """
    from ._extended import geth1line
    if srt not in (0, 1, 2, 3):
        raise ValueError('srt must be zero through three')
    folder = Path(directory)
    if not folder.is_dir():
        raise NotADirectoryError(str(folder))
    output = folder / cfile if isinstance(cfile, str) else None
    old = output.read_text(encoding='utf-8') if output is not None and output.exists() else ''
    if old and output.suffix.lower() != '.m':
        import ast
        old = ast.get_docstring(ast.parse(old)) or ''
    # Both generated Python docstrings and MATLAB comments carry these headers.
    oldlines = old.strip().splitlines()
    oldheader = oldlines[0].lstrip('%# ').strip() if oldlines else ''
    version_match = re.search(r'\bVersion\s+([^\s]+)', old)
    if h1line is None:
        h1line = oldheader or f'{folder.resolve().name.upper()} Toolbox'
    if wver is None:
        if version_match:
            pieces = version_match.group(1).split('.')
            if not pieces[-1].isdigit():
                raise ValueError('previous version must end with a numeric component')
            pieces[-1] = str(int(pieces[-1])+1)
            wver = '.'.join(pieces)
        else:
            wver = '1.0.0'
    rows = []
    for source in folder.iterdir():
        if source.suffix.lower() not in ('.m', '.py') or source.stem.lower() == 'contents':
            continue
        if output is not None and source.resolve() == output.resolve():
            continue
        h1 = geth1line(source)
        if h1:
            rows.append((source.name, h1))
    if srt == 1:
        rows.sort(key=lambda row: row[1].casefold())
    elif srt == 2:
        rows.sort(key=lambda row: row[0].casefold())
    elif srt == 3:
        rows.sort(key=lambda row: row[0][::-1].casefold())
    width = max((len(Path(name).stem) for name, _ in rows), default=0)
    entries = '\n'.join(f'{Path(name).stem:<{width}} - {h1}' for name, h1 in rows)
    if output is not None:
        today = _date.today() if date is None else date
        stamp = today.strftime('%d-%b-%Y') if hasattr(today, 'strftime') else str(today)
        text = f'{h1line}\nVersion {wver}   {stamp}\n\n{entries}\n'
        if output.exists():
            shutil.copy2(output, str(output)+'old')
        if output.suffix.lower() == '.m':
            text = '\n'.join('% '+line for line in text.splitlines())+'\n'
        else:
            # repr avoids broken modules when an H1 line itself contains quotes.
            text = repr(text)+'\n'
        output.write_text(text, encoding='utf-8')
    return entries


def wafostamp(caption=None, flag='', *, fig=None, stamp=None, date=None):
    """Add a replaceable caption/date stamp; return ``(texts, overlay_axes)``.

    No arguments, or numeric zero, is a no-op.  A caption or numeric one
    enables the stamp.  Text artists can be edited with ``set_text``; the
    original MATLAB ``edtext`` GUI callback is not installed.  An explicit
    ``date`` makes figures reproducible.
    """
    enabled = caption is not None if stamp is None else bool(stamp)
    if isinstance(caption, (bool, int, float, np.number)):
        enabled = bool(caption) if stamp is None else bool(stamp)
        caption = ''
    if not enabled:
        return [], None
    if caption is not None and not isinstance(caption, str) or not isinstance(flag, str):
        raise TypeError('caption and flag must be strings')
    fig = _pyplot().gcf() if fig is None else fig
    oldax = fig.gca()
    for old in tuple(fig.axes):
        if old.get_gid() == 'wafostmptxt':
            old.remove()
    overlay = fig.add_axes([.01, .01, .98, .05], label='wafo stamp')
    overlay.set_gid('wafostmptxt')
    overlay.set_axis_off()
    texts = []
    if caption:
        texts.append(figtext(0, 0, caption, 'norm', y_justify='bottom', ax=overlay,
                             fontsize=10, gid='wafostmptxt'))
    today = _date.today() if date is None else date
    stampdate = today.strftime('%d-%b-%Y') if hasattr(today, 'strftime') else str(today)
    texts.append(figtext(1, 0, f'made by WAFO {stampdate} {flag}'.rstrip(), 'norm',
                         x_justify='right', y_justify='bottom', ax=overlay,
                         fontsize=10, fontstyle='italic', gid='wafostmptxt'))
    if oldax in fig.axes:
        _pyplot().sca(oldax)
    return texts, overlay
