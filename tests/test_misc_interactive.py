"""Source-based numerical, headless rendering and widget-event checks."""
import ast
from datetime import date
from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from wafo import misc


@pytest.fixture
def plots():
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    yield plt
    plt.close('all')


def render(fig, tmp_path, name):
    path = tmp_path / (name+'.png')
    fig.savefig(path)
    assert path.stat().st_size > 1000


def test_findpeaks_source_rainflow_ranking_and_threshold():
    y = [0, 4, 2, 3, 0, 5, -1, 2, 0]
    assert_array_equal(misc.findpeaks(y, np.inf, 0), [5, 1, 3, 7])
    assert_array_equal(misc.findpeaks(y, np.inf, 2), [5, 1, 7])
    assert_array_equal(misc.findpeaks(y, 4, 0, .6), [5, 1])
    assert_array_equal(misc.findpeaks([1, 2, 3]), [2])
    assert_array_equal(misc.findpeaks([2, 2, 2]), [0])
    assert misc.findpeaks(y, 0).size == 0
    assert misc.findpeaks([]).size == 0
    with pytest.raises(ValueError):
        misc.findpeaks(y, min_h=-1)
    with pytest.raises(ValueError):
        misc.findpeaks([np.nan, 1])


def test_findpeaks_matrix_strict_row_neighbors_and_matlab_layout():
    S = np.array([[0, 3, 0, 2, 0], [0, 5, 0, 1, 0], [0, 4, 0, 6, 0]])
    rows, cols = misc.findpeaks(S, 8, 0, return_subscripts=True)
    assert_array_equal(rows, [2, 1, 0])
    assert_array_equal(cols, [3, 1, 3])
    assert_array_equal(misc.findpeaks(S, 8, 0), [11, 4, 9])
    assert_array_equal(misc.findpeaks([[0], [2], [0]], min_h=0, return_subscripts=True), ([1], [0]))


def test_choice_menu_click_events_close_registration_and_reentrancy(plots, tmp_path):
    from matplotlib.backend_bases import MouseEvent
    called = []
    menu = misc.choices('demo', 'Choose a demonstration', ['A', 'B'],
                        [lambda: called.append('a'), lambda: called.append('b')])
    assert misc.choices('demo', 'ignored', ['X'], [lambda: None]) is menu
    menu.figure.canvas.draw()
    ax = menu.buttons[1].ax
    x, y = ax.transAxes.transform((.5, .5))
    for name in ('button_press_event', 'button_release_event'):
        event = MouseEvent(name, menu.figure.canvas, x, y, button=1)
        menu.figure.canvas.callbacks.process(name, event)
    assert called == ['b']
    assert menu.select(0) is None
    assert called == ['b', 'a']
    render(menu.figure, tmp_path, 'choices')
    assert misc.choicex('demo')
    assert not plots.fignum_exists(menu.figure.number)
    assert not misc.choicex('demo')
    # Agg may not dispatch close_event; a recycled figure number must not
    # cause a stale registered menu to be returned.
    stale = misc.choices('recycled', 'First menu', ['A'], [lambda: None])
    plots.close(stale.figure)
    other = plots.figure(stale.figure.number)
    replacement = misc.choices('recycled', 'Second menu', ['B'], [lambda: None])
    assert replacement is not stale
    assert replacement.figure is not other
    replacement.close()
    recursive = misc.choices('recursive', 'No recursion', ['Run'], [lambda: recursive.select(0)])
    assert recursive.select(0) is None
    recursive.close()
    with pytest.raises(ValueError, match='callable'):
        misc.choices('invalid', 'No eval', ['A'], ['print(1)'])


def test_figtext_blends_coordinates_and_contour_text_replaces(plots, tmp_path):
    fig, ax = plots.subplots()
    ax.set_xlim(10, 20)
    ax.set_ylim(-5, 5)
    text = misc.figtext(15, .25, 'mixed', 'data', 'norm', 'center', 'top', ax=ax)
    fig.canvas.draw()
    assert_allclose(text.get_transform().transform(text.get_position()), ax.transAxes.transform((.5, .25)))
    texts, overlay = misc.cltext([.1, .3, .5], 1, ax=ax)
    assert texts[0].get_text() == 'Level curves enclosing:'
    assert texts[1].get_text() == '0.1\n0.3\n0.5'
    assert plots.gca() is ax
    newtext, newoverlay = misc.cltext([1, 2], ax=ax)
    assert overlay not in fig.axes
    assert len([a for a in fig.axes if a.get_gid() == 'cltxt']) == 1
    assert newtext[0].get_text() == 'Level curves at:'
    assert misc.cltext([]) == ([], None)
    render(fig, tmp_path, 'annotations')


@pytest.mark.parametrize('loc', ['r', 'l', 't', 'b', 'or', 'ol', 'ot', 'ob', 'a'])
def test_fcolorbar_discrete_ticks_layout_limits_and_delete(plots, tmp_path, loc):
    fig, ax = plots.subplots()
    levels = np.array([-2, -.5, 0, 1, 3])
    field = np.arange(25).reshape(5, 5) / 4 - 2
    contour = ax.contourf(field, levels=levels)
    original = ax.get_position().bounds
    dataax, cax = misc.fcolorbar(levels, loc, 'intensity', ax=ax)
    assert dataax is ax and cax.get_gid() == 'Colorbar'
    assert plots.gca() is ax
    extension = np.diff(levels).mean()/2
    assert_allclose(contour.get_clim(), (levels[0]-extension, levels[-1]+extension))
    ticks = cax.get_yticks() if loc[-1] in 'rla' else cax.get_xticks()
    assert_allclose(ticks, levels)
    if loc.startswith('o'):
        assert_allclose(ax.get_position().bounds, original)
    if loc == 'a':
        assert not ax.get_visible()
    render(fig, tmp_path, f'colorbar-{loc}')
    assert misc.fcolorbar('delete', ax=ax) == (ax, None)
    assert cax not in fig.axes and ax.get_visible()
    assert_allclose(ax.get_position().bounds, original)


def test_fcolorbar_contour_matrix_and_reuse(plots, tmp_path):
    fig, ax = plots.subplots()
    ax.imshow(np.arange(9).reshape(3, 3))
    # Source contour headers are level, segment length.
    C = np.array([[1, 0, 1, 3, 0, 1], [2, 0, 1, 2, 0, 1]])
    _, old = misc.fcolorbar(C, ax=ax)
    _, new = misc.fcolorbar([2], 'b', ax=ax)
    assert old not in fig.axes
    assert_allclose(new.get_xticks(), [2])
    render(fig, tmp_path, 'colorbar-replacement')
    otherfig, otherax = plots.subplots()
    contour = otherax.contourf(np.arange(9).reshape(3, 3), levels=[0, 2, 5, 8])
    _, cax = misc.fcolorbar(contour, ax=otherax)
    assert_allclose(cax.get_yticks(), contour.levels)


def test_progress_updates_and_maximized_export_sizes(plots, tmp_path):
    fig = misc.fwaitbar(-1, 'Computation', 'starting')
    assert fig._wafo_progress == 0
    assert misc.fwaitbar(.8, fig, 'nearly done') is fig
    patch, message = fig._wafo_waitbar
    assert patch.get_width() == .8 and message.get_text() == 'nearly done'
    misc.fwaitbar(.3, fig)
    assert patch.get_width() == .3 and message.get_text() == 'nearly done'
    misc.fwaitbar(2, fig, '')
    assert patch.get_width() == 1 and message.get_text() == ''
    original = fig.get_size_inches().copy()
    assert misc.maximizefigs(fig, 'left', screen_size=(1000, 800)) == [fig]
    assert_allclose(fig.get_size_inches()*fig.dpi, [916, 722])
    assert_allclose(fig._wafo_original_size, original)
    misc.maximizefigs('bottom', [fig.number], screen_size=(1200, 900))
    assert_allclose(fig.get_size_inches()*fig.dpi, [1200, 789])
    render(fig, tmp_path, 'progress')


def test_mkcontents_generates_importable_index_sorted_and_backed_up(tmp_path):
    (tmp_path/'z.py').write_text('"""Alpha module."""\nx = 1\n', encoding='utf-8')
    (tmp_path/'b.py').write_text('"""Beta module."""\nx = 2\n', encoding='utf-8')
    (tmp_path/'legacy.m').write_text('function y=legacy(x)\n% LEGACY old MATLAB helper\n', encoding='utf-8')
    (tmp_path/'undocumented.py').write_text('x = 3\n', encoding='utf-8')
    text = misc.mkcontents('My Toolbox', '2.4.9', 2, directory=tmp_path, date=date(2020, 1, 2))
    assert [line.split()[0] for line in text.splitlines()] == ['b', 'legacy', 'z']
    path = tmp_path/'Contents.py'
    doc = ast.get_docstring(ast.parse(path.read_text(encoding='utf-8')))
    assert doc.startswith('My Toolbox\nVersion 2.4.9   02-Jan-2020')
    old = path.read_text(encoding='utf-8')
    misc.mkcontents(srt=1, directory=tmp_path, date=date(2020, 1, 3))
    assert (tmp_path/'Contents.pyold').read_text(encoding='utf-8') == old
    doc = ast.get_docstring(ast.parse(path.read_text(encoding='utf-8')))
    assert doc.startswith('My Toolbox\nVersion 2.4.10')
    dry = misc.mkcontents(srt=1, cfile=None, directory=tmp_path)
    assert [line.split()[0] for line in dry.splitlines()] == ['z', 'b', 'legacy']
    misc.mkcontents('MATLAB index', 1.1, 3, 'Contents.m', directory=tmp_path)
    assert (tmp_path/'Contents.m').read_text(encoding='utf-8').startswith('% MATLAB index\n% Version 1.1')


def test_wafostamp_replaces_overlay_and_keeps_plot_active(plots, tmp_path):
    fig, ax = plots.subplots()
    ax.plot([0, 1], [1, 0])
    assert misc.wafostamp() == ([], None)
    texts, overlay = misc.wafostamp('A reproducible curve', '(ER)', fig=fig, date=date(2020, 1, 2))
    assert texts[0].get_text() == 'A reproducible curve'
    assert texts[1].get_text() == 'made by WAFO 02-Jan-2020 (ER)'
    assert plots.gca() is ax
    updated, replacement = misc.wafostamp(1, fig=fig, date='fixed date')
    assert overlay not in fig.axes
    assert len(updated) == 1
    assert updated[0].get_text() == 'made by WAFO fixed date'
    assert misc.wafostamp(0, fig=fig) == ([], None)
    assert replacement in fig.axes
    render(fig, tmp_path, 'stamp')
