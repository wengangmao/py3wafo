"""Installed offline source documentation and nonblocking help menus."""
import json as _json
from pathlib import Path as _Path

_TOPICS = [
    ('Installation', 'install'), ('About WAFO', 'aboutwafo'),
    ('Source release notes', 'Readme'), ('Getting started', 'getstart'),
    ('Adding features', 'addfeatr'), ('Known source limitations', 'limitations'),
    ('Bug report form', 'bugreport'), ('Spectrum/covariance/PDF structures', 'datastructures'),
    ('Crossing definitions', 'crossdef'), ('Turning-point definitions', 'tpdef'),
    ('Wave definitions', 'wavedef'), ('Amplitude definitions', 'ampdef'),
    ('Period definitions', 'perioddef'),
]


def _data():
    return _Path(__file__).with_name('data')


def _open(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ('yes', 'y', 'no', 'n'):
        return str(value).lower().startswith('y')
    raise ValueError('ButtonName must be yes/no or a Boolean')


def wafohelp(ButtonName='no', *, topic=None):
    """Return an installed offline help URI, or original topic text.

    topic names are drawn from the original toolbox documentation, including
    datastructures/crossdef/tpdef/wavedef/ampdef/perioddef. The help index also
    records Python conversion status so original docs are not mistaken for a
    promise of supported Python options. Explicit yes opens the local index;
    the default returns its URI without browser side effects.
    """
    launch = _open(ButtonName)
    if topic is not None:
        topics = _json.loads((_data()/'topics.json').read_text(encoding='utf-8'))
        key = next((k for k in topics if k.lower() == str(topic).lower()), None)
        if key is None:
            raise ValueError('Unknown documentation topic: '+str(topic))
        return topics[key]
    target = (_data()/'index.html').resolve()
    if not target.is_file():
        raise FileNotFoundError('Offline help index is missing from this installation')
    uri = target.as_uri()
    if launch:
        import webbrowser
        webbrowser.open(uri)
    return uri


def wafoweb(ButtonName='no'):
    """Return the historical website URL recorded in the supplied toolbox.

    No assertion is made about availability of that legacy site. Explicit yes
    opens the URL in a browser; the default only returns it.
    """
    url = 'http://www.maths.lth.se/matstat/wafo/'
    if _open(ButtonName):
        import webbrowser
        webbrowser.open(url)
    return url


def wafomenu(kind=0):
    """Return source help topics (kind=1), or a nonblocking button menu (0).

    Graphical mode returns figure/buttons/text handles; clicking a topic
    updates a scroll-free text panel with its first 45 lines. Complete text is
    available through wafohelp(topic=...). No input(), eval(), or show() call
    is used; the caller controls its GUI event loop.
    """
    topics = {label: wafohelp(topic=name) for label, name in _TOPICS}
    if kind == 1:
        return topics
    if kind != 0:
        raise ValueError('kind must be 0 (graphical) or 1 (text mapping)')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    figure = plt.figure(figsize=(12, 8))
    panel = figure.add_axes([.37, .06, .61, .88]); panel.set_axis_off()
    text = panel.text(0, 1, 'Select a documentation topic.\nThese are original MATLAB source notes.\nSee the conversion catalog for Python compatibility.', va='top', fontsize=8, family='monospace', wrap=True)
    buttons = []
    for index, (label, content) in enumerate(topics.items()):
        target = figure.add_axes([.02, .91-index*.065, .32, .05])
        button = Button(target, label)
        def select(event, content=content):
            text.set_text('\n'.join(content.splitlines()[:45]))
            figure.canvas.draw_idle()
        button.on_clicked(select)
        buttons.append(button)
    figure.suptitle('WAFO original toolbox documentation')
    # Keep widget callbacks alive for callers retaining only the figure.
    figure._wafo_help_buttons = buttons
    return {'figure': figure, 'buttons': buttons, 'text': text}
