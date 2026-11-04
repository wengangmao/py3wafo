from pathlib import Path
import os
import sys
import pytest
import wafo
from wafo import root, docs


def test_installed_path_helpers_are_portable_and_preserve_search_path():
    before = list(sys.path)
    assert Path(wafo.waforoot()).resolve() == Path(wafo.__file__).resolve().parent
    assert wafo.wafoexepath().endswith(os.sep)
    assert (Path(wafo.wafoexepath())/'__init__.py').is_file()
    for part in ('tools', 'docs', 'wdemos', 'data', 'papers'):
        assert all(Path(p).is_dir() for p in wafo.wafopath(part))
    all_paths = wafo.pathdef().split(os.pathsep)
    for kind in ('minimum', 'normal', 'full'):
        result = wafo.initwafo(kind)
        assert result['version'] == wafo.__version__
        assert set(result['paths']) <= set(all_paths)
    assert len(wafo.initwafo('minimum')['paths']) < len(wafo.initwafo('normal')['paths']) < len(wafo.initwafo('full')['paths'])
    assert sys.path == before
    assert root.parzen(5).shape == (5,)
    for action in (lambda: wafo.wafopath('missing'), lambda: wafo.wafopath(add=1), lambda: wafo.initwafo('bad')):
        with pytest.raises(ValueError):
            action()


def test_installed_source_documentation_topics_and_explicit_browser(monkeypatch):
    import webbrowser
    seen = []
    monkeypatch.setattr(webbrowser, 'open', seen.append)
    uri = docs.wafohelp()
    assert uri.startswith('file:') and 'index.html' in uri
    assert 'WAFO' in docs.wafohelp(topic='aboutwafo')
    assert 'cross' in docs.wafohelp(topic='crossdef').lower()
    assert not seen
    assert docs.wafohelp('yes') == uri
    assert seen == [uri]
    url = docs.wafoweb()
    assert url == 'http://www.maths.lth.se/matstat/wafo/'
    docs.wafoweb(True)
    assert seen[-1] == url
    assert len(docs.wafomenu(1)) == 13
    with pytest.raises(ValueError):
        docs.wafohelp(topic='absent')
    with pytest.raises(ValueError):
        docs.wafohelp('invalid')
    with pytest.raises(ValueError):
        docs.wafomenu(2)


def test_nonblocking_help_buttons_update_and_render(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    menu = docs.wafomenu()
    assert len(menu['buttons']) == 13
    menu['buttons'][1]._observers.process('clicked', None)
    assert 'WAFO' in menu['text'].get_text()
    path = tmp_path/'help.png'
    menu['figure'].savefig(path)
    assert path.stat().st_size > 1000
    plt.close(menu['figure'])
