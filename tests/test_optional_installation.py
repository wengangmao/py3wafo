import os
from pathlib import Path
import subprocess
import sys


def test_package_imports_without_optional_plot_or_projection_libraries():
    import wafo
    # Carry the actually selected package parent through to the subprocess,
    # so this check also validates an installed wheel with source imports off.
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join([str(Path(wafo.__file__).resolve().parent.parent), *sys.path])
    program = '''
import importlib.abc
import sys
class NoOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib', 'pyproj'}:
            raise ModuleNotFoundError('Optional dependency deliberately unavailable: '+fullname)
sys.meta_path.insert(0, NoOptional())
import wafo
import py3wafo
assert wafo.onedim is py3wafo.onedim
assert wafo.waforoot is py3wafo.waforoot
assert wafo.__version__ == py3wafo.__version__
assert 'matplotlib' not in sys.modules and 'pyproj' not in sys.modules
assert py3wafo.initwafo()['root'] == wafo.waforoot()
'''
    result = subprocess.run([sys.executable, '-c', program], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
