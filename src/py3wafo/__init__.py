"""Python 3 WAFO package, with backwards-compatible ``wafo`` imports.

This package exposes the same implementation and submodules. Consult
``wafo.coverage()`` and the validation reports for implementation limits;
the package name does not imply complete MATLAB-toolbox parity.
"""
import importlib as _importlib
import pkgutil as _pkgutil
import sys as _sys
import wafo as _wafo
from wafo import __version__,coverage,source_info

for _name, _value in vars(_wafo).items():
    if not _name.startswith('_'):
        globals()[_name] = _value

for _entry in _pkgutil.walk_packages(_wafo.__path__,'wafo.'):
    _module = _importlib.import_module(_entry.name)
    _relative = _entry.name[len('wafo.'):]
    _sys.modules[__name__+'.'+_relative] = _module
    if '.' not in _relative and not _relative.startswith('_'):
        globals()[_relative] = _module

__all__ = ['__version__','coverage','source_info']+[name for name in globals()
    if not name.startswith('_') and name not in ('coverage','source_info')]
