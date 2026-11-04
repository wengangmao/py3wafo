"""Discover replacements by original C/Fortran/binary filename.

Mappings identify equivalent capabilities, not identical low-level signatures
or old numerical algorithm selections. Consult each entry's limitations.
"""
import importlib
import json
from importlib.resources import files


def native_files(status=None):
    """All native-source/binary records, usable without the MATLAB checkout."""
    data = json.loads(files('wafo').joinpath('native_manifest.json').read_text(encoding='utf-8'))
    return data if status is None else [r for r in data if r['python_status']==status]


def native_info(name):
    """Find exact relative paths or basenames, case insensitively."""
    name = str(name).replace('\\','/').lower()
    return [r for r in native_files() if r['path'].lower()==name or r['path'].rsplit('/',1)[-1].lower()==name]


def resolve_native(name):
    """Return the Python callable for a source/binary filename.

    Read native_info(name) and the returned function's documentation for its
    Python signature; this function does not accept old binary file protocols.
    """
    records = native_info(name)
    if not records:
        raise KeyError(f"No native-file entry for {name!r}")
    apis = {r['python_api'] for r in records}
    if len(apis)!=1 or not next(iter(apis)):
        raise ValueError(f"No unique Python capability replacement for {name!r}")
    module,attribute = next(iter(apis)).rsplit('.',1)
    return getattr(importlib.import_module(module),attribute)
