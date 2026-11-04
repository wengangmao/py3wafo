"""Inspectable migration coverage; pending entries are never executable stubs."""
import json
from importlib.resources import files


def coverage(status=None):
    """Return source records, optionally filtered by implemented/adapted/pending."""
    data = json.loads(files('wafo').joinpath('conversion_manifest.json').read_text(encoding='utf-8'))
    return data if status is None else [r for r in data if r['status'] == status]


def source_info(name):
    """All MATLAB source records matching a function name or source path."""
    return [r for r in coverage() if r['name'] == name or r['matlab'] == name]
