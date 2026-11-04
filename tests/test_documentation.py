"""Execute the Python code blocks of the README and user guides.

Blocks in one document share a namespace and run in order, as a reader would
run them. Figures are written to a temporary directory.
"""
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ['README.md', 'docs/GETTING_STARTED.md']


def _blocks(path):
    return re.findall(r'```python\n(.*?)```', path.read_text(encoding='utf-8'), flags=re.S)


@pytest.mark.parametrize('document', DOCUMENTS)
def test_documentation_examples_run(document, tmp_path, monkeypatch):
    path = ROOT/document
    if not path.exists():
        pytest.skip(f'{document} is not part of this checkout')
    blocks = _blocks(path)
    assert blocks, f'no Python examples found in {document}'
    monkeypatch.chdir(tmp_path)
    namespace = {'__name__': '__doc_example__'}
    for number, code in enumerate(blocks, 1):
        try:
            exec(compile(code, f'{document} block {number}', 'exec'), namespace)
        except Exception as error:
            raise AssertionError(f'{document} block {number} failed: {error!r}\n{code}') from error
    import matplotlib.pyplot as plt
    plt.close('all')
