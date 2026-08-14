Publishing py3wafo
==================

The PyPI distribution is named ``py3wafo``.  The Python import namespace is
kept compatible with WAFO::

    python -m pip install py3wafo
    python -c "import wafo; print(wafo.__version__)"

Build and check
---------------

Run these commands from a clean checkout::

    python -m venv .venv
    .venv/Scripts/python -m pip install --upgrade pip build twine
    .venv/Scripts/python -m pip install -e ".[test]"
    .venv/Scripts/python -m pytest src/wafo
    .venv/Scripts/python -m build
    .venv/Scripts/python -m twine check dist/*

On Linux and macOS use ``.venv/bin/python`` instead.

TestPyPI upload
---------------

Create an API token at TestPyPI, then run::

    .venv/Scripts/python -m twine upload --repository testpypi dist/*
    python -m pip install --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple py3wafo

PyPI upload
-----------

After the TestPyPI install and GitHub Actions pass, create a PyPI project/token
and upload the immutable release artifacts::

    .venv/Scripts/python -m twine upload dist/*

PyPI does not allow replacing files for an existing version.  Increment
``wafo.__version__`` before rebuilding every subsequent release.

Backend status
--------------

The distribution contains the pure-Python ``rindmod`` compatibility backend.
The historical CROSSPACK ``cov2mmpdfreg`` solver is not reimplemented because
an unvalidated approximation could produce misleading scientific densities;
that call raises ``NotImplementedError`` with a direct explanation.
