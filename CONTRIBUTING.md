# Contributing to py3wafo

Bug reports, numerical comparisons with MATLAB WAFO, and fixes are welcome.

## Reporting a problem

Open an issue with:

- the py3wafo, Python, NumPy and SciPy versions:
  `python -c "import wafo, numpy, scipy, sys; print(wafo.__version__, numpy.__version__, scipy.__version__, sys.version)"`;
- a minimal script that reproduces the problem;
- if you have MATLAB WAFO, its result for the same input. Comparisons of this
  kind are the most valuable contribution, because MATLAB parity has not yet
  been certified.

## Development setup

```bash
git clone https://github.com/YOUR-GITHUB-USER/py3wafo.git
cd py3wafo
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest
```

## Guidelines

- Keep MATLAB function names and module locations, so that WAFO users can
  find functions. Pythonic additions go in keyword-only arguments.
- Every change to numerical code needs a test that checks a value derived
  independently: an analytic result, a published value, a simulation or an
  identity. A test that only re-runs the implementation is not enough.
- If you deliberately deviate from the MATLAB source, document it in the
  docstring, keep the original behaviour available where practical, and add
  an entry to `docs/migration/REPAIR_LOG.md`.
- Raise a clear exception for unsupported options instead of silently
  approximating.
- Run `python tools/validate_functions.py` before opening a pull request. It
  runs the suite and regenerates the per-function evidence reports in
  `docs/migration`.

## Licence of contributions

By contributing, you agree that your contributions are licensed under the
GNU General Public License v3.0 or later, the licence of this project.
