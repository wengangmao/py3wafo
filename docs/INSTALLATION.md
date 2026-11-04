# Installing py3wafo

py3wafo is a pure-Python package: it needs no compiler, MATLAB or external
executable. These instructions cover Windows, macOS and Linux.

## 1. Requirements

| Component | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.13 | 64-bit recommended |
| NumPy | ≥ 1.26 | installed automatically |
| SciPy | ≥ 1.15 | installed automatically |
| Matplotlib | ≥ 3.7 | optional, `[plot]` extra: plotting, demos, paper figures |
| pyproj | ≥ 3.6 | optional, `[map]` extra: ellipsoidal map projections in `wafo.m_map` |

Check your Python version:

```bash
python --version
```

On Windows, if `python` is not found, use `py` (the Python launcher), for example
`py -m pip install ...`.

## 2. Create an isolated environment (recommended)

Installing into a virtual environment keeps py3wafo and its dependencies
separate from other projects.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell refuses to run the activation script, allow local scripts for
your user once with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use the Command
Prompt version, `.venv\Scripts\activate.bat`.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**conda**

```bash
conda create -n py3wafo python=3.12 numpy scipy matplotlib pyproj
conda activate py3wafo
```

## 3. Install py3wafo

Choose one of the following.

### a) From GitHub, without cloning

```bash
python -m pip install "py3wafo[all] @ git+https://github.com/YOUR-GITHUB-USER/py3wafo.git"
```

Pin a release by appending a tag, for example `...py3wafo.git@v0.4.0`.

### b) From a local clone

```bash
git clone https://github.com/YOUR-GITHUB-USER/py3wafo.git
cd py3wafo
python -m pip install ".[all]"
```

### c) Editable install for development

Changes to the source take effect without reinstalling:

```bash
python -m pip install -e ".[test]"
```

### d) From a wheel file (offline installs)

Build the wheel once on a machine with internet access:

```bash
python -m pip install build
python -m build                     # creates dist/py3wafo-0.4.0-py3-none-any.whl
```

Copy the wheel to the target machine and install it:

```bash
python -m pip install py3wafo-0.4.0-py3-none-any.whl
```

For a fully offline target, also download the dependencies on the connected
machine with `python -m pip download py3wafo-0.4.0-py3-none-any.whl -d wheels`,
copy the `wheels` folder across, and install with
`python -m pip install --no-index --find-links wheels py3wafo`.

### Choosing extras

`pip install py3wafo` installs only the numerical core. Add extras in square
brackets: `[plot]` for Matplotlib, `[map]` for Matplotlib and pyproj, `[all]`
for both, and `[test]` for pytest plus both.

## 4. Verify the installation

```bash
python -c "import wafo, py3wafo; print(wafo.__version__)"
```

This should print `0.4.0`. A short functional check:

```bash
python -c "from wafo.data import load; from wafo.onedim import dat2tp; from wafo.cycles import tp2rfc; print(len(tp2rfc(dat2tp(load('sea.dat')))), 'rainflow cycles')"
```

The expected output is `1086 rainflow cycles`.

To run the full test suite (about two minutes), from a clone:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

All tests should pass. Tests that need the original MATLAB toolbox are skipped
automatically when it is absent.

## 5. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'wafo'` | The environment where py3wafo was installed is not active. Activate it, or run `python -m pip show py3wafo` to see where it is installed. |
| `ImportError` mentioning `matplotlib` | Install the plotting extra: `python -m pip install "py3wafo[plot]"`. |
| `ImportError` mentioning `pyproj` | Only `wafo.m_map` ellipsoidal projections need it: install `py3wafo[map]`. |
| Plot windows do not appear | Scripts run without a display. Call `matplotlib.pyplot.show()`, or save figures with `fig.savefig(...)`. |
| `ImportError` loading NumPy's C extensions | NumPy was built for a different Python version, for example from a shared folder. Reinstall NumPy and SciPy inside the active environment. |
| pip reports `Can not combine '--user' and '--target'` | Your pip configuration forces user installs. Use a virtual environment, or add `--no-user`. |

## 6. Uninstall

```bash
python -m pip uninstall py3wafo
```
