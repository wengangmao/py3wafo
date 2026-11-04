# py3wafo — WAFO for Python 3

[![CI](https://github.com/YOUR-GITHUB-USER/py3wafo/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR-GITHUB-USER/py3wafo/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)

**py3wafo** is a Python 3 port of **WAFO** (*Wave Analysis for Fatigue and
Oceanography*), the MATLAB toolbox from Lund University for the statistical
analysis and simulation of random waves and random loads. It covers:

- wave spectra, spectral moments and sea-state parameters;
- simulation of linear and second-order (non-linear) Gaussian seas;
- turning points, level crossings and rainflow cycle counting;
- Palmgren–Miner fatigue damage and Markov models of load cycles;
- Rice-type distributions of wave periods, heights and crest velocities;
- transformed-Gaussian models, saddlepoint crossing intensities for quadratic seas;
- extreme-value and other distributions, kernel density estimation;
- directional spectra, spreading functions and map projections;
- the original WAFO measurement data sets, demos and paper reproductions.

Every function of the MATLAB WAFO 2.1.1 toolbox has a Python counterpart.
No MATLAB, MEX files, C/Fortran compilers or external executables are needed:
the original native routines are reimplemented with NumPy and SciPy.

> **Status — beta (0.4.0).** 775 regression tests pass, and every public
> function has passing-test evidence. Numerical agreement with MATLAB WAFO has
> **not** been certified, because the tests could not be run against MATLAB.
> Known deviations and limits are listed in [Known limitations](#known-limitations).

## Installation

Requires Python 3.10 or newer. Full instructions, including virtual
environments, conda and offline installs, are in
[docs/INSTALLATION.md](docs/INSTALLATION.md).

From a clone of this repository:

```bash
git clone https://github.com/YOUR-GITHUB-USER/py3wafo.git
cd py3wafo
python -m pip install ".[all]"
```

Or install straight from GitHub without cloning:

```bash
python -m pip install "py3wafo[all] @ git+https://github.com/YOUR-GITHUB-USER/py3wafo.git"
```

| Extra | Adds | Needed for |
|---|---|---|
| *(none)* | NumPy, SciPy | all numerical functions |
| `[plot]` | Matplotlib | plotting functions, demos and paper figures |
| `[map]` | Matplotlib, pyproj | `wafo.m_map` ellipsoidal projections |
| `[all]` | both of the above | everything |
| `[test]` | pytest, Matplotlib, pyproj | running the test suite |

Check the installation:

```bash
python -c "import wafo; print(wafo.__version__)"
```

## Quick start

```python
import numpy as np
from wafo.data import load
from wafo.spec import jonswap, spec2char
from wafo.wsim import spec2sdat
from wafo.onedim import dat2tp
from wafo.cycles import tp2rfc
from wafo.damage import cc2dam

# A JONSWAP spectrum with Hm0 = 4 m and Tp = 8 s, and its sea-state parameters
S = jonswap(sdata=[4.0, 8.0])
Hm0, Tm02 = spec2char(S, ['Hm0', 'Tm02'])          # 4.00 m, 6.28 s

# Simulate 4096 samples of the sea surface: columns are time and elevation
x = spec2sdat(S, 4096, seed=1)

# Rainflow cycles and fatigue damage of a measured record shipped with WAFO
sea = load('sea.dat')
rfc = tp2rfc(dat2tp(sea))                           # 1086 cycles
damage = cc2dam(rfc, [3, 4, 5])                     # Woehler exponents 3, 4, 5
```

`import py3wafo` and `import wafo` give the same package; the examples use
`wafo`, matching the MATLAB function names. More examples:

- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md): a guided tour of every
  module, conventions (units, indexing, PDF structures) and help functions.
- [examples/quickstart.py](examples/quickstart.py): one calculation from each
  main module.
- [examples/sea_fatigue.py](examples/sea_fatigue.py): a fatigue workflow on measured data.
- [examples/native_capabilities.py](examples/native_capabilities.py): the
  Python replacements of WAFO's C/Fortran routines.

## Documentation

| Guide | For |
|---|---|
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | installing, verifying and troubleshooting |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | conventions and a tour of every module |
| [docs/migration](docs/migration) | how the MATLAB toolbox was ported and validated |
| [CONTRIBUTING.md](CONTRIBUTING.md) | development setup and guidelines |
| [docs/PUBLISHING_TO_GITHUB.md](docs/PUBLISHING_TO_GITHUB.md) | maintainers: pushing, CI and releases |
| [CHANGELOG.md](CHANGELOG.md) | release history |

## Package layout

| Module | Contents |
|---|---|
| `wafo.spec` | wave spectra (JONSWAP, Torsethaugen, Ochi–Hubble, …), moments, directional spectra |
| `wafo.wsim` | simulation of Gaussian, transformed and second-order seas, Markov loads |
| `wafo.onedim` | time-series tools: turning points, crossings, wave parameters, spectral estimation, reconstruction |
| `wafo.cycles` | rainflow and other cycle counts, cycle matrices, level-crossing spectra |
| `wafo.damage` | Palmgren–Miner damage, S–N curves, damage intensity |
| `wafo.markov` | Markov chains of turning points and switching Markov loads |
| `wafo.trgauss` | transformed Gaussian models, Rice wave distributions, multivariate normal integrals, quadratic seas |
| `wafo.wstats` | distributions, fitting, extreme values, 2-D wave models |
| `wafo.kdetools` | kernel density estimation and bandwidth selection |
| `wafo.wavemodels` | parametric distributions of wave characteristics |
| `wafo.multidim` | directional spreading estimation and sensor transfer functions |
| `wafo.m_map` | M_Map map projections and coastlines |
| `wafo.misc` | numerical utilities and physical constants |
| `wafo.data` | the original WAFO measurement data sets (`load('sea.dat')`, …) |
| `wafo.papers`, `wafo.wdemos` | reproductions of the WAFO papers and course demos |
| `wafo.source`, `wafo.exec` | the reimplemented native routines and a command-line driver |
| `wafo.docs` | offline help (`wafohelp`) built from the original documentation |

## Running the tests

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The full suite takes about two minutes. `python tools/validate_functions.py`
runs the same tests and regenerates the per-function validation reports in
[docs/migration](docs/migration).

## Known limitations

- **No MATLAB parity certification.** Tests check analytic identities,
  published values, simulations and the original data sets. They were not
  compared with MATLAB WAFO output.
- **Documented deviations from the MATLAB source.** A few routines contained
  defects that are corrected here, each with a regression test. The
  corrections are listed in [docs/migration/REPAIR_LOG.md](docs/migration/REPAIR_LOG.md):
  - `dirsp2chitwo`: second-order coefficients halved to match Stokes theory;
    `source_scaling=True` keeps the original behaviour.
  - `mctp2tc`: counting convention and reference-level indexing.
  - Saddlepoint CDF evaluation near the distribution mean.
- **Restricted options.** Some routines support fewer legacy options than the
  MATLAB originals. For example, the Rice wave drivers accept untransformed 1-D
  angular-frequency spectra only. Each docstring states its limits.
- **Paper figures.** Figure 6 of the WAFO paper needs a conditional density
  that is not yet available; Figure 8 never existed in the MATLAB source.
- **Unported scripts.** 15 standalone MATLAB scripts are not ported: the
  tutorial chapters, course labs and module self-tests.

## Background and credits

WAFO was developed at the Centre for Mathematical Sciences, Lund University,
by the WAFO group (P. A. Brodtkorb, P. Johannesson, G. Lindgren, I. Rychlik,
J. Rydén, E. Sjö and others). This repository is an independent Python port of
WAFO 2.1.1 (2005) and is not an official WAFO release. See [NOTICE.md](NOTICE.md)
for attribution of the original toolbox, M_Map and the bundled data.

If you use py3wafo in research, please cite the original WAFO publication:

> Brodtkorb, P.A., Johannesson, P., Lindgren, G., Rychlik, I., Rydén, J. and
> Sjö, E. (2000). *WAFO — a Matlab toolbox for analysis of random waves and
> loads.* Proc. 10th Int. Offshore and Polar Eng. Conf. (ISOPE), Seattle, Vol. III, 343–350.

## License

py3wafo is free software released under the
[GNU General Public License v3.0 or later](LICENSE).
