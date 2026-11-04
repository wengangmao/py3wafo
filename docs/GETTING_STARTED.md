# Getting started with py3wafo

This guide tours the main modules with short, runnable examples. Every Python
block below runs as written and is executed by the test suite
(`tests/test_documentation.py`), so the examples stay correct. Install the
package first; see [INSTALLATION.md](INSTALLATION.md). The plotting examples
need the `[plot]` extra.

## 1. Conventions

These conventions are what most often differ from the MATLAB toolbox.

- **Imports.** `import wafo` and `import py3wafo` give the same package.
  Functions keep their MATLAB names and live in the same modules, so MATLAB
  `tp2rfc` is `wafo.cycles.tp2rfc`.
- **Indices are zero-based.** Functions that return sample indices, such as
  `dat2tp(..., return_indices=True)` and crossing indices, follow Python.
- **Time series** are NumPy arrays with time in the first column and one or
  more signal columns, as in WAFO: `x[:, 0]` is time and `x[:, 1]` the signal.
- **Units.** Spectra use angular frequency (rad/s) unless they are explicitly
  converted with `ttspec`. Lengths are in metres and times in seconds.
- **Structures.** WAFO's MATLAB structs, such as spectra and PDFs, are
  dictionaries that also allow attribute access: `S.w`, `S.S`, `f.x[0]`,
  `f.f`. They print their fields and can be copied like dicts.
- **No global state.** Random generators take a `seed=` argument. Demos and
  paper figures keep their state in an explicit session object rather than in
  MATLAB globals.
- **Multiple outputs.** MATLAB `[a, b] = f(...)` becomes a tuple. Where MATLAB
  returned optional extra outputs, Python usually asks for them with
  `full_output=True`.
- **Help.** Every function has a docstring (`help(wafo.cycles.tp2rfc)`),
  which also states its supported options and known limits.

## 2. Wave spectra

```python
import numpy as np
from wafo.spec import jonswap, torsethaugen, spec2mom, spec2char, ttspec

S = jonswap(sdata=[4.0, 8.0])            # Hm0 = 4 m, Tp = 8 s, angular frequency grid
print(S.w[:3], S.S[:3])                  # frequencies (rad/s) and spectral density (m^2 s/rad)

m0, m2, m4 = spec2mom(S, 4)              # spectral moments m0, m2, m4
Hm0, Tm02, Tp = spec2char(S, ['Hm0', 'Tm02', 'Tp'])
print(f'Hm0={Hm0:.2f} m  Tm02={Tm02:.2f} s  Tp={Tp:.2f} s')

St = torsethaugen(np.linspace(0, 3, 257), [6.0, 8.0])   # double-peaked swell + wind sea
Sf = ttspec(St, 'f')                     # the same spectrum in Hz
```

Directional spectra combine a frequency spectrum with a spreading function:

```python
from wafo.spec import jonswap, spreading, mkdspec

S = jonswap(sdata=[4.0, 8.0])
D = spreading(51, 'cos2s', 0, [15, 15, 2*np.pi/8, 0, 0], S.w, 0)   # cos-2s, s = 15
Sd = mkdspec(S, D)
print(Sd.type, np.shape(Sd.S))           # 'dir', (directions, frequencies)
```

## 3. Simulating sea surfaces

```python
from wafo.spec import jonswap
from wafo.wsim import spec2sdat, spec2nlsdat

S = jonswap(sdata=[4.0, 8.0])
x = spec2sdat(S, 2048, seed=7)           # linear Gaussian sea: columns time, elevation
print(x.shape, round(float(x[:, 1].std()), 2))

S.h = 30.0                               # water depth for second-order (Stokes) effects
xnl = spec2nlsdat(S, 2048, seed=7)       # second-order non-linear sea
print('skewness', round(float(((xnl[:, 1]-xnl[:, 1].mean())**3).mean()/xnl[:, 1].std()**3), 2))
```

## 4. Measured data, turning points and waves

WAFO's original data sets are bundled: `sea.dat`, `gfaks89.dat`,
`gfaksr89.dat`, `yura87.dat`, `atlantic.dat`, `japansea.dat`, `northsea.dat`,
`sfa89.dat` and `sn.dat`.

```python
from wafo.data import load
from wafo.onedim import dat2tp, dat2wa, dat2steep, dat2spec

sea = load('sea.dat')                    # time [s], surface elevation [m]
tp, ind = dat2tp(sea, return_indices=True)    # turning points and their zero-based indices
Tc = dat2wa(sea, 0, 'u2d')               # crest periods (up- to down-crossing)
steepness, height = dat2steep(sea)       # wave steepness and wave height per zero-crossing wave
Se = dat2spec(sea)                       # smoothed spectral estimate
print(len(tp), 'turning points;', len(height), 'waves; mean crest period', round(float(Tc.mean()), 2), 's')
```

## 5. Cycle counting and fatigue

```python
from wafo.data import load
from wafo.onedim import dat2tp
from wafo.cycles import tp2rfc, tp2mm, cc2cmat, tp2lc
from wafo.damage import cc2dam

tp = dat2tp(load('sea.dat'))
rfc = tp2rfc(tp)                         # rainflow cycles as rows (min, max)
mM = tp2mm(tp)                           # min-to-max cycles
F = cc2cmat((-2, 2, 33), rfc)            # 33 x 33 rainflow matrix on the levels -2..2
lc = tp2lc(tp)                           # level crossing spectrum (level, count)
D = cc2dam(rfc, [3, 4, 5])               # Palmgren-Miner damage for Woehler exponents 3, 4, 5
print(len(rfc), 'cycles; damage', D.round(1))
```

## 6. Theoretical wave distributions

The Rice-type routines compute exact Gaussian densities of wave periods and
amplitudes from a spectrum by multivariate normal integration.

```python
from wafo.spec import jonswap
from wafo.trgauss import spec2thpdf, spec2mmtpdf

S = jonswap(sdata=[4.0, 8.0])
fTc = spec2thpdf(S, 0, 'Tc', [0, 12, 25])            # density of the crest period Tc
fTcAc = spec2thpdf(S, 0, 'TcAc', [0, 12, 13], np.linspace(0, 6, 9))   # joint Tc / crest amplitude
print(fTc.labx, np.shape(fTcAc.f))
```

Transformed-Gaussian models capture moderate non-linearity in measured data:

```python
from wafo.data import load
from wafo.trgauss import dat2tr

g, test, *_ = dat2tr(load('sea.dat'), full_output=True)   # g: table (Gaussian level -> data level)
print(np.shape(g), float(np.ravel(test)[0]))
```

## 7. Statistics and kernel density estimation

```python
from wafo.wstats import wweibfit, wweibcdf, wgevfit
from wafo.kdetools import kdebin, kdeoptset

rng = np.random.default_rng(0)
heights = 3*rng.weibull(2, 500)
phat = wweibfit(heights)                 # Weibull fit (parameters, covariance, ...)
annual_maxima = rng.gumbel(10, 2, 60)
gev = wgevfit(annual_maxima)             # generalized extreme value fit

f = kdebin(heights, kdeoptset(kernel='epan'))   # binned kernel density estimate
print(f.x[0][np.argmax(f.f)])            # mode of the estimate
```

## 8. Markov models of load sequences

```python
from wafo.wsim import mctpsim
from wafo.markov import mctp2rfm

n = 8
F = np.triu(np.ones((n, n)), 1)          # min-to-max transition counts on 8 levels
x = mctpsim((F, None), 1000, seed=3)     # simulated Markov chain of turning points (level indices)
rfm = mctp2rfm((F, None))                # expected rainflow matrix of that chain
print(len(x), np.shape(rfm))
```

## 9. Plotting

Most results can be passed to WAFO's plotting helpers, which accept an `ax=`
argument:

```python
import matplotlib
matplotlib.use('Agg')                    # omit this line in an interactive session
import matplotlib.pyplot as plt
from wafo.spec import jonswap, wspecplot
from wafo.trgauss import spec2thpdf, pdfplot

S = jonswap(sdata=[4.0, 8.0])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
wspecplot(S, ax=ax1)
pdfplot(spec2thpdf(S, 0, 'Tc', [0, 12, 25]), ax=ax2)
fig.savefig('jonswap.png')
plt.close(fig)
```

## 10. Demos and paper reproductions

The figures of the two WAFO papers are reproduced by `wafo.papers`. Each call
returns a `PaperFigure` holding the Matplotlib figure, the numbers plotted and
any limitations. A session caches the expensive calculations between figures.

```python
from wafo import papers

session = papers.wafoinit(Np=17, NNp=9)  # smaller grids than the paper, for speed
fig2 = papers.wafofig2(session)          # Torsethaugen and estimated spectra
print(fig2.title, fig2.status)
papers.wafocleanup(session)              # close the session's figures and free its cache
```

In an interactive session, `papers.wafodemo()` and `papers.recdemo()` open
menus for all figures. `papers.papermenu()` lets you choose between the two
papers.

## 11. Getting help

```python
import wafo
from wafo.docs import wafohelp

help(wafo.cycles.tp2rfc)                 # Python docstring: arguments, returns, limits
print(wafohelp(topic='tpdef')[:200])     # original WAFO documentation of turning points
print(wafo.source_info('tp2rfc'))        # which MATLAB file it ports, and its status
```

`wafohelp(ButtonName='yes')` opens the offline HTML index of the whole toolbox
in a browser. `wafo.coverage()` returns the conversion record of every MATLAB
source file.

## 12. Command-line numerical drivers

The reimplemented native routines can also be called from the command line
with NumPy `.npz` files:

```bash
python -m wafo.exec --help
python -m wafo.exec findrfc input.npz output.npz
```

## Where to go next

- The docstrings in each module are the reference documentation.
- [migration/FUNCTION_CATALOG.md](migration/FUNCTION_CATALOG.md) maps every
  MATLAB file to its Python function.
- [migration/COMPATIBILITY.md](migration/COMPATIBILITY.md) and
  [migration/REPAIR_LOG.md](migration/REPAIR_LOG.md) describe differences from
  the MATLAB toolbox.
