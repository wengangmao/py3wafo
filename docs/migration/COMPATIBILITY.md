# Python API and numerical compatibility

The library follows the original module names. Each implemented MATLAB name maps
to `wafo.<module>.<function>` in the conversion manifest. No runtime interprets
MATLAB files or launches legacy binaries.

## Common conventions

| Item | Python convention |
|---|---|
| Array indices and Markov states | Zero based, including `findrfc`, `csort`, and `cc2dcc` results |
| `disufq` active bounds | Zero-based inclusive `nmin,nmax`; MATLAB fallbacks retain one-based bounds |
| Multiple outputs | Tuples or explicit flags such as `return_indices=True`, `full_output=True`, `both=True` |
| `tp2rfc(..., full_output=True)` | `(all_cycles, closed_cycles, residual)` |
| Default residual cycles | Upward min-to-max excursions, matching WAFO's `up` option |
| Spectrum representation | Mapping with attribute access (`S.w`, `S.S`); 1D, directional and Cartesian support depends on the function |
| Random number generators | NumPy `seed=`; results reproduce within this implementation, not MATLAB/Fortran random streams |
| Distribution random dimensions | Keyword `size=`, rather than positional MATLAB trailing dimensions |
| Plot flags | Explicit supported modes in each docstring; graphics return axes/controllers without show() |
| Fitting functions | Parameter estimates only; confidence bounds, covariance estimates, and method selectors are generally pending |

## Distribution parameter mappings

Functions use legacy names `w<family>pdf`, `cdf`, `inv`, `rnd`, and `stat` where
the original source has them. `stat` returns `(mean, variance)`.

| Family | Parameters in order |
|---|---|
| `norm`, `logn` | mean and **variance** (of log(X) for lognormal); defaults 0,1 |
| `weib`, `frech` | scale a, shape c |
| `gam` | shape a, scale b (default b=1) |
| `ggam` | shape a, scale b, exponent c (default c=1) |
| `beta` | shapes a,b |
| `exp` | mean/scale m (default 1) |
| `rayl` | scale b |
| `chi2`, `t` | degrees of freedom |
| `f` | numerator and denominator degrees of freedom |
| `gev` | shape k, scale s, location m; same shape sign as SciPy genextreme |
| `gpd` | shape k, scale s, location m; **opposite** shape sign to SciPy genpareto |
| `gumb` | scale a, location b, optional truncation flag at zero |
| `invg` | mean m, shape lambda |

SciPy may differ from legacy routines in tolerance handling near zero shape,
invalid parameters, extreme tails, and floating-point endpoint behavior. The
generalized-gamma evaluator follows SciPy's supported shape domain, which is
broader than the positive-exponent domain used in the legacy file. Near-zero
GEV/GPD shapes use SciPy's continuous evaluation rather than MATLAB's explicit
`1e-4` cutoff.

## Specific limitations

* `dat2tp`: retains the source endpoint rule; fewer than two internal extrema
  yields no turning points. Uses C-MEX crossing plateau semantics, which can
  differ from the old `.m` fallback at exact-level plateaus.
* `findrfc`: intentionally ports the C routine's terminal zero padding and sorted
  output, including repeated indices. It is not interchangeable with general
  hysteresis filtering or weighted half-cycle counting.
* `res2arfc('CS')`: re-extracts extrema in a doubled residual. Its handling of
  flat endpoints needs broader comparison with the original `rfcfilter`.
* `dat2crossind`: supports all/u/d/none/du and complete dw/uw/tw/cw wave
  definitions, with zero-based preceding crossing indices.
* `detrendma`: retains the source running-mean update, including its initial
  update convention. All columns are treated as data. Optional returned trend
  is expanded to input shape.
* `dat2spec`: uses Welch estimation. `dat2spec2` separately supplies a
  covariance-window estimate; its experimental high-frequency chopping branch
  is explicitly unsupported.
* `dat2cov`, `dat2cor`: no plotting/error bands; missing values use finite-pair
  normalization for the unbiased estimator.
* `spec2mom`, `spec2bw`, `spec2char`: supported one-dimensional spectrum
  representations are documented individually. `spec2char` does not return
  characteristic covariance estimates. `spec2cov` now routes directional
  spectra to the mixed spatial/temporal quadrature engine; its Python Nt
  sample-count convention differs from direct `dspec2dcov`'s Nt+1 convention.
* `jonswap`: default `A=-1` normalizes by numerical integration, matching the
  supplied source. Explicit `A=0` chooses the legacy approximate formula.
* `spec2sdat`: interpolates onto an FFT grid; not legacy random-stream parity.
  Uses an explicit `np_` sample-count parameter to avoid shadowing NumPy.
* `duffsim`: Euler integration with 100 substeps per output sample. Python uses
  an explicit default `b=1`; MATLAB changes its default according to whether
  `a` was supplied. Set `b=-1` explicitly for a double-well oscillator. Output
  sampling follows the Fortran driver's indexing and can extend one interval
  beyond T, as the original wrapper does.
* `mvnormprb`: returns probability by default; `full_output=True` uses the
  replicated Sobol backend and returns `(value,error,inform)`. The direct
  `mvnormprb_full` API also provides this tuple. Estimates are not equivalent
  to legacy deterministic bounds. RIND weighted expectations are now available
  separately as `rind`, with their supported Jacobian and options documented in
  [native Python ports](NATIVE_PYTHON_PORTS.md).
* `tranproc`: PCHIP scalar interpolation and chain-rule derivatives through
  order four, with explicit linear extrapolation. Matrix columns represent
  successive process derivatives, not independent signals.
* `cc2cmat`: histogram method only. Kernel-smoothed cycle matrices are pending.
* `cmat2dam`, `cmat2amp`: use the upper-triangular cycle-matrix convention in the
  MATLAB source. Lower-triangular counts are not added to damage.
* `genchol`: pivoted Cholesky with `(L, permutation, rank)`; a different pivot
  sequence is possible from the legacy implementation.
* `gaussq`: SciPy adaptive quadrature; does not implement the legacy vectorized
  weighted-rule selection API.
* `comnsize`: NumPy broadcasting with exceptions instead of a MATLAB error-code
  output; NumPy broadcasting accepts more shapes than the legacy helper.
* `kde`: the original compact Python convenience API retains SciPy Gaussian
  estimation and univariate explicit bandwidths. The new `kdefun` supports
  normalized radial/product kernels, full bandwidth matrices, power/log
  transforms, and sample-adaptive factors. `kdebin` supplies the PDF grid via
  exact evaluation, replacing the old FFT acceleration. Kernel and bandwidth
  helpers cover their documented subsets; unimplemented selectors raise.
* `mc2stat`: rejects nonunique stationary laws and zero-row transitions rather
  than silently deleting states.

Version 0.2.0 added native-code capability replacements, including conditional
weighted normal expectations, noncentral-t integration, finite-grid wave
drivers, and source-missing cycle reconstruction. See
[native Python ports](NATIVE_PYTHON_PORTS.md) for their detailed limitations.

Version 0.3.0 adds many source-based statistical, fatigue, spectral,
simulation, transformation, directional-estimation and empirical-wave APIs.
The per-source conversion manifest and function docstrings identify supported
branches. Public sensor IDs and design-of-experiment factor labels retain
their mathematical one-based labels; array/state indices remain zero based.
Source correctness repairs (including Hermite inversion, conditional density
Jacobians, column fits and small-mass normalization) are recorded in
[REPAIR_LOG.md](REPAIR_LOG.md). Numeric model tables are bundled without pickle
payloads, with original-file hashes in `wafo/wavemodels/data/PROVENANCE.json`.

Version 0.4.0 adds nonblocking graphics, pyproj projections and geodesy, map
data readers, KDE selectors/sampling/explorers, course demos, additional fatigue
and Gaussian algorithms, and directional/encounter spectral conversions.
Encounter folds use conservative bins; moving finite-depth conversion is
explicitly unsupported. Missing external map databases are not synthesized.
Map point/patch clipping does not reproduce every original segment-intersection
rule. `reconstruct` preserves observations and reports exact/local Gaussian
conditioning, replacing the source's whole-record detrending and hidden
positional diagnostics. Local windows do not preserve cross-window draw
dependence. PDF structures use original meshgrid/xy storage unless
`indexing='ij'` is specified; Python Gaussian drivers identify their ij output.

Installation path helpers inspect the installed namespaces without modifying
sys.path. Offline help includes the source notes and conversion status, so
original MATLAB options are not presented as supported Python arguments.

`implemented` in the manifest means an implementation exists; it is not a
certificate of MATLAB numerical equivalence. `adapted` highlights known
restrictions or API changes; it does not imply every other routine is bitwise
identical. The tests are a baseline, and MATLAB/Octave parity testing is pending.
