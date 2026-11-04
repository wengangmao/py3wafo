# Python replacements for non-MATLAB code (0.2.0)

Every one of the **121 native-source/binary files** in this checkout now has a
Python capability mapping. Duplicate sources, algorithm versions, MEX files,
and platform binaries are consolidated into the APIs below. This is a
capability-based port: a mapping does not mean the old executable or every
legacy algorithm branch has been reproduced. No original binary is loaded.

[The per-file table](NATIVE_DEPENDENCIES.md) and
[CSV inventory](non_matlab_files.csv) give the exact mapping, callers, status,
and limitations for each original file. `native_manifest.json` is included in
the wheel, so mappings remain available without the MATLAB checkout.

## Algorithms and entry points

| Original source / executable family | Integrated Python capability | Status |
|---|---|---|
| `findcross.c`, associated MEX/DLL | `wafo.onedim.findcross` | C crossing logic, including level plateaus; zero-based indices |
| `findrfc.c`, associated MEX/DLL | `wafo.onedim.findrfc` | C rainflow-filter logic and terminal-zero convention |
| `csort.c`, associated MEX/DLL | `wafo.misc.csort` | Same stable-sort capability using NumPy stable sorting |
| `disufq.c`, associated MEX/DLL | `wafo.wsim.disufq` | Finite/deep-water sum/difference interaction formulas |
| `simduff.f`, `simduff.exe` | `wafo.wsim.simduff`, `ParkMiller` | Euler driver and original Gaussian/stable RNG formulas, using double precision |
| `rind*.f`, `rindinterf.f`, `alanpab*.f`, `mexrind*`, `jacobmod.f` | `wafo.trgauss.rind`, `rindoptset` | Conditional truncated Gaussian expectation with absolute derivative product; new Sobol backend |
| `mexGenzMvnPrb`, `mexmvnprb*`, `mulnormex.f` | `wafo.trgauss.mvnormprb`, `mvnormprb_full` | Normal rectangle probability; modern SciPy/Sobol implementations |
| `mvnprodcorrprbmex*` | `wafo.trgauss.mvnormpcprb`, `mvnormpcprb_full` | Product-correlation normal probability, including common-factor quadrature |
| `mvnprdmex*` | `wafo.trgauss.mvnortpcprb` | Normal/noncentral Student-t probability with product correlations |
| `mexmvtnormprb`, `mexmvtprb`, `mvdist.f90` | `wafo.trgauss.mvtprb` | Noncentral-t/normal probability with linear constraints and shared chi-square scaling |
| `intmodule.f`, `mvbrvmod.f90` | `wafo.source.integrate_box`, plus RIND's internal integration | Vector cubature using independent scrambled Sobol replicas instead of Korobov rules |
| `sp2Acdf*.f`, `sp2acdf*.exe` | `wafo.trgauss.spec2Acdf` | Finite-horizon Gaussian crest/trough amplitude CDF |
| `sp2mM*.f`, `sp2mmtpdf.f`, matching executables | `wafo.trgauss.spec2mmtpdf` | Consecutive maximum/minimum density, optionally joint with elapsed period |
| `sp2tccpdf*.f`, matching executables | `wafo.trgauss.spec2tccpdf` | Upcrossing-wave period subdensity with amplitude thresholds |
| `sp2thpdf.f`, matching executable | `wafo.trgauss.spec2thpdf` | Selected period/amplitude Rice densities |
| `minmax.f`, `twog.f`, `minmax.exe` | `wafo.trgauss.wminmax` | Minimum/following-maximum density by Gaussian sign symmetry |
| `down2cc.exe` (source missing) | `wafo.damage.down2cc`, `lc2dplus` | Reconstructed nested-cycle slicing; original executable parity unverified |
| `ConvertFig.csh` | `wafo.docs.convert_fig` | Python orchestration of Xfig-to-EPS/LaTeX conversion; requires fig2dev |
| `ConvertToPDF.csh` | `wafo.docs.convert_ps` | Python orchestration of PS/EPS-to-PDF conversion; requires Ghostscript |

The numerical APIs are also exported together from `wafo.source`, preserving
the toolbox's source-directory role without one Python file per binary.

## Gaussian expectations

```python
import numpy as np
from wafo.trgauss import rind, mvtprb

# Variables [Xd1, Xd2], so Nt=0: E[X1+ * X2+].
rho = 0.3
value, error, elapsed, info = rind(
    [[1, rho], [rho, 1]], [0, 0],
    0, np.inf, [0, 2], Nt=0,
    options={"seed": 7, "maxpts": 65536}, full_output=True)
print(value, error, info["inform"], info["evaluations"])

# Numerator noncentrality, not a shifted central t distribution.
p, error, status = mvtprb(
    np.eye(2), [-1, -1], [2, 2], nu=5, delta=[0.5, 1],
    seed=7, full_output=True)
```

RIND orders variables `[Xt, Xd, Xc]`. `Nt` counts unweighted integrated
variables; all remaining integrated variables contribute `abs(product(Xd))`.
`xc` has shape `(Nc, Nx)`. The result includes the conditioning density
`f_Xc(xc)`. Barrier row zero contains constants; subsequent rows multiply
conditioning values. `indI` is a vector of cumulative counts starting at zero,
as in the original mathematical interface.

RIND supports positive semidefinite integrated covariance, with rank and
factorization residual reported. Singular conditioning covariance is rejected
because it has no ordinary joint density. Only the stated absolute-product
Jacobian is supported. Alternative native Jacobians and numeric method
selectors are not silently accepted. Python options use `method='sobol'`.

`error` is an estimated 99% interval from independently scrambled replicas,
not a rigorous error bound or proof of convergence. `inform=1` identifies
an unmet tolerance at the point budget. A zero sampled rare event under
singular constraints is not reported as certified zero. `maxpts` applies
per conditioning point. The covariance rank tolerance introduces an additional
approximation; inspect `covariance_residual` for nearly singular problems.

## Wave drivers and their practical limits

The drivers construct spectral derivative covariances and Gaussian Rice
expectations. They support **untransformed one-dimensional temporal Gaussian
spectra**. Supported definitions are:

* `spec2Acdf`: `Tc`, `Tt`.
* `spec2mmtpdf`: `Mm`, `MmTMm`.
* `spec2thpdf`: `Tc`, `Tt`, `TcAc`, `TtAt`, `TcfAc`, `TtbAt`.
* `spec2tccpdf`: `t<`, `t>`.

They return mappings with attribute access: `f`, `x` (list of coordinate
vectors), `err`, `labx`, `title`, and numerical-method notes. Spatial,
transformed, rainflow, level-separated and continuous upper/lower-bound
branches remain unsupported and raise explicit errors when selected.

```python
from wafo.spec import jonswap
from wafo.trgauss import spec2thpdf

S = jonswap(sdata=[4, 8])
density = spec2thpdf(
    S, definition="Tc", paramt=[0, 12, 13],
    options={"seed": 7, "maxpts": 8192})
print(density.x[0], density.f, density.err)
```

The path restrictions hold on discrete time nodes. Refining `paramt` and
extending its horizon is necessary to investigate continuous-path convergence.
With a positive first requested time, the internal path grid still starts at
zero and output periods are selected from that grid. Small default grids are
for an initial calculation, not publication-ready accuracy. Errors in `err`
cover RIND sampling only: they exclude time discretization, omitted time tails,
covariance rank truncation, and spectral quadrature. Densities/CDFs are not
silently clipped or renormalized to hide these errors.

## Missing-source reconstruction

`down2cc.exe` has no corresponding `down2cc.f` in the supplied source. The new
Python routine follows the wrapper's capability specification: take midpoint
horizontal slices of a piecewise-linear crossing-intensity curve and form one
cycle from each connected superlevel interval. It supports multiple lobes and
requires zero crossing intensity at both boundary levels.

`down2cc` returns `(cycles, delta)` with equal slice weight `delta`.
`lc2dplus` uses `abs(max-min)/2` and these midpoint weights. This deliberately
fixes the supplied MATLAB expression `cc(:,1)-cc(:,2)/2` and does not reproduce
its unexplained final weight of `0.75*delta`. Refine the slice count to assess
error. The result is a reconstructed nested-cycle damage estimate, **not a
certified finite-slice upper bound** or an emulation of the unknown executable.

## Script and executable use

No WAFO executable file protocol or temporary files such as `out.dat` are
needed. Use the Python APIs, or portable NPZ numerical drivers:

```powershell
python -m wafo.exec --help
python -m wafo.exec findcross input.npz output.npz
```

NPZ keys are function keyword parameter names. Scalars are converted to Python
scalars. `options_json` supplies a JSON options object. Spectrum fields are
`S__w`, `S__S`, and optionally `S__h`. Output mappings/tuples are flattened into
NPZ keys separated by `__`; no pickle payloads are accepted or written.

Resolve a source filename programmatically:

```python
from wafo.native import native_info, resolve_native

print(native_info("rind70.f"))  # Includes limitations and dependent WAFO functions.
rind_function = resolve_native("exec/pcwin64/rindd70.exe")
```

Documents, images, datasets, compiled help, format indexes, backups, and
version-control metadata are not algorithms to rewrite as Python scripts.
They remain inventoried assets. Publication conversion helpers use argument
lists without a shell and explicit output directories; original hard-coded
publishing/moving commands are removed. Tests check their command construction;
actual rendering requires the optional external format converters.
