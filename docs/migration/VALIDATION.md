# Validation

**Current results are generated in [FUNCTION_VALIDATION.md](FUNCTION_VALIDATION.md).**
That report supersedes the historical 0.1/0.2 checkpoint counts below. It links
the complete per-API passed/failed/untested lists, the remaining source-function
list, actual test-to-function evidence, JUnit output, and repair history.

For version 0.4, run `python tools/validate_functions.py` to reproduce the
report. Run it with `--require-complete` to enforce the source/test completeness
gate. The distribution is named `py3wafo` and also preserves `wafo` imports.
The [0.3 checkpoint](CHECKPOINT_0_3.md) records 415 passing tests both from the
source tree and from an independently installed wheel, plus the remaining
276 source functions and 42 then-unreviewed source records. It is historical;
the current audit distinguishes executable scripts from comment-only help and
pager wrappers. Reviewed documentation is bundled in the offline help and
listed in [REVIEWED_SOURCE_DOCUMENTATION.md](REVIEWED_SOURCE_DOCUMENTATION.md).

The latest checkpoint is [CHECKPOINT_0_4.md](CHECKPOINT_0_4.md) (775 tests
passing from source and from the installed 0.4.0 wheel). The historical notes
below refer to earlier checkpoints.

Execution environment (0.1-0.3): bundled CPython 3.12 on Windows, with NumPy 2.5.4,
SciPy 1.18.1, and pytest 9.2.0 installed into this folder's `.test-deps`.

* `python -m pytest -q`: **134 passed** at the 0.2.0 native-port checkpoint.
* `python tools/audit_toolbox.py`: inventories **762 MATLAB source files**,
  **1,271 non-MATLAB files**, including **121 native source/binary files**.
* Coverage at this checkpoint: **113 implemented**, **62 adapted**,
  **508 pending functions**, **79 documentation/script files**. A script record
  does not mean that script's computations have been converted.
* At the earlier 0.1.0 checkpoint, `python -m pip wheel . --no-deps
  --no-build-isolation --wheel-dir dist` successfully built the wheel.
  Installed it into a separate `.wheel-check` directory and ran
  `examples/sea_fatigue.py` against the installed package: **passed**. The
  original sea dataset produced 9,524 samples, 2,172 turning points, and
  1,086 rainflow cycles. Damage for beta 3,4,5 was approximately
  `[202.50321456, 206.59473846, 233.36502725]`. JONSWAP m0 was 1 for Hm0=4.
* Built `dist/wafo_py3-0.2.0-py3-none-any.whl`, installed it independently into
  `.wheel-check/v020`, and ran `examples/native_capabilities.py` against the
  installed package: **passed**. Its native registry contained all 121 entries;
  weighted normal, noncentral-t, direct Duffing, wave-density, and reconstructed
  damage examples all ran without a WAFO native executable.

Tests use source-derived examples, mathematical identities, distribution
round trips, and the original `sea.dat` measurements. They cover core numerical
behavior and basic packaging; they do not establish complete MATLAB parity.
The original C MEX/Fortran binaries were not executed or used as test oracles.

The 0.2.0 tests additionally check Park-Miller checkpoints, source Duffing
Euler evolution, finite-depth wave-interaction identities, analytic Rice
intensities, Gaussian positive-part moments, conditional affine barriers,
rank-deficient covariance, honest exhausted-budget reporting, noncentral t
identities and shared scaling, linear probability constraints, all native-file
registry mappings, vector cubature, missing-source cycle reconstruction,
NPZ numerical drivers, and publication converter argument construction.

Native replacement statuses report capability coverage, not binary validation.
Randomized integration error estimates do not bound path-grid/time-tail errors
in the wave drivers. The publication converters were tested with mocked
external executables; actual EPS/LaTeX/PDF rendering requires fig2dev/Ghostscript.

MATLAB and Octave were not available in the execution environment. The supplied
`tests/test_matlab_fallbacks.m` is therefore **not run**. Pure MATLAB fallbacks
need execution and comparison with the original kernels before being treated
as validated replacements. Singular/resonant second-order wave interactions,
extreme statistical tails, all rainflow residual variants, and complete
cross-platform behavior need broader validation.
