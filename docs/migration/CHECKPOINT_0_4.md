# Checkpoint 0.4.0 (3 November 2026)

Environment: CPython 3.11.9 on Windows, NumPy 2.4.2, SciPy 1.15.2,
Matplotlib 3.10.1, pyproj 3.7.1, pytest 9.2.0 (in `.test-deps-py311`).

* Source-tree suite (`python tools/validate_functions.py`): **775 passed**,
  0 failed. Public API aliases with passing-test evidence: **731**; failing: 0;
  untested: 0. Unimplemented MATLAB source functions: **0**.
* Built `dist/py3wafo-0.4.0-py3-none-any.whl`, installed it with `--no-deps`
  into `.wheel-check/v040`, and ran the full suite with source paths disabled
  (`-o pythonpath=`): **775 passed** (`wheel_test_results.xml`). Both `wafo`
  and `py3wafo` imported from the installed wheel; both example scripts ran
  against it and reproduced the 0.1 checkpoint numbers.
* Starting point this session: 751 passed, 3 failed (SciPy 1.15 lacks the
  `rng=` keyword of `multivariate_normal.cdf`). See REPAIR_LOG items 11-15.

New in 0.4.0 at this checkpoint: the remaining trgauss second-order-sea
functions (`dirsp2chitwo`, `ewwdir`, `chitwo2lc_sp`, `specq2lc`,
`rqlf_asympt`, `writecov`, `mctp2tc`, `chitwo2lc_sorm(ass>0)`), direct tests
for the M_Map private projection engines, and the `wafo.papers` demos
(`wafodemo`, `recdemo`, `papermenu`, `wafofig1`-`10`, `recfig1`-`13`).

Documented deviations from the source, each covered by a test (REPAIR_LOG
items 11-20):

* `dirsp2chitwo` halves the source quadratic coefficients so the
  narrow-band skewness equals the Stokes value 3*k*sigma
  (`source_scaling=True` keeps the source).
* `mctp2tc` uses the WAFO 1 counting convention it was written for and
  locates utc by its natural grid index (off-grid levels are supported).
* The Lugannani-Rice cdf uses its small-w expansion next to s=0.
* Paper sessions replace MATLAB globals; figure 7 computes Lc/Ac through the
  temporal driver on the wavenumber spectrum.

Still open, so the migration is **not certified complete**:

* Paper figure 6 lacks its conditional theory curve (supply
  `session.theory[6]`); figure 8 does not exist in the MATLAB source.
* 15 executable source scripts still need review/porting
  (UNREVIEWED_SCRIPTS.md): the tutorial chapters, itmkurs labs, `democc`,
  module test scripts, `startup.m` and `polplot.m`.
* No MATLAB/Octave parity testing was possible in this environment.
