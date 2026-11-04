# Test and repair iterations

The reproducible command is `python tools/validate_functions.py`. It runs the
full suite and records per-API test evidence and remaining implementation gaps.
Subsequent runs are preserved in `validation_history.json` with actual failure
tracebacks, rather than erasing earlier failures when a rerun passes.

1. Existing package baseline: 134 tests passed. The function audit found 50
   exposed APIs with no successful test-call evidence and 508 unimplemented
   MATLAB functions. Missing functions were not reclassified as passing.
2. Added source-moment, parameter-recovery, numerical-utility, cycle-helper,
   KDE, spectral-estimation and multivariate tests. 167 tests passed; all
   212 then-exposed public function aliases had execution evidence.
3. Added normalized compact/noncompact kernels, radial/product distinctions,
   derivatives, linear binning, density quantile levels and bandwidth APIs.
   Initial run: 186 passed, one failed. The failure was in the new scalar
   derivative test: it indexed a scalar result with `[0]`. The function was
   returning the correct scalar. Corrected the test's shape assumption; the
   analytic expected values remain unchanged.

Passing means the listed tested scenarios passed. It does not certify every
parameter branch or full numerical agreement with the original toolbox.

4. Expanded the missing numerical modules and found a directional-spreading
   normalization failure for a subnormal spectral tail column (relative mass
   error about 7.4e-7). Rewrote `multidim.dspec2spreading` to scale columns by
   their maxima before integration/division. This preserves relative precision
   in near-zero frequency tails. Also removed an eagerly evaluated invalid
   `inf*0` expression from the stable hyperbolic-ratio helper.
5. The wave midpoint test initially assumed every turning-point midpoint was
   zero. Endpoints can represent partial waves, so that expectation was wrong.
   Replaced it with the exact midpoint of each actual pair of turning values;
   the test still checks absolute sample indices and crossing bracketing.
6. Source-based additions also repaired the Hermite inverse branch selection,
   Ochi transformation normalization, matrix-column fitting semantics,
   truncated-Gumbel boundary handling, and switching-Markov counting-matrix
   differentiation. Their dedicated regression tests are in the corresponding
   `test_*_complete.py` files.
7. A transformed adaptive-KDE normalization test omitted probability beyond
   its fixed integration interval at alpha=1. Replaced that assumption with
   an independent exact Gaussian-mixture CDF mass on the finite interval,
   and added a separate adaptive-width-domain integral-to-one check. The
   estimator is not renormalized to conceal truncated tails.
8. Source-based 0.4 additions repair covariance-window small-sample indexing,
   independent-column transformation, bivariate plug-in derivative signs and
   bandwidth powers, GPD/turning-point edge cases, and native packed-correlation
   row ordering. Their numerical tests compare explicit objectives, analytic
   Gaussian derivatives, dimensional scaling and probability integrals.
9. Closed fatigue figures immediately after rendering, eliminating the open
   figure warning. Removed test-local source-path insertion so the same suite
   can genuinely validate an independently installed wheel. Offline help now
   distinguishes executable workflows from comment-only and pager documents.
10. A new directional sea-field test exposed an unsuitable quadrature choice:
    Simpson weights need not be nonnegative on irregular grids, while spectral
    random-mode variances must be. Changed direct synthesis to nonnegative
    trapezoid weights and added an irregular-grid regression. Analytic
    single-mode covariance, traveling phase, ensemble variance and transformed
    field checks pass. Fields are never rescaled to hide discretization loss.
11. (3 Nov 2026) The SciPy 1.15 environment exposed `multivariate_normal.cdf`
    calls using the newer `rng=` keyword (3 failing tests). `mvnormprb` now
    uses the frozen distribution, which accepts a seed on every supported
    SciPy release; the affected test oracle uses the same portable form.
    `tools/validate_functions.py` prefers an interpreter-specific
    `.test-deps-pyXY` folder, because compiled 3.12 wheels cannot load on 3.11.
12. Ported `dirsp2chitwo`, `ewwdir`, `chitwo2lc_sp`, `specq2lc` (with its
    private `eww`/`kwaves`), `rqlf_asympt`, `writecov` and `mctp2tc`, and
    enabled `chitwo2lc_sorm(ass>0)` through `rqlf_asympt`. `kchitwo` judged
    rounding in V=S22-S12'S12 against V's own scale; V is exactly zero for the
    random-phase second-order model, so it rejected valid parameters. The
    tolerance now scales with S22.
13. Validation against second-order Stokes theory (narrow-band skewness
    3*k*sigma) showed that the source `dirsp2chitwo` doubles all quadratic
    coefficients (it symmetrizes Q+Q' where `specq2lc` uses (Q+Q')/2):
    skewness 0.60 against 0.306. The default now halves them (skewness 0.299)
    and agrees with `specq2lc` to about 1 % in densities and crossing
    intensities; `source_scaling=True` reproduces the literal source.
14. `mctp2tc` was written for the WAFO 1 counting convention on column-flipped
    cycle matrices. With WAFO 2's `cmat2nt` the literal translation drops the
    highest crest level, puts mass at the reference level and yields negative
    frequencies (max error 0.085 against simulated chains). Using the legacy
    convention internally reproduces simulated trough-crest frequencies to
    sampling error (about 0.002) and the exact upcrossing total.
15. The source Lugannani-Rice cdf subtracts two large finite-difference terms
    next to s=0 and became non-monotone there. For |w|<0.1 the standard
    expansion Phi(w)+phi(w)*lambda3/6 is used, exact in the limit. Tests use
    exact exponential (scaled chi-square) laws, for which the saddlepoint
    density and one-term Rice intensity are exact. The source three-term
    intensity multiplies an already normalized density by Daniels' 1+R0
    (11/12 for that law); this source behaviour is kept and documented.
16. (3 Nov 2026, paper demos) `wafo.papers` had a session framework whose
    dispatcher imported a missing `_figures` module, and nothing was exported.
    Added `_figures.py` (all ten WAFO-paper and thirteen reconstruction-paper
    figures), the package exports and numbered `wafofigN`/`recfigN` wrappers.
    `recinit` converted `dat2tr`'s one-element test array with `float()`,
    which fails on current NumPy; it now unpacks the element.
17. `mctp2tc` (see 14) also counted levels from the top to locate utc while
    indexing them from the bottom: only the centre level of a symmetric grid
    worked (errors about 0.08 off-centre). The source's figure 10 grid
    (-6,6,20) puts utc=0 between levels, where troughs just below utc were
    lost. utc is now located by its natural index, and an off-grid utc gets
    an inserted zero-frequency level. Simulated chains agree to about 0.002
    for centre, off-centre and between-level cases.
18. `th2vhpdf` returned (height,velocity) meshgrid storage but kept the
    input's `indexing='ij'` tag, so `pdfplot` rejected its output. The stale
    tag is dropped.
19. `rind` divided its absolute tolerance by the conditioning density; a
    subnormal density made the tolerance infinite and aborted the whole
    integral. Such points now contribute zero. Its Schur complement
    `S12 cc^-1 S21` was not exactly symmetric, and the symmetry check (scaled
    to tiny conditional variances) rejected it; it is now formed as W'W from
    the Cholesky factor and symmetrized. Both surfaced in figure 7.
20. Figure 7 (Lc/Ac) uses the temporal TcAc driver on the wavenumber
    spectrum, which is exact for a spatial Gaussian record. Lengths are
    measured in units sqrt(m0/m2) for the driver's absolute tolerances and
    mapped back. Figure 6's conditional Tcf/Ac-given-Tc theory is still not
    available (supply `session.theory[6]`); figure 8 was never implemented in
    the MATLAB source and stays unavailable.
