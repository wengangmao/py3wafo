# Changelog

All notable changes to py3wafo are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.4.0] — 2026-11-04

### Added
- Python counterparts for all remaining MATLAB WAFO 2.1.1 functions.
- Second-order (quadratic) sea: `dirsp2chitwo`, `ewwdir`, `chitwo2lc_sp`,
  `specq2lc`, `rqlf_asympt`, `writecov`, `mctp2tc`, and the higher-order
  Breitung expansion in `chitwo2lc_sorm(ass>0)`.
- `wafo.papers`: reproductions of the WAFO (ISOPE 2000) and reconstruction
  (ISOPE 1999) paper figures, with `wafodemo`, `recdemo` and `papermenu`.
- Geographic projections, geodesy and coastlines in `wafo.m_map`, with direct
  tests for the projection engines.
- Statistics, fatigue and Markov graphics; directional spectral
  transformations; time-series reconstruction; KDE bandwidth selection.
- Getting-started guide, installation guide, quick-start example, and a test
  that runs every documentation example.

### Fixed
- Python 3.10 support: removed starred unpacking inside indexing
  (`np.r_[1, *xs]`), which needs Python 3.11.
- The minimum Matplotlib version is now 3.8; contour handling fails on 3.7.
- Tests no longer use `np.trapezoid`, so they run on NumPy 1.26.
- `mvnormprb` works on SciPy 1.15, which lacks the `rng=` keyword.
- `kchitwo` accepts random-phase models whose conditional derivative
  covariance is exactly zero.
- `rind`: subnormal conditioning densities, and rounding asymmetry in the
  conditional covariance, no longer abort the integration.
- `th2vhpdf` reports its output array layout correctly.
- `dirsp2chitwo` second-order coefficients now match Stokes theory; the MATLAB
  scaling is available with `source_scaling=True`.
- `mctp2tc` uses the counting convention it was designed for and supports any
  reference level.
- Lugannani–Rice CDF evaluation is stable near the distribution mean.

See [docs/migration/REPAIR_LOG.md](docs/migration/REPAIR_LOG.md) for the
validation behind each fix.

## [0.3.0]

### Added
- Distribution renamed to `py3wafo`; both `import py3wafo` and `import wafo`
  expose the same implementation.
- Empirical wave-model coefficient tables as numeric-only files, with the
  original files' SHA-256 provenance.
- Validation tooling that records, per public function, the tests that
  exercise it. 415 tests passing; 225 MATLAB functions implemented and 182
  adapted at this checkpoint.

## [0.2.0]

### Added
- Python replacements for all 121 native C/Fortran files and executables:
  Gaussian weighted expectations (`rind`), normal and non-central t
  probabilities, finite-grid Rice wave-density drivers, the direct Duffing
  oscillator, and a reconstruction of the missing-source crossing-to-cycle
  program.

## [0.1.0]

### Added
- First core modules: spectra, simulation, turning points, rainflow counting
  and fatigue damage, with the original WAFO data sets and the `sea.dat`
  fatigue example.
