
|wafo_logo|
==========================================
Wave Analysis for Fatigue and Oceanography
==========================================

Description
===========

``py3wafo`` is the Python 3 distribution of WAFO, a toolbox for statistical
analysis and simulation of random waves and random loads.  The distribution is
installed as ``py3wafo`` and retains the historical ``wafo`` import namespace.
WAFO is distributed under the GNU General Public License v3.

Fatigue Analysis
----------------

- Fatigue life prediction for random loads
- Theoretical density of rainflow cycles

Sea modelling
-------------

- Simulation of linear and non-linear Gaussian waves
- Estimation of seamodels (spectrums)
- Joint wave height, wave steepness, wave period distributions

Statistics
------------

- Extreme value analysis
- Kernel density estimation
- Hidden markov models

Classes
-------
A short description of the main classes found in WAFO:


* TimeSeries:
    Data analysis of time series. Example: extraction of turning points,
    estimation of spectrum and covariance function. Estimation transformation
    used in transformed Gaussian model.

* CovData:
    Computation of spectral functions, linear and non-linear time series
    simulation.

* SpecData:
    Computation of spectral moments and covariance functions, linear and
    non-linear time series simulation. Ex: common spectra implemented,
    directional spectra, bandwidth measures, exact distributions for wave
    characteristics.

* CyclePairs:
    Cycle counting, discretization, and crossings, calculation of damage.
    Simulation of discrete Markov chains, switching Markov chains,
    harmonic oscillator. Ex:  Rainflow cycles and matrix, discretization of
    loads. Damage of a rainflow count or matrix, damage matrix, S-N plot.


Subpackages
-----------
A short descriptions the subpackages of WAFO:

* TRANSFORM
    Modelling with linear or transformed Gaussian waves.
* STATS
    Statistical tools and extreme-value distributions. Ex: generation of random
    numbers, estimation of parameters, evaluation of pdf and cdf
* KDETOOLS
    Kernel-density estimation.
* MISC
    Miscellaneous routines.
* DOCS
    Documentation of toolbox, definitions. An overview is given in the routine
    wafomenu.
* DATA
    Measurements from marine applications.

WAFO homepage: <http://www.maths.lth.se/matstat/wafo/>
On the WAFO home page you will find:
- The WAFO Tutorial
- List of publications related to WAFO.

Installation
============

Install the released package from PyPI::

    python -m pip install py3wafo

Then verify the import::

    python -c "import wafo; print(wafo.__version__)"

For development from a checkout::

    python -m venv .venv
    .venv/Scripts/python -m pip install -e ".[test]"
    .venv/Scripts/python -m pytest src/wafo

On Linux and macOS use ``.venv/bin/python`` instead.  See ``PUBLISHING.rst``
for reproducible build and repository upload instructions.

Scientific backends
-------------------

The package includes SciPy and Numba replacements for several historical
C/Fortran extensions, including a pure-Python ``rindmod`` backend.  The legacy
CROSSPACK ``cov2mmpdfreg`` path-integral solver is exposed through a compatibility
module but deliberately raises ``NotImplementedError``; no unvalidated density
approximation is returned.

Getting started
===============

A quick introduction to some of the many features of wafo can be found in the Tutorial IPython notebooks in the
`tutorial scripts folder`_:

* Chapter 1 - `Some applications of WAFO`_

* Chapter 2 - `Modelling random loads and stochastic waves`_

* Chapter 3 - `Demonstrates distributions of wave characteristics`_

* Chapter 4 - `Fatigue load analysis and rain-flow cycles`_

* Chapter 5 - `Extreme value analysis`_

.. _tutorial scripts folder: https://nbviewer.org/github/wengangmao/py3wafo/tree/main/src/wafo/doc/tutorial_scripts/

.. _Some applications of WAFO: https://nbviewer.org/github/wengangmao/py3wafo/blob/main/src/wafo/doc/tutorial_scripts/WAFO%20Chapter%201.ipynb

.. _Modelling random loads and stochastic waves: https://nbviewer.org/github/wengangmao/py3wafo/blob/main/src/wafo/doc/tutorial_scripts/WAFO%20Chapter%202.ipynb

.. _Demonstrates distributions of wave characteristics: https://nbviewer.org/github/wengangmao/py3wafo/blob/main/src/wafo/doc/tutorial_scripts/WAFO%20Chapter%203.ipynb

.. _Fatigue load analysis and rain-flow cycles: https://nbviewer.org/github/wengangmao/py3wafo/blob/main/src/wafo/doc/tutorial_scripts/WAFO%20Chapter%204.ipynb

.. _Extreme value analysis: https://nbviewer.org/github/wengangmao/py3wafo/blob/main/src/wafo/doc/tutorial_scripts/WAFO%20Chapter%205.ipynb


Unit tests
==========

Run the maintained test suite with::

    python -m pytest src/wafo


.. |wafo_logo| image:: https://github.com/wengangmao/py3wafo/blob/main/src/wafo/data/wafoLogoNewWithoutBorder.png
    :target: https://github.com/wengangmao/py3wafo


.. |docs_img| image:: https://readthedocs.org/projects/pip/badge/?version=latest
    :target: http://pywafo.readthedocs.org/en/latest/

.. |health_img| image:: https://codeclimate.com/github/wengangmao/py3wafo/badges/gpa.svg
   :target: https://codeclimate.com/github/wengangmao/py3wafo
   :alt: Code Climate

.. |coverage_img| image:: https://coveralls.io/repos/wengangmao/py3wafo/badge.svg?branch=main
   :target: https://coveralls.io/github/wengangmao/py3wafo?branch=main

.. |versions_img| image:: https://img.shields.io/pypi/pyversions/py3wafo.svg
   :target: https://github.com/wengangmao/py3wafo


.. |downloads_img| image:: https://img.shields.io/pypi/dm/py3wafo.svg
   :alt: PyPI - Downloads

