"""Wave Analysis for Fatigue and Oceanography (WAFO), Python 3 port.

Modules follow the MATLAB WAFO 2.1.1 toolbox. Every MATLAB source function
has a Python counterpart; ``coverage()`` and the validation reports describe
documented deviations and limits. MATLAB parity has not been certified.
"""
from . import cycles, damage, kdetools, markov, misc, onedim, spec, trgauss, wavemodels, wsim, wstats
from ._registry import coverage, source_info
from .root import initwafo, waforoot, wafopath, wafoexepath, pathdef, parzen

__version__ = "0.5.0"
