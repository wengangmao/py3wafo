"""Compatibility API for WAFO's legacy CROSSPACK covariance solver.

The original ``cov2mod`` implementation is a large Fortran path-integral
solver.  Its configuration entry point is retained here.  The scientific
solver deliberately raises a clear exception rather than returning an
unvalidated approximation.
"""

from __future__ import annotations


_OPTIONS = {}
IS_PYTHON_FALLBACK = True
HAS_CROSSPACK_SOLVER = False


def initinteg(eps=0.01, epss=0.00005, eps0=0.00005, c=4.5, iac=1, isq=0):
    """Store CROSSPACK integration settings using the historical API."""
    _OPTIONS.update(eps=eps, epss=epss, eps0=eps0, c=c, iac=iac, isq=isq)


def cov2mmpdfreg(t, cov, ulev, vlev, tg, xg, nit=2):
    """Raise an explicit error for the unavailable exact CROSSPACK solver."""
    del t, cov, ulev, vlev, tg, xg, nit
    raise NotImplementedError(
        "cov2mmpdfreg requires WAFO's legacy CROSSPACK Fortran solver. "
        "The Python compatibility module prevents import failures but does "
        "not fabricate an approximate maximum/minimum density."
    )
