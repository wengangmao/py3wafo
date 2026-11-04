# Extras

`matlab_fallbacks/` holds pure-MATLAB versions of six WAFO MEX routines
(`csort`, `disufq`, `duffsim`, `findcross`, `findrfc`, `simduff`), written
during the port so that the MATLAB toolbox can run without its compiled
binaries. They are not used by the Python package. They have not been run in
MATLAB or Octave, so test them before relying on them.
