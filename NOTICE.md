# Notices and attribution

## py3wafo

py3wafo is an independent Python 3 port of the MATLAB toolbox WAFO 2.1.1
(Wave Analysis for Fatigue and Oceanography, September 2005). It is released
under the GNU General Public License, version 3 or (at your option) any later
version; see [LICENSE](LICENSE). It is not an official release of the WAFO
group or of Lund University.

## The original WAFO toolbox

WAFO was developed at the Centre for Mathematical Sciences, Mathematical
Statistics, Lund University, Sweden, by the WAFO group, including
P. A. Brodtkorb, P. Johannesson, G. Lindgren, I. Rychlik, J. Rydén and E. Sjö,
with contributions from many others named in the individual source files.
Later WAFO releases and the original Python WAFO project (pywafo) are
distributed under the GNU GPL.

The algorithms, function names, documentation text in `wafo.docs` and default
parameters follow the MATLAB toolbox. Where a MATLAB routine called compiled
C or Fortran code (MEX files and executables), py3wafo contains a new
implementation of the documented capability, written with NumPy and SciPy. No
original binaries are included.

Some WAFO source files carry their own notices, which are kept in the
corresponding documentation records. For example, `wstats/identify.m` is
marked "GPL Copyright (c) Anders Holtsberg, 1998".

## M_Map

`wafo.m_map` ports the mapping capabilities that WAFO bundled from **M_Map**,
a mapping package for MATLAB by Rich Pawlowicz (University of British
Columbia). Its bundled coastline data derive from the M_Map distribution.
These parts remain subject to the terms of the original M_Map distribution.

## Data sets

The measurement data in `src/wafo/data` (`sea.dat`, `gfaks89.dat`,
`gfaksr89.dat`, `yura87.dat`, `atlantic.dat`, `japansea.dat`, `northsea.dat`,
`sfa89.dat`, `sn.dat`) and the parameter tables in
`src/wafo/wavemodels/data` are redistributed unchanged in content from the WAFO
2.1.1 toolbox, converted to NumPy formats where noted. Their provenance files
record the original source files and SHA-256 checksums. Please credit the
original data providers named in the WAFO documentation when you publish
results based on them.

## Citation

If you use py3wafo in research, please cite the original WAFO paper:

Brodtkorb, P.A., Johannesson, P., Lindgren, G., Rychlik, I., Rydén, J. and
Sjö, E. (2000). WAFO — a Matlab toolbox for analysis of random waves and loads.
*Proceedings of the 10th International Offshore and Polar Engineering
Conference (ISOPE)*, Seattle, USA, Vol. III, pp. 343–350.
