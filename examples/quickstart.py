"""py3wafo quick start: one short calculation from each main module.

Run with:  python examples/quickstart.py
Figures are written to the current directory when Matplotlib is installed.
"""
import numpy as np

import wafo
from wafo.data import load
from wafo.spec import jonswap, spec2mom, spec2char
from wafo.wsim import spec2sdat
from wafo.onedim import dat2tp, dat2spec
from wafo.cycles import tp2rfc
from wafo.damage import cc2dam
from wafo.trgauss import spec2thpdf
from wafo.kdetools import kdebin, kdeoptset

print('py3wafo', wafo.__version__)

# 1. Wave spectrum and sea-state parameters (angular frequency, rad/s).
S = jonswap(sdata=[4.0, 8.0])                 # Hm0 = 4 m, Tp = 8 s
m0, m2, m4 = spec2mom(S, 4)
Hm0, Tm02 = spec2char(S, ['Hm0', 'Tm02'])
print(f'JONSWAP: Hm0 = {Hm0:.2f} m, Tm02 = {Tm02:.2f} s, m0 = {m0:.3f} m^2')

# 2. Simulate a Gaussian sea surface from the spectrum.
x = spec2sdat(S, 4096, seed=1)                # columns: time, elevation
print('simulated record:', x.shape, f'std = {x[:, 1].std():.2f} m')

# 3. Measured data: turning points, rainflow cycles and fatigue damage.
sea = load('sea.dat')
tp = dat2tp(sea)
rfc = tp2rfc(tp)
damage = cc2dam(rfc, [3, 4, 5])
print(f'sea.dat: {len(sea)} samples, {len(rfc)} rainflow cycles, damage(beta=3,4,5) = {np.round(damage, 1)}')

# 4. Spectral estimate from the measured record.
Se = dat2spec(sea)
print(f'estimated Hm0 of sea.dat = {4*np.sqrt(spec2mom(Se, 0)[0]):.2f} m')

# 5. Theoretical wave-period density (Rice integral) for the JONSWAP sea.
f = spec2thpdf(S, 0, 'Tc', [0, 12, 25])
print(f'crest period density: peak at Tc = {f.x[0][np.argmax(f.f)]:.2f} s')

# 6. Kernel density estimate of the measured crest heights.
crests = tp[tp[:, 1] > 0, 1]
density = kdebin(crests, kdeoptset(kernel='epan'))
mode = density.x[0][np.argmax(density.f)]
print(f'crest-height KDE: mode at {mode:.2f} m on {np.size(density.f)} grid points')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from wafo.spec import wspecplot
    fig, ax = plt.subplots()
    wspecplot(S, ax=ax)
    fig.savefig('quickstart_spectrum.png', dpi=100)
    print('wrote quickstart_spectrum.png')
except ImportError:
    print('install py3wafo[plot] for figures')
