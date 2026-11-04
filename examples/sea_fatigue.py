"""Run after installing wafo-py3: python examples/sea_fatigue.py."""
from wafo.data import load
from wafo.onedim import dat2tp
from wafo.cycles import tp2rfc
from wafo.damage import cc2dam
from wafo.spec import jonswap, spec2mom
from wafo.wsim import spec2sdat

sea = load('sea.dat')
turning_points = dat2tp(sea)
cycles = tp2rfc(turning_points)
print(f'{len(sea)} samples, {len(turning_points)} turning points, {len(cycles)} cycles')
print('Damage for beta=3,4,5:', cc2dam(cycles, [3,4,5]))
spectrum = jonswap(sdata=[4,8])
print('Spectral moments m0,m2,m4:', spec2mom(spectrum,4))
simulation = spec2sdat(spectrum,4096,seed=42)
print('Simulation shape:', simulation.shape)
