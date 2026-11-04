"""Random process simulation; generators accept NumPy random seeds."""
from ._sea import seasim, seamovie
from ._linear_spectrum import spec2linspec
import numpy as np
from ..markov import mc2stat, _transition
from ._second_order import disufq
from ._legacy_rng import ParkMiller, simduff
from ._cycle_helpers import fr2nt,fr2comb,comb2pro,getulcc,fr2res,rfc2load_fat


def mcsim(P, T, x0=None, *, seed=None):
    """Markov states are zero based in Python."""
    P = _transition(P)
    if int(T) != T or T < 1:
        raise ValueError("T must be a positive integer")
    rng = np.random.default_rng(seed)
    x = np.empty(int(T), dtype=int)
    x[0] = rng.choice(len(P), p=mc2stat(P)) if x0 is None else x0
    if not 0 <= x[0] < len(P):
        raise ValueError("Invalid initial state")
    for i in range(1, len(x)):
        x[i] = rng.choice(len(P), p=P[x[i-1]])
    return x


def spec2sdat(S, np_=None, dt=None, *, cases=1, seed=None):
    """Gaussian spectral synthesis; return time followed by simulated columns.

    Uses independent Gaussian Fourier coefficients and interpolation onto an FFT
    grid. This is a modern implementation, not legacy random-stream parity.
    """
    from ..spec import _validate, spec2dt
    w, s = _validate(S)
    n = int(np_ or 2*(len(w)-1))
    if n < 2 or cases < 1:
        raise ValueError("Need at least 2 samples and 1 case")
    dt = spec2dt(S) if dt is None else dt
    if dt <= 0:
        raise ValueError("dt must be positive")
    omega = 2*np.pi*np.fft.rfftfreq(n, dt)
    dw = 2*np.pi/(n*dt)
    density = np.interp(omega, w, s, left=0, right=0)
    rng = np.random.default_rng(seed)
    coef = (rng.normal(size=(len(omega), cases))+1j*rng.normal(size=(len(omega), cases)))*np.sqrt(density[:, None]*dw)/2*n
    coef[0] = 0
    if n % 2 == 0:
        coef[-1] = rng.normal(size=cases)*np.sqrt(density[-1]*dw)*n
    y = np.fft.irfft(coef, n=n, axis=0)
    return np.column_stack((np.arange(n)*dt, y))


def duffsim(T, dt, z, a=0, b=1, alf=2, *, seed=None):
    """Euler Duffing simulation, with 100 internal steps per output sample.

    The MATLAB wrapper used b=-1 when a was explicitly supplied. Python makes
    b explicit to avoid that context-dependent default. RNG streams differ
    from the original Fortran generator; the stable-noise scale is preserved.
    """
    from scipy.stats import levy_stable
    if T < 0 or dt <= 0 or not 0 < z < 1 or not 0 < alf <= 2:
        raise ValueError("Invalid oscillator parameters")
    n = (int(np.floor(T/dt))+1)*100
    if n > 5000000:
        raise ValueError("Too many internal steps")
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=n+2) if alf > 1.99999 else levy_stable.rvs(alf, 0, size=n+2, random_state=rng)/np.sqrt(2)
    x = noise[0]/(np.sqrt(a) if a > .01 else 1)
    velocity = noise[1]
    step = dt/100
    scale = np.sqrt(4*z)*step**(1/alf)
    result = []
    for i in range(2, n+1):
        newx = x+velocity*step
        velocity += (-b*x-a*x**3-2*z*velocity)*step+scale*noise[i]
        x = newx
        if not np.isfinite(x) or not np.isfinite(velocity):
            raise FloatingPointError("Euler scheme diverged; reduce dt")
        if i % 100 == 0:
            result.append((i*step, x, velocity))
    return np.asarray(result)


from ._simulation import (cov2sdat,cov2csdat,smcsim,smctpsim,mctpsim,sarmasim,
                          spec2nlsdat,rfm2dtp,lc2sdat)
