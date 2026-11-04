"""Portable numerical replacement for ``source/simduff.f``.

The Park--Miller integer recurrence and the source's Box--Muller/symmetric
stable formulas are preserved. Calculations use Python/NumPy double precision,
not the old default REAL arithmetic or compiler-dependent REAL/INTEGER seed
aliasing. This is an algorithm port, not a bitwise emulator of old executables.
"""
from __future__ import annotations

import math
import numbers
import numpy as np


class ParkMiller:
    """Minimal-standard generator with the explicit integer seed WAFO intended.

    No Box--Muller value is cached: each Gaussian or stable draw consumes two
    uniform draws, just as in ``simduff.f``. ``state`` can be saved to resume.
    """

    modulus = 2147483647
    multiplier = 16807

    def __init__(self, seed=123456789):
        if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral)
                or not 0 < int(seed) < self.modulus):
            raise ValueError("seed must be an integer in (0, 2147483647)")
        self.state = int(seed)

    def random(self):
        """Next value strictly inside (0,1)."""
        self.state = self.multiplier*self.state % self.modulus
        return self.state/self.modulus

    def normal(self):
        """Source Box--Muller cosine draw; standard normal in distribution."""
        u1, u2 = self.random(), self.random()
        return math.sqrt(-2*math.log(u1))*math.cos(2*math.pi*u2)

    def stable(self, alpha):
        """Source symmetric stable variate, characteristic exponent alpha.

        The source's alpha=2 stable distribution has variance 2. The oscillator
        driver divides stable draws by sqrt(2) and selects normal draws near 2.
        """
        if not np.isfinite(alpha) or not 0 < alpha <= 2:
            raise ValueError("alpha must lie in (0,2]")
        v = math.pi*(self.random()-.5)
        w = -math.log(self.random())/math.cos((1-alpha)*v)
        # Logarithmic evaluation keeps the original formula while postponing
        # overflow for heavy-tailed alpha and near-endpoint uniforms.
        try:
            value = math.sin(alpha*v)*math.exp(-math.log(w*math.cos(v))/alpha)*w
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError("Stable draw exceeded floating-point range") from exc
        if not math.isfinite(value):
            raise FloatingPointError("Stable draw exceeded floating-point range")
        return value


def simduff(number_of_points, dt, z, a=0, b=1, alf=2, seed=123456789,
            *, return_seed=False):
    """Direct replacement for the native Duffing driver, without file I/O.

    Parameters are the seven values formerly read from ``simduff.in``.
    ``dt`` is the **internal** Euler step, and ``number_of_points`` is the
    number of internal points. Returns ``[i*dt, position, velocity]`` at internal
    indices 100, 200, ... through number_of_points. The initial random state is
    at index 1; source time labels ``i*dt`` are retained despite that offset.

    The oscillator solves ``x'' + 2*z*x' + b*x + a*x**3 = noise``. Standard
    Gaussian draws are used for alf>1.99999; otherwise the source symmetric
    stable draws divided by sqrt(2) are used. Noise increments have scale
    ``sqrt(4*z)*dt**(1/alf)``. If a>.01, the initial position is divided by
    sqrt(a), matching the Fortran code.

    ``return_seed=True`` returns ``(output, final_integer_rng_state)``. The
    seed alone does not contain the final oscillator position/velocity and is
    therefore insufficient to continue a trajectory.
    """
    if (isinstance(number_of_points, (bool, np.bool_))
            or not isinstance(number_of_points, numbers.Integral)
            or number_of_points < 2):
        raise ValueError("number_of_points must be an integer at least 2")
    values = np.asarray([dt, z, a, b, alf], dtype=float)
    if not np.all(np.isfinite(values)) or dt <= 0 or z < 0 or not 0 < alf <= 2:
        raise ValueError("Require finite parameters, dt>0, z>=0, and 0<alf<=2")
    generator = ParkMiller(seed)

    def draw():
        return generator.normal() if alf > 1.99999 else generator.stable(alf)/math.sqrt(2)

    position, velocity = draw(), draw()
    if a > .01:
        position /= math.sqrt(a)
    noise_scale = math.sqrt(4*z)*dt**(1/alf)
    result = np.empty((int(number_of_points)//100, 3))
    out = 0
    for i in range(2, int(number_of_points)+1):
        increment = draw()
        try:
            new_position = position+velocity*dt
            new_velocity = velocity+(-b*position-a*position**3-2*z*velocity)*dt+noise_scale*increment
        except OverflowError as exc:
            raise FloatingPointError("Euler simulation diverged; reduce dt") from exc
        if not math.isfinite(new_position) or not math.isfinite(new_velocity):
            raise FloatingPointError("Euler simulation diverged; reduce dt")
        position, velocity = new_position, new_velocity
        if i % 100 == 0:
            result[out] = (i*dt, position, velocity)
            out += 1
    return (result, generator.state) if return_seed else result
