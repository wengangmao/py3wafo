"""Source-inspired Monte Carlo separation of second-order wave spectra."""
import warnings
import numpy as np


def spec2linspec(S, np_=None, dt=None, iseed=None, fnLimit=np.sqrt(2), *,
                 cases=20, seed=None, max_iter=30, tolerance=5e-5,
                 damping=.5, full_output=False):
    """Separate target S into nonnegative linear and nonlinear components.

    Iteratively simulate Gaussian first- and second-order waves with DISUFQ,
    estimate their ensemble one-sided FFT power spectra, and update the linear
    input by the source's spectral transfer-ratio principle. Gaussian linear
    and centered quadratic terms are uncorrelated: the denominator uses their
    separate estimated powers, avoiding noisy finite-sample cross terms and
    negative inferred nonlinear spectra. Damping stabilizes the fixed point.
    The same innovations are used across iterations to reduce Monte Carlo
    convergence noise. This replaces the source covariance-window estimator
    and graphical pauses; no plot is created.

    S is a one-dimensional angular-frequency spectrum without a marginal
    transformation. np_ is the sample count or [count,cases]; dt defaults to the spectrum's
    Nyquist interval. fnLimit is the finite positive convergence cutoff used
    by spec2nlsdat. Seeds reproduce this Python algorithm. Default returns
    (linear,nonlinear) spectral mappings with S=linear+nonlinear exactly.
    full_output adds iteration changes, convergence flag and estimated powers.
    A convergence warning is issued on exhausted iterations. Results remain
    statistical estimates and depend on record length and ensemble size.
    """
    from ..spec import Spectrum, _validate, spec2dt
    from ._simulation import _sizes, spec2nlsdat
    w, target = _validate(S)
    if np.size(S.get('tr', [])):
        raise ValueError('Second-order spectral separation cannot also apply a marginal transformation')
    n, cases = _sizes(np_, 2*(len(w)-1), cases)
    n += n % 2
    dt = spec2dt(S) if dt is None else dt
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(fnLimit) or fnLimit <= 0:
        raise ValueError('dt and fnLimit must be positive and finite')
    if isinstance(max_iter, (bool, np.bool_)) or int(max_iter) != max_iter or max_iter < 1:
        raise ValueError('max_iter must be a positive integer')
    if not np.isfinite(tolerance) or tolerance <= 0 or not np.isfinite(damping) or not 0 < damping <= 1:
        raise ValueError('tolerance must be positive and damping in (0,1]')
    if seed is not None and iseed is not None:
        raise ValueError('Supply seed or iseed, not both')
    stream = np.random.default_rng(seed if seed is not None else iseed)
    common_seed = int(stream.integers(0, np.iinfo(np.int64).max))
    omega = 2*np.pi*np.fft.rfftfreq(n, dt)
    linear = Spectrum({key: np.copy(value) if isinstance(value, np.ndarray) else value for key, value in S.items()})
    linear.S = np.asarray(target, dtype=float).copy()
    changes = []
    converged = not np.any(target)
    p1 = p2 = np.zeros_like(omega)

    def power(values):
        values = values-np.mean(values, axis=0, keepdims=True)
        result = np.mean(abs(np.fft.rfft(values, axis=0))**2, axis=1)*dt/(np.pi*n)
        result[[0, -1]] *= .5
        return result

    for _ in range(0 if converged else int(max_iter)):
        total, first, *_ = spec2nlsdat(linear, [n, cases], dt, seed=common_seed,
                                     truncationLimit=fnLimit, full_output=True)
        p1 = power(first[:, 1:])
        p2 = power(total[:, 1:]-first[:, 1:])
        total_power = p1+p2
        ratio = np.divide(p1, total_power, out=np.ones_like(p1), where=total_power > total_power.max()*1e-14)
        update = target*np.interp(w, omega, ratio, left=1, right=1)
        update = (1-damping)*linear.S+damping*update
        delta = float(np.max(abs(update-linear.S))/np.max(target))
        changes.append(delta)
        linear.S = update
        if delta <= tolerance:
            converged = True
            break
    if not converged:
        warnings.warn('Linear-spectrum separation exhausted max_iter before tolerance; inspect full_output diagnostics', RuntimeWarning, stacklevel=2)
    nonlinear = Spectrum(linear)
    nonlinear.S = target-linear.S
    linear.note = str(S.get('note', ''))+' linear component (spec2linspec)'
    nonlinear.note = str(S.get('note', ''))+' nonlinear component (spec2linspec)'
    diagnostics = dict(iterations=len(changes), converged=converged,
                       change_history=np.asarray(changes), omega=omega,
                       linear_power=p1, quadratic_power=p2,
                       samples=n, cases=cases, method='common-innovation Monte Carlo transfer-ratio fixed point')
    return (linear, nonlinear, diagnostics) if full_output else (linear, nonlinear)
