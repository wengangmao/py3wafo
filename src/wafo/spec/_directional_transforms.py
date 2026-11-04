"""Directional spectral changes of variables and space/time covariances.

These are adaptations of ``spec/private/{dir2enc,dir2k1d,dspec2dcov,
spa2time,time2spa}.m``. Densities are one-sided in angular frequency, angles
are radians, and Cartesian density arrays have shape (len(k2), len(k)).
Encounter and projected-wave-number conversions use conservative finite-volume
ordinates: the trapezoidal integral on the returned grid preserves variance,
including at Doppler folds where a pointwise density is infinite.
"""
from copy import deepcopy
from itertools import product
import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson, trapezoid
from scipy.interpolate import RegularGridInterpolator


def _copy(S):
    from . import Spectrum
    return Spectrum(deepcopy(dict(S)))


def _axis(x, name, nonnegative=False):
    x = np.asarray(x, dtype=float)
    if (x.ndim != 1 or len(x) < 2 or np.any(~np.isfinite(x))
            or np.any(np.diff(x) <= 0) or (nonnegative and x[0] < 0)):
        raise ValueError(f'{name} must be a finite, strictly increasing vector')
    return x


def _density(y, shape):
    y = np.asarray(y, dtype=float)
    if y.shape != shape or np.any(~np.isfinite(y)) or np.any(y < 0):
        raise ValueError(f'S must be finite, nonnegative, and have shape {shape}')
    return y


def _parameters(S, g=None):
    grav = np.atleast_1d(np.asarray(S.get('g', 9.81) if g is None else g, dtype=float))
    h = float(S.get('h', np.inf))
    if (grav.ndim != 1 or len(grav) not in (1, 2) or np.any(~np.isfinite(grav))
            or np.any(grav <= 0) or np.isnan(h) or h <= 0):
        raise ValueError('Gravity and depth must be positive; gravity has one or two entries')
    if len(grav) == 2 and not np.isinf(h) and grav[0] != grav[1]:
        raise NotImplementedError('Anisotropic gravity is supported only in deep water')
    return h, float(grav[0]), float(grav[-1])


def _temporal(S):
    if bool('w' in S and np.size(S['w'])) == bool('f' in S and np.size(S['f'])):
        raise ValueError('Supply exactly one frequency coordinate w or f')
    if 'w' in S and np.size(S['w']):
        w = _axis(S['w'], 'w', True)
        density = np.asarray(S['S'], dtype=float)
    else:
        w = 2*np.pi*_axis(S['f'], 'f', True)
        density = np.asarray(S['S'], dtype=float)/(2*np.pi)
    directional = 'theta' in S and np.size(S['theta']) > 1
    if directional:
        theta = _axis(S['theta'], 'theta')
        density = _density(density, (len(theta), len(w)))
        if str(S.get('theta_units', 'radians')).lower().startswith('d'):
            theta = np.deg2rad(theta)
            density = density*180/np.pi
        if np.ptp(theta) > 2*np.pi+1e-10:
            raise ValueError('Directions must span at most one full period')
    else:
        theta = np.atleast_1d(np.asarray(S.get('theta', 0.), dtype=float))
        if theta.size != 1 or not np.all(np.isfinite(theta)):
            raise ValueError('Expected a scalar propagation direction')
        if str(S.get('theta_units', 'radians')).lower().startswith('d'):
            theta = np.deg2rad(theta)
        density = _density(density.ravel(), (len(w),))[None, :]
    phi = float(S.get('phi', 0.))
    if not np.isfinite(phi):
        raise ValueError('phi must be finite')
    if str(S.get('theta_units', 'radians')).lower().startswith('d'):
        phi = np.deg2rad(phi)
    return w, theta, density, directional, phi


def _output_grid(grid, upper, count, name):
    if grid is None or np.size(grid) == 0:
        return np.linspace(0., upper, count)
    if np.ndim(grid) == 0 or np.size(grid) == 1:
        step = float(np.asarray(grid).ravel()[0])
        if not np.isfinite(step) or step <= 0:
            raise ValueError(f'{name} spacing must be positive')
        return np.arange(count)*step
    return _axis(grid, name, True)


def _wave_number(w, h, g):
    from . import w2k
    return w2k(w, h=h, g=g)[0]


def _omega(k, h, g):
    from . import k2w
    return k2w(k, h=h, g=g)[0]


def _group_speed(k, w, h, g):
    if np.isinf(h):
        return np.divide(g, 2*w, out=np.zeros_like(w), where=w > 0)
    kh = k*h
    th = np.tanh(kh)
    return np.divide(g*(th+kh*(1-th*th)), 2*w,
                     out=np.full_like(w, np.sqrt(g*h)), where=w > 0)


def _clean_frequency(out):
    for key in ('w', 'f', 'k', 'k2', 'theta', 'theta_units'):
        out.pop(key, None)


def time2spa(S, k=None, k2=None, g=None, rate=2):
    """Convert frequency/directional density to 1D/Cartesian wave numbers.

    Uses ``dω/dk`` in 1D and ``(dω/dk)/k`` in 2D. Scalar coordinate
    arguments denote spacing, as in MATLAB. Directional grids are measured
    relative to the retained ``phi`` rotation. A two-component gravity vector
    reproduces the original deep-water normalized-coordinate transformation.
    Encounter spectra cannot be inverted with an intrinsic dispersion relation.
    """
    kind = str(S.get('type', 'freq')).lower()
    if kind not in ('freq', 'dir', 'rotdir'):
        raise ValueError('time2spa requires an intrinsic freq or dir spectrum')
    if not np.isfinite(rate):
        raise ValueError('rate must be finite')
    rate = max(1, int(np.floor(abs(rate)+.5)))
    w, theta, density, directional, phi = _temporal(S)
    h, gx, gy = _parameters(S, g)
    out = _copy(S)
    _clean_frequency(out)
    out.phi = phi
    out.g = gx if gx == gy else np.array([gx, gy])
    if not directional:
        if gx != gy:
            raise ValueError('A 1D spectrum requires scalar gravity')
        kk = _output_grid(k, float(_wave_number(w[-1], h, gx)),
                          rate*len(w) if k is None or np.size(k) == 0 else len(w), 'k')
        wk = _omega(kk, h, gx)
        sf = np.interp(wk, w, density[0], left=0., right=0.)
        if np.isinf(h) and kk[0] == 0 and sf[0] > 0:
            raise ValueError('A nonzero deep-water zero-frequency density has a singular k density')
        out.S = sf*_group_speed(kk, wk, h, gx)
        out.k = kk
        out.type = 'k1d'
        return out
    nw = max(rate-1, 1)*len(w)
    maxk = float(_wave_number(w[-1], h, gx))
    if k is None or np.size(k) == 0:
        kx = np.linspace(-maxk if np.any(np.cos(theta) < -1e-12) else 0., maxk,
                         2*nw-1 if np.any(np.cos(theta) < -1e-12) else nw)
    elif np.size(k) == 1:
        pos = _output_grid(k, maxk, nw, 'k')
        kx = np.r_[-pos[:0:-1], pos] if np.any(np.cos(theta) < -1e-12) else pos
    else:
        kx = _axis(k, 'k')
    if k2 is None or np.size(k2) == 0:
        ky = np.linspace(-float(_wave_number(w[-1], h, gy)),
                         float(_wave_number(w[-1], h, gy)), 2*nw-1)
    elif np.size(k2) == 1:
        pos = _output_grid(k2, maxk, nw, 'k2')
        ky = np.r_[-pos[:0:-1], pos]
    else:
        ky = _axis(k2, 'k2')
    KX, KY = np.meshgrid(kx, ky)
    if gx == gy:
        radius = np.hypot(KX, KY)
        # Symmetric linspace grids can miss zero by one floating-point ulp.
        # The polar Jacobian is singular there; treat that ordinate as zero,
        # not as a physically meaningful tiny-radius sample.
        radius = np.where(radius <= 16*np.finfo(float).eps*max(np.max(abs(kx)), np.max(abs(ky))), 0., radius)
        W = _omega(radius, h, gx)
        TH = np.arctan2(KY, KX)
        factor = np.divide(_group_speed(radius, W, h, gx), radius,
                           out=np.zeros_like(W), where=radius > 0)
    else:
        scaled_radius = np.hypot(gx*KX, gy*KY)
        scaled_radius = np.where(scaled_radius <= 16*np.finfo(float).eps*max(gx*np.max(abs(kx)), gy*np.max(abs(ky))), 0., scaled_radius)
        W = np.sqrt(scaled_radius)
        TH = np.arctan2(gy*KY, gx*KX)
        factor = np.divide(gx*gy, 2*W**3, out=np.zeros_like(W), where=W > 0)
    if np.isclose(np.ptp(theta), 2*np.pi):
        TH = (TH-theta[0]) % (2*np.pi)+theta[0]
    interp = RegularGridInterpolator((theta, w), density, bounds_error=False, fill_value=0.)
    out.S = interp(np.stack((TH, W), axis=-1))*factor
    out.k, out.k2 = kx, ky
    out.type = 'rotk2d' if kind == 'rotdir' else 'k2d'
    return out


def spa2time(S, w=None, theta=None, g=None):
    """Convert a 1D/Cartesian wave-number density to frequency/direction.

    The inverse polar Jacobian is ``k/(dω/dk)``. Finite-depth conversion
    with scalar gravity is supported; anisotropic gravity is deep-water only.
    Interpolation outside the supplied Cartesian rectangle contributes zero.
    """
    kind = str(S.get('type', '')).lower()
    if not kind.endswith(('k1d', 'k2d')):
        raise ValueError('spa2time requires a k1d or k2d spectrum')
    h, gx, gy = _parameters(S, g)
    directional = kind.endswith('k2d')
    kx = _axis(S['k'], 'k', not directional)
    out = _copy(S)
    _clean_frequency(out)
    out.g = gx if gx == gy else np.array([gx, gy])
    if not directional:
        if gx != gy:
            raise ValueError('A 1D spectrum requires scalar gravity')
        density = _density(S['S'], (len(kx),))
        w = _output_grid(w, float(_omega(kx[-1], h, gx)), len(kx), 'w')
        kk = _wave_number(w, h, gx)
        sk = np.interp(kk, kx, density, left=0., right=0.)
        cg = _group_speed(kk, w, h, gx)
        out.S = np.divide(sk, cg, out=np.zeros_like(w), where=cg > 0)
        out.w, out.type = w, 'freq'
        return out
    ky = _axis(S['k2'], 'k2')
    density = _density(S['S'], (len(ky), len(kx)))
    if gx == gy:
        wmax = float(_omega(np.hypot(np.max(abs(kx)), np.max(abs(ky))), h, gx))
    else:
        wmax = np.sqrt(np.hypot(gx*np.max(abs(kx)), gy*np.max(abs(ky))))
    w = _output_grid(w, wmax, len(kx), 'w')
    if theta is None or np.size(theta) == 0:
        theta = np.linspace(-np.pi, np.pi, 129) if kx[0] < 0 else np.linspace(-np.pi/2, np.pi/2, 65)
    theta = _axis(theta, 'theta')
    if np.ptp(theta) > 2*np.pi+1e-10:
        raise ValueError('theta must span at most one full period')
    W, TH = np.meshgrid(w, theta)
    if gx == gy:
        radius = _wave_number(W, h, gx)
        KX, KY = radius*np.cos(TH), radius*np.sin(TH)
        cg = _group_speed(radius, W, h, gx)
        factor = np.divide(radius, cg, out=np.zeros_like(W), where=cg > 0)
    else:
        KX, KY = W*W*np.cos(TH)/gx, W*W*np.sin(TH)/gy
        factor = 2*W**3/(gx*gy)
    interp = RegularGridInterpolator((ky, kx), density, bounds_error=False, fill_value=0.)
    out.S = interp(np.stack((KY, KX), axis=-1))*factor
    out.w, out.theta, out.theta_units = w, theta, 'radians'
    out.type = 'rotdir' if kind.startswith('rot') else 'dir'
    return out


def _cdf(w, s):
    """Exact antiderivative of the piecewise linear input spectrum."""
    integral = cumulative_trapezoid(s, w, initial=0.)
    slope = np.diff(s)/np.diff(w)
    def evaluate(x):
        x = np.clip(np.asarray(x), w[0], w[-1])
        index = np.clip(np.searchsorted(w, x, side='right')-1, 0, len(w)-2)
        d = x-w[index]
        return integral[index]+s[index]*d+slope[index]*d*d/2
    return evaluate


def _cells(x):
    edges = np.r_[x[0], (x[:-1]+x[1:])/2, x[-1]]
    return edges, np.diff(edges)


def dir2k1d(S, k=None):
    """Project directional waves onto the rotated x axis as one-sided |kx|.

    Opposite propagation directions are folded together. Integration uses the
    actual finite/deep-water dispersion relation and theta-phi rotation.
    Returned ordinates are conservative bin averages, including the near-zero
    projection singularity, rather than the original fragile split integral.
    """
    if str(S.get('type', '')).lower() not in ('dir', 'rotdir'):
        raise ValueError('dir2k1d requires a directional spectrum')
    w, theta, density, directional, phi = _temporal(S)
    if not directional:
        raise ValueError('At least two directions are required')
    h, gx, gy = _parameters(S)
    if gx != gy:
        raise NotImplementedError('Projected spectra require scalar gravity')
    kmax = float(_wave_number(w[-1], h, gx))
    kk = _output_grid(k, kmax, len(w), 'k')
    edges, widths = _cells(kk)
    rows = np.zeros((len(theta), len(kk)))
    for j, angle in enumerate(theta-phi):
        cosine = abs(np.cos(angle))
        cdf = _cdf(w, density[j])
        if cosine < 1e-14:
            if edges[0] == 0:
                rows[j, 0] = cdf(w[-1])/widths[0]
        else:
            inverse = _omega(edges/cosine, h, gx)
            rows[j] = np.diff(cdf(inverse))/widths
    out = _copy(S)
    _clean_frequency(out)
    out.k, out.S, out.type = kk, trapezoid(rows, x=theta, axis=0), 'rotk1d'
    out.phi = phi
    out.discretization = 'conservative trapezoidal bin averages'
    return out


def _encounter_row(w, density, a, edges):
    """Push a ray through |w+a*w**2|, splitting all stationary/zero folds."""
    splits = [w[0], w[-1]]
    if a != 0:
        splits += [x for x in (-1/(2*a), -1/a) if w[0] < x < w[-1]]
    splits = sorted(splits)
    cdf = _cdf(w, density)
    positive = np.zeros(len(edges)-1)
    negative = positive.copy()
    for lo, hi in zip(splits[:-1], splits[1:]):
        sign = 1. if (lo+hi)/2+a*((lo+hi)/2)**2 >= 0 else -1.
        qlo, qhi = abs(lo+a*lo*lo), abs(hi+a*hi*hi)
        q = np.clip(edges, min(qlo, qhi), max(qlo, qhi))
        y = sign*q
        if abs(a) < 1e-15:
            inverse = y
        else:
            root = np.sqrt(np.maximum(1+4*a*y, 0.))
            r1 = 2*y/(1+root)
            r2 = (-1-root)/(2*a)
            inverse = np.where((r1 >= lo-1e-10*(1+hi)) & (r1 <= hi+1e-10*(1+hi)), r1, r2)
        mass = abs(np.diff(cdf(np.clip(inverse, lo, hi))))
        if sign > 0:
            positive += mass
        else:
            negative += mass
    return positive, negative


def dir2enc(S, d=1, v=0, *, w=None):
    """Encounter density under the source convention ωe=ω+v*k*cos(theta-phi).

    Negative encountered frequencies are folded onto the positive axis; for
    directional output their propagation direction is shifted by pi. Conservative
    bin integration captures both roots at Doppler folds without an infinite
    sampled Jacobian. Moving finite-depth conversion is explicitly unsupported.
    Moving directional output requires a uniform full-period theta grid with an
    even number of intervals. Unidirectional input supports d=1; at v=0 d=2
    retains its scalar direction. Custom w grids intentionally discard mass
    outside their limits. Input metadata is copied, never modified.
    """
    if str(S.get('type', 'freq')).lower() not in ('freq', 'dir', 'rotdir'):
        raise ValueError('dir2enc requires an intrinsic frequency or directional spectrum')
    if d not in (1, 2) or not np.isscalar(v) or not np.isfinite(v):
        raise ValueError('d must be 1 or 2 and v must be finite')
    wi, theta, density, directional, phi = _temporal(S)
    h, gx, gy = _parameters(S)
    if gx != gy:
        raise NotImplementedError('Encounter conversion requires scalar gravity')
    out = _copy(S)
    _clean_frequency(out)
    out.v, out.phi = float(v), phi
    if v == 0 and (w is None or np.size(w) == 0):
        out.w = wi.copy()
        if d == 1:
            out.S = trapezoid(density, x=theta, axis=0) if directional else density[0].copy()
            out.type = 'enc'
        else:
            out.S = density.copy() if directional else density[0].copy()
            out.theta, out.theta_units, out.type = theta.copy(), 'radians', 'encdir'
        return out
    if v != 0 and not np.isinf(h):
        raise NotImplementedError('Moving finite-depth encounter conversion is not supported')
    if d == 2 and v != 0:
        if not directional:
            raise NotImplementedError('Moving unidirectional d=2 has delta direction masses; use d=1')
        step = np.diff(theta)
        if (not np.isclose(np.ptp(theta), 2*np.pi) or (len(theta)-1) % 2
                or not np.allclose(step, step[0])):
            raise NotImplementedError('Moving d=2 requires a uniform full-period grid with even intervals')
        if not np.allclose(density[0], density[-1], atol=1e-12):
            raise ValueError('The full-period directional density must have equal endpoint rows')
    a = float(v)*np.cos(theta-phi)/gx
    maxima = [np.max(abs(wi[-1]+a*wi[-1]**2)), np.max(abs(wi[0]+a*wi[0]**2))]
    critical = np.divide(-1., 2*a, out=np.full_like(a, np.inf), where=a != 0)
    valid = (critical >= wi[0]) & (critical <= wi[-1])
    if np.any(valid):
        maxima.append(np.max(abs(critical[valid]+a[valid]*critical[valid]**2)))
    wo = _output_grid(w, max(maxima), 2*len(wi), 'w')
    edges, widths = _cells(wo)
    pos, neg = np.zeros((len(theta), len(wo))), np.zeros((len(theta), len(wo)))
    for j in range(len(theta)):
        pos[j], neg[j] = _encounter_row(wi, density[j], a[j], edges)
    if d == 1:
        rows = (pos+neg)/widths
        out.S = trapezoid(rows, x=theta, axis=0) if directional else rows[0]
        out.type = 'enc'
    else:
        if v == 0:
            out.S = (pos+neg)/widths
            if not directional:
                out.S = out.S[0]
        else:
            n = len(theta)-1
            folded = pos[:-1]+np.roll(neg[:-1], n//2, axis=0)
            out.S = np.vstack((folded, folded[0]))/widths
        out.theta, out.theta_units, out.type = theta.copy(), 'radians', 'encdir'
    out.w = wo
    out.discretization = 'conservative trapezoidal bin averages'
    return out


def _quadweights(x):
    """Nonnegative covariance quadrature: Simpson uniform, trapezoid irregular.

    Harmonic covariance is a positive weighted sum of sinusoidal outer products.
    Unequally spaced Simpson formulas can have negative weights and destroy
    positive semidefiniteness. Uniform-grid Simpson preserves source accuracy;
    irregular grids deliberately use positive trapezoidal control volumes.
    """
    intervals = np.diff(x)
    if np.allclose(intervals, intervals[0], rtol=1e-10, atol=0.):
        weights = simpson(np.eye(len(x)), x=x, axis=1)
        if np.all(weights >= 0):
            return weights
    return np.r_[intervals[0]/2, (intervals[:-1]+intervals[1:])/2, intervals[-1]/2]


def dspec2dcov(S, nr=0, Nt=None, rate=1, Nx=10, Ny=0, dx=None, dy=None, *, dt=None):
    """Space/time covariance and all mixed derivatives through order four.

    Integrates ``cos(kx*x+ky*y+omega*t)`` with the original WAFO phase sign,
    finite-depth dispersion and theta-phi rotation. Positive harmonic quadrature
    replaces the source FFT: uniform axes use Simpson weights, while irregular
    axes use nonnegative trapezoidal weights so covariance stays positive
    semidefinite. Output axes
    are (y,x,t) when both spatial axes are active, (x,t)/(y,t) for one, and (t,)
    otherwise. Nt is the maximum time-lag index, hence Nt+1 samples. ``rate``
    retains the original power-of-two FFT-based default time spacing. ``dt``
    optionally supplies an explicit positive time spacing for Python callers.
    """
    if str(S.get('type', 'freq')).lower() not in ('freq', 'dir', 'rotdir'):
        raise ValueError('dspec2dcov requires intrinsic freq or dir data; first use spa2time')
    for name, value, upper in (('nr', nr, 4), ('Nx', Nx, None), ('Ny', Ny, None)):
        if not isinstance(value, (int, np.integer)) or value < 0 or (upper is not None and value > upper):
            raise ValueError(f'{name} must be a nonnegative integer'+(' at most four' if upper else ''))
    if not isinstance(rate, (int, np.integer)) or rate < 1 or rate & (rate-1):
        raise ValueError('rate must be a positive power of two')
    if dt is not None and (not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0):
        raise ValueError('dt must be positive and finite')
    w, theta, density, directional, phi = _temporal(S)
    h, gx, gy = _parameters(S)
    if gx != gy:
        raise NotImplementedError('Directional covariance requires scalar gravity')
    if Nt is None:
        Nt = len(w)-1
    if not isinstance(Nt, (int, np.integer)) or Nt < 0:
        raise ValueError('Nt must be a nonnegative integer')
    k = _wave_number(w, h, gx)
    kx, ky = np.cos(theta[:, None]-phi)*k, np.sin(theta[:, None]-phi)*k
    weights = _quadweights(w)
    angular = _quadweights(theta) if directional else np.ones(1)
    sd = density*angular[:, None]
    m0 = np.sum(sd*weights)
    mx = np.sum(sd*kx*kx*weights)
    my = np.sum(sd*ky*ky*weights)
    if dx is None:
        dx = 4*np.pi*np.sqrt(m0/mx)/Nx if Nx and mx > 0 and m0 > 0 else 1.
    if dy is None:
        dy = dx if Nx else 4*np.pi*np.sqrt(m0/my)/Ny if Ny and my > 0 and m0 > 0 else 1.
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(dy) or dy <= 0:
        raise ValueError('dx and dy must be positive and finite')
    nbase = 2*len(w)-2
    nfft = int(rate)*2**int(np.ceil(np.log2(nbase)))
    t = np.arange(Nt+1)*(np.pi/w[-1]*(nbase/nfft) if dt is None else dt)
    x, y = np.arange(-Nx, Nx+1)*dx, np.arange(-Ny, Ny+1)*dy
    X, Y = np.meshgrid(x, y)
    spatial = np.exp(1j*(X.ravel()[:, None, None]*kx+Y.ravel()[:, None, None]*ky))
    temporal = np.exp(1j*w[:, None]*t)
    from . import Spectrum
    out = Spectrum({key: deepcopy(S[key]) for key in ('tr', 'h', 'norm', 'note') if key in S})
    out.setdefault('h', h)
    out.setdefault('norm', 0)
    out.setdefault('tr', [])
    out.type, out.t = 'none', t
    if Nx:
        out.x = x
    if Ny:
        out.y = y
    for px, py, pt in product(range(nr+1), repeat=3):
        if px+py+pt > nr or (px and not Nx) or (py and not Ny):
            continue
        amplitude = np.sum(spatial*sd*(1j*kx)**px*(1j*ky)**py, axis=1)
        values = np.real((amplitude*weights*(1j*w)**pt) @ temporal)
        shape = ((len(y), len(x)) if Nx and Ny else (len(x),) if Nx else (len(y),) if Ny else ())+(len(t),)
        out['R'+'x'*px+'y'*py+'t'*pt] = values.reshape(shape)
    return out
