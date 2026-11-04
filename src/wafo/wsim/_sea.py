"""Directional Gaussian sea fields and portable recorded movie frames."""
from __future__ import annotations
import warnings
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or int(value) != value or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _weights(grid):
    """Nonnegative trapezoid weights: spectral mode variances cannot be negative."""
    step = np.diff(grid)
    return np.r_[step[0]/2, (step[:-1]+step[1:])/2, step[-1]/2]


def seasim(spec, Nx=None, Ny=None, Nt=None, dx=None, dy=None, dt=None,
           fftdim=2, plotflag=0, *, seed=None, full_output=False):
    """Simulate a Gaussian sea on an (x,y,t) grid from a directional spectrum.

    Ny=0 selects an x/t section. Output Z follows MATLAB (Ny,Nx,Nt), with
    singleton axes squeezed; x,y,t always retain the complete coordinate axes.
    fftdim=1 uses independent Gaussian harmonic coefficients with nonnegative
    trapezoid weights on the original frequency/direction grid. fftdim=2 interpolates to a Cartesian
    wave-number FFT grid and advances the same coefficients by dispersion.
    Both use phase kx*x+ky*y+omega*t, matching dspec2dcov's covariance convention.
    Gravity is scalar; finite depth is supported. A supplied tr table is
    inverted pointwise after Gaussian simulation.

    No record is rescaled to force a sample variance. target_variance and
    discrete_variance expose quadrature/grid changes; FFT grid variance changes
    above 10 percent produce a warning. Seeded results are reproducible within
    this implementation, not MATLAB random-stream equivalents. Resource limits
    reject excessive direct work or FFT grids rather than dropping wave modes.

    plotflag=1 records the first frame, 2 records all frames, 0 creates no
    graphics. full_output returns (field,movie), with movie None for mode 0.
    Graphics do not play automatically or start a GUI event loop.
    """
    from ..spec import Spectrum, spec2spec
    from ..spec._directional_transforms import (_temporal, _parameters,
        _wave_number, _omega, _group_speed)
    kind = str(spec.get('type', '')).lower()
    if kind.endswith('k2d'):
        spec = spec2spec(spec, 'dir')
    elif kind not in ('dir', 'rotdir'):
        raise ValueError('seasim requires an intrinsic directional or Cartesian spectrum')
    w, theta, density, directional, phi = _temporal(spec)
    if not directional:
        raise ValueError('A two-dimensional directional density is required')
    h, gravity, gy = _parameters(spec)
    if gravity != gy:
        raise ValueError('seasim requires scalar gravity')
    default = 2**int(np.ceil(np.log2(len(w))))
    Nx, Nt = _count(default if Nx is None else Nx, 'Nx'), _count(default if Nt is None else Nt, 'Nt')
    Ny = 1 if Ny == 0 else _count(default if Ny is None else Ny, 'Ny')
    k = _wave_number(w, h, gravity)
    dx = np.pi/(2*k[-1]) if dx is None else dx
    dy = dx if dy is None else dy
    dt = np.pi/w[-1] if dt is None else dt
    if any(not np.isscalar(v) or not np.isfinite(v) or v <= 0 for v in (dx, dy, dt)) or fftdim not in (1, 2) or plotflag not in (0, 1, 2):
        raise ValueError('Positive finite spacings, fftdim 1/2, and plotflag 0/1/2 are required')
    x, y, t = np.arange(Nx)*dx, np.arange(Ny)*dy, np.arange(Nt)*dt
    rng = np.random.default_rng(seed)
    weights = density*_weights(theta)[:, None]*_weights(w)[None, :]
    target = float(weights.sum())
    if Nx*Ny*Nt > 20_000_000:
        raise ValueError('Output exceeds 20 million field values; reduce the requested grid')
    if fftdim == 1:
        if Nx*Ny*len(theta)*len(w) > 200_000_000:
            raise ValueError('Direct harmonic work exceeds the limit; select fftdim=2 or reduce the grid')
        coeff = np.sqrt(weights)*(rng.normal(size=weights.shape)+1j*rng.normal(size=weights.shape))
        kx, ky = k[None, :]*np.cos(theta[:, None]-phi), k[None, :]*np.sin(theta[:, None]-phi)
        xx, yy = np.meshgrid(x, y)
        spatial = np.empty((Nx*Ny, len(w)), dtype=complex)
        for first in range(0, Nx*Ny, 64):
            phase = xx.ravel()[first:first+64, None, None]*kx+yy.ravel()[first:first+64, None, None]*ky
            spatial[first:first+64] = np.sum(np.exp(1j*phase)*coeff, axis=1)
        values = np.real(spatial @ np.exp(1j*w[:, None]*t)).reshape(Ny, Nx, Nt)
        discrete = target
        algorithm = 'Gaussian harmonic quadrature on original directional grid'
    else:
        nfx = 2**int(np.ceil(np.log2(max(Nx, len(w)))))
        nfy = 2**int(np.ceil(np.log2(max(Ny, len(w)))))
        if nfx*nfy > 1_048_576:
            raise ValueError('FFT grid exceeds one million wave-number cells; reduce the spectral/grid resolution')
        kx = 2*np.pi*np.fft.fftfreq(nfx, dx)
        ky = 2*np.pi*np.fft.fftfreq(nfy, dy)
        xx, yy = np.meshgrid(kx, ky)
        radius = np.hypot(xx, yy)
        omega = _omega(radius, h, gravity)
        angle = (np.arctan2(yy, xx)+phi-theta[0]) % (2*np.pi)+theta[0]
        interp = RegularGridInterpolator((theta, w), density, bounds_error=False, fill_value=0.)
        jacobian = np.divide(_group_speed(radius, omega, h, gravity), radius,
                             out=np.zeros_like(radius), where=radius > 0)
        cellmass = interp(np.stack((angle, omega), axis=-1))*jacobian*(2*np.pi/(nfx*dx))*(2*np.pi/(nfy*dy))
        discrete = float(cellmass.sum())
        if target > 0 and abs(discrete/target-1) > .1:
            warnings.warn('Cartesian FFT quadrature variance differs from input by more than 10%; adjust spatial grid/spacing or use fftdim=1', RuntimeWarning, stacklevel=2)
        coeff = np.sqrt(cellmass)*(rng.normal(size=cellmass.shape)+1j*rng.normal(size=cellmass.shape))
        values = np.empty((Ny, Nx, Nt))
        for j, time in enumerate(t):
            values[:, :, j] = np.real(np.fft.ifft2(coeff*np.exp(1j*omega*time))[:Ny, :Nx])*(nfx*nfy)
        algorithm = 'Cartesian Gaussian FFT with dispersion evolution'
    if np.size(spec.get('tr', [])):
        from ..trgauss import tranproc
        values = tranproc(values.ravel(), np.asarray(spec['tr'])[:, ::-1]).reshape(values.shape)
    result = Spectrum(Z=np.squeeze(values), x=x, y=y, t=t, shape=(Ny, Nx, Nt),
                      target_variance=target, discrete_variance=discrete, method=algorithm,
                      note='Variance fields refer to the Gaussian process before any marginal transformation')
    movie = seamovie(result, frames=[0] if plotflag == 1 else None) if plotflag else None
    return (result, movie) if full_output else result


def seamovie(Y, s=1, *, frames=None, ax=None):
    """Record portable RGBA movie frames from a seasim field without playback.

    s=1 shows a surface, 2 a zero-level contour, 3 a grayscale image; x/t
    sections use line plots. Returns a mapping with frames (n,h,w,4), selected
    times, figure and axes. Pass frames as explicit zero-based time indices to
    record a subset. The caller can save/play the returned pixels as needed.
    """
    import matplotlib.pyplot as plt
    x, y, t = [np.asarray(Y.get(key, [0]), dtype=float) for key in ('x', 'y', 't')]
    if any(v.ndim != 1 or not len(v) or not np.all(np.isfinite(v)) or np.any(np.diff(v) <= 0) for v in (x, y, t)):
        raise ValueError('Movie coordinates must be finite increasing vectors')
    values = np.asarray(Y['Z'], dtype=float)
    if values.size != len(x)*len(y)*len(t) or not np.all(np.isfinite(values)) or max(len(x), len(y)) < 2:
        raise ValueError('Field values must match its spatial/time coordinates')
    values = values.reshape(len(y), len(x), len(t))
    if s not in (1, 2, 3):
        raise ValueError('Movie style must be 1, 2, or 3')
    index = np.arange(len(t)) if frames is None else np.asarray(frames)
    if index.ndim != 1 or not len(index) or not np.all(np.isfinite(index)) or np.any(index != np.floor(index)) or np.any(index < 0) or np.any(index >= len(t)):
        raise ValueError('frames must contain valid zero-based time indices')
    index = index.astype(int)
    surface = len(x) > 1 and len(y) > 1
    if ax is None:
        ax = plt.figure().add_subplot(111, projection='3d' if surface and s == 1 else None)
    if surface and s == 1 and not hasattr(ax, 'plot_surface'):
        raise ValueError('Surface movie style requires a 3-D axes')
    width, height = ax.figure.canvas.get_width_height()
    if len(index)*width*height*4 > 512*1024**2:
        raise ValueError('Movie pixels exceed 512 MiB; pass a smaller frames subset or figure')
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        lo, hi = lo-1, hi+1
    xx, yy = np.meshgrid(x, y)
    images = []
    for j in index:
        ax.clear()
        if surface:
            if s == 1:
                ax.plot_surface(xx, yy, values[:, :, j], cmap='winter', vmin=lo, vmax=hi)
                ax.set_zlim(lo, hi)
            elif s == 2:
                if values[:, :, j].min() < 0 < values[:, :, j].max():
                    ax.contour(x, y, values[:, :, j], levels=[0])
                ax.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
            else:
                ax.pcolormesh(x, y, values[:, :, j], cmap='gray', vmin=lo, vmax=hi, shading='auto')
            ax.set(xlabel='x (m)', ylabel='y (m)')
        else:
            grid = x if len(x) > 1 else y
            ax.plot(grid, values[:, :, j].ravel())
            ax.axhline(0, color='k', linestyle=':')
            ax.set(xlim=(grid[0], grid[-1]), ylim=(lo, hi), xlabel='Distance (m)', ylabel='Elevation (m)')
        ax.set_title(f't = {t[j]:g} s')
        ax.figure.canvas.draw()
        images.append(np.asarray(ax.figure.canvas.buffer_rgba()).copy())
    return {'frames': np.stack(images), 'times': t[index], 'figure': ax.figure, 'axes': ax}
