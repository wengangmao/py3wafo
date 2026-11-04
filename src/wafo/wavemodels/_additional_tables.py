"""Original Ochi-Hubble/Torsethaugen conditional wave-characteristic fits.

Stored coefficient encodings, marginal laws and dimensional scales follow the
active MATLAB sources.  SciPy cubic interpolation adapts their interpolants;
this is not bitwise MATLAB ``*cubic`` parity.  Height curves extend linearly
as in ``smooth(...,1,...,1)``; invalid parameters raise.  Sea-state coordinates
never extrapolate beyond the available data or the stated physical domain.
"""
from functools import lru_cache

import numpy as np
from scipy import integrate, stats
from scipy.interpolate import CubicSpline, RectBivariateSpline

from ..spec import Spectrum, ohspec2, spec2char
from ..wstats import wtweibpdf, wtweibcdf
from ._tables import _data, ohhgparfun, thwparfun


_TABLES = dict(ohhs='ohhspar.mat', ohhss='ohhsspar.mat', ohhv='ohhvpar.mat',
               ths='thspar19-Jul-2004.mat', thss='thsspar27-Jul-2004.mat',
               thsnl='thsnlpar06-Aug-2004.mat', thv='thvpar.mat')


def _table(family):
    return _data('joint-'+_TABLES[family])


def _state(family, Hm0, coordinate):
    H, state = np.broadcast_arrays(np.asarray(Hm0, dtype=float), np.asarray(coordinate, dtype=float))
    if not np.all(np.isfinite(H)) or np.any((H < .5) | (H > 12)):
        raise ValueError('Fitted Hm0 range is 0.5 through 12 metres')
    if family.startswith('oh'):
        if not np.all(np.isfinite(state)) or np.any(state != np.floor(state)) or np.any((state < 1) | (state > 11)):
            raise ValueError('Ochi-Hubble definition must be an integer from1 through11')
    else:
        if not np.all(np.isfinite(state)) or np.any((state < 3.5) | (state > 20)) or np.any(H >= (state-2)*12/11):
            raise ValueError('Torsethaugen fit needs 3.5<=Tp<=20 and Hm0<(Tp-2)*12/11')
        if family == 'thv':
            bandwidth = _surface(family, 'eps2').ev(H.ravel(), state.ravel())
            grid = _table(family)['e2'].ravel()
            if np.any((bandwidth < grid[0]) | (bandwidth > grid[-1])):
                raise ValueError('Interpolated spectral bandwidth is outside the thv conditional table')
    return H, state


@lru_cache(maxsize=None)
def _surface(family, key):
    data = _table(family)
    return RectBivariateSpline(data['Hm0'].ravel(), data['Tp'].ravel(), data[key], s=0)


@lru_cache(maxsize=None)
def _height_surfaces(family, key):
    data = _table(family)
    values = np.log(data[key])
    return tuple(RectBivariateSpline(data['Hm0'].ravel(), data['Tp'].ravel(), values[:, :, i], s=0)
                 for i in range(len(data['h2'].ravel())))


@lru_cache(maxsize=256)
def _curve(family, H, coordinate, key):
    data = _table(family)
    height = data['h2'].ravel()
    if family.startswith('oh'):
        raw = data[key][:, int(coordinate)-1, :]
        logarithmic = key == 'A11s'
        if family == 'ohhs':
            # ohhspar stores BOTH logarithms, but its source interpolates
            # exp(B11) before smoothing the height-dependent shape.
            raw = raw if logarithmic else np.exp(raw)
        elif logarithmic:
            raw = np.log(raw)
        values = CubicSpline(data['Hm0'].ravel(), raw, axis=0)(H)
    elif family == 'thv':
        bandwidth = float(_surface(family, 'eps2').ev(H, coordinate))
        e2 = data['e2'].ravel()
        if not e2[0] <= bandwidth <= e2[-1]:
            raise ValueError('Interpolated spectral bandwidth is outside the thv conditional table')
        logarithmic = False
        values = CubicSpline(e2, data[key], axis=1)(bandwidth)
    else:
        logarithmic = True
        values = np.asarray([surface.ev(H, coordinate) for surface in _height_surfaces(family, key)])
    return CubicSpline(height, values, bc_type='natural'), height, logarithmic


def _conditional_parameters(family, height, H, coordinate):
    height, H, coordinate = np.broadcast_arrays(height, H, coordinate)
    result = []
    states = np.column_stack((H.ravel(), coordinate.ravel()))
    unique, inverse = np.unique(states, axis=0, return_inverse=True)
    for key in ('A11s', 'B11s'):
        values = np.empty(height.size)
        for j, (h0, state0) in enumerate(unique):
            mask = inverse == j
            curve, grid, logarithmic = _curve(family, float(h0), float(state0), key)
            point = np.maximum(height.ravel()[mask], 0.)
            fitted = (curve(np.clip(point, grid[0], grid[-1]))+
                      np.minimum(point-grid[0], 0)*curve(grid[0], 1)+
                      np.maximum(point-grid[-1], 0)*curve(grid[-1], 1))
            with np.errstate(over='ignore'):
                values[mask] = np.exp(fitted) if logarithmic else fitted
        if np.any(values <= 0) or not np.all(np.isfinite(values)):
            raise ValueError('Height interpolation/extrapolation produced invalid conditional fit parameters')
        result.append(values.reshape(height.shape))
    return tuple(result)


@lru_cache(maxsize=128)
def _oh_velocity_period(H, definition):
    # Exact source grid for ohhvpdf.  At stored sea-state knots these moments
    # agree with ohhspar.Tm02 to roundoff; off-grid this avoids an extra fit.
    return float(spec2char(ohspec2(np.linspace(0, 100, 16385), [H, definition]), ['Tm02'])[0])


def _scales(family, H, coordinate, normalized):
    if normalized:
        return np.ones_like(H), np.ones_like(H)
    period = np.empty(H.shape)
    if family.startswith('oh'):
        states, inverse = np.unique(np.column_stack((H.ravel(), coordinate.ravel())), axis=0, return_inverse=True)
        for j, (height, definition) in enumerate(states):
            height, definition = float(height), int(definition)
            if family == 'ohhv':
                value = _oh_velocity_period(height, definition)
            else:
                data = _table(family)
                value = CubicSpline(data['Hm0'].ravel(), data['Tm02'][:, definition-1])(height)
            period.ravel()[inverse == j] = value
    else:
        period = _surface(family, 'Tm02').ev(H.ravel(), coordinate.ravel()).reshape(H.shape)
    if np.any(period <= 0) or not np.all(np.isfinite(period)):
        raise ValueError('Interpolated normalizing period/wavelength is invalid')
    scale = 1.25*H/period**2 if family in ('ohhs', 'ths', 'thsnl') else 2*H/period
    return H/np.sqrt(2), scale


@lru_cache(maxsize=256)
def _scalar_marginal(family, H, coordinate):
    if family.startswith('oh'):
        a, b, c = ohhgparfun(H, coordinate, 'space' if family == 'ohhss' else 'time')
        return tuple(float(value) for value in (a, b, c))
    return tuple(float(value) for value in thwparfun(H, coordinate, 'space' if family == 'thss' else 'time'))


def _marginal(family, H, coordinate):
    H, coordinate = np.broadcast_arrays(H, coordinate)
    states, inverse = np.unique(np.column_stack((H.ravel(), coordinate.ravel())), axis=0, return_inverse=True)
    values = np.asarray([_scalar_marginal(family, float(height), float(state)) for height, state in states])
    parameters = tuple(values[inverse, j].reshape(H.shape) for j in range(3))
    return ('gengamma' if family.startswith('oh') else 'tweibull'), parameters


def _marginal_pdf(h, kind, parameters):
    a, b, c = parameters
    with np.errstate(invalid='ignore'):
        result = stats.gengamma.pdf(h, a, c, scale=b) if kind == 'gengamma' else wtweibpdf(h, a, b, c)
    return np.where(np.isposinf(h), 0., result)


def _marginal_cdf(h, kind, parameters):
    a, b, c = parameters
    return stats.gengamma.cdf(h, a, c, scale=b) if kind == 'gengamma' else wtweibcdf(h, a, b, c)


def _marginal_ppf(p, kind, parameters):
    a, b, c = parameters
    return stats.gengamma.ppf(p, a, c, scale=b) if kind == 'gengamma' else a*((c/a)**b-np.log1p(-p))**(1/b)-c


def _conditional(family, a, b):
    return stats.gamma(a, scale=b) if family in ('ths', 'thss', 'thsnl') else stats.weibull_min(b, scale=a)


def _parameter_table(family, key):
    data = _table(family)
    axes = [data['Hm0'].ravel(), np.arange(1, 12), data['h2'].ravel()] if family.startswith('oh') else (
        [data['e2'].ravel(), data['h2'].ravel()] if family == 'thv' else
        [data['Tp'].ravel(), data['Hm0'].ravel(), data['h2'].ravel()])
    return Spectrum(f=data[key].copy(), x=axes, title=family+' stored '+key,
                    note='Original stored coefficients; ohhs stores log(A) and log(B); other tables store raw parameters')


def _pdf(family, Hd, characteristic, Hm0, coordinate, normalized, condon, full_output):
    if normalized not in (False, True) or condon not in (0, 1, 3, 4, 5, 6, 7):
        raise ValueError('Use a boolean normalizedInput and condon0,1,3,4,5,6,7')
    H, state = _state(family, Hm0, coordinate)
    Hd, V, H, state = np.broadcast_arrays(Hd, characteristic, H, state)
    hs, vs = _scales(family, H, state, normalized)
    h, v = Hd/hs, V/vs
    if condon in (1, 3, 4, 6) and np.any(~np.isfinite(h)):
        raise ValueError('Conditional distributions require finite heights')
    a, b = _conditional_parameters(family, np.where(np.isfinite(h), h, 0.), H, state)
    distribution = _conditional(family, a, b)
    with np.errstate(invalid='ignore'):
        density = np.where(np.isposinf(v), 0., distribution.pdf(v)/vs)
    cumulative, survival = distribution.cdf(v), distribution.sf(v)
    marginal = _marginal_pdf(h, *_marginal(family, H, state))/hs
    if condon == 0:
        # Some spatial height fits have an integrable infinite density at
        # zero.  At support corners define 0*inf as0 (a measure-zero choice)
        # while retaining legitimate infinite joint/conditional densities.
        result = np.multiply(marginal, density, out=np.zeros_like(density),
                             where=(marginal != 0) & (density != 0))
    elif condon == 1:
        result = density
    elif condon in (3, 4):
        with np.errstate(invalid='ignore'):
            result = np.where(np.isinf(V), 0., V**(condon-2)*density)
    elif condon == 5:
        result = marginal*cumulative
    elif condon == 6:
        result = cumulative
    else:
        result = marginal*survival
    result = np.where(h < 0, 0., result)
    result = np.where(np.isnan(h) | np.isnan(v), np.nan, result)
    return (result, hs, vs, _parameter_table(family, 'A11s'), _parameter_table(family, 'B11s')) if full_output else result


def _cdf(family, Hd, characteristic, Hm0, coordinate, tail, tol, full_output):
    if tail not in (0, 1) or not np.isfinite(tol) or tol <= 0:
        raise ValueError('Use tail0/1 and a positive finite integration tolerance')
    H, state = _state(family, Hm0, coordinate)
    Hd, V, H, state = np.broadcast_arrays(Hd, characteristic, H, state)
    hs, vs = _scales(family, H, state, False)
    out, error = np.zeros(H.shape), np.zeros(H.shape)
    for index in np.ndindex(H.shape):
        height, v, H0, c0 = Hd[index]/hs[index], V[index]/vs[index], H[index], state[index]
        if np.isnan(height) or np.isnan(v):
            out[index] = error[index] = np.nan; continue
        kind, params = _marginal(family, H0, c0)
        cutoff = float(_marginal_cdf(max(0., height), kind, params))
        if tail == 0 and (height <= 0 or v <= 0):
            continue
        if tail == 1 and (np.isposinf(height) or np.isposinf(v)):
            continue
        if tail == 0 and np.isposinf(v):
            out[index] = cutoff; continue
        if tail == 1 and v <= 0:
            out[index] = 1-cutoff; continue
        lower, upper = (0., cutoff) if tail == 0 else (cutoff, 1.)
        def function(probability):
            h = _marginal_ppf(probability, kind, params)
            a, b = (float(value) for value in _conditional_parameters(family, h, H0, c0))
            distribution = _conditional(family, a, b)
            return float(distribution.cdf(v) if tail == 0 else distribution.sf(v))
        out[index], error[index] = integrate.quad(function, lower, upper, epsabs=tol, epsrel=tol, limit=150)
    return (out, error) if full_output else out


def _grid(family, Hd, characteristic, Hm0, coordinate, normalized, full_output):
    h, v = np.asarray(Hd, dtype=float), np.asarray(characteristic, dtype=float)
    if h.ndim != 1 or v.ndim != 1 or not len(h) or not len(v) or not np.all(np.isfinite(h)) or not np.all(np.isfinite(v)) or np.any(np.diff(h) <= 0) or np.any(np.diff(v) <= 0):
        raise ValueError('Grid coordinates must be finite increasing nonempty vectors')
    vv, hh = np.meshgrid(v, h)
    density, hs, vs, a, b = _pdf(family, hh, vv, Hm0, coordinate, normalized, 0, True)
    label = 'Vcf' if family in ('ohhv', 'thv') else 'Scf'
    pdf = Spectrum(f=density, x=[v, h], norm=int(normalized), labx=[label, 'Hd'],
                   title=family+' joint wave distribution', note='Original fitted coefficients; cubic interpolation adapted to SciPy')
    return (pdf, hs, vs, a, b) if full_output else pdf


def ohhspdf(Hd, Scf, Hm0=6, definition=1, normalizedInput=False, condon=0, *, full_output=False):
    """Ochi-Hubble time steepness/height fit, definitions1..11, Hm0 in[.5,12].

    Weibull conditionals, generalized-gamma height marginal, and source
    1.25*Hm0/Tm02² steepness scale.  condon0/1/3/4/5/6/7 selects joint PDF /
    conditional PDF / first or second moment integrand / marginal*CDF /
    conditional CDF / marginal*survival.  Conditional Jacobians and moments
    use physical units, repairing the source common scaling error.  These
    conditioning and full-output conventions apply to every pointwise model
    in this module.  Full output adds (Hrms,scale,stored_A_table,stored_B_table).
    Undefined 0*inf support corners take value0; integrable infinite densities
    elsewhere are preserved rather than erased as in the source.
    """
    return _pdf('ohhs', Hd, Scf, Hm0, definition, normalizedInput, condon, full_output)


def ohhsspdf(Hd, Scf, Hm0=6, definition=1, normalizedInput=False, condon=0, *, full_output=False):
    """Ochi-Hubble spatial steepness fit; same modes/domain as ohhspdf.

    Uses the distinct spatial marginal, conditional table and wavelength
    normalization 2*Hm0/Tm02 from ohhsspar.mat.
    """
    return _pdf('ohhss', Hd, Scf, Hm0, definition, normalizedInput, condon, full_output)


def ohhvpdf(v, h, Hm0=6, definition=1, norm=False, *, full_output=False):
    """Original grid API for OH front velocity: meshgrid(v,h) PDF mapping.

    Uses Weibull velocity conditionals and the time generalized-gamma height
    marginal.  Physical scale 2*Hm0/Tm02 uses actual ohspec2 spectral moments
    on the source frequency grid.  norm=True accepts normalized coordinates.
    Full output returns (pdf,stored_A_table,stored_B_table), as in the source.
    """
    result = _grid('ohhv', h, v, Hm0, definition, norm, full_output)
    return (result[0], result[3], result[4]) if full_output else result


def thspdf(Hd, Scf, Hm0=6, Tp=8, normalizedInput=False, condon=0, *, full_output=False):
    """Torsethaugen linear time steepness/height fit with gamma conditionals.

    Active thspar19-Jul-2004 coefficients and thwparfun time height marginal;
    log interpolation of conditional shape and scale.  Requires .5<=Hm0<=12,
    3.5<=Tp<=20 and Hm0<(Tp-2)*12/11.  Modes/outputs follow ohhspdf.
    """
    return _pdf('ths', Hd, Scf, Hm0, Tp, normalizedInput, condon, full_output)


def thsspdf(Hd, Scf, Hm0=6, Tp=8, normalizedInput=False, condon=0, *, full_output=False):
    """Torsethaugen spatial gamma steepness fit, active July27 source table.

    Distinct spatial truncated-Weibull marginal and wavelength normalization;
    domain and conditioning modes follow thspdf/ohhspdf.
    """
    return _pdf('thss', Hd, Scf, Hm0, Tp, normalizedInput, condon, full_output)


def thsnlpdf(Hd, Scf, Hm0=6, Tp=8, normalizedInput=False, condon=0, *, full_output=False):
    """Second-order nonlinear Torsethaugen steepness fit from August6 table.

    Gamma conditionals and time truncated-Weibull marginal; same domains and
    modes as thspdf.  No linear table is substituted for nonlinear coefficients.
    """
    return _pdf('thsnl', Hd, Scf, Hm0, Tp, normalizedInput, condon, full_output)


def thvpdf(Hd, Vcf, Hm0=6, Tp=8, normalizedInput=False, condon=0, *, full_output=False):
    """Torsethaugen front-velocity Weibull fit indexed by spectral bandwidth.

    Sea-state domain follows thspdf; interpolated eps2 must additionally lie
    within the actual conditional table0..1.2.  Other modes follow ohhspdf.
    """
    return _pdf('thv', Hd, Vcf, Hm0, Tp, normalizedInput, condon, full_output)


def ohhscdf(Hd, Scf, Hm0=6, definition=1, tail=0, *, tol=1e-8, full_output=False):
    """OH time steepness joint CDF/tail; integrates without a hard height cap."""
    return _cdf('ohhs', Hd, Scf, Hm0, definition, tail, tol, full_output)


def ohhsscdf(Hd, Scf, Hm0=6, definition=1, tail=0, *, tol=1e-8, full_output=False):
    """OH spatial steepness joint CDF/tail, matching ohhsspdf."""
    return _cdf('ohhss', Hd, Scf, Hm0, definition, tail, tol, full_output)


def thscdf(Hd, Scf, Hm0=6, Tp=8, tail=0, *, tol=1e-8, full_output=False):
    """Torsethaugen time steepness CDF/tail, matching thspdf."""
    return _cdf('ths', Hd, Scf, Hm0, Tp, tail, tol, full_output)


def thsscdf(Hd, Scf, Hm0=6, Tp=8, tail=0, *, tol=1e-8, full_output=False):
    """Torsethaugen spatial steepness CDF/tail, matching thsspdf."""
    return _cdf('thss', Hd, Scf, Hm0, Tp, tail, tol, full_output)


def thsnlcdf(Hd, Scf, Hm0=6, Tp=8, tail=0, *, tol=1e-8, full_output=False):
    """Nonlinear Torsethaugen steepness CDF/tail, matching thsnlpdf."""
    return _cdf('thsnl', Hd, Scf, Hm0, Tp, tail, tol, full_output)


def thvcdf(Hd, Vcf, Hm0=6, Tp=8, tail=0, *, tol=1e-8, full_output=False):
    """Torsethaugen front-velocity CDF/tail, matching thvpdf."""
    return _cdf('thv', Hd, Vcf, Hm0, Tp, tail, tol, full_output)


def ohhspdf2(Hd, Scf, Hm0=6, definition=1, normalizedInput=False, *, full_output=False):
    """OH time steepness PDF mapping on meshgrid(Scf,Hd)."""
    return _grid('ohhs', Hd, Scf, Hm0, definition, normalizedInput, full_output)


def ohhsspdf2(Hd, Scf, Hm0=6, definition=1, normalizedInput=False, *, full_output=False):
    """OH spatial steepness PDF mapping on meshgrid(Scf,Hd)."""
    return _grid('ohhss', Hd, Scf, Hm0, definition, normalizedInput, full_output)


def thspdf2(Hd, Scf, Hm0=6, Tp=8, normalizedInput=False, *, full_output=False):
    """Torsethaugen time steepness PDF mapping on meshgrid(Scf,Hd)."""
    return _grid('ths', Hd, Scf, Hm0, Tp, normalizedInput, full_output)


def thsspdf2(Hd, Scf, Hm0=6, Tp=8, normalizedInput=False, *, full_output=False):
    """Torsethaugen spatial steepness PDF mapping on meshgrid(Scf,Hd)."""
    return _grid('thss', Hd, Scf, Hm0, Tp, normalizedInput, full_output)


def thsnlpdf2(Hd, Scf, Hm0=6, Tp=8, normalizedInput=False, *, full_output=False):
    """Nonlinear Torsethaugen steepness PDF mapping on meshgrid(Scf,Hd)."""
    return _grid('thsnl', Hd, Scf, Hm0, Tp, normalizedInput, full_output)


def thvpdf2(Hd, Vcf, Hm0=6, Tp=8, normalizedInput=False, *, full_output=False):
    """Torsethaugen front-velocity PDF mapping on meshgrid(Vcf,Hd)."""
    return _grid('thv', Hd, Vcf, Hm0, Tp, normalizedInput, full_output)
