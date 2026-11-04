"""Tabulated transformed-Gaussian models and source-based estimators.

Moment models follow hermitetr/ochitr and their private MATLAB helpers.
Smoothing minimizes the same roughness objective through wafo.misc.smooth;
Python returns explicit optional tuples instead of nargout-dependent outputs.
"""
from datetime import datetime, timezone
import numpy as np
from scipy import interpolate, optimize, stats
from scipy.integrate import trapezoid


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _table(g):
    if isinstance(g, dict):
        g = g['tr']
    a = np.asarray(g, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2 or not np.all(np.isfinite(a)):
        raise ValueError('Expected a finite two-column transformation table')
    a = a[np.argsort(a[:, 0], kind='stable')]
    if np.any(np.diff(a[:, 0]) <= 0):
        raise ValueError('Transformation abscissas must be distinct')
    return a


def _linear(x, y, points):
    points = np.asarray(points, dtype=float)
    result = np.interp(points, x, y)
    return (result+np.minimum(points-x[0], 0)*(y[1]-y[0])/(x[1]-x[0])
            +np.maximum(points-x[-1], 0)*(y[-1]-y[-2])/(x[-1]-x[-2]))


def createpdf(ndim=1):
    """Create an attribute-access mapping for an ndim-dimensional density."""
    from ..spec import Spectrum
    if not np.isscalar(ndim) or not np.isfinite(ndim) or int(ndim) != ndim or ndim < 1:
        raise ValueError('ndim must be a positive integer')
    return Spectrum(f=np.asarray([]), x=[np.asarray([]) for _ in range(int(ndim))],
                    labx=['']*int(ndim), title='', note='', date=_stamp())


def trmak(tr=None, ma=None, sa=None, note=''):
    """Create a transformation mapping with form,tr,mean,std,note,date fields.

    Tables use form='table'; Python spline/callable objects use form='pp'.
    The empty constructor leaves mean/std unset, as the original does.
    """
    from ..spec import Spectrum
    if isinstance(tr, dict) and {'form','tr','mean','std','note','date'} <= set(tr):
        return Spectrum(tr)
    if tr is None:
        value, form = np.empty((0, 2)), 'table'
    elif callable(tr):
        value, form = tr, 'pp'
    else:
        value, form = _table(tr), 'table'
    if ma is not None and (not np.isscalar(ma) or not np.isfinite(ma)):
        raise ValueError('Mean must be a finite scalar')
    if sa is not None and (not np.isscalar(sa) or not np.isfinite(sa) or sa <= 0):
        raise ValueError('Standard deviation must be positive and finite')
    return Spectrum(form=form, tr=value, mean=ma, std=sa, note=note, date=_stamp())


def trunmak(tro):
    """Return (tr,mean,std,form,note,date) from a transformation mapping."""
    required = ('tr','mean','std','form','note','date')
    if not isinstance(tro, dict) or not set(required) <= set(tro):
        raise ValueError('Expected a transformation mapping created by trmak')
    return tuple(tro[key] for key in required)


def trangood(ff, nmin=None, mini=None, maxi=None, nmax=np.inf):
    """Uniformly resample a sorted table and linearly extend its endpoints.

    nmin/nmax bound the number of points inside the original table range;
    endpoint extension may add points beyond that count, as in the source.
    """
    f = _table(ff)
    nmin = len(f) if nmin is None else nmin
    mini = f[0, 0] if mini is None else mini
    maxi = f[-1, 0] if maxi is None else maxi
    if (not np.isscalar(nmin) or not np.isfinite(nmin) or int(nmin) != nmin or nmin < 2
            or not np.isscalar(nmax) or np.isnan(nmax) or nmax == -np.inf
            or (np.isfinite(nmax) and (int(nmax) != nmax or nmax < 2))
            or not np.isfinite(mini+maxi) or maxi < mini):
        raise ValueError('Invalid point counts or extension bounds')
    steps = np.diff(f[:, 0])
    if len(f) < nmin or len(f) > nmax or not np.allclose(steps, steps[0], rtol=1e-12, atol=0):
        count = int(min(nmin, nmax))
        x = np.linspace(f[0, 0], f[-1, 0], count)
        f = np.column_stack((x, np.interp(x, f[:, 0], f[:, 1])))
    step = f[1, 0]-f[0, 0]
    first = min(0, int(np.floor((mini-f[0, 0])/step)))
    last = max(len(f)-1, int(np.ceil((maxi-f[0, 0])/step)))
    x = f[0, 0]+np.arange(first, last+1)*step
    return np.column_stack((x, _linear(f[:, 0], f[:, 1], x)))


def tranproc(x, ff, *, extrapolate=True):
    """Transform [X,X',...,X'''' ] using the chain rule, up to order four.

    A vector denotes scalar process values; a matrix's columns are successive
    derivatives, not independent signals. For no derivatives, a PCHIP table
    interpolation preserves the previous Python API. Derivative transforms use
    a degree-up-to-five interpolating spline rather than the source's recursive
    finite differences. Values outside a table extend linearly by default.
    """
    a = np.asarray(x, dtype=float)
    vector = a.ndim <= 1
    if not vector and (a.ndim != 2 or not 1 <= a.shape[1] <= 5):
        raise ValueError('Expected values or a matrix [X,X1,...,X4]')
    if not np.all(np.isfinite(a)):
        raise ValueError('Process values and derivatives must be finite')
    columns = a.reshape(-1, 1) if vector else a
    order = columns.shape[1]-1
    base = ff['tr'] if isinstance(ff, dict) else ff
    values = columns[:, 0]
    if callable(base):
        if order and not hasattr(base, 'derivative'):
            raise ValueError('Derivative transformation needs a differentiable spline or table')
        derivatives = [np.asarray(base(values))]
        derivatives += [np.asarray(base.derivative(k)(values)) for k in range(1, order+1)]
    else:
        g = _table(base)
        outside = (values < g[0, 0]) | (values > g[-1, 0])
        if outside.any() and not extrapolate:
            raise ValueError('Values outside the transformation table')
        if not order:
            result = interpolate.PchipInterpolator(g[:, 0], g[:, 1])(np.clip(values, g[0, 0], g[-1, 0]))
            result[outside] = _linear(g[:, 0], g[:, 1], values[outside])
            return result.reshape(a.shape)
        degree = min(5, len(g)-1)
        if order > degree:
            raise ValueError('Not enough table points for requested derivative order')
        spline = interpolate.make_interp_spline(g[:, 0], g[:, 1], k=degree)
        derivatives = [spline(values, nu=k) for k in range(order+1)]
        if outside.any():
            derivatives[0][outside] = _linear(g[:, 0], g[:, 1], values[outside])
            derivatives[1][values < g[0, 0]] = (g[1, 1]-g[0, 1])/(g[1, 0]-g[0, 0])
            derivatives[1][values > g[-1, 0]] = (g[-1, 1]-g[-2, 1])/(g[-1, 0]-g[-2, 0])
            for k in range(2, order+1):
                derivatives[k][outside] = 0
    result = np.empty_like(columns)
    result[:, 0] = derivatives[0]
    if order:
        x1 = columns[:, 1]
        result[:, 1] = derivatives[1]*x1
    if order >= 2:
        x2 = columns[:, 2]
        result[:, 2] = derivatives[2]*x1*x1+derivatives[1]*x2
    if order >= 3:
        x3 = columns[:, 3]
        result[:, 3] = derivatives[3]*x1**3+3*derivatives[2]*x1*x2+derivatives[1]*x3
    if order >= 4:
        x4 = columns[:, 4]
        result[:, 4] = (derivatives[4]*x1**4+6*derivatives[3]*x1*x1*x2
                       +derivatives[2]*(3*x2*x2+4*x1*x3)+derivatives[1]*x4)
    return result.reshape(a.shape)


def _parameters(data, defaults):
    if isinstance(data, dict):
        raise NotImplementedError('Compute process moments explicitly; spectral spec2skew is not implemented here')
    parameters = np.asarray(defaults, dtype=float)
    if data is not None:
        supplied = np.asarray(data, dtype=float).ravel()
        for i, value in enumerate(supplied[:len(parameters)]):
            if not np.isnan(value):
                parameters[i] = value
    if not np.all(np.isfinite(parameters)) or parameters[0] <= 0:
        raise ValueError('Moment parameters must be finite and standard deviation positive')
    return parameters


def hermitetr(x=None, data=None, definition=1, *, full_output=False):
    """Winterstein Hermite transform, data=[sigma,skew,kurtosis,mean].

    definition=1 is the 1994 parametrization, 2 the 1988 parametrization.
    The softening polynomial is inverted on the increasing branch containing
    zero, matching the source's central-root convention. Values beyond that
    branch raise instead of switching to a remote polynomial branch.
    full_output returns (table, integral((standardized_x-g(x))**2)).
    """
    sigma, skew, kurt, mean = _parameters(data, [1, .16, 3.04, 0])
    if definition not in (1, 2) or kurt <= 0:
        raise ValueError('Require definition 1/2 and positive kurtosis')
    x = np.linspace(mean-5*sigma, mean+5*sigma, 501) if x is None or np.size(x) == 0 else np.asarray(x, dtype=float).ravel()
    if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError('x must be a finite increasing vector')
    z, excess = (x-mean)/sigma, kurt-3
    if excess < 0:
        c3, c4 = skew/6, excess/24
        values = z-c3*(z*z-1)-c4*(z**3-3*z)
    else:
        if definition == 2:
            c4 = (np.sqrt(1+1.5*excess)-1)/18
            c3 = skew/(6*(1+6*c4))
        else:
            c3 = skew/6*(1-.015*abs(skew)+.3*skew*skew)/(1+.2*excess)
            if excess == 0:
                c4 = 0.
            else:
                base = 1-1.43*skew*skew/excess
                if base < 0:
                    raise ValueError('Skew/kurtosis combination gives a non-real Hermite coefficient')
                c4 = .1*(np.cbrt(1+1.25*excess)-1)*base**(1-.1*kurt**.8)
        K = 1/np.sqrt(1+2*c3*c3+6*c4*c4)
        polynomial = np.trim_zeros(np.asarray([c4, c3, 1-3*c4, -c3])*K, 'f')
        derivative = np.polyder(polynomial)
        critical = np.roots(derivative)
        critical = critical.real[abs(critical.imag) < 1e-8]
        left = np.max(critical[critical < 0], initial=-np.inf)
        right = np.min(critical[critical > 0], initial=np.inf)
        if np.polyval(derivative, 0) <= 0:
            raise ValueError('Hermite central branch is not increasing')
        values = np.empty_like(z)
        for i, target in enumerate(z):
            coefficients = polynomial.copy(); coefficients[-1] -= target
            roots = np.roots(coefficients)
            roots = roots.real[abs(roots.imag) < 1e-8]
            roots = roots[(np.polyval(derivative, roots) > 0) & (roots > left) & (roots < right)]
            if len(roots) != 1:
                raise ValueError('Hermite inverse is not unique on an increasing branch')
            values[i] = roots[0]
    if np.any(np.diff(values) <= 0):
        raise ValueError('Hermite transform is not strictly increasing over x')
    table = np.column_stack((x, values))
    departure = float(trapezoid((z-values)**2, z))
    return (table, departure) if full_output else table


def ochitr(x=None, data=None, *, full_output=False):
    """Ochi exponential transformation, data=[sigma,skew,mean].

    Solves the three normalization equations in ochitr.m. Uses the stable
    small quadratic root for mean2, correcting that file's inconsistent root
    algebra. No bitwise equivalence to that algebraic defect is claimed.
    """
    sigma, skew, mean = _parameters(data, [1, .16, 0])
    if abs(skew) > np.sqrt(8):
        raise ValueError('Ochi skewness requires abs(skew)<=sqrt(8)')
    x = np.linspace(mean-5*sigma, mean+5*sigma, 513) if x is None or np.size(x) == 0 else np.asarray(x, dtype=float).ravel()
    if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError('x must be a finite increasing vector')
    if skew == 0:
        a, variance2, mean2 = 0., 1., 0.
    else:
        variance2 = optimize.brentq(lambda v: np.sqrt(2*(v-1))*(4-v)-abs(skew), 1, 2)
        a = np.sign(skew)*np.sqrt((variance2-1)/(2*variance2**2))
        discriminant = max(0, 1-4*a*a*variance2)
        mean2 = -2*a*variance2/(1+np.sqrt(discriminant))
    z = (x-mean)/sigma
    gamma = np.where(z >= 0, 1.28*a, 3*a)
    transformed = z.copy()
    active = gamma != 0
    transformed[active] = -np.expm1(-gamma[active]*z[active])/gamma[active]
    values = (transformed-mean2)/np.sqrt(variance2)
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
        raise ValueError('Ochi transform overflows or is not increasing on x')
    table = np.column_stack((x, values))
    departure = float(trapezoid((z-values)**2, z))
    return (table, departure) if full_output else table


_DEFAULTS = dict(chkder='on', csm=.95, gsm=.05, crossdef='uM', param=[-5, 5, 513],
                 linextrap='on', cvar=1, gvar=1, ne=7, ntr=2000,
                 plotflag='off', delay=0, multip=0)


def troptset(*updates, **kwargs):
    """Transformation options with unique case-insensitive name prefixes.

    Defaults follow troptset.m except plotting is off. Source function-name
    tokens are accepted but do not dispatch arbitrary callables.
    """
    from ..misc import parseoptions
    updates = tuple(v for v in updates if not isinstance(v, str) or v.lower() not in ('dat2tr', 'lc2tr', 'cdf2tr', 'reconstruct'))
    out = parseoptions(_DEFAULTS, *updates, **kwargs)
    for key in ('csm', 'gsm'):
        if not np.isscalar(out[key]) or not np.isfinite(out[key]) or not 0 <= out[key] <= 1:
            raise ValueError(f'{key} must lie in [0,1]')
    for key in ('ne', 'ntr'):
        if not np.isscalar(out[key]) or not np.isfinite(out[key]) or out[key] < 0 or int(out[key]) != out[key]:
            raise ValueError(f'{key} must be a nonnegative integer')
    if out['multip'] not in (0, 1):
        raise ValueError('multip must be zero or one')
    lo, hi, count = np.asarray(out['param'], dtype=float)
    if not np.all(np.isfinite([lo, hi, count])) or hi <= lo or count < 2 or int(count) != count:
        raise ValueError('param must be [minimum,maximum,integer_count>=2]')
    return out


def _enabled(value):
    return str(value).lower() not in ('0', 'off', 'none', 'false')


def _options(options):
    out = troptset({} if options is None else options)
    if _enabled(out['plotflag']):
        raise NotImplementedError('Plot returned transformation tables explicitly; estimator plotting is not implemented')
    return out


def _grid(mean, sigma, options):
    if not np.isfinite(mean) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Mean must be finite and standard deviation positive')
    lo, hi, count = options['param']
    u = np.linspace(lo, hi, int(count))
    return u, mean+sigma*u


def _variance(values, count):
    values = np.atleast_1d(np.asarray(values, dtype=float))
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('Smoothing variances must be positive finite vectors')
    return np.interp(np.linspace(0, 1, count), np.linspace(0, 1, len(values)), values)


def _finish(grid, transformed, u, empirical, options, full_output):
    if not np.all(np.isfinite(transformed)):
        raise ValueError('Nonfinite transformation estimate')
    if _enabled(options['chkder']) and np.any(np.diff(transformed) <= 0):
        raise ValueError('Estimated transform is not increasing; adjust smoothing or input distribution')
    table = np.column_stack((grid, transformed))
    departure = float(np.sqrt(trapezoid((u-transformed)**2, u)))
    return (table, departure, empirical) if full_output else table


def cdf2tr(Fx1, ma=0, sa=1, options=None, *, full_output=False):
    """Estimate g(x)=Phi^-1(F(x)) with source roughness-penalty smoothing.

    Repeated x values retain the last CDF value. Infinite Gaussian scores at
    F=0/1 are excluded. ne removes finite-score endpoints; gvar supplies the
    source smoothing variances. full_output returns (g,test,empirical_g).
    """
    from ..misc import smooth
    if isinstance(Fx1, str) and Fx1 == 'defaults':
        return troptset()
    opt = _options(options)
    F = np.asarray(Fx1, dtype=float)
    if F.ndim != 2 or F.shape[1] != 2 or len(F) < 2 or not np.all(np.isfinite(F)) or np.any(np.diff(F[:, 0]) < 0) or np.any(np.diff(F[:, 1]) < 0) or np.any((F[:, 1] < 0) | (F[:, 1] > 1)):
        raise ValueError('CDF must have nondecreasing finite x/probability columns and probabilities in [0,1]')
    keep = np.r_[np.diff(F[:, 0]) > 0, True]
    F = F[keep]
    F = F[(F[:, 1] > 0) & (F[:, 1] < 1)]
    if len(F) < 2:
        raise ValueError('At least two distinct finite Gaussian-score points are required')
    empirical = np.column_stack((F[:, 0], stats.norm.ppf(F[:, 1])))
    ne = int(opt['ne'])
    if len(F) <= 2*ne+1:
        raise ValueError('Not enough CDF points after endpoint trimming; reduce ne')
    sample = empirical[ne:len(empirical)-ne if ne else None]
    if opt['ntr'] and len(sample) > opt['ntr']:
        xx = np.linspace(sample[0, 0], sample[-1, 0], int(opt['ntr']))
        sample = np.column_stack((xx, np.interp(xx, sample[:, 0], sample[:, 1])))
    u, grid = _grid(ma, sa, opt)
    result = smooth(sample[:, 0], sample[:, 1], opt['gsm'], grid,
                    _enabled(opt['linextrap']), _variance(opt['gvar'], len(sample)))
    return _finish(grid, result, u, empirical, opt, full_output)


def lc2tr(cross, ma=0, sa=1, options=None, *, full_output=False):
    """Estimate a transform from a unimodal level-crossing intensity.

    Follows the source's two smoothings of -log(cross), signed square-root
    around its minimum, then the transformation itself. Zero intensities are
    excluded before trimming. Nonmonotone estimates raise rather than silently
    certifying an invalid transform. full_output returns (g,test,empirical_g).
    """
    from ..misc import smooth
    if isinstance(cross, str) and cross == 'defaults':
        return troptset()
    opt = _options(options)
    c = np.asarray(cross, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2 or not np.all(np.isfinite(c)) or np.any(np.diff(c[:, 0]) <= 0) or np.any(c[:, 1] < 0):
        raise ValueError('Crossing intensity must have increasing levels and nonnegative finite counts')
    c = c[c[:, 1] > 0]
    ne = int(opt['ne'])
    if len(c) <= 2*ne+2:
        raise ValueError('Not enough positive crossing points after trimming; reduce ne')
    u, grid = _grid(ma, sa, opt)
    x, logintensity = (c[:, 0]-ma)/sa, -np.log(c[:, 1])
    sample = np.column_stack((x, logintensity))[ne:len(c)-ne if ne else None]
    if opt['ntr'] and len(sample) > opt['ntr']:
        xx = np.linspace(sample[0, 0], sample[-1, 0], int(opt['ntr']))
        sample = np.column_stack((xx, np.interp(xx, sample[:, 0], sample[:, 1])))
    smoothed = smooth(sample[:, 0], sample[:, 1], opt['csm'], sample[:, 0],
                      _enabled(opt['linextrap']), _variance(opt['cvar'], len(sample)))
    center = int(np.argmin(smoothed))
    signed = np.sqrt(2*abs(smoothed-smoothed[center]))
    signed[:center] *= -1
    result = smooth(sample[:, 0], signed, opt['gsm'], u,
                    _enabled(opt['linextrap']), _variance(opt['gvar'], len(sample)))
    center2 = int(np.argmin(logintensity))
    raw = np.sqrt(2*abs(logintensity-logintensity[center2])); raw[:center2] *= -1
    empirical = np.column_stack((c[:, 0], raw))
    return _finish(grid, result, u, empirical, opt, full_output)


def dat2tr(x, definition='nonlinear', options=None, *, full_output=False):
    """Estimate transformed-Gaussian models from sampled processes.

    definition supports linear, hermite, ochitr, nonlinear/empirical crossing
    models, and mnonlinear/mempirical marginal models. Input is a value vector
    or time followed by signal columns. multip=1 fits columns separately;
    otherwise columns are pooled as one sea state, as in the source.
    full_output returns (g,test,cmax,irregularity,empirical_g).
    """
    from .. import cycles, onedim
    if isinstance(x, str) and x == 'defaults':
        return troptset()
    opt = _options(options)
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        signals = a[:, None]
    elif a.ndim == 2 and a.shape[1] >= 2:
        signals = a[:, 1:]
    else:
        raise ValueError('Expected a value vector or time/signal matrix')
    if len(signals) < 3 or not np.all(np.isfinite(signals)):
        raise ValueError('At least three finite samples are required')
    definition = str(definition).lower()[:3]
    if definition not in ('lin','her','och','non','emp','mno','mem'):
        raise ValueError('Unknown transformation estimator definition')
    columns = [signals[:, i:i+1] for i in range(signals.shape[1])] if opt['multip'] else [signals]
    transforms, empirical, tests, maxima, irregularity = [], [], [], [], []
    crossdef = {'u':1, 'uM':2, 'umM':3, 'um':4}.get(opt['crossdef'], opt['crossdef'])
    for values in columns:
        ma = float(np.mean(values))
        sa = float(np.sqrt(np.mean(np.var(values, axis=0, ddof=1))))
        u, grid = _grid(ma, sa, opt)
        counts = []
        for i in range(values.shape[1]):
            tp = onedim.dat2tp(values[:, i])
            counts.extend(cycles.tp2mm(tp).tolist())
        if counts:
            crossing = cycles.cc2lc(np.asarray(counts), crossdef)
            cmax = float(np.max(crossing[:, 1]))
            irr = len(counts)/cmax if cmax else 0.
        else:
            crossing, cmax, irr = np.empty((0, 2)), 0., 0.
        if definition == 'lin':
            g, test, g2 = np.column_stack((grid, u)), 0., np.column_stack((grid, u))
        elif definition in ('her','och'):
            data = values.ravel()
            skew = float(stats.skew(data))
            if definition == 'her':
                excess = float(stats.kurtosis(data))
                kurt = min(min(4*(4*skew/3)**2, 12), max(excess, 1.5*skew*skew))+3
                g, test = hermitetr(grid, [sa, skew, kurt, ma], full_output=True)
            else:
                g, test = ochitr(grid, [sa, skew, ma], full_output=True)
            g2 = g.copy()
        elif definition in ('mno','mem'):
            data = np.sort(values.ravel())
            cdf = np.column_stack((data, (np.arange(len(data))+.5)/len(data)))
            local = dict(opt)
            if definition == 'mem':
                local['gsm'] = 1.
            g, test, g2 = cdf2tr(cdf, ma, sa, local, full_output=True)
        else:
            if not len(crossing):
                raise ValueError('No cycles available for a crossing-based transform')
            local = dict(opt)
            if definition == 'emp':
                local.update(csm=1., gsm=1.)
            g, test, g2 = lc2tr(crossing, ma, sa, local, full_output=True)
        transforms.append(g); empirical.append(g2)
        tests.append(test); maxima.append(cmax); irregularity.append(irr)
    table = np.column_stack(transforms)
    if not full_output:
        return table
    raw = empirical[0] if len(empirical) == 1 else empirical
    return table, np.asarray(tests), np.asarray(maxima), np.asarray(irregularity), raw
