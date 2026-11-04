"""Four-point rainflow counting with WAFO residual conventions."""
from ._extrapolation import cmat2extralc, extralc, rfmextrapolate, tpextrapolate
import numpy as np
from ._remaining import (dtp2arfm_sid,dtp2rfm_sid,smoothcmat_hnorm,smoothcmat,lc2rfmextreme,wgpdfit_mld)
from ._plotting import ccplot,cmatplot,cocc,lsplot


def _rows(x):
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return np.empty((0, 1))
    return a[:, None] if a.ndim == 1 else a


def _pair(a, b, timed):
    return [a[-1], b[-1], a[0], b[0]] if timed else [a[-1], b[-1]]


def tp2arfc4p(x, res0=None, def_time=0):
    """Return (closed asymmetric cycles, residual); supports streaming."""
    a = _rows(x)
    stack = list(_rows(res0)) if res0 is not None and np.size(res0) else []
    cycles = []
    for row in a:
        stack.append(row.copy())
        while len(stack) >= 4:
            inner = sorted((stack[-3][-1], stack[-2][-1]))
            outer = sorted((stack[-4][-1], stack[-1][-1]))
            if inner[0] < outer[0] or inner[1] > outer[1]:
                break
            cycles.append(_pair(stack[-3], stack[-2], def_time))
            del stack[-3:-1]
    return (np.asarray(cycles).reshape(-1, 4 if def_time else 2),
            np.asarray(stack).reshape(-1, a.shape[1]))


def res2arfc(res, definition='up', def_time=0):
    a = _rows(res)
    definition = definition.lower()
    if definition not in ('up', 'down', 'cs'):
        raise ValueError("Residual definition must be up, down, or CS")
    if len(a) < 2:
        return np.empty((0, 4 if def_time else 2))
    if definition == 'cs':
        from ..onedim import findextrema
        twice = np.vstack((a, a))
        inds = np.r_[0, findextrema(twice[:, -1]), len(twice)-1]
        return tp2arfc4p(twice[inds], def_time=def_time)[0]
    up = a[1, -1] > a[0, -1]
    start = 0 if up == (definition == 'up') else 1
    return np.asarray([_pair(a[i], a[i+1], def_time) for i in range(start, len(a)-1, 2)]).reshape(-1, 4 if def_time else 2)


def tp2arfc(x, definition=None, ARFC0=None, res0=None, *, full_output=False):
    """Asymmetric cycles; optional (all cycles, closed cycles, residual)."""
    options = {'res': 'up', 'time': 0}
    if isinstance(definition, str):
        options['res'] = definition
    elif isinstance(definition, dict):
        options.update(definition)
    elif definition is not None:
        options['time'] = int(definition)
    a = _rows(x)
    if options['time'] and a.shape[1] == 1:
        a = np.column_stack((np.arange(len(a)), a[:, 0]))
    closed, residual = tp2arfc4p(a, res0, options['time'])
    if ARFC0 is not None and np.size(ARFC0):
        closed = np.vstack((ARFC0, closed))
    all_cycles = np.vstack((closed, res2arfc(residual, options['res'], options['time'])))
    return (all_cycles, closed, residual) if full_output else all_cycles


def tp2rfc(x, definition=None, RFC0=None, res0=None, *, full_output=False):
    """Symmetric min/max cycles; default residual counts upward excursions."""
    result = tp2arfc(x, definition, res0=res0, full_output=True)
    asymmetric = isinstance(definition, dict) and definition.get('asymmetric', False)
    converted = []
    for c in result[:2]:
        c = c.copy()
        if not asymmetric:
            swap = c[:, 0] > c[:, 1]
            c[swap, :2] = c[swap, :2][:, ::-1]
            if c.shape[1] == 4:
                c[swap, 2:] = c[swap, 2:][:, ::-1]
        if RFC0 is not None and np.size(RFC0):
            c = np.vstack((RFC0, c))
        converted.append(c)
    return (*converted, result[2]) if full_output else converted[0]


def tp2mm(tp, *, both=False):
    """Adjacent minimum-to-maximum pairs (not rainflow cycles)."""
    up = res2arfc(tp, 'up')
    return (up, res2arfc(tp, 'down')) if both else up


def levels(param):
    """Grid for [minimum, maximum, number of levels]."""
    lo, hi, n = param
    if n < 2 or int(n) != n or hi <= lo:
        raise ValueError("Invalid level parameters")
    return np.linspace(lo, hi, int(n))


def cc2dcc(param, cc, ddef=1):
    """Discrete cycle indices; classes are zero based in Python."""
    u = levels(param)
    c = np.asarray(cc, dtype=float).reshape(-1, 2)
    swap = c[:, 0] > c[:, 1]
    d = (np.sort(c, axis=1)-u[0])/(u[1]-u[0])
    n = len(u)
    if ddef == 0:
        d = np.clip(np.floor(d+.5), 0, n-1)
    elif ddef == 1:
        d[:, 0] = np.clip(np.floor(d[:, 0]), 0, n-2)
        d[:, 1] = np.clip(np.ceil(d[:, 1]), 1, n-1)
    elif ddef == -1:
        d[:, 0] = np.clip(np.ceil(d[:, 0]), 1, n-1)
        d[:, 1] = np.clip(np.floor(d[:, 1]), 0, n-2)
    else:
        raise ValueError("ddef must be -1, 0, or 1")
    d[swap] = d[swap, ::-1]
    return d.astype(int)


def dcc2cc(param, dcc):
    return levels(param)[np.asarray(dcc, dtype=int)]


def dcc2cmat(dcc, n):
    d = np.asarray(dcc)
    if d.size == 0:
        return np.zeros((n, n))
    if np.any(d != np.floor(d)) or np.any((d < 0) | (d >= n)):
        raise ValueError("Discrete indices must be integers in [0,n)")
    F = np.zeros((n, n))
    np.add.at(F, tuple(d.astype(int).T), 1)
    return F


def cc2cmat(param, cc, ddef=1, method=0):
    if method != 0:
        raise NotImplementedError("Only histogram cycle matrices are currently ported")
    return dcc2cmat(cc2dcc(param, cc, ddef), int(param[2]))


def cmat2amp(param, F):
    u = levels(param)
    return np.column_stack(((u-u[0])/2, [np.diag(F, i).sum() for i in range(len(u))]))


def cmat2nt(F, definition=1):
    if definition not in (1, 11):
        raise NotImplementedError("Only definition 1/11 is implemented in the source")
    F = np.asarray(F)
    out = np.zeros_like(F)
    out[1:, :-1] = np.cumsum(np.cumsum(F[:-1, 1:], axis=0)[:, ::-1], axis=1)[:, ::-1]
    return out


def nt2cmat(NT, definition=1):
    if definition not in (1, 11):
        raise NotImplementedError("Only definition 1/11 is implemented in the source")
    a = np.asarray(NT)
    out = np.zeros_like(a)
    out[:-1, 1:] = a[1:, :-1]-a[:-1, :-1]-a[1:, 1:]+a[:-1, 1:]
    return out


from ._extended import (cc2amp,cc2lc,tp2lc,nt2lc,cmat2lc,cmat2rmcmat,
                       rmcmat2cmat,cmatcombine,rfcfilter,dat2dtp,dtp2arfm4p,
                       dtp2arfm,dtp2rfm,dat2rfm,cmatresamp)
