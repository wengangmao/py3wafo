"""Direct complex-arithmetic port of source/disufq.c (no compiled WAFO code)."""
import numpy as np
import numbers


def disufq(A, w, kw, h, g, nmin, nmax, *, components=False):
    """Second-order sum/difference coefficients.

    nmin/nmax are ZERO-BASED INCLUSIVE frequency indices. A has simulations in
    rows and FFT coefficients in columns. components=True returns (sum,difference).
    """
    a = np.asarray(A, dtype=complex)
    vector = a.ndim == 1
    if a.ndim not in (1, 2):
        raise ValueError("A must be a vector or a simulations-by-frequency matrix")
    a = np.atleast_2d(a)
    w, k = np.asarray(w, dtype=float), np.asarray(kw, dtype=float)
    if w.ndim != 1 or k.ndim != 1:
        raise ValueError("w and kw must be one-dimensional frequency vectors")
    if (not np.isscalar(h) or not np.isscalar(g) or np.isnan(h)
            or not np.isfinite(g) or h <= 0 or g <= 0):
        raise ValueError("Depth must be positive (possibly infinite), gravity finite and positive")
    if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, numbers.Integral)
            for i in (nmin, nmax)) or not 0 <= nmin <= nmax < min(len(w), len(k), a.shape[1])):
        raise ValueError("Invalid depth, gravity, or active frequency indices")
    if 2*nmax >= a.shape[1]:
        raise ValueError("FFT buffer is too small for active sum frequencies")
    active_w, active_k = w[nmin:nmax+1], k[nmin:nmax+1]
    if (not np.all(np.isfinite(a)) or not np.all(np.isfinite(active_w))
            or not np.all(np.isfinite(active_k)) or np.any(active_w <= 0) or np.any(active_k <= 0)):
        raise ValueError("Active frequencies and wavenumbers must be positive and finite")
    if np.any(np.diff(active_w) <= 0) or np.any(np.diff(active_k) <= 0):
        raise ValueError("Active frequencies and wavenumbers must be strictly increasing")
    def divide(numerator, denominator):
        # The source contains true resonant denominators. Reject unresolved
        # singular interactions instead of silently emitting NaN/Inf values.
        if not np.isfinite(denominator) or abs(denominator) <= 32*np.finfo(float).eps:
            raise FloatingPointError("Singular or unresolved second-order interaction denominator")
        value = numerator/denominator
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite second-order interaction coefficient")
        return value
    s, d = np.zeros_like(a), np.zeros_like(a)
    for i in range(nmin, nmax+1):
        k1, w1 = k[i], w[i]
        if h > 10000:
            ep, ed = k1, 0
        else:
            th = np.tanh(k1*h)
            cg = .5*g*(th+k1*h*(1-th*th))/w1
            t1, t2 = .5*g*(k1/w1)**2, .5*w1*w1/g
            t3 = g*k1/(w1*cg)
            t4 = k1/np.sinh(2*k1*h) if k1*h < 300 else 0
            ed = divide(t1-t2+t3, 1-g*h/cg**2)-t4
            ep = divide(3*(t1-t2), 1-t1/k1*np.tanh(2*k1*h))+3*t2-t1
        s[:, 2*i] += 2*a[:, i]**2*ep
        d[:, 0] += 2*abs(a[:, i])**2*ed
        for j in range(i+1, nmax+1):
            k2, w2 = k[j], w[j]
            if h > 10000:
                ep, ed = .5*(k1+k2), -.5*(k2-k1)
            else:
                if w1 == w2:
                    raise ValueError("Active frequencies must be distinct")
                t1 = g*(k1/w1)*(k2/w2)
                t2 = .5/g*(w1*w1+w2*w2+w1*w2)
                t3 = .5*g*(w1*k2*k2+w2*k1*k1)/(w1*w2*(w1+w2))
                t4 = 1-g*(k1+k2)/(w1+w2)**2*np.tanh((k1+k2)*h)
                ep = divide(t1-t2+t3, t4)+t2-.5*t1
                t2 = .5/g*(w1*w1+w2*w2-w1*w2)
                t3 = -.5*g*(w1*k2*k2-w2*k1*k1)/(w1*w2*(w1-w2))
                t4 = 1-g*(k1-k2)/(w1-w2)**2*np.tanh((k1-k2)*h)
                ed = divide(t1-t2+t3, t4)+t2-.5*t1
            s[:, i+j] += 4*a[:, i]*a[:, j]*ep
            d[:, j-i] += 4*a[:, i]*np.conj(a[:, j])*ed
    if vector:
        s, d = s[0], d[0]
    return (s, d) if components else s+d
