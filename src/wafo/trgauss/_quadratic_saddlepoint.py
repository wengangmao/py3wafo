"""Second-order (quadratic) sea: chi-two parameters and saddlepoint crossings.

Ports dirsp2chitwo, ewwdir, chitwo2lc_sp, specq2lc (with its private eww and
kwaves), rqlf_asympt, writecov and mctp2tc. Cumulant derivatives use the
source finite-difference stencils and marching rules. Tilts outside the
finite cumulant domain end a march instead of producing complex values.
"""
from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid
from scipy.special import ndtr

from ._quadratic_moments import (holmquist1, holmquist2, holmquist3,
    holmberg1, holmberg2, holmberg3, kchitwo)
from ._quadratic_geometry import mindist
from ._transformations import createpdf


def _gravity():
    from ..misc import gravity
    return float(gravity())


def _wavenumber(w, h, g):
    from ..spec import w2k
    return w2k(w, 0, h, g)[0]


def _depth(h):
    h = 5000. if h is None or not np.isfinite(h) or h <= 0 else float(h)
    return min(h, 5000.)


def ewwdir(omega, theta, omegat, thetat, h=5000):
    """Directional quadratic transfer function E(w,th,wt,tht) (Prevosto).

    Returns the meshgrid(omega,omegat) matrix with rows indexed by omegat.
    Reduces to the long-crested eww for constant directions. Pairs with
    w*wt=0 or w+wt=0 (difference-frequency singularity) are set to zero.
    """
    w0, th0, wt0, tht0 = (np.asarray(v, dtype=float).reshape(-1) for v in (omega, theta, omegat, thetat))
    if not (len(w0) == len(wt0) == len(th0) == len(tht0)) or not len(w0):
        raise ValueError('Frequency and direction vectors must have equal positive length')
    if not all(np.all(np.isfinite(v)) for v in (w0, th0, wt0, tht0)):
        raise ValueError('Frequencies and directions must be finite')
    g, h = _gravity(), (5000. if h is None or h <= 0 else float(h))
    eps0 = 1e-6
    w, wt = np.meshgrid(w0, wt0)
    th, tht = np.meshgrid(th0, tht0)
    wpl = w+wt
    ind = abs(w*wt) < eps0
    ind1 = abs(wpl) < eps0
    wpl[ind] = 1; w[ind] = 1; wt[ind] = 1
    kw, kwt = _wavenumber(w, h, g), _wavenumber(wt, h, g)
    kwx, kwy = kw*np.cos(th), kw*np.sin(th)
    kwtx, kwty = kwt*np.cos(tht), kwt*np.sin(tht)
    kk = np.hypot(kwx+kwtx, kwy+kwty)
    kkh = g*kk*np.tanh(kk*h)
    dot = kwx*kwtx+kwy*kwty
    with np.errstate(divide='ignore', invalid='ignore'):
        D = (2*wpl*(g**2*dot-(w*wt)**2)+g**2*(kw**2*wt+kwt**2*w)-wt*w*(wt**3+w**3))/(2*w*wt*(wpl**2-kkh))
    D[ind1] = 0.
    out = (1/2/g)*(-g**2*dot/(w*wt)+w**2+wt**2+w*wt+2*wpl*D)
    out[ind] = 0.
    if not np.all(np.isfinite(out)):
        raise ValueError('Transfer function is singular for the supplied frequency pairs')
    return out


def _eww(omega, omegat, h=5000):
    """Long-crested transfer function E(w,wt), Eq. 6 of Butler et al. (2003)."""
    g, h = _gravity(), (5000. if h is None or h <= 0 else float(h))
    w, wt = np.meshgrid(np.asarray(omega, dtype=float).reshape(-1), np.asarray(omegat, dtype=float).reshape(-1))
    wpl = w+wt
    ind = abs(w*wt*wpl) < 1e-6
    wpl[ind] = 1; w[ind] = 1; wt[ind] = 1
    kw, kwt = _wavenumber(w, h, g), _wavenumber(wt, h, g)
    top = g*kw*kwt/(w*wt)-(w**2+wt**2+w*wt)/(2*g)+(g/2)*(w*kwt**2+wt*kw**2)/(w*wt*wpl)
    bottom = 1-g*(kw+kwt)/wpl**2*np.tanh((kw+kwt)*h)
    out = top/bottom-g*kw*kwt/(2*w*wt)+(w**2+wt**2+w*wt)/(2*g)
    out[ind] = 0.
    return out


def dirsp2chitwo(s=None, w=None, L0=None, L2=None, th=None, h=5000, eps=1e-5, dthdw=None,
                 *, source_scaling=False):
    """Non-central chi-two parameters (gamma,beta,S12,S22) for directional Stokes waves.

    s,w,th give the (truncated) spectrum on equally spaced vectors; L0,L2 are
    the untruncated linear moments whose excess is added to the first linear
    component. Components carrying a relative quadratic variance below eps are
    merged into one linear component. Defaults follow the source (JONSWAP,
    long-crested, depth 5000 m).

    The source symmetrizes the transfer matrices as Q+Q.T, which doubles the
    second-order terms: its narrow-band skewness is twice the Stokes value
    3*k*sigma. The default uses (Q+Q.T)/2, as specq2lc does, and reproduces
    that value; source_scaling=True keeps the literal source matrices.
    """
    if s is None:
        from ..spec import jonswap
        spectrum = jonswap()
        s, w = spectrum.S, spectrum.w
    s, w = np.asarray(s, dtype=float).reshape(-1), np.asarray(w, dtype=float).reshape(-1)
    if len(s) != len(w) or len(w) < 3 or not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)) or np.any(s < 0):
        raise ValueError('s and w must be finite equal-length vectors (at least 3 values), s nonnegative')
    th = np.zeros_like(w) if th is None else np.asarray(th, dtype=float).reshape(-1)
    if th.shape != w.shape or not np.all(np.isfinite(th)):
        raise ValueError('th must be a finite vector matching w')
    if dthdw is None:
        dthdw = (w[2]-w[1])*(th[2]-th[1])
    if dthdw < 1e-10:
        dthdw = w[2]-w[1]
    if L0 is None or L2 is None:
        # Source defaults use the w spacing, independent of the direction spacing.
        dw = w[2]-w[1]
        L0, L2 = dw*s.sum(), dw*(w**2*s).sum()
    h = _depth(h)
    root = np.sqrt(s)
    spec = np.outer(root, root)
    W = np.diag(-w)
    Em, Ep = ewwdir(w, th, -w, th, h), ewwdir(w, th, w, th, h)
    Q = (Em+Ep)*spec*dthdw
    R = (Em-Ep)*spec*dthdw
    Q, R = Q+Q.T, R+R.T
    if not source_scaling:
        Q, R = Q/2, R/2
    Delta, P1c = np.linalg.eigh(Q)
    Gama, P2c = np.linalg.eigh(R)
    P1, P2 = P1c.T, P2c.T
    N = len(w)
    gam = np.concatenate((Delta, Gama))/2
    variance = 2*np.sum(gam**2)
    sigma = np.sqrt(s*dthdw)
    bet = np.concatenate((P1@sigma, np.zeros(N)))
    Z = np.zeros((N, N))
    PP = np.block([[P1, Z], [Z, P2]])
    S12 = PP@np.block([[Z, -W], [W, Z]])@PP.T
    S22 = PP@np.block([[W**2, Z], [Z, W**2]])@PP.T
    n = 2*N
    if eps > 0 and variance > 0:
        order = np.argsort(abs(gam), kind='stable')
        gs, bs = gam[order], bet[order]
        S12s, S22s = S12[np.ix_(order, order)], S22[np.ix_(order, order)]
        mm = int(np.sum(2*np.cumsum(gs**2)/variance < eps))
        if mm > 0:
            b1 = bs[:mm]
            lead = np.sqrt(np.sum(b1**2))
            bet = np.concatenate(([lead], bs[mm:]))
            gam = np.concatenate(([0.], gs[mm:]))
            S12 = np.zeros((n-mm+1, n-mm+1)); S22 = np.zeros_like(S12)
            S22[1:, 1:] = S22s[mm:, mm:]
            S12[1:, 1:] = S12s[mm:, mm:]
            if lead > 0:
                S22[0, 0] = b1@S22s[:mm, :mm]@b1/lead**2
                S22[0, 1:] = b1@S22s[:mm, mm:]/lead
                S12[0, 1:] = b1@S12s[:mm, mm:]/lead
            S12[1:, 0] = -S12[0, 1:]
            S22[1:, 0] = S22[0, 1:]
    dL0 = L0-s.sum()*dthdw
    dL2 = L2-(s@w**2)*dthdw
    if bet[0]**2+dL0 < 0:
        raise ValueError('L0 is smaller than the variance of the truncated linear spectrum')
    bet[0] = np.sqrt(bet[0]**2+dL0)
    if dL2 > 0.001 and dL0 > 0:
        S22[0, 0] += dL2/dL0
    return gam, bet, S12, S22


def _derivatives(K, s, ds, dt):
    """Source stencils for K and its mixed derivatives at (s,0)."""
    K0, K1, K2, K3, K4 = K(s, 0), K(s-ds, 0), K(s+ds, 0), K(s-2*ds, 0), K(s+2*ds, 0)
    K1t, K2t = K(s, -dt), K(s, dt)
    K11, K22, K12t, K21t = K(s-ds, -dt), K(s+ds, dt), K(s-ds, dt), K(s+ds, -dt)
    K2tt, K1tt = K(s, 2*dt), K(s, -2*dt)
    return dict(K=K0, DK=(K2-K1)/(2*ds), DDK=(K1+K2-2*K0)/ds**2,
        DK111=(K4-2*K2+2*K1-K3)/(2*ds**3), K40=(K4-4*K2+6*K0-4*K1+K3)/ds**4,
        DK122=(K22-2*K2+K21t-K12t+2*K1-K11)/(2*ds**3),
        DDKt=(K1t+K2t-2*K0)/dt**2, DK2222=(K2tt-4*K2t+6*K0-4*K1t+K1tt)/dt**4,
        DK1122=(K22-2*K2t+K12t-2*K2+4*K0-2*K1+K21t-2*K1t+K11)/ds**4)


def _try(K, s, ds, dt):
    try:
        return _derivatives(K, s, ds, dt)
    except (ValueError, np.linalg.LinAlgError):
        return None


def _stack(rows):
    keys = rows[0].keys()
    return {key: np.asarray([row[key] for row in rows]) for key in keys}


def _lugannani_rice(S, D, small=.1):
    """Lugannani-Rice cdf.

    Near s=0 the source formula subtracts two large, finite-difference
    contaminated terms. For |w|<small the standard expansion
    Phi(w)+phi(w)*lambda3/6 replaces it; it is exact in the s=0 limit.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        wh = np.sign(S)*np.sqrt(np.maximum(2*(S*D['DK']-D['K']), 0))
        pdf = np.exp(-wh**2/2)/np.sqrt(2*np.pi)
        F = ndtr(wh)-pdf*(1/S/np.sqrt(D['DDK'])-1/wh)
    near = abs(wh) < small
    F[near] = ndtr(wh[near])+pdf[near]*D['DK111'][near]/(6*D['DDK'][near]**1.5)
    return F


def _density(S, D, title, label):
    f = np.exp(D['K']-S*D['DK'])/np.sqrt(2*np.pi*D['DDK'])
    area = trapezoid(f, D['DK'])
    if not np.isfinite(area) or area <= 0:
        raise ValueError('Saddlepoint density could not be normalized')
    fu = createpdf()
    fu.title, fu.labx[0], fu.x[0] = title, label, D['DK']
    return f, area, fu


def _corrections(D):
    eq22 = -.25*(D['DK1122']*D['DDK']-D['DK111']*D['DK122'])/(D['DDK']**2*D['DDKt'])
    eq23 = (D['DK2222']*D['DDK']-3*D['DK122']**2)/(D['DDK']*D['DDKt']**2)
    return eq22, eq23


def chitwo2lc_sp(gamma, beta=None, S12=None, S22=None, ds0=0.1):
    """Saddlepoint crossing intensity of X(t)=sum(beta*Z+gamma*Z**2).

    Returns (mu, mu2, mu1, muG, beta, cc, fu, Fu) as in the source:
    mu, mu2, mu1, muG are (level, intensity) columns for the three-term
    (with density correction R0), alternative three-term, two-term and
    one-term (Gaussian) approximations; beta is the generalized reliability
    index; cc=sqrt(K_tt)/sqrt(2 pi); fu is the Daniels density; Fu the
    Lugannani-Rice cdf. Levels are the saddlepoint means K_s and are not
    equally spaced; reduce ds0 for a finer grid.
    """
    g = np.asarray(gamma, dtype=float).reshape(-1)
    n = len(g)
    b = np.zeros(n) if beta is None else np.asarray(beta, dtype=float).reshape(-1)
    S12 = np.zeros((n, n)) if S12 is None else np.asarray(S12, dtype=float)
    S22 = np.eye(n) if S22 is None else np.asarray(S22, dtype=float)
    if ds0 is None or ds0 <= 0:
        ds0 = 0.1
    variance = 2*np.sum(g**2)
    ucrt = 5*np.sqrt(np.sum(b**2)+2*variance)
    K = lambda s, t: kchitwo(s, t, g, b, S12, S22)
    if _try(K, 0., .05*ds0, .05*ds0) is None:
        raise ValueError('Invalid chi-two parameters')

    def stop(s, D):
        x = s*D['DK']-D['K']
        return x > 25 or x < 0 or abs(D['DK']) > ucrt or D['DDK'] < 0

    positive, s, j, ds1, old = [], 0., 0, ds0, 0.
    while True:
        j += 1
        ds = .05*ds1
        D = _try(K, s, ds, ds)
        if D is None:
            break
        if abs(D['DK']-old) > .075:
            ds1 /= 2
        old = D['DK']
        if j != 1 and stop(s, D):
            break
        positive.append((s, D))
        s += ds1/2
        if j > 400:
            break
    negative, s, j, ds1, old = [], -ds0/2, 0, ds0, positive[0][1]['DK']
    ds = .25*ds1
    while True:
        j += 1
        D = _try(K, s, ds, ds)
        if D is None or (j != 1 and stop(s, D)):
            break
        if abs(D['DK']-old) > .1:
            ds1 /= 2
        old = D['DK']
        negative.append((s, D))
        s -= ds1/2
        ds = .25*ds1
        if j > 400:
            break
    rows = negative[::-1]+positive
    S = np.asarray([row[0] for row in rows])
    D = _stack([row[1] for row in rows])
    for key, value in D.items():
        D[key] = np.asarray(value)
    f, area, fu = _density(S, D, 'Density of X(0) non-central Chi^2 process', 'X(0)')
    fu.f = f/area
    reliability = np.sqrt(2*abs(S*D['DK']-D['K'])-2*np.log(area))
    Fu = np.column_stack((D['DK'], _lugannani_rice(S, D)))
    R0 = D['K40']/D['DDK']**2/8-5*D['DK111']**2/D['DDK']**3/24
    cc = np.sqrt(D['DDKt'])/np.sqrt(2*np.pi)
    muG = f/area*cc
    eq22, eq23 = _corrections(D)
    mu1 = muG*(1+eq22)
    mu = muG*(1+R0)*(1+eq22-eq23/24)
    mu2 = muG*(1+eq22-eq23/24*(1-eq22))
    x = D['DK']
    return (np.column_stack((x, mu)), np.column_stack((x, mu2)), np.column_stack((x, mu1)),
            np.column_stack((x, muG)), reliability, cc, fu, Fu)


def _kwaves(s1, s2, A22, A44, A24, CT, CC, sum1, sum2, x1, x2, x3, x4):
    """Cumulant generating function, Eq. 36 of Machado and Rychlik (2003)."""
    r = np.concatenate((s1*x2, s2*x4))+s2*CT@np.concatenate((s1*x1, s2*x3))
    B = np.eye(len(r))-np.block([[s1*A22, s2*A24], [s2*A24.T, s1*A44]])-s2**2*CC
    B = (B+B.T)/2
    root = np.linalg.cholesky(B)
    z = np.linalg.solve(root, r)
    return float(-np.log(np.diag(root)).sum()+s1**2*sum1+s2**2*sum2+z@z/2)


def specq2lc(Spec=None, ds0=0.1, nb=0, *, max_steps=20000):
    """Saddlepoint crossing intensity and density of a long-crested quadratic sea.

    Returns (mu, fu, Fu, muG, mu1, mu2). mu has columns (level, quadratic-sea
    three-term saddlepoint intensity, Gaussian Rice intensity); fu.f holds the
    normalized Daniels density and the Gaussian density; Fu is the
    Lugannani-Rice cdf. nb=1 uses the narrow-band transfer approximation.
    The source's nb=99 diagnostic printout is not reproduced.
    """
    if Spec is None:
        from ..spec import jonswap
        Spec = jonswap()
    if ds0 is None or ds0 <= 0:
        ds0 = 0.1
    from ..spec import spec2mom
    omega = np.asarray(Spec.w, dtype=float).reshape(-1)
    s = np.asarray(Spec.S, dtype=float).reshape(-1)
    if len(s) != len(omega) or len(omega) < 2 or np.any(s < 0) or not np.all(np.isfinite(s)):
        raise ValueError('Spectrum must be a finite nonnegative one-dimensional density')
    h = _depth(Spec.get('h', np.inf))
    root = np.sqrt(s)
    spec = np.outer(root, root)
    dw = omega[1]-omega[0]
    w = omega.copy()
    if nb == 1:
        w = np.full_like(omega, omega[np.argmax(s)])
    W = np.diag(-omega)
    Em, Ep = _eww(w, -w, h), _eww(w, w, h)
    Q = (Em+Ep)*spec*dw
    R = (Em-Ep)*spec*dw
    Q, R = (Q+Q.T)/2, (R+R.T)/2
    Delta, P1c = np.linalg.eigh(Q)
    Gama, P2c = np.linalg.eigh(R)
    Il = np.argsort(abs(Delta), kind='stable')
    Ill = np.argsort(abs(Gama), kind='stable')
    lam, P1 = Delta[Il], P1c.T[Il]
    gmd, P2 = Gama[Ill], P2c.T[Ill]
    N = len(omega)
    variance = 2*(np.sum(lam**2)+np.sum(gmd**2))
    if variance <= 0:
        raise ValueError('Spectrum has no quadratic component')
    mm = max(1, int(np.sum(2*(np.cumsum(lam**2)+np.cumsum(gmd**2))/variance < 1e-5)))
    lamt, gmdt = lam.copy(), gmd.copy()
    lamt[:mm] = 0; gmdt[:mm] = 0
    Dt, Gt = np.diag(lamt), np.diag(gmdt)
    sigma = np.sqrt(s*dw)
    r = np.concatenate((P1@sigma, P2@W@sigma))
    x1, x2, x3, x4 = r[:mm], r[mm:N], r[N:N+mm], r[N+mm:]
    sum1, sum2 = np.sum(x1**2)/2, np.sum(x3**2)/2
    cross = P1@W@P2.T
    Sn = Dt@cross-cross@Gt
    A22, A44, A24 = Dt[mm:, mm:], Gt[mm:, mm:], Sn[mm:, mm:]
    C = np.zeros((2*mm, 2*N-2*mm))
    C[:mm, N-mm:] = Sn[:mm, mm:]
    C[mm:, :N-mm] = Sn[mm:, :mm].T
    CT, CC = C.T, C.T@C
    K = lambda a, b: _kwaves(a, b, A22, A44, A24, CT, CC, sum1, sum2, x1, x2, x3, x4)
    ds = .01

    def march(start, step):
        rows, s, j = [], start, 0
        while True:
            j += 1
            if j > max_steps:
                raise RuntimeError('Saddlepoint march did not reach its tail criterion')
            D = _try(K, s, ds, ds)
            if D is None:
                break
            x = s*D['DK']-D['K']
            if j != 1 and (x > 25 or x < 0):
                break
            rows.append((s, D))
            s += step
        return rows

    positive = march(0., ds0/2)
    if not positive:
        raise ValueError('Cumulant generating function is undefined at zero tilt')
    rows = march(-ds0/2, -ds0/2)[::-1]+positive
    S = np.asarray([row[0] for row in rows])
    D = _stack([row[1] for row in rows])
    f, area, fu = _density(S, D, 'Density of X(0) for guadratic sea', 'X(0) [m]')
    L0 = spec2mom(Spec, 2)
    x = D['DK']
    gaussian = np.exp(-x**2/2/L0[0])/np.sqrt(2*np.pi*L0[0])
    fu.f = np.column_stack((f/area, gaussian))
    Fu = np.column_stack((x, _lugannani_rice(S, D)))
    muG = f*np.sqrt(D['DDKt'])/np.sqrt(2*np.pi)/area
    eq22, eq23 = _corrections(D)
    mu1 = muG*(1+eq22)
    mu = muG*(1+eq22-eq23/24)
    mu2 = muG*(1+eq22-eq23/24*(1-eq22))
    rice = np.exp(-x**2/2/L0[0])*np.sqrt(L0[1]/L0[0])/(2*np.pi)
    return (np.column_stack((x, mu, rice)), fu, Fu, np.column_stack((x, muG)),
            np.column_stack((x, mu1)), np.column_stack((x, mu2)))


def rqlf_asympt(u, g, b, S22, S21):
    """Breitung two-term asymptotic crossing rate of a quadratic+linear form.

    Returns (Cb0, Cb1, a0, a1): leading and correction terms with their
    amplitude factors. S21 is the source's derivative cross covariance (the
    S12 argument of chitwo2lc_sorm); V=S22+S21@S21.
    """
    g, b = np.asarray(g, dtype=float).reshape(-1), np.asarray(b, dtype=float).reshape(-1)
    if len(g) != len(b):
        raise ValueError('The lengths of b and g should be the same.')
    S22, S21 = np.asarray(S22, dtype=float), np.asarray(S21, dtype=float)
    n = len(S22)
    if S22.shape != (n, n) or S21.shape != (n, n) or n != len(g):
        raise ValueError('Covariance matrices must match the process dimension')
    xJ = np.asarray(mindist(g, b, u, 20), dtype=float).reshape(-1)
    b0 = np.linalg.norm(xJ)
    if b0 == 0:
        raise ValueError('Asymptotic expansion requires a nonzero design point')
    I = np.eye(n)
    yJ = xJ/b0
    PJ = I-np.outer(yJ, yJ)
    dg = -b0*(b+2*g*xJ)
    GJ = -2*b0**2*np.diag(g)/np.linalg.norm(dg)
    V = S22+S21@S21
    sJ2 = yJ@V@yJ
    if sJ2 <= 0:
        raise ValueError('Derivative variance at the design point must be positive')
    sJ = np.sqrt(sJ2)
    M = I+PJ@GJ@PJ
    a0 = sJ*np.sqrt(1-yJ@S21@(I+GJ)@S21@yJ/sJ2)/np.sqrt(np.linalg.det(M))
    aJ = -1/sJ*PJ@(I+GJ)@S21@yJ
    WJ1 = np.linalg.inv(M+np.outer(aJ, aJ))
    WJ2 = np.linalg.inv(M)
    vJ1 = PJ@S21@PJ@GJ@yJ
    vJ2 = PJ@GJ@yJ
    vJ3 = PJ@GJ@S21@yJ
    vJ4 = PJ@GJ@PJ@V@yJ
    KJ1 = yJ@S21@GJ@yJ
    KJ2 = yJ@GJ@yJ
    QJ1 = PJ@GJ@PJ
    QJ2 = PJ@S21@PJ@GJ@PJ
    QJ3 = QJ2+.5*KJ1*QJ1-np.outer(vJ2, vJ3)
    QJ4 = -(yJ@V@PJ@GJ@yJ)*QJ1-2*QJ1@V@np.outer(yJ, yJ)@GJ@PJ+QJ1@V@QJ1
    o = np.outer
    hq1, hq2, hq3 = holmquist1, holmquist2, holmquist3
    a11 = (.5*hq1(WJ1, QJ4)/sJ2-.5*hq1(WJ1, o(vJ4, vJ4))/sJ2**2
           + .5*hq2(WJ1, QJ3, QJ3)/sJ2+hq2(WJ1, o(aJ, vJ4), QJ3)/sJ**3
           + .5*hq2(WJ1, o(aJ, aJ), o(vJ4, vJ4))/sJ2**2
           + hq3(WJ1, o(vJ2, vJ2), QJ1, QJ1)/8-.5*hq2(WJ1, o(vJ2, vJ2), QJ1)
           - (KJ2+1)*hq2(WJ1, QJ1, QJ1)/8+.5*hq2(WJ1, o(vJ2, vJ4), QJ1)/sJ2)
    W2 = WJ2
    a12 = ((.5*holmberg1(W2, aJ, vJ1, QJ1)+holmberg1(W2, aJ, vJ2, QJ2)-.5*holmberg1(W2, aJ, vJ3, QJ1)
            + KJ1*holmberg1(W2, aJ, vJ2, QJ1)-holmberg1(W2, aJ, vJ3, o(vJ2, vJ2))
            - .5*KJ2*holmberg1(W2, aJ, vJ3, QJ1))/sJ
           - .5*holmberg2(W2, aJ, vJ2, QJ3, QJ1)/sJ-.5*holmberg2(W2, aJ, aJ, o(vJ2, vJ2), QJ1)
           - (KJ2+1)*holmberg2(W2, aJ, aJ, QJ1, QJ1)/8+holmberg3(W2, aJ, aJ, o(vJ2, vJ2), QJ1, QJ1)/8)
    product = np.linalg.det(WJ1)*a11
    if product < 0:
        raise ValueError('Second-order Breitung term is not real for this level')
    a1 = sJ*(np.sqrt(product)+np.sqrt(np.linalg.det(WJ2))*a12)
    Cb0 = a0*np.exp(-b0**2/2)/np.pi
    Cb1 = a1*np.exp(-b0**2/2)/np.pi/b0**2
    return float(Cb0), float(Cb1), float(a0), float(a1)


def writecov(cov, nr=None, directory='.'):
    """Write covariance columns [T,r,r1,...] to Cd0.in ... Cd<nr>.in.

    These ASCII spline-input files were read by the legacy minmax/wave_t
    executables. Returns the written paths.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[1] < 2 or not np.all(np.isfinite(cov)):
        raise ValueError('cov must be a finite matrix [T,r,r1,...]')
    nr = cov.shape[1]-2 if nr is None else int(nr)
    if nr < 0 or cov.shape[1] < 2+nr:
        raise ValueError(f'You must supply at least nr={nr} derivatives.')
    directory = Path(directory)
    paths = []
    for k in range(nr+1):
        path = directory/f'Cd{k}.in'
        with open(path, 'w') as stream:
            for t, value in zip(cov[:, 0], cov[:, k+1]):
                stream.write(f'{t:12.10f} {value:12.10E} {0:4.2f} {0:4.2f} {0:4.2f}\n')
        paths.append(path)
    return paths


def _normalize_rows(A):
    A = A.copy()
    total = A.sum(axis=1)
    nonzero = total != 0
    A[nonzero] /= total[nonzero, None]
    return A


def _flipped_cmat2nt(F):
    """WAFO 1 counts NT(i,j)=sum(F(1:i-1,1:j-1)) of a column-flipped matrix."""
    NT = np.zeros_like(F)
    NT[1:, 1:] = np.cumsum(np.cumsum(F[:-1, :-1], axis=0), axis=1)
    return NT


def _flipped_nt2cmat(NT):
    F = np.zeros_like(NT)
    F[:-1, :-1] = NT[1:, 1:]-NT[:-1, 1:]-NT[1:, :-1]+NT[:-1, :-1]
    return F


def mctp2tc(freqPVR, utc, param, freqPVL=None):
    """Upcrossing trough-crest frequency matrix from a Markov chain of turning points.

    freqPVR is the max2min frequency matrix, freqPVL the min2max matrix
    (default freqPVR), both on the levels(param) grid in the source's
    column-flipped layout (callers pass fliplr of a min-max matrix and flip
    the result back). utc is the reference level. Indexing follows the
    source, with MATLAB ranges translated, apart from three repairs, each
    validated against simulated turning-point chains:

    * The source was written for the WAFO 1 counting convention on
      column-flipped matrices; WAFO 2's cmat2nt drops the highest crest level
      and misplaces the excursion corrections, so the legacy convention is
      used.
    * The source counts levels from the top to locate utc but then indexes
      levels from the bottom, which is only consistent for the centre level of
      a symmetric grid. utc is located by its natural grid index instead.
    * A utc between grid levels gets an inserted zero-frequency level, which
      the chain never visits; it is removed from the result.
    """
    from ..cycles import levels
    R = np.asarray(freqPVR, dtype=float)
    L = R.copy() if freqPVL is None else np.asarray(freqPVL, dtype=float)
    u = levels(param)
    n = len(R)
    if R.shape != (n, n) or L.shape != (n, n) or n != len(u):
        raise ValueError('Frequency matrices must be square on the levels(param) grid')
    if not np.isfinite(utc) or not u[0] < utc < u[-1]:
        raise ValueError('the reference level out of range')
    on_grid = np.isclose(u, utc, rtol=0, atol=1e-10*(u[-1]-u[0]))
    if on_grid.any():
        return _mctp2tc(R, L, int(np.flatnonzero(on_grid)[0])+1)
    k = int(np.searchsorted(u, utc))
    # Natural row/column k becomes flipped column n-k of the enlarged matrix.
    pad = lambda A: np.insert(np.insert(A, k, 0., axis=0), n-k, 0., axis=1)
    F = _mctp2tc(pad(R), pad(L), k+1)
    return np.delete(np.delete(F, k, axis=0), n-k, axis=1)


def _mctp2tc(R, L, ntc):
    """Source mctp2tc core; ntc is the 1-based natural index of the utc level."""
    n = len(R)
    if ntc < 2 or ntc > n-1:
        raise ValueError('the reference level out of range')
    P = _normalize_rows(R)[:, ::-1]
    Ph = _normalize_rows(np.rot90(L[:, ::-1], -1))[:, ::-1]
    F = np.zeros((n, n))
    F[:ntc-1, :n-ntc] = L[:ntc-1, :n-ntc]
    F = _flipped_cmat2nt(F)
    tempp = tempm = None
    for i in range(2, ntc+1):
        for j in range(ntc, n):
            if i < ntc:
                Ap, Bp = P[i-1:ntc-1, i:ntc], Ph[i:ntc, i-1:ntc-1]
                e = Ph[i:ntc, 0] if i == 2 else Ph[i:ntc, :i-1].sum(axis=1)
                tempp = np.zeros(ntc-i)
                if np.max(abs(e)) > 1e-10:
                    tempp = Ap@np.linalg.solve(np.eye(len(Ap))-Bp@Ap, e)
            if j > ntc:
                Am, Bm = P[ntc-1:j-1, ntc:j], Ph[ntc:j, ntc-1:j-1]
                em = P[ntc-1:j-1, n-1] if j == n-1 else P[ntc-1:j-1, j:n].sum(axis=1)
                tempm = np.zeros(j-ntc)
                if np.max(abs(em)) > 1e-10:
                    tempm = Bm@np.linalg.solve(np.eye(len(Am))-Am@Bm, em)
            columns = np.arange(n-ntc-1, n-j-1, -1)
            col = n-j
            if j > ntc and i < ntc:
                F[i-1, col] += tempp@L[np.ix_(range(i-1, ntc-1), columns)]@tempm
                F[i-1, col] += tempp@L[i-1:ntc-1, :n-j].sum(axis=1)
                F[i-1, col] += L[np.ix_(range(i-1), columns)].sum(axis=0)@tempm
            if j == ntc and i < ntc:
                F[i-1, col] += tempp@L[i-1:ntc-1, :n-j].sum(axis=1)
                F[i-1, n-ntc:] = F[i-1, n-ntc]
            if j > ntc and i == ntc:
                F[i-1, col] += L[np.ix_(range(i-1), columns)].sum(axis=0)@tempm
                F[ntc-1:, col] = F[ntc-1, col]
    F = _flipped_nt2cmat(F)
    # Differencing the counts leaves rounding-level negatives.
    F[(F < 0) & (F > -1e-12*max(np.max(abs(F)), np.finfo(float).tiny))] = 0.
    return F
