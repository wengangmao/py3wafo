"""Gaussian quadratic moments and quadratic-process cumulant generators.

Holmquist trace formulas follow the original source. Holmberg moments use an
equivalent exact mixed-derivative calculation in a truncated polynomial ring,
replacing the source's very long expanded expressions without differencing.
"""
import math
import numpy as np


def _covariance(S):
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or not len(S) or not np.all(np.isfinite(S)):
        raise ValueError('Expected a finite nonempty square covariance matrix')
    S = (S+S.T)/2
    scale = max(np.max(abs(S)), np.finfo(float).tiny)
    if np.linalg.eigvalsh(S).min() < -1e-11*scale:
        raise ValueError('Covariance must be positive semidefinite')
    return S


def _quadratics(S, *matrices):
    result = []
    for matrix in matrices:
        a = np.asarray(matrix, dtype=float)
        if a.shape != S.shape or not np.all(np.isfinite(a)):
            raise ValueError('Quadratic forms must be finite matrices matching the covariance')
        result.append((a+a.T)/2)
    return result


def holmquist1(S, Q1):
    """E[X.T Q1 X] for centered Gaussian X with covariance S."""
    S = _covariance(S)
    Q1, = _quadratics(S, Q1)
    return float(np.trace(Q1@S))


def holmquist2(S, Q1, Q2):
    """E[(X.T Q1 X)(X.T Q2 X)] for centered Gaussian X."""
    S = _covariance(S)
    Q1, Q2 = _quadratics(S, Q1, Q2)
    A, B = Q1@S, Q2@S
    return float(2*np.trace(A@B)+np.trace(A)*np.trace(B))


def holmquist3(S, Q1, Q2, Q3):
    """E[product(X.T Qj X,j=1..3)] including singular covariances."""
    S = _covariance(S)
    Q1, Q2, Q3 = _quadratics(S, Q1, Q2, Q3)
    A, B, C = Q1@S, Q2@S, Q3@S
    return float(8*np.trace(A@B@C)+2*np.trace(A@B)*np.trace(C)
                 +2*np.trace(A@C)*np.trace(B)+2*np.trace(B@C)*np.trace(A)
                 +np.trace(A)*np.trace(B)*np.trace(C))


def _product(p, q):
    """Subset convolution with each formal variable squared equal to zero."""
    output = np.zeros_like(p)
    for mask in range(len(p)):
        subset = mask
        while True:
            output[mask] += p[subset]*q[mask ^ subset]
            if not subset:
                break
            subset = (subset-1) & mask
    return output


def _series(p, coefficients):
    power = np.zeros_like(p); power[0] = 1
    result = np.zeros_like(p)
    for coefficient in coefficients:
        result += coefficient*power
        power = _product(power, p)
    return result


def _holmberg(S, a, b, *matrices):
    # For Q=sum(tj Qj), Gaussian exponential tilting gives exactly
    # det(I-2SQ)^(-1/2) * a.T C(Q)b / sqrt(1+a.T C(Q)a),
    # C(Q)=(I-2SQ)^(-1)S. Its mixed coefficient is the desired moment.
    S = _covariance(S)
    matrices = _quadratics(S, *matrices)
    a, b = np.asarray(a, dtype=float).reshape(-1), np.asarray(b, dtype=float).reshape(-1)
    if a.shape != (len(S),) or b.shape != a.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('a and b must be finite vectors matching the covariance')
    k = len(matrices); size = 1 << k
    covariance = [S]
    logdet = np.zeros(size)
    for mask in range(1, size):
        value = np.zeros_like(S)
        for j in range(k):
            if mask & (1 << j):
                value += 2*S@matrices[j]@covariance[mask ^ (1 << j)]
        covariance.append(value)
        j = (mask & -mask).bit_length()-1
        logdet[mask] = np.trace(matrices[j]@covariance[mask ^ (1 << j)])
    determinant = _series(logdet, [1/math.factorial(j) for j in range(k+1)])
    numerator = np.asarray([a@C@b for C in covariance])
    variance = np.asarray([a@C@a for C in covariance])
    constant = 1+variance[0]
    variance[0] = 0
    # Binomial coefficients for (1+z)^(-1/2), through order three.
    inverse_root = _series(variance/constant, [1, -.5, .375, -.3125][:k+1])/np.sqrt(constant)
    return float(_product(_product(determinant, numerator), inverse_root)[-1])


def holmberg1(S, a, b, Q1):
    """sqrt(pi/2) E[(b.T X)(X.T Q1 X)(2 Phi(a.T X)-1)]."""
    return _holmberg(S, a, b, Q1)


def holmberg2(S, a, b, Q1, Q2):
    """Gaussian sign-weighted product of two quadratic forms; exact analytic moment."""
    return _holmberg(S, a, b, Q1, Q2)


def holmberg3(S, a, b, Q1, Q2, Q3):
    """Gaussian sign-weighted product of three quadratic forms; no finite differences."""
    return _holmberg(S, a, b, Q1, Q2, Q3)


def kchitwo(s, t, gam, bet, S12, S22):
    """Joint cumulant generator of a quadratic Gaussian process and derivative.

    X=sum(gam*Z**2+bet*Z), X'=sum((2*gam*Z+bet)*Z'),
    Cov(Z)=I, Cov(Z,Z')=S12, Cov(Z')=S22. s,t are real scalars.
    Reject invalid covariance and tilts outside the finite moment domain.
    Uses Cholesky solves/log-determinants instead of inverse/det products.
    """
    g, b = np.asarray(gam, dtype=float).reshape(-1), np.asarray(bet, dtype=float).reshape(-1)
    if not len(g) or b.shape != g.shape or not np.all(np.isfinite(g)) or not np.all(np.isfinite(b)):
        raise ValueError('gam and bet must be finite vectors of equal positive length')
    if any(np.ndim(v) or not np.isfinite(v) for v in (s, t)):
        raise ValueError('s and t must be finite real scalars')
    n = len(g); I = np.eye(n); G = np.diag(g)
    cross = np.asarray(S12, dtype=float)
    derivative = np.asarray(S22, dtype=float)
    if cross.shape != (n, n) or derivative.shape != (n, n) or not np.all(np.isfinite(cross)) or not np.all(np.isfinite(derivative)) or not np.allclose(derivative, derivative.T):
        raise ValueError('Derivative covariance matrices must be finite and shape-compatible; S22 symmetric')
    # Judge rounding against S22: V is exactly zero for random-phase models.
    V = derivative-cross.T@cross
    V = (V+V.T)/2
    if np.linalg.eigvalsh(V).min() < -1e-11*max(np.max(abs(derivative)), np.finfo(float).tiny):
        raise ValueError('Covariance must be positive semidefinite')
    A = I-2*s*G-2*t*(G@cross.T+cross@G)-4*t*t*G@V@G
    T = (s*I+t*cross+2*t*t*G@V)@b
    try:
        root = np.linalg.cholesky(A)
    except np.linalg.LinAlgError as error:
        raise ValueError('Tilt lies outside the finite cumulant-generating domain') from error
    z = np.linalg.solve(root, T)
    return float(-np.log(np.diag(root)).sum()+t*t*(b@V@b)/2+(z@z)/2)


def rfadd(R1, R2):
    """Add rational functions stored as [numerator;denominator] coefficient rows.

    Coefficients are ascending powers, as in the source help and mindist3.
    Common initial zero factors are removed to preserve source behavior. No
    cancellation of other polynomial factors is attempted.
    """
    values = []
    for R in (R1, R2):
        R = np.asarray(R, dtype=float)
        if R.ndim != 2 or R.shape[0] != 2 or not R.shape[1] or not np.all(np.isfinite(R)) or not np.any(R[1]):
            raise ValueError('Rational functions need finite 2 by N coefficient arrays and nonzero denominators')
        values.append(R)
    A, B = values
    output = np.vstack((np.convolve(A[0], B[1])+np.convolve(B[0], A[1]), np.convolve(A[1], B[1])))
    start = np.flatnonzero(np.any(output != 0, axis=0))[0]
    return output[:, start:]
