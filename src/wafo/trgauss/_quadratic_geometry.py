"""Closest-point and FORM/SORM geometry of Gaussian quadratic surfaces."""
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import brentq
from ._quadratic_moments import _covariance


def _parameters(g, b, u, tolerance):
    g, b = np.asarray(g, dtype=float).reshape(-1), np.asarray(b, dtype=float).reshape(-1)
    if not len(g) or g.shape != b.shape or not np.all(np.isfinite(g)) or not np.all(np.isfinite(b)):
        raise ValueError('g and b must be finite vectors of equal positive length')
    if np.ndim(u) or not np.isfinite(u) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('u must be finite and tolerance positive')
    return g, b


def mindist3(g, b, u, *, tolerance=1e-12, full_output=False):
    """Globally closest point on sum(g*x**2+b*x)=u.

    Solves the monotone secular equation on the globally optimal multiplier
    branch, including zero linear coefficients and repeated largest diagonal
    values. This avoids the source high-degree polynomial roots and their
    unstable ordering. Negative levels are solved by sign reversal. At a hard
    case with several equally close points, a reproducible positive component
    is chosen in the first extremal coordinate. full_output returns (point,
    multiplicity_mask), preserving the meaning of the source's second output.
    Infeasible levels raise ValueError rather than returning an infinite point.
    """
    g, b = _parameters(g, b, u, tolerance)
    x, multiplicity = np.zeros(len(g)), np.zeros(len(g), dtype=int)
    if u == 0:
        return (x, multiplicity) if full_output else x
    sign = 1 if u > 0 else -1
    g, b, target = sign*g, sign*b, abs(float(u))
    edge = max(0., float(g.max()))
    extreme = g == edge
    regular = ~extreme
    endpoint = np.zeros(len(g))
    endpoint[regular] = .5*b[regular]/(edge-g[regular])
    unbounded = np.any(b[extreme] != 0)
    limit = np.inf if unbounded else float(np.sum(g*endpoint**2+b*endpoint))
    slack = tolerance*max(target, abs(limit) if np.isfinite(limit) else target, np.finfo(float).tiny)
    if not unbounded and target >= limit-slack:
        remaining = target-limit
        if remaining <= slack:
            x = endpoint
        elif edge > 0:
            x = endpoint
            chosen = np.flatnonzero(extreme)[0]
            x[chosen] = np.sqrt(remaining/edge)
            multiplicity[extreme] = 1
        else:
            raise ValueError('Requested level is outside the range of the quadratic form')
    else:
        def residual(delta):
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                candidate = np.divide(.5*b, delta+(edge-g), out=np.zeros_like(b), where=b != 0)
                value = np.sum(g*candidate*candidate+b*candidate)-target
            return float(value) if np.isfinite(value) else np.inf
        lower = np.nextafter(0., np.inf)
        upper = max(np.max(abs(g)), np.linalg.norm(b)/np.sqrt(target), np.finfo(float).tiny)
        for _ in range(1024):
            if residual(upper) < 0:
                break
            upper *= 2
            if not np.isfinite(upper):
                raise FloatingPointError('Could not bracket a finite quadratic-surface multiplier')
        else:
            raise RuntimeError('Could not bracket the closest-point secular root')
        delta = brentq(residual, lower, upper, xtol=np.nextafter(0., np.inf),
                            rtol=max(4*np.finfo(float).eps, tolerance*.01), maxiter=1024)
        x = np.divide(.5*b, delta+(edge-g), out=np.zeros_like(b), where=b != 0)
    # Check the physical constraint rather than trusting a solver termination.
    actual = float(np.sum(g*x*x+b*x))
    if not np.isfinite(actual) or abs(actual-target) > 50*tolerance*max(target, np.finfo(float).tiny):
        raise RuntimeError('Closest-point solve did not satisfy the requested surface tolerance')
    return (x, multiplicity) if full_output else x


def mindist(g, b, u, n0=10, epsi=1e-12):
    """Closest quadratic-surface point using the global secular solver.

    The source n0 initial-coordinate count is accepted for call compatibility;
    all coordinates now enter the exact global solve, so it does not select a
    local starting point. epsi controls the physical constraint tolerance.
    """
    if isinstance(n0, (bool, np.bool_)) or np.ndim(n0) or not np.isfinite(n0) or int(n0) != n0 or n0 < 1:
        raise ValueError('n0 must be a positive integer')
    return mindist3(g, b, u, tolerance=epsi)


def chitwo2lc_sorm(u, g, b, S22, S12, ass=0, n0=10, epsi=1e-9, *, full_output=False):
    """FORM (ass<0) or first-order SORM (ass=0) quadratic crossing intensity.

    X=sum(g*Z**2+b*Z), with Cov(Z)=I and stationary derivative covariances
    S12=Cov(Z,Z'), S22=Cov(Z'). Returns columns (level,intensity). Uses the
    source closest-point formula, evaluated with a tangent-space determinant
    instead of unstable cofactor products. This is an asymptotic single-design-
    point approximation, not an exact crossing integral. Nonunique closest
    points, focal/indefinite curvature, and zero non-Gaussian levels explicitly
    reject that approximation. Linear Gaussian levels, including zero, are
    exact. Positive ass returns the source mean of Breitung's two asymptotic
    terms (rqlf_asympt). full_output adds closest points and distances.
    """
    levels = np.atleast_1d(np.asarray(u, dtype=float))
    if levels.ndim != 1 or not len(levels) or not np.all(np.isfinite(levels)):
        raise ValueError('u must be a finite nonempty level vector')
    g, b = _parameters(g, b, levels[0], epsi)
    if np.ndim(ass) or not np.isfinite(ass):
        raise ValueError('ass must be finite')
    if ass > 0:
        # Source: mean of Breitung's leading and corrected terms (rqlf_asympt).
        from ._quadratic_saddlepoint import rqlf_asympt
        terms = np.asarray([rqlf_asympt(level, g, b, S22, S12)[:2] for level in levels])
        result = np.column_stack((levels, .5*terms.sum(axis=1)))
        points = np.column_stack([mindist(g, b, level, 20) for level in levels])
        details = dict(points=points, distances=np.linalg.norm(points, axis=0), method='Breitung', terms=terms)
        return (result, details) if full_output else result
    n = len(g)
    derivative = _covariance(S22)
    cross = np.asarray(S12, dtype=float)
    if derivative.shape != (n, n) or cross.shape != (n, n) or not np.all(np.isfinite(cross)) or not np.allclose(cross+cross.T, 0, atol=1e-11):
        raise ValueError('S12 must be finite skew-symmetric and covariances must match the process dimension')
    _covariance(derivative-cross.T@cross)
    if np.any(levels == 0) and np.any(g):
        raise ValueError('Zero levels require a separate non-Gaussian crossing approximation')
    if not np.any(g):
        variance = b@b
        if variance <= 0:
            raise ValueError('Constant process has no regular Gaussian crossing law')
        intensity = np.sqrt((b@derivative@b)/variance)/(2*np.pi)*np.exp(-levels**2/(2*variance))
        points = b[:, None]*levels[None, :]/variance
    else:
        points = np.empty((n, len(levels))); intensity = np.empty(len(levels))
        for j, level in enumerate(levels):
            point, multiplicity = mindist3(g, b, level, tolerance=epsi, full_output=True)
            if np.any(multiplicity):
                raise ValueError('Single-design-point FORM/SORM is invalid for nonunique closest points')
            distance = np.linalg.norm(point)
            gradient = b+2*g*point
            gradient_length = np.linalg.norm(gradient)
            if gradient_length == 0:
                raise ValueError('Crossing surface has a stationary gradient at its closest point')
            normal = point/distance
            velocity_variance = float(normal@derivative@normal)
            logdet = 0.
            if ass == 0:
                curvature = -2*distance*g/gradient_length*np.sign(level)
                tangent = null_space(normal[None, :])
                metric = tangent.T@np.diag(1+curvature)@tangent
                if len(metric):
                    try:
                        root = np.linalg.cholesky(metric)
                    except np.linalg.LinAlgError as error:
                        raise ValueError('SORM tangent curvature is not positive definite') from error
                    logdet = 2*np.log(np.diag(root)).sum()
                velocity_variance += curvature@((cross.T@normal)**2)
            if velocity_variance < 0:
                raise ValueError('SORM derivative correction has negative variance')
            intensity[j] = np.exp(-.5*distance**2-.5*logdet)*np.sqrt(velocity_variance)/(2*np.pi)
            points[:, j] = point
    result = np.column_stack((levels, intensity))
    details = dict(points=points, distances=np.linalg.norm(points, axis=0), method='SORM' if ass == 0 else 'FORM')
    return (result, details) if full_output else result
