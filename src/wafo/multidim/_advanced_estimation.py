"""Fourier EMEM and transfer calibration from the original private routines.

EMEM fits a normalized exponential *Fourier* density to sensor transfer
moments and selects its order using the source AIC. This is a different model
from MEM's exponential family generated directly by transfer constraints.
Original GUI loops and unchecked Newton steps are replaced by bounded,
backtracked least-squares Newton steps with explicit fit diagnostics.
"""
import numbers
import warnings
import numpy as np
from scipy.integrate import simpson
from scipy.special import logsumexp


def memfun(x, P1, P2, P3, P4=None):
    """Source MEM objective: sum(abs(integral P2*exp(P3@x) dtheta)).

    Uses the MATLAB helper's Simpson integration and unnormalized exponential;
    it is not the normalized MEM dual objective. P1 is the angular coordinate,
    P2 has one column per moment equation, and P3 has one column per coefficient.
    P4 optionally checks the number of P2 columns. Complex moment residuals are
    supported. Exponential scaling avoids overflow; a mathematically too-large
    objective returns infinity. Invalid inputs are rejected instead of using
    the source's dimension-dependent NaN penalty ``y(ind)=x(ind)*100``.
    """
    theta = np.asarray(P1, dtype=float).reshape(-1)
    coef = np.asarray(x, dtype=float).reshape(-1)
    features = np.asarray(P3, dtype=float)
    residuals = np.asarray(P2)
    if residuals.ndim == 1:
        residuals = residuals[:, None]
    if (len(theta) < 2 or np.any(np.diff(theta) <= 0) or np.any(~np.isfinite(theta))
            or features.shape != (len(theta), len(coef)) or residuals.ndim != 2
            or residuals.shape[0] != len(theta) or np.any(~np.isfinite(coef))
            or np.any(~np.isfinite(features)) or np.any(~np.isfinite(residuals))):
        raise ValueError('Expected increasing angles, finite coefficients, and compatible P2/P3 matrices')
    if P4 is not None and (not isinstance(P4, numbers.Integral) or P4 != residuals.shape[1]):
        raise ValueError('P4 must equal the number of moment equations in P2')
    exponent = features@coef
    shift = float(np.max(exponent))
    integrated = simpson(residuals*np.exp(exponent-shift)[:, None], x=theta, axis=0)
    magnitude = float(np.sum(abs(integrated)))
    if magnitude == 0:
        return 0.
    logarithm = shift+np.log(magnitude)
    return np.inf if logarithm > np.log(np.finfo(float).max) else float(np.exp(logarithm))


def hwestimate(Sf, SfBest, Hw, pos, *, strict=True, full_output=False):
    """Estimate common sensor transfer amplitudes from their autospectra.

    This is an autospectral ratio estimator, not a wavelet transform. Independent
    sensor types 2,3,9,12,15,18 use sqrt(mean(Sf)/SfBest); orthogonal pairs
    (4,5),(10,11),(13,14),(16,17) use their summed autospectra. Curvatures
    (6,7,8) use xx+yy+2*xy. Matching is by occurrence order, as in the source;
    each family is assumed to share its frequency response and use orthogonal
    axes. All family members receive the pooled estimate, including unpaired
    extras. Elevation type 1 and zero-SfBest frequencies retain the supplied Hw.

    Fixes the source's undefined curvature lengths, active-column indexing,
    and omitted acceleration pair 13/14. Missing required members raise by
    default; strict=False warns and leaves that family unchanged.
    """
    sf, best, hw, position = (np.asarray(v, dtype=float) for v in (Sf, SfBest, Hw, pos))
    if (sf.ndim != 2 or best.shape != (sf.shape[1],) or hw.shape != sf.shape
            or position.ndim != 2 or position.shape[0] != len(sf) or position.shape[1] not in (4, 5)
            or any(np.any(~np.isfinite(v)) for v in (sf, best, hw, position))
            or np.any(sf < 0) or np.any(best < 0) or np.any(hw < 0)):
        raise ValueError('Expected nonnegative Sf/Hw=(M,Nf), SfBest=(Nf,), pos=(M,4 or5)')
    identifiers = position[:, 3]
    if np.any(identifiers != np.floor(identifiers)) or np.any((identifiers < 1) | (identifiers > 18)):
        raise ValueError('Sensor identifiers must be integers 1 through 18')
    if not isinstance(strict, (bool, np.bool_)):
        raise ValueError('strict must be a boolean')
    result = hw.copy()
    active = np.flatnonzero(best > 0)
    estimated = np.zeros_like(hw, dtype=bool)
    families, missing = [], []
    if len(active):
        for types in ((2,), (3,), (9,), (12,), (15,), (18,),
                      (4, 5), (10, 11), (13, 14), (16, 17), (6, 7, 8)):
            members = [np.flatnonzero(identifiers == kind) for kind in types]
            if not any(len(group) for group in members):
                continue
            if any(not len(group) for group in members):
                missing.append(types)
                if strict:
                    raise ValueError('Transfer estimation requires complete sensor family '+str(types))
                warnings.warn('Incomplete sensor transfer family '+str(types)+' left unchanged', RuntimeWarning, stacklevel=2)
                continue
            count = min(map(len, members))
            total = np.zeros(len(active))
            for j, group in enumerate(members):
                weight = 2 if len(types) == 3 and j == 2 else 1
                total += weight*sf[np.ix_(group[:count], active)].mean(axis=0)
            amplitude = np.sqrt(total/best[active])
            rows = np.concatenate(members)
            result[np.ix_(rows, active)] = amplitude
            estimated[np.ix_(rows, active)] = True
            families.append(dict(sensor_types=types, matched_observations=count, sensors=rows.copy()))
    info = dict(estimated=estimated, active_frequencies=active, families=families,
                missing_families=missing, assumption='common response within each orthogonal sensor family')
    return (result, info) if full_output else result


def _constraints(S, G, weights):
    features, targets = [], []
    constant_residual = 0.
    for i in range(len(S)):
        for j in range(i, len(S)):
            product = G[i]*G[j].conj()
            for component in (np.real, np.imag):
                feature, target = component(product), float(component(S[i, j]))
                if np.ptp(feature) > np.sqrt(np.finfo(float).eps):
                    features.append(feature)
                    targets.append(target)
                else:
                    constant = float(weights@feature/(2*np.pi))
                    constant_residual = max(constant_residual, abs(target-constant))
    return np.asarray(features).reshape(-1, len(weights)).T, np.asarray(targets), constant_residual


def _fit_fourier(H, target, weights, B, options):
    """Damped source Newton step using the exact transfer/Fourier covariance."""
    tolerance = float(options['errortol'])/2
    logweights = np.log(weights)
    coef = np.zeros(B.shape[1])
    maxcoef = float(options['maxcoef'])
    def evaluate(c):
        exponent = B@c
        logZ = logsumexp(logweights+exponent)
        density = np.exp(exponent-logZ)
        probability = weights*density
        moment, basis_moment = probability@H, probability@B
        residual = moment-target
        jacobian = H.T@(probability[:, None]*B)-np.outer(moment, basis_moment)
        return density, residual, jacobian
    D, residual, jacobian = evaluate(coef)
    reason = 'iteration limit'
    iteration = 0
    for iteration in range(1, int(options['maxiter'])+1):
        if np.max(abs(residual), initial=0.) <= tolerance:
            reason = 'moment tolerance'
            iteration -= 1
            break
        delta = np.linalg.lstsq(jacobian, -residual, rcond=1e-12)[0]
        objective = float(residual@residual)
        step = float(options['relax'])
        accepted = False
        for _ in range(24):
            trial = np.clip(coef+step*delta, -maxcoef, maxcoef)
            if np.all(np.isfinite(trial)):
                trial_D, trial_residual, trial_jacobian = evaluate(trial)
                if float(trial_residual@trial_residual) < objective or np.max(abs(trial_residual), initial=0.) <= tolerance:
                    accepted = True
                    break
            step /= 2
        if not accepted:
            reason = 'no decreasing Newton step'
            break
        change = float(np.max(abs(trial-coef), initial=0.))
        coef, D, residual, jacobian = trial, trial_D, trial_residual, trial_jacobian
        if change <= options['coefabstol']:
            reason = 'coefficient tolerance'
            break
    error = float(np.max(abs(residual), initial=0.))
    return D, dict(coefficients=coef, moment_residual=error, residual_vector=residual,
                   iterations=iteration, converged=error <= options['errortol'], reason=reason)


def emem(Sxyn, Gwt, theta, fi=None, k=None, opt=None, *, full_output=False):
    """Extended maximum entropy: Fourier model fitted to transfer moments.

    D(theta)=exp(sum(a_n*cos(n*theta)+b_n*sin(n*theta)))/Z. Each order fits
    real/imaginary Gi*conj(Gj) moments by a bounded, damped Newton least-squares
    solve, then uses the source score M*(log(2*pi*var(residual))+1)+4*N+2.
    Residual variance is floored at errortol**2, so model selection does not
    chase roundoff below the requested physical accuracy. The first AIC
    improvement smaller than aictol stops order growth, as in the source.

    Complete increasing 2*pi angle grids are required; trapezoidal weights
    also permit nonuniform spacing. k uses zero-based active frequency indices.
    maxmodelorder defaults to floor(M/2+1), capped by angular identifiability;
    minmodelorder is also bounded by half the preceding selected order.
    maxcoef bounds coefficient growth. maxiter is a total budget per order.

    full_output adds per-frequency convergence, physical cross-spectrum and
    moment residuals, Fourier coefficients, model orders, and every AIC trial.
    Infeasible/noisy moments return their best fitted model with converged=False;
    without full_output a warning reports such fits. No MEM/MLM fallback occurs.
    """
    from ._estimation import _inputs, specoptset, getcrossspectra
    S, G, theta, weights, active = _inputs(Sxyn, Gwt, theta, k)
    options = specoptset(opt or {})
    for name in ('maxiter', 'minmodelorder'):
        if not isinstance(options[name], numbers.Integral) or options[name] < 1:
            raise ValueError(name+' must be a positive integer')
    for name in ('coefabstol', 'errortol', 'relax'):
        if not np.isfinite(options[name]) or options[name] <= 0:
            raise ValueError(name+' must be positive and finite')
    if np.isnan(options['maxcoef']) or options['maxcoef'] <= 0:
        raise ValueError('maxcoef must be positive')
    if not np.isfinite(options['aictol']) or options['aictol'] < 0:
        raise ValueError('aictol must be finite and nonnegative')
    nf = S.shape[2]
    frequency = np.arange(nf) if fi is None else np.asarray(fi, dtype=float)
    if frequency.shape != (nf,) or np.any(~np.isfinite(frequency)):
        raise ValueError('fi must be a finite frequency vector matching the input')
    angular_limit = (len(theta)-2)//2
    supplied_max = options['maxmodelorder']
    if supplied_max is not None and (not isinstance(supplied_max, numbers.Integral)
            or not options['minmodelorder'] <= supplied_max <= angular_limit):
        raise ValueError('maxmodelorder must be an integer >=minmodelorder with 2*order<ntheta-1')
    if options['minmodelorder'] > angular_limit:
        raise ValueError('minmodelorder is not identifiable on this angular grid')
    D = np.full((len(theta), nf), 1/(2*np.pi))
    orders, iterations, constraint_count = np.zeros(nf, int), np.zeros(nf, int), np.zeros(nf, int)
    residual, moment_error = np.zeros(nf), np.zeros(nf)
    converged = np.ones(nf, bool)
    coefficients, trials = [np.empty(0) for _ in range(nf)], [[] for _ in range(nf)]
    previous_order = 0
    for index in active:
        H, target, constant_error = _constraints(S[:, :, index], G[:, :, index], weights)
        M = H.shape[1]
        constraint_count[index] = M
        if not M:
            residual[index] = moment_error[index] = constant_error
            converged[index] = constant_error <= options['errortol']
            continue
        maximum = min(angular_limit, M//2+1) if supplied_max is None else int(supplied_max)
        maximum = max(maximum, int(options['minmodelorder']))
        minimum = min(maximum, max(int(options['minmodelorder']), (previous_order+1)//2))
        selected = None
        for order in range(minimum, maximum+1):
            harmonics = theta[:, None]*np.arange(1, order+1)
            B = np.column_stack((np.cos(harmonics), np.sin(harmonics)))
            if np.linalg.matrix_rank(B-B.mean(axis=0)) < 2*order:
                raise ValueError('Fourier model is not identifiable on the supplied angular grid')
            model_D, fit = _fit_fourier(H, target, weights, B, options)
            variance = np.var(fit['residual_vector'], ddof=1) if M > 1 else 0.
            score = float(M*(np.log(2*np.pi*max(variance, options['errortol']**2))+1)+4*order+2)
            fit.update(order=order, aic=score)
            trials[index].append(fit)
            if selected is not None and score+options['aictol'] > selected[2]['aic']:
                break
            selected = order, model_D, fit
        order, model_D, fit = selected
        D[:, index] = model_D
        orders[index], iterations[index] = order, fit['iterations']
        coefficients[index] = fit['coefficients'].copy()
        moment_error[index] = max(constant_error, fit['moment_residual'])
        modeled = getcrossspectra(theta, G[:, :, index:index+1], model_D[:, None])[:, :, 0]
        residual[index] = float(np.max(abs(modeled-S[:, :, index])))
        converged[index] = residual[index] <= options['errortol']
        previous_order = order
    info = dict(method='EMEM', converged=converged, residual=residual,
                moment_residual=moment_error, model_order=orders, coefficients=coefficients,
                iterations=iterations, constraint_count=constraint_count, trials=trials,
                active_frequencies=active, frequency=frequency, aic_variance_floor=options['errortol']**2)
    if not full_output and np.any(~converged[active]):
        warnings.warn('EMEM moment fit did not converge at all active frequencies; request full_output diagnostics', RuntimeWarning, stacklevel=2)
    return (D, info) if full_output else D
