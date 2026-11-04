"""Source asymmetric-rainflow inversion and switching-MCTP model fitting."""
from time import perf_counter
import numpy as np
from scipy import optimize


def arfm2mctp(Frfc, *, full_output=False, tolerance=1e-8):
    """Invert a regular asymmetric rainflow matrix by increasing cycle range.

    Implements the source elementary-event recursion and its two bilinear
    standing/hanging equations.  Physical polynomial roots replace the
    source's unchecked quadratic square root; linear/degenerate cases use the
    same equations with bounded solves.  The source swapped forward/reverse
    budgets at the lowest-minimum boundary; correcting them also permits
    nonreversible loads.  Returns a normalized from/to joint
    matrix whose upward and downward masses are each one, irrespective of
    input count scale.  Full output also returns transition matrices, elapsed
    seconds and a forward-reconstruction residual instead of MATLAB clock rows.

    Input must follow ``mctp2arfm``'s standalone outer-cycle convention and
    have positive minimum/maximum marginals on every possible interior class.
    Inactive classes, incompatible cycle masses and singular inverses raise
    explicit errors; an inverse is not manufactured for unidentifiable data.
    """
    from . import mctp2arfm
    from ._remaining import _solve
    started = perf_counter()
    rainflow = np.asarray(Frfc, dtype=float)
    if (rainflow.ndim != 2 or rainflow.shape[0] != rainflow.shape[1] or len(rainflow) < 2
            or not np.all(np.isfinite(rainflow)) or np.any(rainflow < 0)
            or rainflow.sum() <= 0 or np.any(np.diag(rainflow) != 0)
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('a finite nonnegative asymmetric rainflow matrix with zero diagonal is required')
    n, total = len(rainflow), float(rainflow.sum())
    rainflow = rainflow/total
    upper, lower = np.triu(rainflow, 1), np.tril(rainflow, -1)
    minimum, maximum = upper.sum(axis=1)+lower.sum(axis=0), upper.sum(axis=0)+lower.sum(axis=1)
    if np.any(minimum[:-1] <= 0) or np.any(maximum[1:] <= 0):
        raise ValueError('inactive minimum/maximum classes make the source inverse unidentifiable')
    if np.any(lower[:, 0] > tolerance) or np.any(upper[1:, -1] > tolerance):
        raise ValueError('rainflow matrix does not follow the standalone outer-cycle convention')
    Q, Qh, Qr, Qrh = (np.zeros_like(rainflow) for _ in range(4))
    result = np.zeros_like(rainflow)
    for distance in range(1, n):
        for i in range(n-distance):
            j = i+distance
            pS, pH, Min, Max = rainflow[i, j], rainflow[j, i], minimum[i], maximum[j]
            if i == 0:
                x = Max*(1-Qrh[j, 1:j].sum())
                y = Max*(1-Qh[j, 1:j].sum())
            elif j == n-1:
                x = Min*(1-Q[i, i+1:-1].sum())
                y = Min*(1-Qr[i, i+1:-1].sum())
            elif pS == 0:
                x, y = 0., pH
            elif pH == 0:
                x, y = pS, 0.
            elif distance == 1:
                if Min <= pH or Max <= pS:
                    raise ValueError('adjacent rainflow masses exceed their marginal probability budgets')
                x, y = Min*pS/(Min-pH), Max*pH/(Max-pS)
            else:
                c0 = c0h = 0.
                c1, c1h = 1/Min, 1/Max
                e0, e0h = 1-Qr[i, i+1:j].sum(), 1-Qrh[j, i+1:j].sum()
                D = Qr[i, i+1]*(1-Qrh[i+1, i])
                Dh = Qrh[j, j-1]*(1-Qr[j-1, j])
                R = Qr[i, i+1]*Qrh[i+1, i]
                Rh = Qrh[j, j-1]*Qr[j-1, j]
                E3, E3h = 1-Q[i, i+1:j].sum(), 1-Qh[j, i+1:j].sum()
                if distance >= 3:
                    I, J = np.arange(i+1, j-1), np.arange(i+2, j)
                    A, Ah = Qr[np.ix_(I, J)], Qrh[np.ix_(J, I)]
                    inv = _solve(np.eye(len(I))-A@Ah, np.eye(len(I)))
                    a, ah, b, bh = Qr[i, J], Qrh[j, I], Qr[I, j], Qrh[J, i]
                    e = 1-Qr[I, i+2:j+1].sum(axis=1)
                    eh = 1-Qrh[J, i:j-1].sum(axis=1)
                    c0, c0h = a@Ah@inv@b, ah@inv@A@bh
                    D += a@eh+a@Ah@inv@A@eh
                    Dh += ah@inv@e
                    e0 += a@Ah@inv@e
                    e0h += ah@inv@A@eh
                    R += a@bh+a@Ah@inv@A@bh
                    Rh += ah@inv@b
                    A3, A3h = Q[np.ix_(I, J)], Qh[np.ix_(J, I)]
                    inv3 = _solve(np.eye(len(I))-A3@A3h, np.eye(len(I)))
                    e3, e3h = 1-Q[I, i+2:j].sum(axis=1), 1-Qh[J, i+1:j-1].sum(axis=1)
                    E3 += Q[i, J]@A3h@inv3@e3
                    E3h += Qh[j, I]@inv3@A3@e3h
                L, H = 1-R, 1-Rh
                u0, u1 = e0*H+c0*Dh, -c1*H+c1*Dh
                v0, v1 = e0h*L+c0h*D, -c1h*L+c1h*D
                # a0+a1*x+a2*y+a3*x*y=0, similarly b: the source
                # polynomials, factored to avoid dozens of cancelling terms.
                a = np.array([pS*H*(L*H-c0*c0h)-Max*E3h*c0h*u0,
                              -pS*H*c0*c1h-Max*E3h*c1h*u0,
                              -pS*H*c1*c0h-Max*E3h*c0h*u1,
                              -pS*H*c1*c1h-Max*E3h*c1h*u1])
                b = np.array([pH*L*(L*H-c0*c0h)-Min*E3*c0*v0,
                              -pH*L*c0*c1h-Min*E3*c0*v1,
                              -pH*L*c1*c0h-Min*E3*c1*v0,
                              -pH*L*c1*c1h-Min*E3*c1*v1])
                coefficients = np.array([a[2]*b[3]-a[3]*b[2],
                                          a[0]*b[3]-a[1]*b[2]+a[2]*b[1]-a[3]*b[0],
                                          a[0]*b[1]-a[1]*b[0]])
                budget = np.array([min(Min*(1-Q[i].sum()), Max*(1-Qrh[j].sum())),
                                   min(Max*(1-Qh[j].sum()), Min*(1-Qr[i].sum()))])
                budget = np.maximum(budget, 0)
                scales = np.maximum(np.max(abs(np.vstack((a, b))), axis=1), 1e-15)
                def equations(pair):
                    xx, yy = pair
                    terms = np.array([1, xx, yy, xx*yy])
                    return np.array([a@terms, b@terms])/scales
                nonzero = np.flatnonzero(abs(coefficients) > 1e-12*max(abs(coefficients).max(), 1e-30))
                roots = np.roots(coefficients[nonzero[0]:]) if len(nonzero) else []
                candidates = []
                for root in roots:
                    if abs(np.imag(root)) > tolerance:
                        continue
                    yy = float(np.real(root))
                    denominator = b[1]+b[3]*yy
                    if abs(denominator) > 1e-15:
                        xx = -(b[0]+b[2]*yy)/denominator
                    else:
                        denominator = a[1]+a[3]*yy
                        if abs(denominator) <= 1e-15:
                            continue
                        xx = -(a[0]+a[2]*yy)/denominator
                    pair = np.array([xx, yy])
                    if (np.all(pair >= -tolerance) and np.all(pair <= budget+tolerance)
                            and np.linalg.norm(equations(pair), ord=np.inf) < tolerance):
                        candidates.append(np.maximum(pair, 0))
                if not candidates and np.all(budget > 0):
                    initial = np.minimum([pS, pH], .5*budget)
                    solved = optimize.least_squares(equations, initial, bounds=(0, budget),
                                                     ftol=1e-13, xtol=1e-13, gtol=1e-13)
                    if np.linalg.norm(equations(solved.x), ord=np.inf) < tolerance:
                        candidates.append(solved.x)
                if not candidates:
                    raise ValueError(f'no physical solution to inverse cycle equations at classes ({i},{j})')
                x, y = min(candidates, key=lambda pair: pair[1])
            if min(x, y) < -tolerance:
                raise ValueError('inverse recursion exceeds a transition probability budget')
            x, y = max(x, 0), max(y, 0)
            result[i, j], result[j, i] = x, y
            Q[i, j], Qh[j, i], Qr[i, j], Qrh[j, i] = x/Min, y/Max, y/Min, x/Max
    if not (np.allclose(Q[:-1].sum(axis=1), 1, atol=tolerance)
            and np.allclose(Qh[1:].sum(axis=1), 1, atol=tolerance)
            and np.allclose(Qr[:-1].sum(axis=1), 1, atol=tolerance)
            and np.allclose(Qrh[1:].sum(axis=1), 1, atol=tolerance)):
        raise ValueError('inverse recursion does not yield stochastic active transitions')
    recovered = mctp2arfm((np.triu(result, 1), np.tril(result, -1)))
    residual = float(np.max(abs(recovered-rainflow)))
    if residual > tolerance*10:
        raise ValueError('inverse fails forward asymmetric-rainflow reconstruction')
    diagnostics = dict(elapsed_seconds=perf_counter()-started, input_mass=total,
                       forward_error=residual, transitions=(Q, Qh), reversed_transitions=(Qr, Qrh))
    return (result, diagnostics) if full_output else result


def _transition_exact(P):
    P = np.asarray(P, dtype=float)
    if (P.ndim != 2 or P.shape[0] != P.shape[1] or not len(P) or not np.all(np.isfinite(P))
            or np.any(P < 0) or not np.allclose(P.sum(axis=1), 1, atol=1e-10)):
        raise ValueError('regime transition probabilities must be nonnegative with unit row sums')
    return P


def _callback(value, default):
    from . import f_funm, tr_m2x, tr_x2m
    registry = dict(f_funm=f_funm, tr_m2x=tr_m2x, tr_x2m=tr_x2m)
    value = default if value is None else value
    if callable(value):
        return value
    if isinstance(value, str) and value in registry:
        return registry[value]
    raise ValueError('model callbacks must be Python callables or f_funm/tr_m2x/tr_x2m')


def _pair(value):
    if isinstance(value, (tuple, list)) and len(value) == 2 and np.ndim(value[0]) == 2:
        return (np.asarray(value[0], dtype=float),
                None if value[1] is None else np.asarray(value[1], dtype=float))
    return np.asarray(value, dtype=float), None


def _mode(whatEst):
    modes = {'P', 'MeanStd', 'P,MeanStd', 'CalcF', 'P,CalcF'}
    if whatEst in ('SimF', 'P,SimF'):
        raise NotImplementedError('SimF needs an unspecified simulation model; the supplied source also omits P,SimF objective dispatch')
    if whatEst not in modes:
        raise ValueError('whatEst must be P, MeanStd, P,MeanStd, CalcF or P,CalcF')
    return whatEst.startswith('P'), whatEst.split(',')[-1] if whatEst != 'P' else None


def _has_constraints(value):
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value)
    return np.size(value) > 0 and bool(np.any(value))


def _decode(X, whatEst, known, init):
    from . import tr_x2p, scalemat
    estimate_P, family = _mode(whatEst)
    X = np.asarray(X, dtype=float).reshape(-1)
    if not np.all(np.isfinite(X)):
        raise ValueError('optimization vector must be finite')
    if estimate_P:
        if init is None or 'P' not in init:
            if family is not None:
                raise ValueError('combined fitting needs initial P to identify the number of regimes')
            r = int((1+np.sqrt(1+4*len(X)))/2)
        else:
            r = len(init['P'])
        size = r*(r-1)
        if len(X) < size:
            raise ValueError('optimization vector is too short for the transition model')
        P, remainder = _transition_exact(tr_x2p(X[:size], 1)), X[size:]
    else:
        P, remainder = _transition_exact(known['P']), X
        r = len(P)
    parameters = dict(P=P)
    if family is None:
        if len(remainder):
            raise ValueError('unexpected parameters after transition probabilities')
        pairs = [_pair(value) for value in known['F']]
    elif family == 'MeanStd':
        if len(remainder) != 2*r:
            raise ValueError('MeanStd requires one location and log scale per regime')
        transformed = remainder.reshape(r, 2, order='F').copy()
        transformed[:, 1] = np.exp(transformed[:, 1])
        parameters['MeanStd'] = transformed
        template = [_pair(value) for value in known['F']]
        if len(template) != r:
            raise ValueError('one template matrix pair is required per regime')
        n = len(template[0][0])
        param = known.get('param', (-1, 1, n))
        def scale_pj(F, mean, scale):
            # scalemat's public source layout is WAT; model kernels are PJ.
            wat = np.flipud(F.T)
            return np.flipud(scalemat(param, wat, mean, scale, param)).T
        pairs = []
        for (up, down), (mean, scale) in zip(template, transformed):
            a = scale_pj(up, mean, scale)
            b = None if down is None else scale_pj(down.T, mean, scale).T
            pairs.append((a, b))
    else:
        inverse = _callback(known.get('trX2Model'), 'tr_x2m')
        calculate = _callback(known.get('Ffun'), 'f_funm')
        sizes = known.get('nM')
        if sizes is None:
            if init is None or 'M' not in init:
                raise ValueError('CalcF needs nM or initial models to delimit each model vector')
            forward = _callback(known.get('trModel2X'), 'tr_m2x')
            sizes = [len(np.asarray(forward(model)).ravel()) for model in init['M']]
        if len(sizes) != r or any(int(size) != size or size < 1 for size in sizes) or sum(sizes) != len(remainder):
            raise ValueError('nM must delimit the complete model parameter vector')
        models, pairs, start = [], [], 0
        for size in sizes:
            model = inverse(remainder[start:start+int(size)], known)
            models.append(model)
            pairs.append(_pair(calculate(known['param'], model)))
            start += int(size)
        parameters['M'] = models
    if len(pairs) != r:
        raise ValueError('one min/max matrix pair is required per regime')
    band = known.get('NOsubzero', 0)
    if not np.isscalar(band) or not np.isfinite(band) or int(band) != band or band < 0:
        raise ValueError('NOsubzero must be a nonnegative integer')
    cleaned = []
    for up, down in pairs:
        if up.ndim != 2 or up.shape[0] != up.shape[1] or not np.all(np.isfinite(up)) or np.any(up < 0):
            raise ValueError('submodels must be finite nonnegative square matrices')
        up = np.triu(up, 1+int(band))
        if down is not None:
            if down.shape != up.shape or not np.all(np.isfinite(down)) or np.any(down < 0):
                raise ValueError('downward submodel matrix must match the upward model')
            down = np.tril(down, -1-int(band))
        cleaned.append((up, down))
    return P, cleaned, parameters


def _side_blocks(value, r, n):
    value = np.asarray(value, dtype=float)
    if value.shape == (r*n, r*n):
        value = value.reshape(r, n, r, n).transpose(0, 2, 1, 3)
    if value.shape != (r, r, n, n):
        raise ValueError('side observations must have shape (regime,regime,level,level) or block-matrix shape')
    value = value.copy()
    value[:, :, 0, 0] += value[:, :, -1, -1]
    value[:, :, -1, -1] = 0
    return value.transpose(0, 2, 1, 3).reshape(r*n, r*n)


def f_smctp(X, Fobs, whatEst, method, known, whatKnown=None, init=None, *, full_output=False):
    """Source switching-MCTP likelihood/distance objective and model decoder.

    Supports transition (P), location/scale (MeanStd), elliptical or callable
    model (CalcF) parameters and their P combinations.  P uses row-major
    off-diagonal log odds; MeanStd uses column-major [means,log-scales], as in
    MATLAB.  ``known.SideInfo`` supports 0 and 11; source-unspecified 12/21/22
    and SimF models raise explicitly.  Callbacks accept their original source
    arguments; recognized built-in names are resolved without eval.

    Full output is ``(objective,predicted_matrix,P,submodel_pairs)``.  Known
    kernels use the package's PJ min/max layout, correcting the MeanStd source
    mismatch with scalemat's WAT layout; explicit downward kernels are retained.
    Nonempty whatKnown is rejected because the source never defines it.
    """
    from . import smctp2rfm, smctp2arfm, loglcmat, chi2cmat, hdcmat, klcmat
    if _has_constraints(whatKnown):
        raise NotImplementedError('whatKnown parameter constraints are undefined in the source')
    P, pairs, _ = _decode(X, whatEst, dict(known), init)
    side = known.get('SideInfo', 0)
    if side == 0:
        predicted, observations = smctp2rfm(P, pairs), np.asarray(Fobs, dtype=float)
    elif side == 11:
        _, marks, _ = smctp2arfm(P, pairs, SideInfo=1, full_output=True)
        r, n = len(P), len(pairs[0][0])
        predicted, observations = _side_blocks(marks, r, n), _side_blocks(Fobs, r, n)
    else:
        raise NotImplementedError('only source SideInfo=0 and SideInfo=11 are defined')
    methods = dict(ML=lambda obs, fit: -loglcmat(obs, fit), chi2=chi2cmat, HD=hdcmat, KL=klcmat)
    if method not in methods:
        raise ValueError('method must be ML, chi2, HD or KL')
    value = float(methods[method](observations, predicted))
    return (value, predicted, P, pairs) if full_output else value


def estsmctp(Fobs, whatEst, method, known, whatKnown=None, init=None, OPTIONS=None, *, full_output=False):
    """Fit a switching turning-point model by source Nelder-Mead optimization.

    Returns ``(Fest,Est)`` or ``(Fest,Est,optimizer_options)`` with full output.
    Fest contains P, F and the predicted rainflow matrix; Est includes fitted
    parameters, objective, vector and iteration counts.  OPTIONS accepts SciPy
    Nelder-Mead options plus MATLAB TolX/TolFun/MaxIter/MaxFunEvals/Display names.
    Domain-invalid trial models receive infinite objective; invalid starts,
    nonconvergence and nonfinite final fits raise explicit exceptions.
    SimF/P,SimF require unspecified external model implementations and are
    rejected rather than approximated.  Label permutations and data-poor
    models can be statistically unidentifiable despite numerical convergence.
    """
    from . import tr_p2x
    estimate_P, family = _mode(whatEst)
    if _has_constraints(whatKnown):
        raise NotImplementedError('whatKnown constraints are undefined in the source')
    if init is None:
        raise ValueError('provide initial model parameters in init')
    known, init = dict(known), dict(init)
    r = len(_transition_exact(init['P'] if estimate_P else known['P']))
    parts = []
    if estimate_P:
        probabilities = _transition_exact(init['P'])
        vector = tr_p2x(probabilities, 1)
        if not np.all(np.isfinite(vector)):
            raise ValueError('initial off-diagonal probabilities must lie strictly between zero and one')
        parts.append(vector)
    if family == 'MeanStd':
        meanstd = np.asarray(init['MeanStd'], dtype=float)
        if meanstd.shape != (r, 2) or not np.all(np.isfinite(meanstd)) or np.any(meanstd[:, 1] <= 0):
            raise ValueError('initial MeanStd must contain a finite location and positive scale per regime')
        meanstd = meanstd.copy()
        meanstd[:, 1] = np.log(meanstd[:, 1])
        parts.append(meanstd.ravel(order='F'))
    elif family == 'CalcF':
        models = init['M']
        if len(models) != r:
            raise ValueError('provide one initial model per regime')
        transform = _callback(known.get('trModel2X'), 'tr_m2x')
        model_vectors = [np.asarray(transform(model), dtype=float).ravel() for model in models]
        known['nM'] = [len(vector) for vector in model_vectors]
        parts.extend(model_vectors)
    initial = np.concatenate(parts) if parts else np.empty(0)
    if not np.all(np.isfinite(initial)):
        raise ValueError('initial parameter transformation must be finite')
    options = dict(xatol=1e-6, fatol=1e-8, maxiter=2000, maxfev=8000, disp=False)
    aliases = dict(TolX='xatol', TolFun='fatol', MaxIter='maxiter', MaxFunEvals='maxfev', Display='disp')
    allowed = set(options) | {'adaptive', 'initial_simplex', 'return_all'}
    for name, value in ({} if OPTIONS is None else OPTIONS).items():
        key = aliases.get(name, name)
        if key not in allowed:
            raise TypeError('unsupported optimizer option: '+name)
        options[key] = (value.lower() not in ('off', 'none') if key == 'disp' and isinstance(value, str) else value)
    initial_value = f_smctp(initial, Fobs, whatEst, method, known, whatKnown, init)
    if not np.isfinite(initial_value):
        raise ValueError('initial model assigns zero probability to observed cells or has nonfinite distance')
    def objective(vector):
        try:
            with np.errstate(over='raise', divide='raise', invalid='raise'):
                return f_smctp(vector, Fobs, whatEst, method, known, whatKnown, init)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return np.inf
    if len(initial):
        result = optimize.minimize(objective, initial, method='Nelder-Mead', options=options)
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError('switching-MCTP optimization did not converge: '+str(result.message))
        vector, iterations, evaluations = result.x, result.nit, result.nfev
        message = str(result.message)
    else:
        vector, iterations, evaluations, message = initial, 0, 1, 'No free parameters'
    value, predicted, P, pairs = f_smctp(vector, Fobs, whatEst, method, known, whatKnown, init, full_output=True)
    _, _, parameters = _decode(vector, whatEst, known, init)
    parameters.update(objective=value, X=vector, nit=iterations, nfev=evaluations,
                       success=True, message=message, initial_objective=initial_value)
    fitted = dict(P=P, F=pairs, rfm=predicted)
    return (fitted, parameters, options) if full_output else (fitted, parameters)
