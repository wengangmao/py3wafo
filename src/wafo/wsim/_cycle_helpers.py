"""FAT rainflow reconstruction helpers, with the original anti-diagonal layout.

``f[i,j]`` counts a cycle with zero-based minimum ``i`` and maximum
``n-1-j``. This differs from WAFO's upper-triangular min/max matrix. Levels
and returned indices are zero based throughout this Python module. Mathematical
count/combination formulas follow the source; NumPy's paired advanced indexing
is deliberately not substituted for MATLAB's Cartesian submatrix indexing.
"""
import warnings
import numpy as np


def _frequency(f, *, physical=False):
    f = np.asarray(f, dtype=float)
    if (f.ndim != 2 or f.shape[0] != f.shape[1] or len(f) < 3
            or np.any(~np.isfinite(f)) or np.any(f < 0)):
        raise ValueError('Frequency matrix must be finite, nonnegative, square, and at least 3 by 3')
    if physical and np.any(f[np.indices(f.shape).sum(axis=0) >= len(f)-1] != 0):
        raise ValueError('Positive cycles must lie strictly above the anti-diagonal')
    return f.copy()


def _levels(r, n):
    r = np.asarray(r, dtype=float).reshape(-1)
    if np.any(~np.isfinite(r)) or np.any(r != np.floor(r)) or np.any(r < 0) or np.any(r >= n):
        raise ValueError('Residual levels must be zero-based integers in [0,n-1]')
    return r.astype(int)


def _limit(value, name):
    if not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name+' must be a positive integer')
    return int(value)


def _anti(f, strict=False):
    return np.fliplr(np.triu(np.fliplr(f), 1 if strict else 0))


def fr2nt(f):
    """Counting distribution from a FAT anti-diagonal frequency matrix.

    In the source's interior cells this is ``sum(f[p,q], p<i, q<j)``.
    The first/last rows and columns are zero, and only cells on or above the
    anti-diagonal are retained. No conversion to a WAFO min/max matrix occurs.
    """
    f = _frequency(f)
    prefix = np.cumsum(np.cumsum(f, axis=1)-f, axis=0)
    result = np.zeros_like(f)
    result[1:-1, 1:-1] = prefix[:-2, 1:-1]
    return _anti(result)


def fr2comb(f, r=()):
    """Source combination matrix, including contributions from a residual.

    The source's ``comb(M,n-1-M)`` expression updates a Cartesian block, not
    paired anti-diagonal entries. That indexing is preserved with ``np.ix_``;
    paired indexing would silently erase the residual contribution when the
    strict anti-diagonal mask is subsequently applied.
    """
    f = _frequency(f)
    n = len(f)
    residual = _levels(r, n)
    rows = np.cumsum(f, axis=0)-f
    cols = np.cumsum(f, axis=1)-f
    combination = 2*fr2nt(f)+rows+cols
    for a, b in zip(residual[:-1], residual[1:]):
        levels = np.arange(b+1, a+1) if a > b else np.arange(a, b)
        combination[np.ix_(levels, n-1-levels)] += 1
    return _anti(combination, True)


def comb2pro(combinations, freq):
    """Probabilities of no insertion, then insertion at each candidate place.

    The exact source survival probability is ``(1-1/(c+add))**freq``, where
    ``add=(c+freq==1)``. The first returned entry is the product of all
    survivals; subsequent entries select the first successful candidate.
    Empty candidate vectors give [1]. Zero frequency gives survival one.
    A positive count with fewer than one available place is rejected, except
    the source's special c=0,freq=1 case (forced insertion).
    """
    c = np.asarray(combinations, dtype=float).reshape(-1)
    f = np.asarray(freq, dtype=float).reshape(-1)
    if c.shape != f.shape or np.any(~np.isfinite(c)) or np.any(~np.isfinite(f)) or np.any(c < 0) or np.any(f < 0):
        raise ValueError('Combinations and frequencies need matching finite nonnegative vectors')
    denominator = c+(c+f == 1)
    active = f > 0
    if np.any(denominator[active] < 1):
        raise ValueError('Positive cycle frequency has no valid candidate places')
    survival = np.ones_like(f)
    survival[active] = (1-1/denominator[active])**f[active]
    previous = np.r_[1., np.cumprod(survival)]
    return np.r_[previous[-1], (1-survival)*previous[:-1]]


def getulcc(M):
    """Return zero-based row, column, count of the upper-left positive entry.

    Anti-diagonals are searched by increasing row+column. The MATLAB helper
    explicitly did not support ties: it returned vectors and then evaluated
    ``M(row,col)`` as a Cartesian submatrix, making subsequent scalar indexing
    fail. This port raises a clear ValueError for a nonunique first entry.
    """
    M = np.asarray(M, dtype=float)
    if (M.ndim != 2 or M.shape[0] != M.shape[1] or not len(M)
            or np.any(~np.isfinite(M)) or np.any(M < 0)):
        raise ValueError('Expected a finite nonnegative square matrix')
    n = len(M)
    for diagonal in range(n):
        row = np.arange(diagonal+1)
        col = diagonal-row
        found = np.flatnonzero(M[row, col] > 0)
        if len(found) > 1:
            raise ValueError('The upper-left positive entry is not unique on its anti-diagonal')
        if len(found):
            i = int(row[found[0]])
            j = int(col[found[0]])
            return i, j, float(M[i, j])
    raise ValueError('No positive entry exists on or above the anti-diagonal')


def _draw(rng, combinations, frequency):
    p = comb2pro(combinations, frequency)
    return min(int(np.searchsorted(np.cumsum(p), rng.random(), side='right')), len(p)-1)


def _advance(f, combination, residual, rng, *, consume):
    """One source FAT placement step; operate on zero-based mutable lists."""
    r = residual
    n = len(f)
    h = 0
    if r[0] > r[1]:
        maximum, minimum = r[0], r[1]
        j = n-1-maximum
        if maximum > minimum+1:
            rows = np.arange(maximum-1, minimum, -1)
            h = _draw(rng, combination[rows, j], f[rows, j])
        if h:
            level = maximum-h
            if f[level, j] <= 0:
                raise RuntimeError('FAT placement selected a cycle with zero remaining frequency')
            if consume:
                f[level, j] -= 1
                combination[np.arange(level+1, maximum+1), j] -= 1
            r = [maximum-1, level]+r
        else:
            if consume and maximum > minimum+1:
                combination[np.arange(minimum+1, maximum), j] -= 1
            r[0] -= 1
    else:
        maximum, minimum = r[1], r[0]
        i = minimum
        if minimum < maximum-1:
            cols = n-1-np.arange(minimum+1, maximum)
            h = _draw(rng, combination[i, cols], f[i, cols])
        if h:
            level = minimum+h
            j = n-1-level
            if f[i, j] <= 0:
                raise RuntimeError('FAT placement selected a cycle with zero remaining frequency')
            if consume:
                f[i, j] -= 1
                cols = n-1-np.arange(minimum+h-1, minimum-1, -1)
                combination[i, cols] -= 1
            r = [minimum+1, level]+r
        else:
            if consume:
                combination[i, n-1-np.arange(minimum, maximum)] -= 1
            r[0] += 1
    if r[0] == r[1]:
        r = r[1:]
    return r


def fr2res(f, *, seed=None, burn_in=100, max_steps=100000):
    """Generate the source stationary residual, with a bounded retry loop.

    Each of the two residual sides must survive ``burn_in`` consecutive source
    placement steps. Reaching a one-level residual restarts that side exactly
    as in MATLAB. Some matrices can never survive (for example, a sole wide
    outer cycle); ``max_steps`` bounds total work and raises RuntimeError rather
    than allowing the original infinite loop. Default burn_in is the source's
    100. NumPy RNG seeds/generators are accepted; MATLAB stream parity is not.
    """
    f = _frequency(f, physical=True)
    burn_in = _limit(burn_in, 'burn_in')
    max_steps = _limit(max_steps, 'max_steps')
    i, j, _ = getulcc(f)
    n = len(f)
    initial = [n-1-j, i]
    if initial[0]-initial[1] < 4:
        return np.asarray(initial, dtype=int)
    rng = np.random.default_rng(seed)
    sides = []
    steps = 0
    for _ in range(2):
        r = initial.copy()
        combination = fr2comb(f, r)
        survived = 0
        while survived < burn_in:
            if steps >= max_steps:
                raise RuntimeError('Stationary residual did not survive burn-in within max_steps')
            r = _advance(f, combination, r, rng, consume=False)
            steps += 1
            survived += 1
            if len(r) < 2:
                r = initial.copy()
                survived = 0
        sides.append(r)
    return np.asarray(sides[0]+[initial[0]]+sides[1][::-1], dtype=int)


def rfc2load_fat(f, res=None, num_cc=None, *, seed=None, max_steps=1000000,
                 residual_burn_in=100, full_output=False):
    """Reconstruct a FAT load with the original random cycle-placement algorithm.

    Signature follows the actual MATLAB function (f,res,num_cc), whose CALL
    comment swapped its last two arguments. Fractional matrices are scaled by
    floor(100*f), with a warning, as in the source. ``num_cc`` requests a
    stationary simulation: after removing the largest residual cycles, the
    remaining matrix is scaled to floor(max(1000*n,1e6)*f/sum(f)), and simulation
    stops once the shaved turning-point count exceeds 2*num_cc+4. This is an
    approximate length request, not exact rainflow counts. With num_cc=None,
    all integer cycles must be exhausted or an explicit failure is raised.

    Default returns load values; full_output returns (load, original/proper
    residual, final combination matrix, unconsumed frequency matrix), matching
    the source output fields. Returned levels use zero-based FAT orientation.
    The input arrays are not modified. Retry/work limits replace unbounded
    source loops; no fallback signal is substituted when reconstruction fails.
    """
    f = _frequency(f, physical=True)
    max_steps = _limit(max_steps, 'max_steps')
    rng = np.random.default_rng(seed)
    if num_cc is not None and (not np.isscalar(num_cc) or not np.isfinite(num_cc) or num_cc < 0):
        raise ValueError('num_cc must be a nonnegative finite expected cycle count')
    if np.any(f != np.floor(f)):
        warnings.warn('Noninteger FAT frequency matrix scaled with floor(100*f), as in MATLAB', RuntimeWarning, stacklevel=2)
        f = np.floor(100*f)
    n = len(f)
    if res is None or np.size(res) == 0:
        from ..cycles import tp2rfc, dtp2rfm
        raw = fr2res(f, seed=rng, burn_in=residual_burn_in, max_steps=max_steps)
        residual = tp2rfc(raw, full_output=True)[2].reshape(-1).astype(int)
        wafo_residual = n-1-residual
        in_residual = np.fliplr(dtp2rfm(wafo_residual, n, 'CS')).T
        f -= in_residual
        if np.any(f < 0):
            raise ValueError('Generated residual contains more cycles than the supplied frequency matrix')
    else:
        residual = _levels(res, n)
    if len(residual) < 2 or np.any(np.diff(residual) == 0):
        raise ValueError('Residual needs at least two distinct consecutive levels')
    if len(residual) > 2 and np.any(np.diff(residual)[:-1]*np.diff(residual)[1:] >= 0):
        raise ValueError('Residual must consist of alternating turning points')
    smallest, largest = int(min(residual)), int(max(residual))
    r = residual.tolist()
    count = int(f[smallest, n-1-largest])
    f[smallest, n-1-largest] = 0
    if 2*count+len(r) > max_steps+2:
        raise RuntimeError('Largest-cycle insertion exceeds max_steps memory/work bound')
    extra = [smallest, largest]*count
    if extra:
        for index, pair in enumerate(zip(r[:-1], r[1:])):
            if pair == (smallest, largest):
                r[index:index] = extra
                break
            if pair == (largest, smallest):
                r[index:index] = extra[::-1]
                break
        else:
            raise ValueError('Residual has no adjacent pair spanning its largest amplitude')
    if num_cc is not None:
        total = f.sum()
        if total <= 0:
            raise ValueError('Stationary resampling needs positive non-outer-cycle frequency')
        f = np.floor(max(1000*n, 1000000)*f/total)
    combination = fr2comb(f, r)
    load = [r[0]]
    for _ in range(max_steps):
        if len(r) < 2:
            break
        r = _advance(f, combination, r, rng, consume=True)
        load.append(r[0])
        if len(load) >= 3 and ((load[-3] < load[-2] < load[-1]) or (load[-3] > load[-2] > load[-1])):
            del load[-2]
        combination = _anti(combination, True)
        if num_cc is not None and len(load) > 2*num_cc+4:
            break
    else:
        if len(r) >= 2:
            raise RuntimeError('FAT reconstruction did not terminate within max_steps')
    if num_cc is None and np.any(f != 0):
        raise ValueError('Residual was exhausted before all cycles could be placed')
    load = np.asarray(load, dtype=int)
    return (load, residual.copy(), combination, f) if full_output else load
