"""WAFO distribution names and parameter conventions, backed by SciPy.

Normal/lognormal second parameters are VARIANCES. GPD shape has the opposite
sign to scipy.stats.genpareto. Random functions accept size= and seed=.
Fit functions return parameters only, not legacy confidence intervals.
"""
import numpy as np
from scipy import stats


def _norm(m=0, v=1):
    return stats.norm(loc=m, scale=np.sqrt(v))


def _logn(m=0, v=1):
    return stats.lognorm(np.sqrt(v), scale=np.exp(m))


def _weib(a, c):
    return stats.weibull_min(c, scale=a)


def _gam(a, b=1):
    return stats.gamma(a, scale=b)


def _gev(k, s=1, m=0):
    return stats.genextreme(k, loc=m, scale=s)


def _gpd(k, s=1, m=0):
    return stats.genpareto(-np.asarray(k), loc=m, scale=s)


def _gumb(a, b, trunc=0):
    base = stats.gumbel_r(loc=b, scale=a)
    return _Truncated(base, 0) if trunc else base


class _Truncated:
    def __init__(self, base, lower):
        self.base, self.lower = base, lower
        self.mass = base.sf(lower)

    def cdf(self, x):
        return np.where(np.asarray(x)>self.lower,-np.expm1(self.base.logsf(x)-self.base.logsf(self.lower)),0)

    def pdf(self, x):
        return np.where(np.asarray(x)>=self.lower,np.exp(self.base.logpdf(x)-self.base.logsf(self.lower)),0)

    def ppf(self, p):
        p=np.asarray(p)
        value=self.base.isf((1-p)*self.mass)
        return np.where((p<0)|(p>1),np.nan,np.where(p==0,self.lower,value))

    def rvs(self, size=None, random_state=None):
        return self.ppf(np.random.default_rng(random_state).random(size))

    def stats(self, moments='mv'):
        from scipy.integrate import quad_vec
        m=quad_vec(lambda u:self.ppf(u),0,1)[0]
        v=quad_vec(lambda u:(self.ppf(u)-m)**2,0,1)[0]
        return m,v


_FAMILIES = {
    'norm': _norm, 'logn': _logn, 'weib': _weib,
    'gam': _gam, 'beta': lambda a, b: stats.beta(a, b),
    'chi2': lambda p: stats.chi2(p), 'exp': lambda m=1: stats.expon(scale=m),
    'rayl': lambda b: stats.rayleigh(scale=b), 'f': lambda a, b: stats.f(a, b),
    't': lambda p: stats.t(p), 'gev': _gev, 'gpd': _gpd, 'gumb': _gumb,
    'frech': lambda a, c: stats.invweibull(c, scale=a),
    'ggam': lambda a, b, c=1: stats.gengamma(a, c, scale=b),
    'invg': lambda m, l: stats.invgauss(np.asarray(m)/l, scale=l),
}


def _make_eval(factory, operation, name):
    def evaluate(x, *parameters):
        return getattr(factory(*parameters), operation)(x)
    evaluate.__name__ = name
    evaluate.__doc__ = f"WAFO {name}; see module documentation for parameter conventions."
    return evaluate


def _make_stat(factory, name):
    def moments(*parameters):
        return factory(*parameters).stats(moments='mv')
    moments.__name__ = name
    return moments


def _make_rnd(factory, name):
    def random(*parameters, size=None, seed=None):
        return factory(*parameters).rvs(size=size, random_state=np.random.default_rng(seed))
    random.__name__ = name
    random.__doc__ = "Use keyword size= for dimensions, seed= for reproducibility."
    return random


for _family, _factory in _FAMILIES.items():
    for _suffix, _operation in [('pdf', 'pdf'), ('cdf', 'cdf'), ('inv', 'ppf')]:
        _name = 'w'+_family+_suffix
        globals()[_name] = _make_eval(_factory, _operation, _name)
    _name = 'w'+_family+'stat'
    globals()[_name] = _make_stat(_factory, _name)
    _name = 'w'+_family+'rnd'
    globals()[_name] = _make_rnd(_factory, _name)


def wnormfit(x):
    a = np.asarray(x, dtype=float)
    if a.ndim==0 or not np.all(np.isfinite(a)):
        raise ValueError('Need finite observations')
    if a.ndim==1 or (a.ndim==2 and 1 in a.shape):
        a=a.ravel()
        if len(a)<2:
            raise ValueError('Need at least two observations')
        return np.asarray([a.mean(),a.var(ddof=1)])
    if a.shape[0]<2:
        raise ValueError('Need at least two observations per column')
    return np.stack((a.mean(axis=0),a.var(axis=0,ddof=1)),axis=-1).reshape(-1,2)


def wlognfit(x):
    a = np.asarray(x, dtype=float)
    if np.any(a <= 0):
        raise ValueError("Lognormal data must be positive")
    return wnormfit(np.log(a))


def wexpfit(x):
    a = np.asarray(x, dtype=float)
    if np.any(a < 0):
        raise ValueError("Exponential data must be nonnegative")
    return a.mean()


def wraylfit(x):
    a = np.asarray(x, dtype=float)
    if np.any(a < 0):
        raise ValueError("Rayleigh data must be nonnegative")
    return np.sqrt(np.mean(a*a)/2)


def wweibfit(x):
    c, _, a = stats.weibull_min.fit(x, floc=0)
    return np.asarray([a, c])


def wgamfit(x):
    a, _, b = stats.gamma.fit(x, floc=0)
    return np.asarray([a, b])


def wbetafit(x):
    a, b, _, _ = stats.beta.fit(x, floc=0, fscale=1)
    return np.asarray([a, b])


def wgevfit(x):
    k, m, s = stats.genextreme.fit(x)
    return np.asarray([k, s, m])


def wgpdfit(x):
    c, m, s = stats.genpareto.fit(x, floc=0)
    return np.asarray([-c, s, m])


def wgumbfit(x):
    b, a = stats.gumbel_r.fit(x)
    return np.asarray([a, b])


def wtfit(x):
    return stats.t.fit(x, floc=0, fscale=1)[0]


def wchi2fit(x):
    return stats.chi2.fit(x, floc=0, fscale=1)[0]


def empdistr(x):
    """Two-column [sorted sample, empirical CDF] with ranks (i-0.5)/n."""
    x = np.sort(np.asarray(x).ravel())
    return np.column_stack((x, (np.arange(len(x))+.5)/len(x)))


def cempdistr(x, c=-np.inf):
    """Conditional empirical CDF of observations at or above c."""
    x = np.asarray(x).ravel()
    return empdistr(x[x >= c])


def wquantile(x, p, method=1):
    methods = {1: 'hazen', 2: 'weibull', 3: 'averaged_inverted_cdf'}
    if method not in methods:
        raise ValueError("method must be 1, 2, or 3")
    return np.quantile(x, p, axis=0, method=methods[method])


def iqr(x, axis=0):
    return stats.iqr(x, axis=axis)


def wskewness(x, dim=0):
    return stats.skew(x, axis=dim, bias=True)


def wkurtosis(x, dim=0):
    return stats.kurtosis(x, axis=dim, fisher=False, bias=True)


def mvnormpdf(x, m, S):
    return stats.multivariate_normal.pdf(x, mean=m, cov=S)


def wnormndpdf(x, m, S):
    return mvnormpdf(x, m, S)


def wmnormrnd(m, S, *, size=1, seed=None):
    return np.random.default_rng(seed).multivariate_normal(m, S, size=size)


from ._extended import (walpharnd, wtweibcdf, wtweibpdf, wtweibfit, wtweibfun,
    wtraylcdf, wtraylpdf, wtraylfit, winvgfit, wggamfit, wgpdfit_ml,
    loglike, wgevlike, wgamafit, wggambfit, wgumbafit, wweibcfit, range, var, comnsize)
from ._bivariate import (weib2dpdf, weib2dcdf, weib2dcinv, weib2drnd,
    weib2dstat, weib2dprb, weib2dlike, weib2dfit, weib2dpdf2,
    mdist2dpdf, mdist2dcdf, mdist2dcinv, mdist2drnd, mdist2dlike, mdist2dpdf2)
from ._design import cl2cnr, cnr2cl, cdr, sudg, ffd, yates, ryates, alias
from ._plotting import (wqqplot,distplot,wnormplot,wexpplot,wraylplot,wweibplot,
    wgumbplot,whisto,pairs,kdeplot,fitmodel,getmodel,nplot,cplot,wgumbtest,identify,
    dist2dcdfplot,mdist2dcdfplot,weib2dcdfplot,dist2dstatplot,mdist2dstatplot,weib2dstatplot,dist2dparamplot)
from ._conditional import (dist2dfun,dist2dpdf,dist2dcdf,dist2dprb,dist2drnd,
    dist2dstat,dist2dpdf2,dist2dfit,dist2dsmfun,dist2dsmfun2,mdist2dfit,mdist2dstat)

__all__ = sorted(name for name in globals() if name.startswith('w') and callable(globals()[name])) + [
    'empdistr', 'cempdistr', 'iqr', 'mvnormpdf', 'loglike', 'range', 'var', 'comnsize',
    'mdist2dpdf', 'mdist2dcdf', 'mdist2dcinv', 'mdist2drnd', 'mdist2dlike', 'mdist2dpdf2',
    'cl2cnr','cnr2cl','cdr','sudg','ffd','yates','ryates','alias',
    'dist2dfun','dist2dpdf','dist2dcdf','dist2dprb','dist2drnd','dist2dstat',
    'dist2dpdf2','dist2dfit','dist2dsmfun','dist2dsmfun2','mdist2dfit','mdist2dstat',
    'distplot','pairs','kdeplot','fitmodel','getmodel','nplot','cplot','identify',
    'dist2dcdfplot','mdist2dcdfplot','weib2dcdfplot','dist2dstatplot','mdist2dstatplot','weib2dstatplot','dist2dparamplot']
