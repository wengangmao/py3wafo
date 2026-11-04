"""Source-based parametric wave characteristic distributions."""
from ._additional_tables import (ohhspdf, ohhscdf, ohhspdf2, ohhsspdf, ohhsscdf,
    ohhsspdf2, ohhvpdf, thspdf, thscdf, thspdf2, thsspdf, thsscdf, thsspdf2,
    thsnlpdf, thsnlcdf, thsnlpdf2, thvpdf, thvcdf, thvpdf2)
from ._remaining_models import tay81cdf, tay81pdf, ltwcpdf, jhvnlcdf, jhvnlpdf, jhvnlpdf2
import numpy as np
from scipy.special import betaln
from scipy.stats import beta
from ._analytic import (ochi98cdf,ochi98pdf,ochi98fit,braylfit,trraylpdf,
    cav76pdf,lh83pdf,tay90fun,tay90pdf,tay90cdf,tay81fun)
from ._mk87 import mk87pdf,mk87pdf2,mk87cdf,mk87rnd
from ._empirical import (b04pdf,b04cdf,b04pdf2,b04jpdf,b04jcdf,b04jpdf2,
    bmr00pdf,bmr00cdf,bmr00pdf2,jhwparfun,jhnlwparfun)
from ._tables import (thgparfun,thwparfun,ohhgparfun,thpdf,thcdf,ohhpdf,ohhcdf,
    jhspdf,jhscdf,jhspdf2,jhsnlpdf,jhsnlcdf,jhsnlpdf2,jhvpdf,jhvcdf,jhvpdf2)


def braylcdf(x, a, b, c):
    """Beta-Rayleigh wave-height CDF: beta CDF of (x/c)², x>=0."""
    x = np.asarray(x)
    return np.where(x >= 0, beta.cdf((x/c)**2, a, b), 0)


def braylpdf(x, a, b, c):
    """Beta-Rayleigh PDF, including finite/infinite endpoint limits."""
    x, a, b, c = np.broadcast_arrays(x, a, b, c)
    z = x/c
    with np.errstate(divide='ignore', invalid='ignore'):
        density = 2/c*np.exp((2*a-1)*np.log(z)+(b-1)*np.log1p(-z*z)-betaln(a,b))
        atzero = np.where(a < .5, np.inf, np.where(a == .5, 2/c*np.exp(-betaln(a,b)), 0))
        atc = np.where(b < 1, np.inf, np.where(b == 1, 2*a/c, 0))
    density = np.where(x == 0, atzero, np.where(x == c, atc, density))
    density = np.where((x < 0) | (x > c), 0, density)
    return np.where((a <= 0) | (b <= 0) | (c <= 0), np.nan, density)
