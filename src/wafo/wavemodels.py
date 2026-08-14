'''
Created on 13. mar. 2018

@author: pab
'''
import numpy as np
from numpy import pi, sqrt
from scipy.special import gammaln
import wafo.transform.estimation as te
import wafo.transform as wt
from wafo.containers import PlotData
from wafo.kdetools.kernels import qlevels
from wafo.misc import tranproc
import warnings


def _set_default_t_h_g(t, h, g, m0, m2):
    if g is None:
        y = np.linspace(-5, 5)
        x = sqrt(m0) * y + 0
        g = wt.TrData(y, x)
    if t is None:
        tt1 = 2 * pi * sqrt(m0 / m2)
        t = np.linspace(0, 1.7 * tt1, 51)
    if h is None:
        px = g.gauss2dat([0, 4.])
        px = abs(px[1] - px[0])
        h = np.linspace(0, 1.3 * px, 41)
    return h, t, g


def lh83pdf(t=None, h=None, mom=None, g=None):
    """
    LH83PDF Longuet-Higgins (1983) approximation of the density (Tc,Ac)
             in a stationary Gaussian transform process X(t) where
             Y(t) = g(X(t)) (Y zero-mean Gaussian, X  non-Gaussian).

      CALL:  f   = lh83pdf(t,h,[m0,m1,m2],g);

            f    = density of wave characteristics of half-wavelength
                   in a stationary Gaussian transformed process X(t),
                   where Y(t) = g(X(t)) (Y zero-mean Gaussian)
           t,h   = vectors of periods and amplitudes, respectively.
                   default depending on the spectral moments
        m0,m1,m2 = the 0'th,1'st and 2'nd moment of the spectral density
                   with angular  frequency.
            g    = space transformation, Y(t)=g(X(t)), default: g is identity
                   transformation, i.e. X(t) = Y(t)  is Gaussian,
                   The transformation, g, can be estimated using lc2tr
                   or dat2tr or given apriori by ochi.

    Examples
    --------
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap()
    >>> w = np.linspace(0,4,256)
    >>> S = Sj.tospecdata(w)   #Make spectrum object from numerical values
    >>> S = sm.SpecData1D(Sj(w),w) # Alternatively do it manually
    >>> mom, mom_txt = S.moment(nr=2, even=False)
    >>> f = lh83pdf(mom=mom)
    >>> f.plot()

    See also
    --------
    cav76pdf,  lc2tr, dat2tr

    References
    ----------
    Longuet-Higgins,  M.S. (1983)
    "On the joint distribution wave periods and amplitudes in a
     random wave field", Proc. R. Soc. A389, pp 24--258

    Longuet-Higgins,  M.S. (1975)
    "On the joint distribution wave periods and amplitudes of sea waves",
    J. geophys. Res. 80, pp 2688--2694
    """

    # tested on: matlab 5.3
    # History:
    # Revised pab 01.04.2001
    # - Added example
    # - Better automatic scaling for h,t
    # revised by IR 18.06.2000, fixing transformation and transposing t and h to fit simpson req.
    # revised by pab 28.09.1999
    #   made more efficient calculation of f
    # by Igor Rychlik

    m0, m1, m2 = mom
    h, t, g = _set_default_t_h_g(t, h, g, m0, m2)

    L0 = m0
    L1 = m1 / (2 * pi)
    L2 = m2 / (2 * pi)**2
    eps2 = sqrt((L2 * L0) / (L1**2) - 1)

    if np.any(~np.isreal(eps2)):
        raise ValueError('input moments are not correct')

    const = 4 / sqrt(pi) / eps2 / (1 + 1 / sqrt(1 + eps2**2))

    a = len(h)
    b = len(t)
    der = np.ones((a, 1))

    h_lh = g.dat2gauss(h.ravel(), der.ravel())

    der = abs(h_lh[1])  # abs(h_lh[:, 1])
    h_lh = h_lh[0]

    # Normalization + transformation of t and h ???????
    # Without any transformation

    t_lh = t / (L0 / L1)
    # h_lh = h_lh/sqrt(2*L0)
    h_lh = h_lh / sqrt(2)
    t_lh = 2 * t_lh

    # Computation of the distribution
    T, H = np.meshgrid(t_lh[1:b], h_lh)
    f_th = np.zeros((a, b))
    tmp = const * der[:, None] * (H / T)**2 * np.exp(-H**2. *
                                                     (1 + ((1 - 1. / T) / eps2)**2)) / ((L0 / L1) * sqrt(2) / 2)
    f_th[:, 1:b] = tmp

    f = PlotData(f_th, (t, h),
                 xlab='Tc', ylab='Ac',
                 title='Joint density of (Tc,Ac) - Longuet-Higgins (1983)',
                 plot_kwds=dict(plotflag=1))

    return _add_contour_levels(f)


def cav76pdf(t=None, h=None, mom=None, g=None):
    """
    CAV76PDF Cavanie et al. (1976) approximation of the density  (Tc,Ac)
             in a stationary Gaussian transform process X(t) where
             Y(t) = g(X(t)) (Y zero-mean Gaussian, X  non-Gaussian).

     CALL:  f = cav76pdf(t,h,[m0,m2,m4],g);

            f    = density of wave characteristics of half-wavelength
                   in a stationary Gaussian transformed process X(t),
                   where Y(t) = g(X(t)) (Y zero-mean Gaussian)
           t,h   = vectors of periods and amplitudes, respectively.
                   default depending on the spectral moments
     m0,m2,m4    = the 0'th, 2'nd and 4'th moment of the spectral density
                   with angular frequency.
            g    = space transformation, Y(t)=g(X(t)), default: g is identity
                   transformation, i.e. X(t) = Y(t)  is Gaussian,
                   The transformation, g, can be estimated using lc2tr
                   or dat2tr or given a priori by ochi.
           []    = default values are used.

     Examples
     --------
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap()
    >>> w = np.linspace(0,4,256)
    >>> S = Sj.tospecdata(w)   #Make spectrum object from numerical values
    >>> S = sm.SpecData1D(Sj(w),w) # Alternatively do it manually
    >>> mom, mom_txt = S.moment(nr=4, even=True)
    >>> f = cav76pdf(mom=mom)
    >>> f.plot()

     See also
     --------
     lh83pdf, lc2tr, dat2tr

     References
     ----------
     Cavanie, A., Arhan, M. and Ezraty, R. (1976)
     "A statistical relationship between individual heights and periods of
      storm waves".
     In Proceedings Conference on Behaviour of Offshore Structures,
     Trondheim, pp. 354--360
     Norwegian Institute of Technology, Trondheim, Norway

     Lindgren, G. and Rychlik, I. (1982)
     Wave Characteristics Distributions for Gaussian Waves --
     Wave-lenght, Amplitude and Steepness, Ocean Engng vol 9, pp. 411-432.
    """
    # tested on: matlab 5.3 NB! note
    # History:
    # revised pab 04.11.2000
    # - fixed xlabels i.e. f.labx={'Tc','Ac'}
    # revised by IR 4 X 2000. fixed transform and normalisation
    # using Lindgren & Rychlik (1982) paper.
    # At the end of the function there is a text with derivation of the density.
    #
    # revised by jr 21.02.2000
    # - Introduced cell array for f.x for use with pdfplot
    # by pab 28.09.1999

    m0, m2, m4 = mom
    h, t, g = _set_default_t_h_g(t, h, g, m0, m2)

    eps4 = 1.0 - m2**2 / (m0 * m4)
    alfa = m2 / sqrt(m0 * m4)
    if np.any(~np.isreal(eps4)):
        raise ValueError('input moments are not correct')

    a = len(h)
    b = len(t)
    der = np.ones((a, 1))

    h_lh = g.dat2gauss(h.ravel(), der.ravel())
    der = abs(h_lh[1])
    h_lh = h_lh[0]

    # Normalization + transformation of t and h

    pos = 2 / (1 + alfa)    # inverse of a fraction of positive maxima
    cons = 2 * pi**4 * pos / sqrt(2 * pi) / m4 / sqrt((1 - alfa**2))
    # Tm=2*pi*sqrt(m0/m2)/alpha; #mean period between positive maxima

    t_lh = t
    h_lh = sqrt(m0) * h_lh

    # Computation of the distribution
    T, H = np.meshgrid(t_lh[1:b], h_lh)
    f_th = np.zeros((a, b))

    f_th[:, 1:b] = cons * der[:, None] * (H**2 / (T**5)) * np.exp(-0.5 * (
        H / T**2)**2. * ((T**2 - pi**2 * m2 / m4)**2 / (m0 * (1 - alfa**2)) + pi**4 / m4))
    f = PlotData(f_th, (t, h),
                 xlab='Tc', ylab='Ac',
                 title='Joint density of (Tc,Ac) - Cavanie et al. (1976)',
                 plot_kwds=dict(plotflag=1))
    return _add_contour_levels(f)


def _add_contour_levels(f):
    p_levels = np.r_[10:90:20, 95, 99, 99.9]
    try:
        c_levels = qlevels(f.data, p=p_levels)
        f.clevels = c_levels
        f.plevels = p_levels
    except ValueError as e:
        msg = "Could not calculate contour levels!. ({})".format(str(e))
        warnings.warn(msg)
    return f

# Let U,Z be the height and second derivative (curvature) at a local maximum in a Gaussian proces
# with spectral moments m0,m2,m4. The conditional density ($U>0$) has the following form
# $$
# f(z,u)=c \frac{1}{\sqrt{2\pi}}\frac{1}{\sqrt{m0(1-\alpha^2)}}\exp(-0.5\left(\frac{u-z(m2/m4)}
# {\sqrt{m0(1-\alpha^2)}}\right)^2)\frac{|z|}{m4}\exp(-0.5z^2/m4), \quad z<0,
# $$
# where $c=2/(1+\alpha)$, $\alpha=m2/\sqrt{m0\cdot m4}$.
#
# The cavanie approximation is based on the model $X(t)=U \cos(\pi t/T)$, consequently
# we have $U=H$ and by twice differentiation $Z=-U(\pi^2/T)^2\cos(0)$. The variable change has
# Jacobian $2\pi^2 H/T^3$ giving the final formula for the density of $T,H$
# $$
# f(t,h)=c \frac{2\pi^4}{\sqrt{2\pi}}\frac{1}{m4\sqrt{m0(1-\alpha^2)}}\frac{h^2}{t^5}
#       \exp(-0.5\frac{h^2}{t^4}\left(\left(\frac{t^2-\pi^2(m2/m4)}
# {\sqrt{m0(1-\alpha^2)}}\right)^2+\frac{\pi^4}{m4}\right)).
# $$
#

def _defaulted_parameters(values, defaults):
    params = np.asarray(defaults, dtype=float).copy()
    if values is None:
        return params
    values = np.asarray(values, dtype=float).ravel()
    n = min(values.size, params.size)
    valid = ~np.isnan(values[:n])
    params[:n][valid] = values[:n][valid]
    return params


def _lognormal_mu_sigma(mean, std):
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    sigma2 = np.log1p((std / mean) ** 2)
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - 0.5 * sigma2
    return mu, sigma


def _lognormal_pdf(x, mean, std):
    x = np.asarray(x, dtype=float)
    mu, sigma = _lognormal_mu_sigma(mean, std)
    out = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(mu), np.shape(sigma)))
    mask = (x > 0) & (mean > 0) & (std > 0)
    x_m = np.broadcast_to(x, out.shape)[mask]
    mu_m = np.broadcast_to(mu, out.shape)[mask]
    sigma_m = np.broadcast_to(sigma, out.shape)[mask]
    out[mask] = np.exp(-0.5 * ((np.log(x_m) - mu_m) / sigma_m) ** 2) / (
        x_m * sigma_m * np.sqrt(2 * np.pi))
    return out


def _lognormal_cdf(x, mean, std):
    from scipy.special import ndtr

    x = np.asarray(x, dtype=float)
    mu, sigma = _lognormal_mu_sigma(mean, std)
    out = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(mu), np.shape(sigma)))
    mask = (x > 0) & (mean > 0) & (std > 0)
    x_m = np.broadcast_to(x, out.shape)[mask]
    mu_m = np.broadcast_to(mu, out.shape)[mask]
    sigma_m = np.broadcast_to(sigma, out.shape)[mask]
    out[mask] = ndtr((np.log(x_m) - mu_m) / sigma_m)
    return out


def _weibull_pdf(x, scale, shape):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = (x > 0) & (scale > 0) & (shape > 0)
    z = x[mask] / scale
    out[mask] = shape / scale * z ** (shape - 1) * np.exp(-(z ** shape))
    return out


def _generalized_gamma_pdf(x, shape1, shape2, scale):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = (x > 0) & (shape1 > 0) & (shape2 > 0) & (scale > 0)
    z = x[mask] / scale
    log_pdf = (
        np.log(shape2) - np.log(scale) - gammaln(shape1 / shape2) +
        (shape1 - 1) * np.log(z) - z ** shape2)
    out[mask] = np.exp(log_pdf)
    return out


def ltwcpdf(t, h, Pt1, Ph1=None, kind=0, condon=0):
    """
    LTWCPDF Long Term Wave Climate PDF of Hm0 and wave period

    CALL: f = ltwcpdf(T,Hs,Pt,Ph,kind)

       f      = density
       T      = peak period, Tp  or mean zerodowncrossing period, Tz (sec)
       Hs     = significant wave height (meter)
       Pt     = [A1 A2 A3 B1 B2 B3 B4] parameters for the conditional
                distribution of T given Hs
                (default [1.59 0.42 2 0.005 0.09 0.13 1.34])
       Ph     = [C1 C2 C3 Hc M1 S1], parameters for the marginal
                distribution for Hs (default [2.82 1.547 0 3.27  0.836 0.376])
       kind    = defines the parametrization used (default 0)

      The probability distribution is given as the product:

       f(T,Hs)=f(T|Hs)*f(Hs)
      where

          f(T|Hs) = pdflognorm(T,my(Hs),S(Hz)) with mean and variance
           my(Hs)  = A1 + A2*log(Hs+A3)          if mod(kind,2)==0
                     A1 + A2*Hs^A3               if mod(kind,2)==1
           S(Hs)   = B1+B2*exp(-B3*Hs.^B4)

       For kind <=1:
          f(Hs)    = pdflognorm(Hs,M1,S1)    for Hs <= Hc
                   = pdfweib(Hs-C3,C1,C2)    for Hs >  Hc
       For kind >1:
          f(Hs)    = pdflognorm(Hs,M1,S1)    for Hs <= Hc
                   = pdfgengam(Hs,C1,C2,C3)  for Hs >  Hc  (Suggested by Ochi)

       The default values for T and Hs are suitable for the Northern North
       Sea for peak period, Tp and significant waveheight, Hs.
       With a suitable change of parameters for Pt this model fit mean wave
       period, Tz, also reasonably well.

      The size of f is the common size of T and Hs

      NOTE: - by specifying nan's  in the vectors Pt or Ph default values
              will be used.
            - if length(Pt) or length(Ph) is shorter than the parameters
              needed then the default values are used for the parameters
              not specified.
            - For tables of fitted parameter values to specific sites see
              the end of this file (type ltwcpdf.m)

    Examples
    --------
    Set  C1 = 2.73,  Hc = 3.95  and the rest to their default values
      Ph = [2.73, NaN,NaN, 3.95]; x = linspace(0, 15);
      [T,H] = meshgrid(x);
      f = ltwcpdf(T,H,[],Ph);
      contour(x,x,f)

    See also
    --------
    pdfweib, pdflognorm, pdfgengam

    References
    ---------
    Haver, S (1980)
    'Analysis of uncertainties related to the stochastic modelling of
     Ocean waves'
    Ph.D. thesis, Norwegian Institute of Technology, NTH, Trondheim, Norway

    Haver,S and Nyhus, K. A.  (1986)
    'A wave climate description for long term response calculation.'
    In Proc. of OMAE'86, tokyo, Japan, pp. 27-34

    Sagli, Gro (2000)
    "Model uncertainty and simplified estimates of long term extremes of
    hull girder loads in ships"
    Ph.D. thesis, Norwegian University of Science and Technology, NTNU,
    Trondheim, Norway, pp 45--47

    Michel K. Ochi (1998),
    "OCEAN WAVES, The stochastic approach",
    OCEAN TECHNOLOGY series 6, Cambridge, pp 127.
    (Generalized Gamma distribution for wave height)

    Bitner-Gregersen, E.M. and Hagen, Ø. (2000)
    "Aspects of joint distribution for metocean Phenoma at the Norwegian
    Continental shelf."
    In Proc. of OMAE'2000,
    """
    ph = _defaulted_parameters(Ph1, [2.82, 1.547, 0.0, 3.27, 0.836, 0.376])
    pt = _defaulted_parameters(Pt1, [1.59, 0.42, 2.0, 0.005, 0.09, 0.13, 1.34])

    t, h = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(h, dtype=float))
    C1, C2, C3, Hc, myHz, sHz = ph
    A1, A2, A3, B1, B2, B3, B4 = pt

    if np.mod(abs(kind), 2) == 0:
        myTp = A1 + A2 * np.log(h + A3)
    else:
        myTp = A1 + A2 * h ** A3
    sTp = B1 + B2 * np.exp(-B3 * h ** B4)

    p = np.zeros_like(h, dtype=float)
    positive = (t > 0) & (h > 0)
    if condon == 5:
        p[positive] = _lognormal_cdf(t[positive], myTp[positive], sTp[positive])
    elif condon == 10:
        p[positive] = 1.0
    else:
        p[positive] = _lognormal_pdf(t[positive], myTp[positive], sTp[positive])

    if condon in (0, 5, 10):
        low = (h > 0) & (h <= Hc)
        if np.any(low):
            p[low] *= _lognormal_pdf(h[low], myHz, sHz)

        high = h > Hc
        if np.any(high):
            if kind <= 1:
                p[high] *= _weibull_pdf(h[high] - C3, C1, C2)
            else:
                p[high] *= _generalized_gamma_pdf(h[high], C1, C2, C3)

    if condon in (0, 1, 5, 10):
        return p
    if condon == 3:
        return t * p
    if condon == 4:
        return t ** 2 * p
    raise ValueError('unknown option')
# The following is taken from Sagli (2000):
# Parameters for the long term description of the wave climate (Tp|Hs):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Area            Pt = [  A1     A2    A3     B1    B2    B3   B4]  kind
#   ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#   Northern North Sea:  [1.59,  0.42, 2.00, 0.005, 0.09, 0.13, 1.34]  0
#   Aasgard           :  [1.72,  0.34, 0.46, 0.005, 0.10, 0.29, 1.00]  1
#   Sleipner          :  [0.23,  1.69, 0.15, 0.005, 0.12, 0.40, 1.00]  1
#   Troms I           :  [1.35,  0.61, 0.34, 0.005, 0.12, 0.36, 1.00]  1
#   Statfjord/Brent   :  [0.40,  1.59, 0.15, 0.005, 0.12, 0.34, 1.00]  1
#   Ekofisk           :  [1.00,  0.90, 0.25, 0.005, 0.10, 0.44, 1.00]  1
#   Ekofisk Zone II   :  [0.03,  1.81, 0.15, 0.005, 0.16, 0.58, 1.00]  1
#   Ekofisk Zone III  :  [0.03,  1.81, 0.15, 0.005, 0.16, 0.58, 1.00]  1
#   Ekofisk Zone IV   :  [1.41,  0.38, 0.54, 0.010, 0.08, 0.41, 1.00]  1
#   Ekofisk Zone VI   :  [0.00,  1.72, 0.15, 0.010, 0.14, 1.00, 1.00]  1
#
#
# Parameters for the long term description of the wave climate (Hs):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Area            Ph = [  C1     C2    C3    Hc    M1    S1  ]  kind
#   ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#   Northern North Sea:  [2.82,  1.55, 0.00, 3.27, 0.84, 0.61^2  ]  0
#   Aasgard           :  [2.73,  1.45, 0.00, 3.95, 0.84, 0.59^2  ]  1
#   Sleipner          :  [2.38,  1.44, 0.00, 3.00, 0.69, 0.60^2  ]  1
#   Troms I           :  [2.26,  1.36, 0.00, 4.25, 0.72, 0.56^2  ]  1
#   Statfjord/Brent   :  [2.82,  1.55, 0.00, 3.27, 0.84, 0.61^2  ]  1
#   Ekofisk           :  [2.08,  1.37, 0.00, 2.65, 0.52, 0.67^2  ]  1
#   Ekofisk Zone II   :  [2.12,  1.37, 0.00, 2.70, 0.54, 0.67^2  ]  1
#   Ekofisk Zone III  :  [2.05,  1.55, 0.00, 2.15, 0.50, 0.65^2  ]  1
#   Ekofisk Zone IV   :  [1.56,  1.46, 0.00, 2.00, 0.25, 0.62^2  ]  1
#   Ekofisk Zone VI   :  [1.18,  1.42, 0.00, 1.25, 0.07, 0.70^2  ]  1
#


# The following is taken from Bitner-Gregersen and Hagen (2000):
# Description of sites:
# Vøring plateau:  (position 67, 27'N 5, 58'E water depth 1460m) NORWAVE
#                   buoy every 3rd hour for a period of 2.3 years
#

# Parameters for the long term description of the wave climate (Tp|Hs):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Area           Pt = [  A1     A2      A3      B1     B2       B3     B4]  kind
#   ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#   Vøring M         :  [1.718,  0.194, 0.642, -288.28, 288.52,0.279e-5 , 1]  1
#   Vøring M/H       :  [1.670,  0.220, 0.617, - 93.13,  93.37,0.330e-4 , 1]  1
#   Vøring H         :  [0.474,  1.438, 0.186,   0.047,   0.46,   0.476 , 1]  1
#   Haltenbanken     :  [1.921,  0.171, 0.665,   0.064,  0.327,   0.227 , 1]  1

# Parameters for the long term description of the wave climate (Tz|Hs)
# (mean zero-down crossing period given Hz)                           :
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Area            Pt = [  A1     A2      A3     B1    B2     B3   B4]  kind
#   ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#   Statfjord&Gullfaks:  [1.790,  0.110, 0.759, 0.106, 0.361,0.969 , 1]  1
#   Statfjord Platform:  [1.771,  0.080, 0.902, 0.001, 0.239,0.169 , 1]  1
#   Ekofisk           :  [0.611,  0.902, 0.289, 0.071, 0.211,0.606 , 1]  1


# Parameters for the long term description of the wave climate (Hs):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Area            Ph = [  C1     C2    C3    Hc  ]  kind
#   ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#   Vøring M          :  [2.371,  1.384, 0.755, 0  ]  1
#   Vøring M/H        :  [2.443,  1.405, 0.820, 0  ]  1
#   Vøring H          :  [2.304,  1.383, 0.899, 0  ]  1
#   Haltenbanken      :  [2.154,  1.273, 0.763, 0  ]  1
#   Statfjord&Gullfaks:  [2.264,  1.398, 0.969, 0  ]  1
#   Statfjord Platform:  [2.502,  1.527, 0.657, 0  ]  1
#   Ekofisk           :  [1.597,  1.184, 0.852, 0  ]  1


# Fitted parameters by splitting the data into wind Sea and swell:

# Parameters for the long term description of the wave climate (Tp|Hs):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Area            Pt = [  A1     A2    A3     B1    B2    B3   B4]  kind
#   ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#   Vøring M         :  [-0.707, 2.421, 0.126, 0.076, 0.063, 0.162, 1]  1    (Wind Sea)
#   Vøring M         :  [ 1.501, 0.882, 0,135, 0.126, 0.000, 0.000, 1]  1    (Swell)
#   Haltenbanken     :  [-4383.9, 4385.6, 0.948e-4, 0.045, 0.369,0.780,1]  1 (Wind Sea)
#   Haltenbanken     :  [ 2.257, 0.088, 0.746, 0.010, 0.229, 0.167, 1]  1    (Swell)

# Parameters for the long term description of the wave climate (Hs):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Area            Ph = [  C1     C2    C3    Hc  ]  kind
#   ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#   Vøring M          :  [2.170,  1.382, 0.203, 0  ]  1  (Wind sea)
#   Vøring M          :  [1.606,  1.252, 0.229, 0  ]  1  (Swell)
#   Haltenbanken      :  [1.473,  1.032, 0.241, 0  ]  1  (Wind sea)
#   Haltenbanken      :  [1.661,  1.117, 0.240, 0  ]  1  (Swell)



def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


if __name__ == '__main__':
    test_docstrings()
