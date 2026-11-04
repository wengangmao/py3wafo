"""Physical Jacobians, variance conservation, and independent wave integrals."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import simpson, trapezoid
from wafo.spec import (Spectrum, dir2enc, dir2k1d, dspec2dcov, spa2time,
                       time2spa, w2k, k2w, spec2cov, spec2spec)


def spectrum(n=401, directional=False, h=np.inf):
    w = np.linspace(0, 3, n)
    s = w**4*np.exp(-3*w*w)
    S = Spectrum(type='freq', w=w, S=s, h=h, phi=0., note='test')
    if directional:
        theta = np.linspace(-np.pi, np.pi, 65)
        D = (1+.65*np.cos(theta))/(2*np.pi)
        S.update(type='dir', theta=theta, S=D[:, None]*s)
    return S


def variance(S):
    density = S.S
    if 'theta' in S and np.size(S.theta) > 1:
        density = trapezoid(density, x=S.theta, axis=0)
    return trapezoid(density, x=S.get('w', S.get('k')))


@pytest.mark.parametrize('h', [np.inf, 8.])
def test_1d_dispersion_jacobian_and_roundtrip(h):
    S = spectrum(h=h)
    k = w2k(S.w, h=h)[0]
    K = time2spa(S, k)
    w = S.w
    if np.isinf(h):
        cg = np.divide(9.81, 2*w, out=np.zeros_like(w), where=w > 0)
    else:
        kh = k*h
        cg = np.divide(9.81*(np.tanh(kh)+kh/np.cosh(kh)**2), 2*w,
                       out=np.full_like(w, np.sqrt(9.81*h)), where=w > 0)
    assert_allclose(K.S, S.S*cg, rtol=2e-12, atol=1e-14)
    back = spa2time(K, S.w)
    assert_allclose(back.S, S.S, rtol=3e-11, atol=1e-14)
    assert_allclose(variance(K), variance(S), rtol=4e-5)
    assert back.note == 'test' and S.type == 'freq'


def test_hz_gravity_and_scalar_spacing():
    S = spectrum(65, h=12)
    S.g = 4.
    Hz = Spectrum(S)
    Hz.f, Hz.S = Hz.pop('w')/(2*np.pi), Hz.S*2*np.pi
    A, B = time2spa(S, .005), time2spa(Hz, .005)
    assert_allclose(A.S, B.S, rtol=2e-14)
    assert_allclose(A.k, np.arange(65)*.005)
    C = spa2time(A, .02)
    assert_allclose(C.w, np.arange(65)*.02)
    with pytest.raises(ValueError, match='singular'):
        time2spa(Spectrum(type='freq', w=[0., 1.], S=[1., 0.]))


@pytest.mark.parametrize('h,g', [(np.inf, 9.81), (7., 9.81), (np.inf, [5., 12.])])
def test_2d_change_of_variables_and_mass(h, g):
    S = spectrum(161, True, h)
    S.g = g
    # A dense Cartesian grid tests the polar Jacobian, not round-trip identity.
    gx, gy = np.atleast_1d(g)[0], np.atleast_1d(g)[-1]
    limitx = w2k(S.w[-1], h=h, g=gx)[0]
    limity = w2k(S.w[-1], h=h, g=gy)[0]
    K = time2spa(S, np.linspace(-limitx, limitx, 401), np.linspace(-limity, limity, 401))
    mass = trapezoid(trapezoid(K.S, x=K.k, axis=1), x=K.k2)
    assert_allclose(mass, variance(S), rtol=.012)
    back = spa2time(K, S.w, S.theta)
    # The zero Cartesian ordinate is intentionally zero; test away from its cell.
    region = (S.w > .4) & (S.w < 2.)
    assert_allclose(back.S[:, region], S.S[:, region], rtol=.07, atol=2e-5)
    assert_allclose(variance(back), variance(S), rtol=.012)
    assert K.S.shape == (401, 401) and back.S.shape == S.S.shape


def test_cartesian_interpolation_analytic_and_rotation_metadata():
    k = np.linspace(-2, 2, 51)
    # A constant Cartesian density has exact directional Jacobian 2*w^3/g^2.
    K = Spectrum(type='rotk2d', k=k, k2=k, S=np.ones((51, 51)), phi=.2, g=3.)
    w = np.linspace(0, 1.4, 31)
    th = np.linspace(-np.pi, np.pi, 33)
    D = spa2time(K, w, th)
    assert_allclose(D.S, np.broadcast_to(2*w**3/9, D.S.shape), atol=1e-15)
    assert D.type == 'rotdir' and D.phi == .2
    assert time2spa(D, k, k).type == 'rotk2d'
    with pytest.raises(NotImplementedError, match='Anisotropic'):
        time2spa(spectrum(h=5), g=[3., 4.])


@pytest.mark.parametrize('v', [0., 3., -3., 20., -20.])
def test_encounter_variance_with_multiple_fold_roots(v):
    S = spectrum(501, True)
    S.phi = .4
    E = dir2enc(S, 1, v)
    assert E.type == 'enc' and np.all(E.S >= 0) and np.all(np.isfinite(E.S))
    assert_allclose(variance(E), variance(S), rtol=1e-13)
    assert E.v == v and S.type == 'dir'
    D = dir2enc(S, 2, v)
    assert_allclose(variance(D), variance(S), rtol=1e-13)
    assert_allclose(D.S[0], D.S[-1], rtol=2e-14)
    assert_allclose(trapezoid(D.S, x=D.theta, axis=0), E.S, rtol=3e-13, atol=1e-15)


def test_encounter_moment_direct_mapping_and_fold_direction():
    S = spectrum(801)
    S.phi = np.pi
    v = 8.
    E = dir2enc(S, 1, v)
    mapped = abs(S.w-v*S.w**2/9.81)
    for p in (1, 2):
        assert_allclose(trapezoid(E.S*E.w**p, x=E.w),
                        trapezoid(S.S*mapped**p, x=S.w), rtol=1.2e-4)
    # All energy lies beyond the negative-frequency crossing and flips direction.
    D = spectrum(101, True)
    D.S[:] = 0
    j = len(D.theta)//2
    D.S[j, D.w > 1.3] = np.sin((D.w[D.w > 1.3]-1.3)*np.pi/1.7)**2
    D.phi = np.pi
    E = dir2enc(D, 2, 10.)
    assert np.sum(E.S[j]) == 0
    assert np.sum(E.S[0]) > 0


def test_encounter_cross_seas_hz_and_rejected_branches():
    S = spectrum(101)
    S.phi = np.pi/2
    E = dir2enc(S, 1, 10.)
    assert_allclose(variance(E), variance(S), rtol=1e-14)
    assert_allclose(trapezoid(E.S*E.w, x=E.w), trapezoid(S.S*S.w, x=S.w), rtol=2e-7)
    F = Spectrum(S)
    F.f, F.S = F.pop('w')/(2*np.pi), F.S*2*np.pi
    assert_allclose(dir2enc(F, v=2).S, dir2enc(S, v=2).S, atol=1e-14)
    with pytest.raises(NotImplementedError, match='finite-depth'):
        dir2enc(spectrum(h=10), v=3)
    with pytest.raises(NotImplementedError, match='delta'):
        dir2enc(S, 2, 3)
    with pytest.raises(ValueError):
        dir2enc(S, d=3)


@pytest.mark.parametrize('h', [np.inf, 6.])
def test_projected_wave_number_energy_and_moments(h):
    S = spectrum(601, True, h)
    S.phi = .3
    K = dir2k1d(S, np.linspace(0, w2k(S.w[-1], h=h)[0], 1601))
    assert K.type == 'rotk1d'
    assert_allclose(variance(K), variance(S), rtol=4e-14)
    kx = abs(np.cos(S.theta[:, None]-S.phi))*w2k(S.w, h=h)[0]
    for p in (1, 2):
        moment = trapezoid(trapezoid(S.S*kx**p, x=S.w, axis=1), x=S.theta)
        assert_allclose(trapezoid(K.S*K.k**p, x=K.k), moment, rtol=8e-5)
    with pytest.raises(ValueError, match='directional'):
        dir2k1d(spectrum())


def test_covariance_matches_temporal_transform_and_moments():
    S = spectrum(201, True, h=5.)
    S.phi = .2
    C = dspec2dcov(S, 4, 15, 2, 0, 0)
    F = Spectrum(S)
    F.type, F.S = 'freq', simpson(S.S, x=S.theta, axis=0)
    F.pop('theta')
    temporal = spec2cov(F, 4, 16, C.t[1])
    for n in range(5):
        assert_allclose(C['R'+'t'*n], temporal['R'+'t'*n], rtol=1e-12, atol=1e-14)
    assert C.R.shape == (16,) and C.R[0] > 0
    assert_allclose(C.Rtt[0], -simpson(F.S*S.w**2, x=S.w), rtol=2e-14)


def test_covariance_all_spatial_derivatives_against_wave_integral():
    S = spectrum(81, True, h=9.)
    S.phi = .3
    C = dspec2dcov(S, 4, 3, 1, 2, 1, .7, .9)
    k = w2k(S.w, h=S.h)[0]
    kx, ky = np.cos(S.theta[:, None]-S.phi)*k, np.sin(S.theta[:, None]-S.phi)*k
    phase = kx*C.x[3]+ky*C.y[0]+S.w*C.t[2]
    for powers in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1), (1, 1, 2), (0, 4, 0)):
        px, py, pt = powers
        amplitude = (1j*kx)**px*(1j*ky)**py*(1j*S.w)**pt
        expected = simpson(simpson(S.S*amplitude*np.exp(1j*phase), x=S.w, axis=1), x=S.theta).real
        field = 'R'+'x'*px+'y'*py+'t'*pt
        assert C[field].shape == (3, 5, 4)
        assert_allclose(C[field][0, 3, 2], expected, rtol=2e-12, atol=1e-14)
    # Real covariance symmetry under simultaneous reversal of all lags.
    Z = dspec2dcov(S, 2, 0, 1, 2, 2, .7, .7)
    assert_allclose(Z.R[:, :, 0], Z.R[::-1, ::-1, 0], atol=1e-14)
    assert_allclose(Z.Rx[:, :, 0], -Z.Rx[::-1, ::-1, 0], atol=1e-14)
    Y = dspec2dcov(S, 1, 2, 1, 0, 1)
    assert Y.R.shape == (3, 3) and 'Ry' in Y and 'Rx' not in Y


def test_covariance_defaults_and_validation():
    S = spectrum(41)
    C = dspec2dcov(S, 1, 2)
    assert C.R.shape == (21, 3) and np.all(np.isfinite(C.R))
    assert_allclose(C.Ry if 'Ry' in C else 0., 0.)
    for kwargs in ({'rate': 3}, {'nr': 5}, {'Nx': -1}, {'Nt': -1}, {'dx': 0}):
        with pytest.raises(ValueError):
            dspec2dcov(S, **kwargs)
    with pytest.raises(ValueError):
        spa2time(S)
    with pytest.raises(ValueError):
        time2spa(Spectrum(type='enc', w=S.w, S=S.S))


@pytest.mark.parametrize('h', [np.inf, 8.])
def test_spec2spec_directional_cartesian_workflows(h):
    S=spectrum(161,True,h)
    S.phi=.2
    K=spec2spec(S,'k2d',phi=.1)
    assert K.type=='k2d' and K.phi==pytest.approx(.3)
    D=spec2spec(K,'dir')
    assert D.type=='dir' and D.phi==pytest.approx(.3)
    assert_allclose(variance(D),variance(S),rtol=.035)
    F=spec2spec(K,'freq')
    assert F.type=='freq' and 'theta' not in F
    assert_allclose(variance(F),variance(S),rtol=.035)
    for source in (S,K):
        projected=spec2spec(source,'k1d')
        assert projected.type=='k1d' and 'theta' not in projected
        assert_allclose(variance(projected),variance(S),rtol=.035)


def test_spec2spec_encounter_workflows_metadata_and_rejected_inversions():
    S=spectrum(161,True)
    S.phi=.2
    E=spec2spec(S,'encdir',.1,-4.)
    F=spec2spec(E,'enc')
    direct=spec2spec(S,'enc',.1,-4.)
    assert E.v==F.v==-4 and E.phi==pytest.approx(.3)
    assert_allclose(F.S,direct.S,rtol=3e-13,atol=1e-14)
    assert_allclose(variance(F),variance(S),rtol=1e-13)
    assert F.note=='test' and 'theta' not in F
    assert S.phi==.2 and 'v' not in S
    K=spec2spec(S,'k2d')
    cart_enc=spec2spec(K,'enc',v=4.)
    assert cart_enc.type=='enc' and cart_enc.v==4
    assert_allclose(variance(cart_enc),variance(S),rtol=.035)
    for src,target in ((E,'dir'),(E,'freq'),(F,'freq'),(F,'k1d')):
        with pytest.raises(NotImplementedError,match='inversion'):
            spec2spec(src,target)
    with pytest.raises(NotImplementedError,match='re-encountered'):
        spec2spec(E,'encdir',v=2.)
    with pytest.raises(ValueError,match='speed'):
        spec2spec(S,'freq',v=2.)
    with pytest.raises(NotImplementedError):
        spec2spec(spectrum(),'dir')


def test_1d_encounter_native_dispersion_and_angle_units():
    S=spectrum(161)
    S.g=4.
    K=spec2spec(S,'k1d')
    assert_allclose(spec2spec(K,'freq').S,S.S,atol=1e-14)
    E=spec2spec(K,'enc',phi=.4,v=2.)
    direct=spec2spec(S,'enc',phi=.4,v=2.)
    assert_allclose(E.S,direct.S,rtol=2e-12,atol=1e-14)
    D=spectrum(101,True)
    D.phi=.3
    degrees=Spectrum(D)
    degrees.theta=np.rad2deg(D.theta)
    degrees.phi=np.rad2deg(D.phi)
    degrees.S=D.S*np.pi/180
    degrees.theta_units='degrees'
    for convert in (lambda s:time2spa(s),lambda s:spec2spec(s,'enc',phi=.2,v=3.),lambda s:dir2k1d(s)):
        a,b=convert(D),convert(degrees)
        assert_allclose(a.S,b.S,rtol=3e-12,atol=1e-13)
        assert_allclose(a.phi,b.phi,atol=1e-14)
    # Explicit gravity overrides update both the numerical transform and metadata.
    C=spa2time(K,S.w,g=5.)
    assert C.g==5.


def test_spec2cov_dispatch_preserves_sample_count_and_spacing():
    S=spectrum(81,True,h=6.)
    C=spec2cov(S,3,7,.15,Nx=2,Ny=1,dx=.7,dy=.9)
    expected=dspec2dcov(S,3,6,1,2,1,.7,.9,dt=.15)
    assert C.R.shape==(3,5,7)
    assert_allclose(C.t,np.arange(7)*.15)
    for field in ('R','Rx','Ry','Rt','Rxyt','Rttt'):
        assert_allclose(C[field],expected[field],atol=1e-14)
    B=spec2cov(S,0,5,rate=2,Nx=1)
    assert_allclose(B.t,np.arange(5)*np.pi/S.w[-1]/2)
    K=spec2spec(S,'k2d')
    assert spec2cov(K,0,3,.1,Nx=1).R.shape==(3,3)
    E=spec2spec(spectrum(),'enc',v=3.)
    T=spec2cov(E,2,4,.1)
    assert T.v==3. and T.R.shape==(4,)
    with pytest.raises(NotImplementedError,match='intrinsic'):
        spec2cov(E,Nx=1)
    ED=spec2spec(spectrum(81,True),'encdir',v=3.)
    assert_allclose(spec2cov(ED,1,4,.1).R,spec2cov(spec2spec(ED,'enc'),1,4,.1).R)


def test_spec2cov_wave_number_returns_space_covariance():
    F=spectrum(101)
    K=spec2spec(F,'k1d')
    C=spec2cov(K,2,6,.2)
    assert C.type=='space' and 't' not in C and 'Rx' in C and 'Rxx' in C
    assert_allclose(C.x,np.arange(6)*.2)
    assert_allclose(C.R[0],trapezoid(K.S,x=K.k),rtol=1e-13)
    assert_allclose(C.Rxx[0],-trapezoid(K.S*K.k**2,x=K.k),rtol=1e-13)
    D=spec2spec(spectrum(81,True),'k2d')
    assert spec2cov(D,0,4).R.shape==(4,)


def test_irregular_grid_covariance_is_positive_semidefinite_in_space_and_time():
    from scipy.linalg import toeplitz
    w=np.array([0.,.01,.04,1.,3.])
    theta=np.array([-np.pi,-np.pi+.005,-2.5,.4,np.pi])
    # These axes give negative Simpson weights. A positive density cannot
    # safely be integrated as a signed mixture of harmonic covariances.
    sw=simpson(np.eye(len(w)),x=w,axis=1)
    st=simpson(np.eye(len(theta)),x=theta,axis=1)
    assert np.any(sw<0) and np.any(st<0)
    density=(1+.3*np.cos(theta[:,None]))*(1+w[None,:]**2)
    S=Spectrum(type='dir',w=w,theta=theta,S=density,h=7.,phi=.2)
    C=dspec2dcov(S,0,3,1,2,1,.7,.9,dt=.3)
    points=[(0,0,0),(1,0,1),(-1,1,2),(0,1,3),(1,1,1)]
    gram=np.empty((len(points),len(points)))
    for i,a in enumerate(points):
        for j,b in enumerate(points):
            ix,iy,it=np.asarray(a)-b
            if it<0:ix,iy,it=-ix,-iy,-it
            gram[i,j]=C.R[iy+1,ix+2,it]
    assert np.linalg.eigvalsh(gram).min()>-1e-12*np.linalg.norm(gram,2)
    # Independently integrate the physical harmonic kernel with positive
    # trapezoid control volumes; this tests values as well as eigenvalues.
    k=w2k(w,h=7.)[0]
    phase=.7*k*np.cos(theta[:,None]-.2)-.9*k*np.sin(theta[:,None]-.2)+.6*w
    expected=trapezoid(trapezoid(density*np.cos(phase),x=w,axis=1),x=theta)
    assert_allclose(C.R[0,3,2],expected,rtol=2e-14,atol=1e-14)
    F=Spectrum(type='freq',w=w,S=1+w*w)
    R=spec2cov(F,0,30,.3)
    assert np.linalg.eigvalsh(toeplitz(R.R)).min()>-1e-12*np.linalg.norm(R.R)
    assert_allclose(R.R[0],trapezoid(F.S,x=w),rtol=1e-14)
