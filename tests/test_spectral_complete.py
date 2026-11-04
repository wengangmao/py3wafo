"""Analytic/invariance checks for every implemented spectral numerical API."""
import inspect
import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from scipy.integrate import simpson,trapezoid
from wafo import spec


TESTED_FUNCTIONS = [
    'Spectrum','createspec','pmspec','getjonswappeakedness','jonswap','spec2mom',
    'spec2bw','spec2dt','spec2char','spec2cov','k2w','w2k','freqtype','lagtype',
    'createcov','phi1','tmaspec','ttspec','scalespec','wnormspec','specinterp',
    'covinterp','cov2spec','spec2cov2','spec2spec','rotspec','spreading','mkdspec',
    'demospec','ohspec','ohspec2','ohspec3','mccormick','wallop','oscspec',
    'torsethaugen','specdiag','dspec2char','wspecplot']


@pytest.fixture
def gaussian():
    w=np.linspace(0,9,1001)
    return spec.createspec(w=w,S=np.sqrt(2/np.pi)*np.exp(-w*w/2))


def test_spectrum_constructor_fields_and_metadata():
    S=spec.Spectrum(a=1)
    assert S.a==1
    S.b=2
    assert S['b']==2
    with pytest.raises(AttributeError):
        _=S.missing
    assert spec.freqtype(spec.createspec())=='w'
    assert spec.freqtype(spec.createspec('freq','f'))=='f'
    assert spec.freqtype(spec.createspec('k1d'))=='k'
    assert 'theta' in spec.createspec('dir')
    assert 'k2' in spec.createspec('k2d')
    assert spec.createspec('enc').v==0
    C=spec.createcov(2,'xyt')
    assert {'Rx','Ry','Rt','Rxy','Rxt','Ryt','Rtt','Rxx','Ryy'}<=C.keys()
    C.t=np.arange(3)
    assert spec.lagtype(C)=='t'
    with pytest.raises(ValueError):
        spec.createspec('nonsense')


def test_moment_covariance_and_characteristic_analytic_values(gaussian):
    assert_allclose(spec.spec2mom(gaussian,4),[1,1,3],atol=1e-12)
    assert_allclose(spec.spec2mom(gaussian,2,even=False),[1,np.sqrt(2/np.pi),1],atol=1e-10)
    C=spec.spec2cov(gaussian,2,11,.1)
    expected=np.exp(-C.t*C.t/2)
    assert_allclose(C.R,expected,atol=1e-12)
    assert_allclose(C.Rt,-C.t*expected,atol=1e-12)
    assert_allclose(C.Rtt,(C.t*C.t-1)*expected,atol=1e-12)
    matrix=spec.spec2cov2(gaussian,2,10,.1)
    assert matrix.shape==(11,3)
    assert_allclose(matrix,np.column_stack((C.R,C.Rt,C.Rtt)),atol=1e-12)
    assert_allclose(spec.spec2char(gaussian,['Hm0','Tm02','alpha']),[4,2*np.pi,1/np.sqrt(3)])
    bw=spec.spec2bw(gaussian,[1,2,3,4])
    assert_allclose(bw[:3],[1/np.sqrt(3),np.sqrt(np.pi/2-1),np.sqrt(2/3)])
    assert_allclose(spec.spec2dt(gaussian),np.pi/9)


@pytest.mark.parametrize('h',[.01,.5,100,np.inf])
def test_dispersion_relation_including_shallow_and_negative_frequencies(h):
    w=np.array([0,.001,.1,1,8])
    kx,ky=spec.w2k(w,.4,h)
    back,angles=spec.k2w(kx,ky,h)
    assert_allclose(back,w,atol=1e-11,rtol=1e-10)
    assert_allclose(angles[1:],.4)
    assert_allclose(spec.w2k(-w,.4,h)[0],-kx,atol=1e-14)


def test_finite_depth_transfer_and_tma():
    w=np.linspace(0,3,501)
    assert_allclose(spec.phi1(w,np.inf),1)
    factor=spec.phi1(w,20)
    assert factor[0]==0 and np.all((factor>=0)&(factor<=1+1e-12))
    k=spec.w2k(w,h=20)[0]
    th=np.tanh(k*20)
    expected=np.divide(th**3,th+20*k*(1-th*th),out=np.zeros_like(w),where=k>0)
    assert_allclose(factor,expected,atol=1e-12)
    T=spec.tmaspec(w,[4,8],20)
    J=spec.jonswap(w,[4,8])
    assert_allclose(T.S,J.S*factor)
    assert T.h==20


def test_unit_changes_preserve_density_and_moments(gaussian):
    F=spec.ttspec(gaussian)
    assert spec.freqtype(F)=='f'
    assert 'w' not in F
    assert_allclose(spec.spec2mom(F,4),[1,1,3],atol=1e-12)
    assert_allclose(spec.spec2cov(F,1,5,.1).R,spec.spec2cov(gaussian,1,5,.1).R)
    W=spec.ttspec(F)
    assert_allclose(W.w,gaussian.w)
    assert_allclose(W.S,gaussian.S)
    assert_allclose(spec.spec2dt(F),spec.spec2dt(gaussian))


def test_scaling_normalization_and_resampling(gaussian):
    scaled=spec.scalespec(gaussian,4,9)
    assert_allclose(spec.spec2mom(scaled,2),[4,9],rtol=1e-12)
    normalized=spec.wnormspec(scaled)
    assert_allclose(spec.spec2mom(normalized,2),[1,1],rtol=1e-12)
    normalized2=spec.wnormspec(scaled,4,9)
    assert_allclose(normalized2.S,normalized.S,rtol=1e-12)
    resampled=spec.specinterp(gaussian,dt=.2)
    assert resampled.w[0]==0
    assert_allclose(resampled.w[-1],np.pi/.2)
    assert len(resampled.w)-1==2**int(np.log2(len(resampled.w)-1))
    assert_allclose(spec.spec2mom(resampled,0),[1],rtol=2e-5)


def test_covariance_inverse_matches_gaussian_spectral_transform():
    R=spec.createcov()
    R.t=np.linspace(0,15,301)
    R.R=np.exp(-R.t*R.t/2)
    S=spec.cov2spec(R,trunc=0,fast=False)
    expected=np.sqrt(2/np.pi)*np.exp(-S.w*S.w/2)
    assert_allclose(S.S,expected,atol=2e-14)
    interpolated=spec.cov2spec(R,rate=4)
    assert len(interpolated.w)==(513-1)*4+1
    C=spec.covinterp(R,[.025,401])
    assert_allclose(C.R,np.exp(-C.t*C.t/2),atol=2e-7)
    assert_allclose(spec.covinterp(R,.1).t,np.arange(151)*.1)
    with pytest.raises(ValueError):
        spec.covinterp(R,[0,16,17])


@pytest.mark.parametrize('h',[10,np.inf])
def test_frequency_wave_number_jacobians_roundtrip(h):
    S=spec.jonswap(np.linspace(0,6,2001),[4,8])
    S.h=h
    K=spec.spec2spec(S,'k1d')
    W=spec.spec2spec(K,'freq')
    assert spec.freqtype(K)=='k'
    assert_allclose(W.w,S.w,atol=1e-12)
    assert_allclose(W.S,S.S,atol=1e-12)
    assert_allclose(spec.spec2mom(K,0),spec.spec2mom(S,0),rtol=2e-6)
    if np.isinf(h):
        encountered=spec.spec2spec(S,'enc',v=10)
        assert encountered.type=='enc'
        assert_allclose(trapezoid(encountered.S,x=encountered.w),
                        trapezoid(S.S,x=S.w),rtol=1e-12)
    else:
        with pytest.raises(NotImplementedError):
            spec.spec2spec(S,'enc',v=10)


@pytest.mark.parametrize('kind',['cos2s','poisson','wnormal','mises','sech2','box'])
def test_all_spreading_kernels_have_unit_mass_and_consistent_first_moment(kind):
    theta=np.linspace(-np.pi,np.pi,4001)
    D=spec.spreading(theta,kind,.3,15,w=[0,.5,1],definition=0)
    assert D.S.shape==(len(theta),1)
    assert_allclose(trapezoid(D.S,x=theta,axis=0),1,atol=1e-12)
    r1=trapezoid(D.S[:,0]*np.cos(theta-.3),x=theta)
    assert_allclose(r1,15/16,atol=.0003)
    assert np.all(D.S>=0)


def test_spreading_frequency_law_directional_composition_rotation_and_units():
    S=spec.jonswap(np.linspace(0,3,101),[4,8])
    D=spec.spreading(401,'cos2s',np.linspace(0,.4,len(S.w)),w=S.w)
    DS=spec.mkdspec(S,D)
    assert DS.S.shape==(401,101)
    assert_allclose(spec.spec2spec(DS,'freq').S,S.S,atol=1e-12)
    assert_allclose(spec.spec2mom(DS,2),spec.spec2mom(S,2),atol=1e-12)
    degrees=spec.ttspec(DS,'f','degrees')
    back=spec.ttspec(degrees,'w','radians')
    assert_allclose(back.S,DS.S,atol=1e-12)
    assert_allclose(back.theta,DS.theta)
    rotated=spec.rotspec(DS,.7)
    assert_allclose(rotated.phi,.7)
    assert_allclose(rotated.S,DS.S)
    physical=spec.rotspec(rotated,0,True)
    assert physical.phi==0
    assert_allclose(trapezoid(physical.S,x=physical.theta,axis=0),S.S,atol=1e-10)


def test_all_directional_characteristics_and_spectral_diagonalization():
    S=spec.demospec('dir')
    values,names=spec.dspec2char(S,np.arange(1,27),return_names=True)
    assert len(values)==len(names)==26
    assert_allclose(spec.dspec2char(S,['Mdir','Spr','S1']),[0,np.sqrt(2/16),15],atol=1e-10)
    assert spec.demospec().type=='freq'
    assert spec.demospec('k1d').type=='k1d'
    matrix=np.array([[2,1+1j],[1-1j,3]],complex)
    largest,smallest,theta,nu,Z=spec.specdiag(matrix)
    assert_allclose(Z@np.diag([largest,smallest])@Z.conj().T,matrix,atol=1e-12)
    assert largest>=smallest>=0 and np.isfinite(theta+nu)
    entries=spec.specdiag(2,3,1+1j)
    assert_allclose(entries[:4],[largest,smallest,theta,nu])
    batch=np.stack((matrix,2*matrix),axis=-1)
    result=spec.specdiag(batch)
    assert result[-1].shape==(2,2,2)
    assert_allclose(result[0],[largest,2*largest])


@pytest.mark.parametrize('name,data',[('pmspec',[4,8]),('jonswap',[4,8]),
    ('ohspec',[4,8,3]),('mccormick',[4,8,8/1.228]),('wallop',[4,8,5])])
def test_parametric_spectral_height_normalizations(name,data):
    S=getattr(spec,name)(np.linspace(0,80,50001),data)
    assert np.all(np.isfinite(S.S)) and np.all(S.S>=0)
    assert_allclose(spec.spec2mom(S,0),[1],rtol=8e-5)
    assert S.S[0]==0


@pytest.mark.parametrize('definition',range(1,12))
def test_ochi_hubble_confidence_family(definition):
    S,parts=spec.ohspec2(np.linspace(0,30,10001),[7,definition],components=True)
    assert_allclose(S.S,parts[0].S+parts[1].S)
    assert np.all(np.isfinite(S.S)) and np.all(S.S>=0)
    assert spec.spec2mom(S,0)[0]>0


@pytest.mark.parametrize('definition',range(1,10))
def test_ochi_hubble_mixed_seas(definition):
    S,parts=spec.ohspec3(definition=definition,components=True)
    assert_allclose(S.S,parts[0].S+parts[1].S)
    assert np.all(np.isfinite(S.S)) and S.S.max()>0
    assert len(spec.ohspec3(definition=[1,2]))==2


@pytest.mark.parametrize('parameters',[[4,6,-1],[4,12,-1],[0,8,-1],[7,11,0]])
def test_torsethaugen_wind_swell_additivity(parameters):
    S,wind,swell=spec.torsethaugen(sdata=parameters,components=True)
    assert_allclose(S.S,wind.S+swell.S)
    assert np.all(np.isfinite(S.S)) and np.all(S.S>=0)
    # A=0 is the source's approximate empirical normalization, not numerical
    # mass normalization. Its allowed moment discrepancy is part of the model.
    assert_allclose(spec.spec2mom(S,0),[(parameters[0]/4)**2],rtol=.04 if parameters[2]==0 else .02,atol=1e-12)


def test_oscillator_formula_and_automatic_peakedness():
    S=spec.oscspec([0,4,401],.1,2,3)
    expected=9/((4-S.w*S.w)**2+.16*S.w*S.w)/(2*np.pi)
    assert_allclose(S.S[1:],expected[1:])
    assert S.S[0]==0
    assert_allclose(spec.getjonswappeakedness([1,4],[10,20]),[1,1])
    gamma=spec.getjonswappeakedness(7,11)
    assert 1<=gamma<=7


@pytest.mark.parametrize('factory,args',[('pmspec',(-1,)),('ohspec',(None,[4,8,-1])),
    ('wallop',(None,[4,8,1])),('oscspec',([0,4,100],0)),
    ('spreading',(5,'not-a-family')),('torsethaugen',(None,[4,8,1]))])
def test_invalid_model_inputs_raise(factory,args):
    with pytest.raises(ValueError):
        getattr(spec,factory)(*args)


def test_invalid_measurements_raise_instead_of_nan_results(gaussian):
    empty=spec.createspec(w=np.arange(3),S=np.zeros(3))
    for f in (spec.spec2bw,spec.spec2char):
        with pytest.raises(ValueError):
            f(empty)
    with pytest.raises(ValueError):
        spec.spec2bw(gaussian,[1.5])
    with pytest.raises(ValueError):
        spec.spec2mom(gaussian,-1)
    with pytest.raises(ValueError):
        spec.w2k([1],h=np.nan)
    with pytest.raises(ValueError):
        spec.k2w([1],g=np.inf)


def test_all_spectrum_plot_modes_render_to_png(tmp_path):
    matplotlib=pytest.importorskip('matplotlib')
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    for mode in (1,2,3):
        axes=spec.wspecplot(spec.demospec(),mode)
        figure=axes[0].figure if isinstance(axes,list) else axes.figure
        destination=tmp_path/f'frequency-{mode}.png'
        figure.savefig(destination)
        assert destination.stat().st_size>1000
        plt.close(figure)
    S=spec.demospec('dir')
    for mode in range(1,8):
        axes=spec.wspecplot(S,mode)
        figure=axes[0].figure if isinstance(axes,list) else axes.figure
        destination=tmp_path/f'directional-{mode}.png'
        figure.savefig(destination)
        assert destination.stat().st_size>1000
        plt.close(figure)
    K=spec.createspec('k2d')
    K.k=K.k2=np.linspace(-2,2,21)
    K.S=np.exp(-(K.k[:,None]**2+K.k2[None,:]**2))
    for mode in (1,2):
        axes=spec.wspecplot(K,mode)
        destination=tmp_path/f'cartesian-{mode}.png'
        axes.figure.savefig(destination)
        assert destination.stat().st_size>1000
        plt.close(axes.figure)
