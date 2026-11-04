"""Source coefficient identity, conditional moments and joint probability laws."""
import hashlib
from importlib.resources import files
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import integrate, stats
from scipy.interpolate import interp1d

from wafo import wavemodels as w
from wafo.wavemodels import _additional_tables as a


FAMILIES = [
    ('ohhs', w.ohhspdf, w.ohhscdf, w.ohhspdf2, 1),
    ('ohhss', w.ohhsspdf, w.ohhsscdf, w.ohhsspdf2, 1),
    ('ths', w.thspdf, w.thscdf, w.thspdf2, 8),
    ('thss', w.thsspdf, w.thsscdf, w.thsspdf2, 8),
    ('thsnl', w.thsnlpdf, w.thsnlcdf, w.thsnlpdf2, 8),
    ('thv', w.thvpdf, w.thvcdf, w.thvpdf2, 8),
]


def test_all_new_assets_are_exact_numeric_source_conversions():
    records = json.loads(files('wafo.wavemodels').joinpath('data', 'additional_joint.PROVENANCE.json').read_text())
    assert len(records) == 7
    for record in records:
        with files('wafo.wavemodels').joinpath('data', record['asset']).open('rb') as stream:
            with np.load(stream, allow_pickle=False) as archive:
                converted = {key: archive[key] for key in archive.files}
        assert set(converted) == set(record['variables'])
        assert all(np.all(np.isfinite(value)) for value in converted.values())
        source = Path(__file__).resolve().parents[2]/'wafo_matlab'/record['source']
        if source.exists():
            from scipy.io import loadmat
            assert hashlib.sha256(source.read_bytes()).hexdigest() == record['sha256']
            original = loadmat(source)
            for key, value in converted.items():
                assert_array_equal(value, original[key])


@pytest.mark.parametrize('family,pdf,cdf,grid,state', FAMILIES)
def test_exact_conditional_table_node_or_source_bandwidth_formula(family, pdf, cdf, grid, state):
    data = a._table(family)
    row = np.flatnonzero(data['Hm0'].ravel() == 6)[0]
    ih = 8
    height, v = float(data['h2'].ravel()[ih]), 1.2
    if family.startswith('oh'):
        raw_a, raw_b = data['A11s'][row, 0, ih], data['B11s'][row, 0, ih]
        A, B = (np.exp(raw_a), np.exp(raw_b)) if family == 'ohhs' else (raw_a, raw_b)
        expected = stats.weibull_min.pdf(v, B, scale=A)
    elif family == 'thv':
        col = np.flatnonzero(data['Tp'].ravel() == state)[0]
        bandwidth = data['eps2'][row, col]
        A = float(interp1d(data['e2'].ravel(), data['A11s'][ih], kind='cubic')(bandwidth))
        B = float(interp1d(data['e2'].ravel(), data['B11s'][ih], kind='cubic')(bandwidth))
        expected = stats.weibull_min.pdf(v, B, scale=A)
    else:
        col = np.flatnonzero(data['Tp'].ravel() == state)[0]
        A, B = data['A11s'][row, col, ih], data['B11s'][row, col, ih]
        expected = stats.gamma.pdf(v, A, scale=B)
    assert_allclose(pdf(height, v, 6, state, True, 1), expected, rtol=3e-13)
    kind, parameters = a._marginal(family, np.asarray(6.), np.asarray(state))
    # Its separate marginal curve must multiply the conditional without a
    # common, dimensionally incorrect Hrms*Vrms divisor.
    marginal = a._marginal_pdf(height, kind, parameters)
    assert_allclose(pdf(height, v, 6, state, True), expected*marginal, rtol=3e-13)


@pytest.mark.parametrize('family,pdf,cdf,grid,state', FAMILIES)
def test_all_conditional_densities_and_physical_moments(family, pdf, cdf, grid, state):
    height = 3.
    _, hs, vs, *_ = pdf(height, .1, 6, state, full_output=True)
    integral = integrate.quad(lambda x: pdf(height, x, 6, state, condon=1), 0, np.inf)[0]
    assert_allclose(integral, 1, atol=2e-9)
    for mode, power in ((3, 1), (4, 2)):
        moment = integrate.quad(lambda x: pdf(height, x, 6, state, condon=mode), 0, np.inf)[0]
        A, B = (float(v) for v in a._conditional_parameters(family, height/hs, 6., float(state)))
        expected = (stats.gamma.moment(power, A, scale=B) if family in ('ths', 'thss', 'thsnl') else
                    stats.weibull_min.moment(power, B, scale=A))*vs**power
        assert_allclose(moment, expected, rtol=1e-8)
    velocity = float(vs)*.8
    physical = pdf(height, velocity, 6, state)
    normalized = pdf(height/hs, velocity/vs, 6, state, True)
    assert_allclose(normalized, physical*hs*vs, rtol=2e-13)
    assert_allclose(pdf(height/hs, velocity/vs, 6, state, True, 1), pdf(height, velocity, 6, state, condon=1)*vs)
    eps = velocity*.0001
    derivative = (pdf(height, velocity+eps, 6, state, condon=6)-pdf(height, velocity-eps, 6, state, condon=6))/(2*eps)
    assert_allclose(derivative, pdf(height, velocity, 6, state, condon=1), rtol=2e-7)


@pytest.mark.parametrize('family,pdf,cdf,grid,state', FAMILIES)
def test_all_cdfs_match_marginal_factorization_and_joint_survival(family, pdf, cdf, grid, state):
    height = 3.
    _, _, scale, *_ = pdf(height, .1, 6, state, full_output=True)
    velocity = .8*float(scale)
    lower, error = cdf(height, velocity, 6, state, full_output=True)
    upper = cdf(height, velocity, 6, state, tail=1)
    marginal_h = cdf(height, np.inf, 6, state)
    marginal_v = cdf(np.inf, velocity, 6, state)
    assert error < 1e-7 and 0 < lower < marginal_h < 1
    assert_allclose(upper, 1-marginal_h-marginal_v+lower, atol=3e-8)
    eps = .0001
    derivative = (cdf(height+eps, velocity, 6, state, tol=1e-10)-cdf(height-eps, velocity, 6, state, tol=1e-10))/(2*eps)
    assert_allclose(derivative, pdf(height, velocity, 6, state, condon=5), rtol=2e-5)
    assert_allclose(cdf(np.inf, np.inf, 6, state), 1)
    assert_allclose(cdf(height, -1, 6, state, tail=1), 1-marginal_h)
    assert_allclose(cdf(0, velocity, 6, state), 0)


@pytest.mark.parametrize('family,pdf,cdf,grid,state', FAMILIES)
def test_all_grid_mappings_and_broadcast_sea_states(family, pdf, cdf, grid, state):
    heights = np.array([.2, 1., 2., 3.])
    characteristic = np.array([.1, .4, 1.2])
    result = grid(heights, characteristic, 6, state, True)
    assert result.f.shape == (4, 3) and result.norm == 1
    assert_array_equal(result.x[0], characteristic)
    assert_array_equal(result.x[1], heights)
    assert_allclose(result.f, pdf(heights[:, None], characteristic, 6, state, True))
    boundary = grid([0., .1, 1.], [0., .1, 1.], 6, state, True)
    assert not np.any(np.isnan(boundary.f))
    states = np.array([5., 6.])
    actual = pdf([1., 2.], [.5, 1.], states, state, True)
    expected = [pdf(h, v, H, state, True) for h, v, H in zip([1., 2.], [.5, 1.], states)]
    assert_allclose(actual, expected)
    assert_array_equal(pdf([1., np.inf], [np.inf, 1.], 6, state), [0., 0.])


def test_oh_velocity_original_grid_interface_and_spectral_scale():
    heights = np.linspace(0, 4, 401)
    velocities = np.linspace(0, 8, 501)
    pdf, scale_table, shape_table = w.ohhvpdf(velocities, heights, 6, 2, True, full_output=True)
    assert pdf.f.shape == (401, 501) and pdf.norm == 1
    assert scale_table.f.shape == shape_table.f.shape == (24, 11, 24)
    assert_allclose(integrate.simpson(integrate.simpson(pdf.f, x=velocities, axis=1), x=heights), 1, atol=2e-4)
    data = a._table('ohhv'); row, ih = 11, 8
    h = float(data['h2'].ravel()[ih]); v = 1.2
    A, B = data['A11s'][row, 1, ih], data['B11s'][row, 1, ih]
    kind, params = a._marginal('ohhv', np.asarray(6.), np.asarray(2.))
    expected = a._marginal_pdf(h, kind, params)*stats.weibull_min.pdf(v, B, scale=A)
    assert_allclose(w.ohhvpdf([v], [h], 6, 2, True).f, [[expected]], rtol=1e-13)
    hs = 6/np.sqrt(2)
    vs = 12/a._oh_velocity_period(6., 2)
    assert_allclose(w.ohhvpdf([v*vs], [h*hs], 6, 2).f*hs*vs, [[expected]], rtol=1e-13)


@pytest.mark.parametrize('family,pdf,cdf,grid,state', FAMILIES)
def test_joint_density_integrates_to_exact_finite_height_probability(family, pdf, cdf, grid, state):
    # Keep heights inside every fitted conditional range.  A geometric
    # characteristic grid resolves both narrow small-wave kernels and tails.
    # The spatial OH marginal has steep curvature near zero; a geometric
    # height grid resolves it instead of relaxing the conservation tolerance.
    h = np.geomspace(.001, 2.3, 501)
    v = np.geomspace(1e-5, 40, 701)
    result = grid(h, v, 6, state, True)
    actual = integrate.simpson(integrate.simpson(result.f, x=v, axis=1), x=h)
    hs = 6/np.sqrt(2)
    expected = cdf(h[-1]*hs, np.inf, 6, state)-cdf(h[0]*hs, np.inf, 6, state)
    assert_allclose(actual, expected, atol=2e-5)


def test_explicit_fitted_domains():
    for family, pdf, cdf, grid, state in FAMILIES:
        with pytest.raises(ValueError):
            pdf(1, .1, .1, state)
    with pytest.raises(ValueError, match='definition'):
        w.ohhspdf(1, .1, 6, 12)
    with pytest.raises(ValueError, match='Torsethaugen'):
        w.thspdf(1, .1, 10, 5)
    with pytest.raises(ValueError, match='bandwidth'):
        w.thvpdf(1, .1, .5, 20)
    with pytest.raises(ValueError):
        w.thspdf2([2, 1], [0, 1])
