import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

import wafo.m_map as m

FAMILIES = ['mp_azim', 'mp_conic', 'mp_cyl', 'mp_omerc', 'mp_tmerc', 'mp_utm']


@pytest.mark.parametrize('family', FAMILIES)
def test_every_projection_engine_round_trips_interior_points(family):
    engine = getattr(m, family)
    assert set(engine('usage')['commands']) >= {'initialize', 'll2xy', 'xy2ll', 'box'}
    for name in engine('name'):
        options = dict(lon=[-12, 12], lat=[25, 65]) if family == 'mp_utm' else {}
        state = engine('initialize', name, **options)
        info = engine('get', state=state)
        lon = np.linspace(*np.asarray(info['lon'], float), 5)[1:-1]
        lat = np.linspace(*np.asarray(info['lat'], float), 5)[1:-1]
        lon, lat = np.meshgrid(lon, lat)
        x, y, _ = engine('ll2xy', lon, lat, 'clip', 'off', state=state)
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        back = engine('xy2ll', x, y, state=state)
        assert_allclose(back[0], lon, atol=1e-9, err_msg=name)
        assert_allclose(back[1], lat, atol=1e-9, err_msg=name)
        bx, by = engine('box', state=state)
        assert bx.shape == by.shape and len(bx) > 4


def test_projection_formulas_match_analytic_values():
    state = m.mp_cyl('initialize', 'Mercator')
    x, y, _ = m.mp_cyl('ll2xy', np.array([30.]), np.array([45.]), state=state)
    assert_allclose([x[0], y[0]], [np.radians(30), np.log(np.tan(np.pi/4+np.radians(45)/2))])
    state = m.mp_tmerc('initialize', 'Sinusoidal')
    x, y, _ = m.mp_tmerc('ll2xy', np.array([30.]), np.array([60.]), state=state)
    assert_allclose([x[0], y[0]], [np.radians(30)*.5, np.radians(60)])
    state = m.mp_azim('initialize', 'Stereographic', lon=10, lat=50, radius=20)
    x, y, _ = m.mp_azim('ll2xy', np.array([10., 10.]), np.array([50., 70.]), state=state)
    assert_allclose(x, 0, atol=1e-15)
    assert_allclose(y, [0, 2*np.tan(np.radians(20)/2)])
    grid = m.mp_cyl('xgrid', [-90, 0, 90], 'bottom', state=m.mp_cyl('initialize', 'Mercator'))
    assert len(grid) == 4


def test_map_utilities_limits_ticks_and_coast_extraction():
    assert_allclose(m.mu_util('axisticks', 0, 10, 5), [0, 2, 4, 6, 8, 10])
    state = m.mp_azim('initialize', 'Stereographic')
    xlim, ylim = m.mu_util('xylimits', state=state)
    assert xlim[0] < xlim[1] and ylim[0] < ylim[1]
    lonlim, latlim = m.mu_util('lllimits', state=state)
    assert_allclose(lonlim, [-180, 180])
    with pytest.raises(NotImplementedError):
        m.mu_util('nonsense', state=state)
    state = m.mp_cyl('initialize', 'Mercator', lon=[-20, 20], lat=[40, 60])
    coast, area, k = m.mu_coast('default', return_data=True, state=state)
    assert coast.shape[1] == 2 and len(area) > 0
    # k holds zero-based NaN delimiter rows between coastline segments.
    assert np.all(np.isnan(coast[np.asarray(k, int)[(k >= 0) & (k < len(coast))]]).any(axis=1))
    finite = coast[np.isfinite(coast).all(axis=1)]
    assert len(finite) > 100


def test_clabel_labels_native_and_packed_contours():
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    labels = m.clabel(ax.contour(X, Y, X**2+Y**2, [.25, .5]), ax=ax)
    assert len(labels) > 0
    packed = np.array([[.5, 0, 1, 2.], [3, 0, 1, 0]])
    artists = m.clabel(packed, ax=ax)
    assert any(getattr(a, 'get_text', lambda: '')() == '0.5' for a in artists)
    plt.close(fig)
