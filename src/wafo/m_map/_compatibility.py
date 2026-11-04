"""Source-compatible numeric M_Map private helpers.

The projection engines themselves are implemented with PROJ and retain their
public m_* entry points. Their private MATLAB graphics/global-state dispatchers
are not represented by aliases here.
"""
import numpy as np
from ._geodesy import m_geo2mag, m_mag2geo


def mc_ellips():
    """Return the twelve historical (semimajor axis, flattening) definitions.

    Values reproduce ``private/mc_ellips.m`` literally, including its rounded
    WGS84 flattening and the unusual GRS67 value. Public geodesy defaults use
    the modern precise WGS84 definition; pass one of these numeric pairs to
    request the original toolbox's historical parameters explicitly. Returned
    arrays are independent copies and can be safely modified by the caller.
    """
    return {name: np.array([a, f], dtype=float) for name, a, f in (
        ('normal', 1., 0.), ('sphere', 6370997., 0.),
        ('grs80', 6378137., 1/298.257), ('grs67', 6378160., 1/247.247),
        ('wgs84', 6378137., 1/298.257), ('wgs72', 6378135., 1/298.260),
        ('wgs66', 6378145., 1/298.250), ('wgs60', 6378165., 1/298.300),
        ('clrk66', 6378206.4, 1/294.97870), ('clrk80', 6378249.1, 1/293.466),
        ('intl24', 6378388., 1/297.), ('intl67', 6378157.5, 1/298.250))}


def mc_coords(option, *args, system='IGRF2000-geomagnetic'):
    """Coordinate-engine dispatcher: name, parameters, geo2mag, or mag2geo.

    ``parameters`` returns the original dipole Euler angles in radians; numeric
    coordinates and east/north vectors use the same rotation as m_geo2mag and
    m_mag2geo. Geographic ``system`` is the identity rotation. The default
    fixed IGRF2000 system replaces the original implicit mutable MATLAB globals
    with an explicit selection and preserves array shapes/broadcasting.
    """
    names = ('geographic', 'IGRF2000-geomagnetic')
    option = str(option).lower()
    if option == 'name':
        if args:
            raise TypeError('name takes no coordinate arguments')
        return {'name': names}
    if option == 'parameters':
        if len(args) != 1:
            raise TypeError('parameters takes one coordinate-system name')
        system = args[0]
    key = str(system).lower()
    if key not in ('geographic', 'igrf2000-geomagnetic'):
        raise ValueError('Unrecognized coordinate system')
    if option == 'parameters':
        if key == 'geographic':
            return {'name': names[0], 'lambda': 0., 'phi': 0.}
        longitude = np.arctan(5186./-1728.)
        tilt = np.arctan((-1728*np.cos(longitude)+5186*np.sin(longitude))/-29615.)
        return {'name': names[1], 'lambda': longitude, 'phi': tilt}
    if option not in ('geo2mag', 'mag2geo'):
        raise ValueError('Unknown mc_coords operation')
    if len(args) not in (2, 4):
        raise TypeError('Provide longitude, latitude, and optionally both east/north vector components')
    if key == 'geographic':
        values = tuple(np.asarray(x, dtype=float).copy() for x in np.broadcast_arrays(*args))
        if any(np.any(~np.isfinite(x)) for x in values) or np.any(abs(values[1]) > 90):
            raise ValueError('Expected finite coordinates/vectors and latitude in [-90,90]')
        return values
    return (m_geo2mag if option == 'geo2mag' else m_mag2geo)(*args)
