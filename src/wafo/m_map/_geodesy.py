"""Spherical, ellipsoidal and fixed-dipole coordinate geometry.

M_Map's sphere radius is retained. Ellipsoid solving uses PROJ's geodesic
implementation instead of the old Vincenty iteration, including antipodes.
IGRF-2000 rotation coefficients are from the supplied mc_coords.m source.
"""
import numpy as np

EARTH_RADIUS = 6378137.0
_ELLIPSOIDS = {
    'normal':(1.,0.),'sphere':(6370997.,0.),'grs80':(6378137.,1/298.257),
    'grs67':(6378160.,1/247.247),'wgs84':(6378137.,1/298.257223563),
    'wgs72':(6378135.,1/298.260),'wgs66':(6378145.,1/298.250),
    'wgs60':(6378165.,1/298.300),'clrk66':(6378206.4,1/294.980),
    'clrk80':(6378249.1,1/293.466),'intl24':(6378388.,1/297.),
    'intl67':(6378157.5,1/298.250),
}


def _pyproj():
    try:
        import pyproj
    except ImportError as exc:
        raise ImportError('Map projections and ellipsoidal geodesics require optional pyproj') from exc
    return pyproj


def _geod(spheroid='wgs84'):
    if isinstance(spheroid,str):
        key=spheroid.lower()
        if key not in _ELLIPSOIDS:
            raise ValueError('Unknown source spheroid: '+spheroid)
        a,f=_ELLIPSOIDS[key]
    else:
        a,f=np.asarray(spheroid,dtype=float)
        if not np.isfinite(a+f) or a<=0 or not 0<=f<1:
            raise ValueError('Spheroid must be (positive semimajor axis, flattening in [0,1))')
    return _pyproj().Geod(a=a,f=f)


def _coordinates(*args):
    values=np.broadcast_arrays(*[np.asarray(a,dtype=float) for a in args])
    if not all(np.all(np.isfinite(a)) for a in values):
        raise ValueError('Coordinates must be finite')
    for latitude in values[1::2]:
        if np.any(abs(latitude)>90):
            raise ValueError('Latitude must lie in [-90,90] degrees')
    return values


def m_idist(lon1,lat1,lon2,lat2,spheroid='wgs84'):
    """Return (metres, forward azimuth, backward azimuth); azimuths in [0,360)."""
    a,b,c,d=_coordinates(lon1,lat1,lon2,lat2)
    az1,az2,distance=_geod(spheroid).inv(a,b,c,d)
    return np.asarray(distance),np.mod(az1,360),np.mod(az2,360)


def m_fdist(lon1,lat1,a12,s,spheroid='wgs84'):
    """Forward ellipsoidal geodesic: return longitude, latitude, back azimuth."""
    a,b,az,distance=np.broadcast_arrays(lon1,lat1,a12,s)
    if not np.all(np.isfinite([a,b,az,distance])) or np.any(abs(b)>90) or np.any(distance<0):
        raise ValueError('Expected finite geodetic coordinates, azimuths and nonnegative distances')
    lon,lat,back=_geod(spheroid).fwd(a,b,az,distance)
    return np.asarray(lon),np.asarray(lat),np.mod(back,360)


def m_geodesic(lon1,lat1,lon2,lat2,Npoints=100,spheroid='wgs84'):
    """Return distance/lon/lat arrays with each geodesic in a column."""
    if int(Npoints)!=Npoints or Npoints<2:
        raise ValueError('Npoints must be an integer >=2')
    a,b,c,d=_coordinates(lon1,lat1,lon2,lat2)
    distance,az,_=m_idist(a,b,c,d,spheroid)
    s=np.linspace(0,1,int(Npoints))[:,None]*distance.ravel()[None,:]
    lon,lat,_=m_fdist(a.ravel()[None,:],b.ravel()[None,:],np.asarray(az).ravel()[None,:],s,spheroid)
    return s,lon,lat


def m_lldist(long,lat,N=None):
    """Source haversine distances (km); optional N+1 point spherical geodesics."""
    lon,latitude=_coordinates(long,lat)
    lon,latitude=lon.ravel(),latitude.ravel()
    radians=np.deg2rad(latitude)
    dlon,dlat=np.diff(np.deg2rad(lon)),np.diff(radians)
    h=np.sin(dlat/2)**2+np.cos(radians[:-1])*np.cos(radians[1:])*np.sin(dlon/2)**2
    angle=2*np.arctan2(np.sqrt(np.clip(h,0,1)),np.sqrt(np.clip(1-h,0,1)))
    distance=EARTH_RADIUS/1000*angle
    if N is None:
        return distance
    if int(N)!=N or N<1:
        raise ValueError('N must be a positive integer')
    # Use the robust spherical geodesic solver for antipodal/identical points.
    _,paths_lon,paths_lat=m_geodesic(lon[:-1],latitude[:-1],lon[1:],latitude[1:],int(N)+1,(EARTH_RADIUS,0))
    return distance,paths_lon.T,paths_lat.T


def m_xydist(x,y):
    from ._projection import m_xy2ll
    return m_lldist(*m_xy2ll(x,y))


def _rotation():
    g10,g11,h11=-29615.,-1728.,5186.
    lam=np.arctan(h11/g11)
    phi=np.arctan((g11*np.cos(lam)+h11*np.sin(lam))/g10)
    z=np.array([[np.cos(lam),np.sin(lam),0],[-np.sin(lam),np.cos(lam),0],[0,0,1]])
    y=np.array([[np.cos(phi),0,-np.sin(phi)],[0,1,0],[np.sin(phi),0,np.cos(phi)]])
    return y@z


def _rotate(lon,lat,east=None,north=None,inverse=False):
    if east is not None and north is not None:
        lon,lat,east,north=np.broadcast_arrays(lon,lat,east,north)
    lon,lat=_coordinates(lon,lat)
    shape=lon.shape
    lo,la=np.deg2rad(lon.ravel()),np.deg2rad(lat.ravel())
    cart=np.array([np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)])
    transform=_rotation().T if inverse else _rotation()
    mapped=transform@cart
    newlo=np.arctan2(mapped[1],mapped[0]); newla=np.arcsin(np.clip(mapped[2],-1,1))
    result=(np.rad2deg(newlo).reshape(shape),np.rad2deg(newla).reshape(shape))
    if east is None and north is None:
        return result
    if east is None or north is None:
        raise ValueError('Both east and north vector components are required')
    east,north=np.broadcast_arrays(east,north,lon)[:2]
    basis_e=np.array([-np.sin(lo),np.cos(lo),np.zeros_like(lo)])
    basis_n=np.array([-np.sin(la)*np.cos(lo),-np.sin(la)*np.sin(lo),np.cos(la)])
    vector=transform@(basis_e*east.ravel()+basis_n*north.ravel())
    new_e=np.array([-np.sin(newlo),np.cos(newlo),np.zeros_like(newlo)])
    new_n=np.array([-np.sin(newla)*np.cos(newlo),-np.sin(newla)*np.sin(newlo),np.cos(newla)])
    return (*result,np.sum(vector*new_e,axis=0).reshape(shape),np.sum(vector*new_n,axis=0).reshape(shape))


def m_geo2mag(longGEO,latGEO,phiVecGEO=None,thetaVecGEO=None):
    """IGRF-2000 dipole rotation; optional east/north vector components."""
    return _rotate(longGEO,latGEO,phiVecGEO,thetaVecGEO)


def m_mag2geo(longMAG,latMAG,phiVecMAG=None,thetaVecMAG=None):
    """Inverse of m_geo2mag, including tangent-vector rotation."""
    return _rotate(longMAG,latMAG,phiVecMAG,thetaVecMAG,True)
