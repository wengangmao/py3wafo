"""Context-local cartographic projections backed by optional PROJ/pyproj."""
from contextvars import ContextVar
from copy import deepcopy
import numpy as np
from ._geodesy import _pyproj,m_geo2mag,m_mag2geo

_projection=ContextVar('wafo_map_projection',default=None)
_coordinates=ContextVar('wafo_map_coordinates',default='geographic')
_PROJECTIONS={
    'stereographic':'stere','orthographic':'ortho','azimuthal equal-area':'laea',
    'azimuthal equidistant':'aeqd','gnomonic':'gnom','satellite':'nsper',
    'albers equal-area conic':'aea','lambert conformal conic':'lcc',
    'equidistant conic':'eqdc','mercator':'merc','miller cylindrical':'mill',
    'equidistant cylindrical':'eqc','sinusoidal':'sinu','gall-peters':'cea',
    'mollweide':'moll','robinson':'robin','hammer-aitoff':'hammer',
    'transverse mercator':'tmerc','utm':'utm','oblique mercator':'omerc',
}
_ALIASES={'miller':'miller cylindrical','lambert':'lambert conformal conic',
          'albers':'albers equal-area conic','hammer':'hammer-aitoff',
          'gall peters':'gall-peters','azimuthal equal area':'azimuthal equal-area'}


def _state():
    state=_projection.get()
    if state is None:
        raise RuntimeError('Initialize a projection with m_proj first')
    return state


def m_coord(coordsys='get'):
    """Select geographic or fixed IGRF2000-geomagnetic input coordinates.

    get returns current name; set/usage return available names. Selection is
    context-local, so independent asynchronous tasks need not share global state.
    """
    value=str(coordsys).lower()
    if value=='get':return _coordinates.get()
    if value in ('set','usage'):return ('geographic','IGRF2000-geomagnetic')
    if value in ('geomagnetic','igrf2000-geomagnetic'):value='IGRF2000-geomagnetic'
    if value not in ('geographic','IGRF2000-geomagnetic'):
        raise ValueError('Unknown coordinate system')
    _coordinates.set(value)
    return value


def _name(name):
    name=str(name).strip().lower()
    name=_ALIASES.get(name,name)
    candidates=[n for n in _PROJECTIONS if n==name]
    if not candidates:candidates=[n for n in _PROJECTIONS if n.startswith(name)]
    if len(candidates)!=1:
        raise ValueError('Unknown or ambiguous projection: '+name)
    return candidates[0]


def m_proj(proj='get',*pairs,**kwargs):
    """Initialize/query a source-named map projection.

    m_proj('set') lists names; m_proj('get') returns current parameters.
    Accepts Python keywords or MATLAB-like name/value pairs. lon/lat give
    two-element bounds or a scalar centre. Standard parallels use parallels=.
    Spherical projections use unit-radius coordinates by default, as M_Map;
    UTM or an explicit ellipsoid use metres. PROJ algorithms replace the
    hand-written source projection formulas.
    """
    if str(proj).lower() in ('set','usage'):
        return tuple(_PROJECTIONS)
    if str(proj).lower()=='get':
        state=_state()
        return {key:deepcopy(value) for key,value in state.items() if key!='projector'}
    if len(pairs)%2:
        raise ValueError('Projection options must be name/value pairs')
    options=dict(zip(pairs[::2],pairs[1::2])); options.update(kwargs)
    aliases={'long':'lon','longitude':'lon','latitude':'lat','clongitude':'lon0',
             'clatitude':'lat0','ellipsoid':'ellipsoid','spheroid':'ellipsoid'}
    opts={aliases.get(str(k).lower(),str(k).lower()):v for k,v in options.items()}
    legal={'lon','lat','lon0','lat0','parallels','zone','hemisphere','ellipsoid',
           'radius','height','angle','k0','rectbox','scale'}
    if set(opts)-legal:
        raise TypeError('Unsupported projection options: '+', '.join(sorted(set(opts)-legal)))
    name=_name(proj); code=_PROJECTIONS[name]
    lon=np.asarray(opts.get('lon',[-180,180]),dtype=float).ravel()
    lat=np.asarray(opts.get('lat',[-80,80]),dtype=float).ravel()
    if lon.size not in (1,2) or lat.size not in (1,2) or not np.all(np.isfinite(np.r_[lon,lat])):
        raise ValueError('lon/lat must be finite centres or two-element bounds')
    lon0=float(opts.get('lon0',np.mean(lon)))
    lat0=float(opts.get('lat0',np.mean(lat)))
    bounds_lon=np.array([-180,180])+lon0 if len(lon)==1 else lon
    bounds_lat=np.array([-80,80]) if len(lat)==1 else lat
    if bounds_lon[1]<=bounds_lon[0] or bounds_lon[1]-bounds_lon[0]>360 or bounds_lat[1]<=bounds_lat[0] or np.any(abs(bounds_lat)>90) or not np.isfinite(lon0+lat0) or abs(lat0)>90:
        raise ValueError('Invalid geographic map bounds/centre')
    radius=float(opts.get('radius',1))
    if not np.isfinite(radius) or radius<=0:
        raise ValueError('radius must be a positive physical projection radius')
    projection=dict(proj=code,lon_0=lon0,lat_0=lat0)
    ellipsoid=opts.get('ellipsoid')
    metres=code=='utm' or ellipsoid is not None
    if metres:projection['ellps']='WGS84' if ellipsoid is None else str(ellipsoid)
    else:projection['R']=radius
    if code in ('aea','lcc','eqdc'):
        parallels=np.asarray(opts.get('parallels',[30,60]),dtype=float)
        if parallels.shape!=(2,) or np.any(abs(parallels)>=90) or abs(parallels.sum())<1e-12:
            raise ValueError('Conic projections need two non-opposite standard parallels')
        projection.update(lat_1=float(parallels[0]),lat_2=float(parallels[1]))
    if code=='cea':projection['lat_ts']=45.
    if code=='nsper':projection['h']=float(opts.get('height',2*radius))
    if code=='utm':
        zone=opts.get('zone',int(np.floor((lon0+180)/6))+1)
        if int(zone)!=zone or not 1<=zone<=60:
            raise ValueError('UTM zone must be an integer from 1 through 60')
        projection.update(zone=int(zone)); projection.pop('lon_0');projection.pop('lat_0')
        hemisphere=str(opts.get('hemisphere','north' if lat0>=0 else 'south')).lower()
        if hemisphere not in ('north','south'):
            raise ValueError('hemisphere must be north or south')
        if hemisphere=='south':projection['south']=True
    if code=='omerc':projection.update(alpha=float(opts.get('angle',45)),gamma=0)
    if 'k0' in opts:projection['k_0']=float(opts['k0'])
    if opts.get('rectbox',True) not in (True,False,'on','off'):
        raise ValueError('rectbox must be boolean/on/off')
    if 'scale' in opts:
        raise ValueError('Use k0 for projection scale or m_scale for physical figure scale')
    projector=_pyproj().Proj(**projection)
    state=dict(name=name,parameters=projection,lon=bounds_lon,lat=bounds_lat,
               lon0=lon0,lat0=lat0,units='metres' if metres else 'radius',
               radius=radius,coordsystem=_coordinates.get(),projector=projector)
    _projection.set(state)
    return m_proj('get')


def _change(lon,lat,source,target):
    if source==target:return lon,lat
    return m_geo2mag(lon,lat) if target!='geographic' else m_mag2geo(lon,lat)


def m_ll2xy(longitude,latitude,*pairs,clip='on',return_mask=False):
    """Project longitude/latitude arrays with point clipping.

    on/point mask out-of-range coordinates; off retains finite projected points;
    patch clamps geographic coordinates to map bounds. Matplotlib artists
    perform projected rectangular clipping; source line-border interpolation
    and arbitrary nonrectangular patch clipping are not reproduced here.
    """
    if pairs:
        if len(pairs)!=2 or str(pairs[0]).lower()!='clip':
            raise ValueError('Only the clip name/value option is supported')
        clip=pairs[1]
    if clip not in ('on','off','point','patch'):
        raise ValueError('clip must be on, off, point or patch')
    state=_state()
    lon,lat=np.broadcast_arrays(np.asarray(longitude,dtype=float),np.asarray(latitude,dtype=float))
    finite=np.isfinite(lon)&np.isfinite(lat)
    if np.any(abs(lat[finite])>90):raise ValueError('Latitude must lie in [-90,90]')
    lo,la=lon.copy(),lat.copy()
    if np.any(finite):
        lo[finite],la[finite]=_change(lon[finite],lat[finite],_coordinates.get(),state['coordsystem'])
    center=np.mean(state['lon'])
    wrapped=(lo-center+180)%360+center-180
    # Preserve +180 endpoint instead of wrapping both seam sides to -180.
    wrapped=np.where((wrapped==center-180)&(lo>center),center+180,wrapped)
    outside=(wrapped<state['lon'][0])|(wrapped>state['lon'][1])|(la<state['lat'][0])|(la>state['lat'][1])|~finite
    if clip=='patch':
        wrapped=np.clip(wrapped,*state['lon']);la=np.clip(la,*state['lat'])
    x,y=state['projector'](wrapped,la)
    x,y=np.asarray(x),np.asarray(y)
    invalid=~np.isfinite(x)|~np.isfinite(y)|~finite
    if clip in ('on','point'):invalid|=outside
    x=np.where(invalid,np.nan,x); y=np.where(invalid,np.nan,y)
    return (x,y,outside|invalid) if return_mask else (x,y)


def m_xy2ll(x,y):
    """Inverse active map projection, returning degrees in current coordinates."""
    state=_state()
    x,y=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
    lon,lat=state['projector'](x,y,inverse=True)
    lon,lat=np.asarray(lon),np.asarray(lat)
    valid=np.isfinite(lon)&np.isfinite(lat)
    if np.any(valid):
        lon[valid],lat[valid]=_change(lon[valid],lat[valid],state['coordsystem'],_coordinates.get())
    return np.where(valid,lon,np.nan),np.where(valid,lat,np.nan)
