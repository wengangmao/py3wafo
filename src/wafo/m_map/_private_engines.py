"""Private M_Map command engines with explicit, independently reusable state.

The six projection dispatchers implement name/usage/set/initialize/get,
ll2xy/xy2ll, xgrid/ygrid and box. Initialize returns a state object; pass it
as ``state=`` to later calls. Without state, each family remembers its last
initialization in the current context. Geographic coordinates are degrees;
these private engines do not apply m_coord's separate geomagnetic conversion.
Source unit-sphere normalizations are retained, including its unusual
equidistant-cylinder/Mollweide/Robinson scales and normalized UTM offsets.
Conic/ellipsoidal UTM solving uses optional PROJ instead of truncated series.
"""
from contextvars import ContextVar
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from ._geodesy import _pyproj
from ._compatibility import mc_ellips


_FAMILIES = {
    'mp_azim': ('Stereographic', 'Orthographic', 'Azimuthal Equal-area', 'Azimuthal Equidistant', 'Gnomonic', 'Satellite'),
    'mp_conic': ('Albers Equal-Area Conic', 'Lambert Conformal Conic'),
    'mp_cyl': ('Mercator', 'Miller Cylindrical', 'Equidistant Cylindrical'),
    'mp_omerc': ('Oblique Mercator',),
    'mp_tmerc': ('Transverse Mercator', 'Sinusoidal', 'Gall-Peters', 'Hammer-Aitoff', 'Mollweide', 'Robinson'),
    'mp_utm': ('UTM',),
}
_OPTIONS = {
    'mp_azim': ('lon','lat','radius','altitude','rectbox','rotation'),
    'mp_conic': ('lon','lat','clongitude','parallels','rectbox','ellipsoid','false_origin'),
    'mp_cyl': ('lon','lat'), 'mp_omerc': ('lon','lat','aspect','direction'),
    'mp_tmerc': ('lon','lat','clongitude','rectbox'),
    'mp_utm': ('lon','lat','zone','hemisphere','ellipsoid','rectbox'),
}
_saved = ContextVar('wafo_private_map_engines', default=None)
_active = ContextVar('wafo_private_active_map', default=None)


@dataclass
class _EngineState:
    family: str
    name: str
    options: dict
    lon: np.ndarray
    lat: np.ndarray
    xlims: np.ndarray
    ylims: np.ndarray
    rectbox: str
    params: dict
    projector: object = None


def _bounds(value, name):
    value = np.asarray(value, dtype=float).ravel()
    if value.size != 2 or np.any(~np.isfinite(value)) or value[0] >= value[1]:
        raise ValueError(name+' must contain two increasing finite limits')
    return value


def _options(family, pairs, kwargs):
    if len(pairs) % 2:
        raise ValueError('Options must be name/value pairs')
    source = dict(zip(pairs[::2], pairs[1::2]))
    source.update(kwargs)
    result = {}
    for key, value in source.items():
        key = str(key).lower()
        aliases = {'rotangle':'rotation','long':'lon','longitude':'lon','latitude':'lat'}
        key = aliases.get(key, key)
        matches = [name for name in _OPTIONS[family] if name == key or name[:3] == key[:3]]
        if len(matches) != 1:
            raise TypeError('Unsupported or ambiguous '+family+' option: '+key)
        result[matches[0]] = value
    return result


def _name(family, value):
    matches = [n for n in _FAMILIES[family] if n.lower().startswith(str(value).strip().lower())]
    if len(matches) != 1:
        raise ValueError('Unknown or ambiguous projection in '+family)
    return matches[0]


def _state(state=None, family=None):
    if state is not None:
        if not isinstance(state, _EngineState) or family is not None and state.family != family:
            raise ValueError('State belongs to a different private projection engine')
        return state
    state = (_saved.get() or {}).get(family) if family else _active.get()
    if state is None:
        raise RuntimeError('Initialize a private projection engine first, or pass state=')
    return state


def _clip_edge(mode, X, edge, outside, Y):
    """Source one-edge line clipping: columns are independent polylines."""
    if mode not in ('on','off','point','patch'):
        raise ValueError('clip must be on/off/point/patch')
    X, Y, outside = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y), np.asarray(outside, bool))
    shape = X.shape
    if X.ndim > 2:
        raise ValueError('Clipping supports scalars, vectors, or columns of line vertices')
    x = X.reshape((-1, 1) if X.ndim < 2 else shape)
    y = Y.reshape(x.shape)
    mask = outside.reshape(x.shape).copy()
    xc, yc = x.copy(), y.astype(np.result_type(y.dtype, float), copy=True)
    if mode == 'off':
        return xc.reshape(shape), yc.reshape(shape)
    if mode != 'point' and len(x) > 1:
        transitions = np.diff(mask.astype(int), axis=0)
        for entering, target_offset in ((-1, 0), (1, 1)):
            i, j = np.where(transitions == entering)
            if len(i):
                dx = x[i+1, j]-x[i, j]
                tiny = abs(dx) < 5*np.finfo(float).eps
                with np.errstate(invalid='ignore', divide='ignore'):
                    interpolation = y[i, j]+(edge-x[i, j])*(y[i+1, j]-y[i, j])/np.where(tiny, 1., dx)
                interpolation = np.where(tiny, (y[i, j]+y[i+1, j])/2, interpolation)
                target = i+target_offset
                xc[target, j], yc[target, j] = edge, interpolation
                mask[target[np.isfinite(interpolation)], j[np.isfinite(interpolation)]] = False
    if mode in ('on','point'):
        xc[mask], yc[mask] = np.nan, np.nan
    else:
        xc[mask] = edge
    return xc.reshape(shape), yc.reshape(shape)


def _box_clip(x, y, xb, yb, mode):
    mask = (x < xb[0]) | (x > xb[1]) | (y < yb[0]) | (y > yb[1]) | ~np.isfinite(x) | ~np.isfinite(y)
    if mode != 'off':
        for bound, comparison in ((xb[0], np.less), (xb[1], np.greater)):
            x, y = _clip_edge(mode, x, bound, comparison(x, bound), y)
        for bound, comparison in ((yb[0], np.less), (yb[1], np.greater)):
            y, x = _clip_edge(mode, y, bound, comparison(y, bound), x)
    return x, y, mask


def _azimuthal(state, lon, lat):
    p = state.params
    delta, phi, phi0 = np.deg2rad(lon-p['lon0']), np.deg2rad(lat), np.deg2rad(p['lat0'])
    cosc = np.sin(phi0)*np.sin(phi)+np.cos(phi0)*np.cos(phi)*np.cos(delta)
    east = np.sin(delta)*np.cos(phi)
    north = np.cos(phi0)*np.sin(phi)-np.sin(phi0)*np.cos(phi)*np.cos(delta)
    sinc = np.hypot(east, north)
    c = np.arctan2(sinc, cosc)
    with np.errstate(divide='ignore', invalid='ignore'):
        code = p['code']
        rho = (2*np.tan(c/2) if code == 0 else sinc if code == 1 else 2*np.sin(c/2)
               if code == 2 else c if code == 3 else np.tan(c) if code == 4
               else sinc/(1+(1-cosc)/p['altitude']))
        direction = np.divide(east+1j*north, sinc, out=np.ones_like(east, dtype=complex), where=sinc > 0)
    return rho, direction, cosc


_ROBX = np.array([1,.9986,.9954,.9900,.9822,.9730,.9600,.9427,.9216,.8962,.8679,.8350,.7986,.7597,.7186,.6732,.6213,.5722,.5322])
_ROBY = np.array([0,.062,.124,.186,.248,.310,.372,.434,.4958,.5571,.6176,.6769,.7346,.7903,.8435,.8936,.9394,.9761,1])*.5072
_ROBLAT = np.arange(0,91,5)


def _forward(state, lon, lat):
    p, name = state.params, state.name.lower()
    lo, la = np.deg2rad(lon-p.get('lon0',0)), np.deg2rad(lat)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if state.family == 'mp_azim':
            rho, azimuth, _ = _azimuthal(state,lon,lat)
            z = rho*azimuth*np.exp(1j*np.deg2rad(p['rotation']))
            return z.real,z.imag
        if state.family == 'mp_cyl':
            y = np.arctanh(np.sin(la)) if name == 'mercator' else np.arctanh(np.sin(.8*la))/.8 if name.startswith('miller') else la/np.cos(np.deg2rad(np.mean(state.lat)))
            return lo,y
        if state.family == 'mp_tmerc':
            if name == 'transverse mercator':
                return np.arctanh(np.cos(la)*np.sin(lo)),np.arctan2(np.tan(la),np.cos(lo))-np.deg2rad(p['lat0'])
            if name == 'sinusoidal':
                return lo*np.cos(la)+np.deg2rad(p['lon0']),la
            if name == 'gall-peters':
                return lo/np.sqrt(2),np.sin(la)*np.sqrt(2)
            if name == 'hammer-aitoff':
                z = np.sqrt((1+np.cos(la)*np.cos(lo/2))/2)
                return 2*np.cos(la)*np.sin(lo/2)/z,np.sin(la)/z
            if name == 'mollweide':
                angle = la.copy()
                pole = abs(lat) >= 90-1e-12
                for _ in range(50):
                    denominator = 2+2*np.cos(2*angle)
                    step = np.divide(2*angle+np.sin(2*angle)-np.pi*np.sin(la),denominator,out=np.zeros_like(angle),where=(denominator>1e-14)&~pole)
                    angle -= step
                    if np.max(abs(step),initial=0)<1e-13:break
                angle = np.where(pole,np.sign(lat)*np.pi/2,angle)
                return 2/np.pi*lo*np.cos(angle)+p['lon0']/90,np.sin(angle)
            return lo*np.interp(abs(lat),_ROBLAT,_ROBX),np.sign(lat)*np.pi*np.interp(abs(lat),_ROBLAT,_ROBY)
        if state.family == 'mp_omerc':
            longitude = np.deg2rad(lon)-p['l0']
            pole = p['polelat']
            A = np.sin(pole)*np.sin(la)-np.cos(pole)*np.cos(la)*np.sin(longitude)
            along = np.arctan2(np.tan(la)*np.cos(pole)+np.sin(pole)*np.sin(longitude),np.cos(longitude))
            across = np.arctanh(A)
            return (along,across) if p['direction']=='horizontal' else (-across,along)
        x,y = state.projector(lon,lat)
        return np.asarray(x),np.asarray(y)


def _inverse(state,x,y):
    p, name = state.params,state.name.lower()
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        if state.family == 'mp_azim':
            z = (x+1j*y)*np.exp(-1j*np.deg2rad(p['rotation']))
            radius = abs(z)
            code = p['code']
            c = (2*np.arctan(radius/2) if code == 0 else np.arcsin(radius) if code == 1
                 else 2*np.arcsin(radius/2) if code == 2 else radius if code == 3
                 else np.arctan(radius) if code == 4 else np.arcsin((p['altitude']+1)/np.sqrt(1+(p['altitude']/radius)**2))-np.arctan(radius/p['altitude']))
            c = np.where(radius==0,0.,c)
            north = np.divide(z.imag,radius,out=np.zeros_like(radius),where=radius>0)
            east = np.divide(z.real,radius,out=np.zeros_like(radius),where=radius>0)
            phi0 = np.deg2rad(p['lat0'])
            lat = np.rad2deg(np.arcsin(np.clip(np.cos(c)*np.sin(phi0)+np.cos(phi0)*np.sin(c)*north,-1,1)))
            lon = p['lon0']+np.rad2deg(np.arctan2(east*np.sin(c),np.cos(phi0)*np.cos(c)-np.sin(phi0)*north*np.sin(c)))
        elif state.family == 'mp_cyl':
            lon = np.rad2deg(x)+p['lon0']
            lat = np.rad2deg(np.arctan(np.sinh(y))) if name=='mercator' else np.rad2deg(np.arctan(np.sinh(.8*y)))/.8 if name.startswith('miller') else np.rad2deg(y)*np.cos(np.deg2rad(np.mean(state.lat)))
        elif state.family == 'mp_tmerc':
            if name=='transverse mercator':
                d = y+np.deg2rad(p['lat0'])
                lon,lat = p['lon0']+np.rad2deg(np.arctan2(np.sinh(x),np.cos(d))),np.rad2deg(np.arcsin(np.sin(d)/np.cosh(x)))
            elif name=='sinusoidal':
                lat = np.rad2deg(y)
                lon = p['lon0']+np.rad2deg((x-np.deg2rad(p['lon0']))/np.cos(y))
            elif name=='gall-peters':
                lon,lat = p['lon0']+np.rad2deg(x*np.sqrt(2)),np.rad2deg(np.arcsin(y/np.sqrt(2)))
            elif name=='hammer-aitoff':
                z = np.sqrt(1-(x/4)**2-(y/2)**2)
                lon,lat = p['lon0']+np.rad2deg(2*np.arctan2(z*x,2*(2*z*z-1))),np.rad2deg(np.arcsin(y*z))
            elif name=='mollweide':
                angle = np.arcsin(y)
                lat = np.rad2deg(np.arcsin(np.clip((2*angle+np.sin(2*angle))/np.pi,-1,1)))
                lon = p['lon0']+np.rad2deg((x-p['lon0']/90)*np.pi/2/np.cos(angle))
            else:
                lat = np.sign(y)*np.interp(abs(y)/np.pi,_ROBY,_ROBLAT,left=np.nan,right=np.nan)
                lon = p['lon0']+np.rad2deg(x/np.interp(abs(lat),_ROBLAT,_ROBX))
        elif state.family=='mp_omerc':
            along,across = (x,y) if p['direction']=='horizontal' else (y,-x)
            pole = p['polelat']
            lat = np.rad2deg(np.arcsin(np.clip(np.sin(pole)*np.tanh(across)+np.cos(pole)*np.sin(along)/np.cosh(across),-1,1)))
            lon = np.rad2deg(p['l0']+np.arctan2(np.sin(pole)*np.sin(along)-np.cos(pole)*np.sinh(across),np.cos(along)))
        else:
            lon,lat = state.projector(x,y,inverse=True)
            lon,lat = np.asarray(lon),np.asarray(lat)
    lon = p.get('lon0',np.mean(state.lon))+(lon-p.get('lon0',np.mean(state.lon))+180)%360-180
    return np.where(np.isfinite(lon),lon,np.nan),np.where(np.isfinite(lat),lat,np.nan)


def _ll2xy(state,lon,lat,clip):
    if clip not in ('on','off','point','patch'):raise ValueError('Unknown clipping mode')
    lon,lat = np.broadcast_arrays(np.asarray(lon,float),np.asarray(lat,float))
    if np.any(abs(lat[np.isfinite(lat)])>90):raise ValueError('Latitude must lie in [-90,90]')
    outside = ~np.isfinite(lon)|~np.isfinite(lat)
    if state.family=='mp_azim':
        rho,az,cosc = _azimuthal(state,lon,lat)
        mask = cosc<np.cos(np.deg2rad(state.params['radius']))
        outside |= mask
        if clip!='off':
            rho,az = _clip_edge(clip,rho,state.params['rhomax'],mask,az)
            az = np.divide(az,abs(az),out=np.full_like(az,np.nan),where=abs(az)>0)
        z = rho*az*np.exp(1j*np.deg2rad(state.params['rotation']))
        x,y = z.real,z.imag
    else:
        if state.rectbox=='off':
            lon,lat,mask = _box_clip(lon,lat,state.lon,state.lat,clip)
            outside |= mask
        x,y = _forward(state,lon,lat)
    if state.rectbox=='on':
        x,y,mask = _box_clip(x,y,state.xlims,state.ylims,clip)
        outside |= mask
    invalid = ~np.isfinite(x)|~np.isfinite(y)
    return np.where(invalid,np.nan,x),np.where(invalid,np.nan,y),outside|invalid


def _perimeter(xb,yb,n):
    x,y = np.linspace(*xb,n),np.linspace(*yb,n)
    return np.r_[x,np.full(n,xb[1]),x[::-1],np.full(n,xb[0])],np.r_[np.full(n,yb[0]),y,np.full(n,yb[1]),y[::-1]]


def _limits(state):
    lon,lat = _perimeter(state.lon,state.lat,91)
    x,y = _forward(state,lon,lat)
    valid = np.isfinite(x)&np.isfinite(y)
    if not valid.any():raise ValueError('Map boundary has no finite projected points')
    return np.array([x[valid].min(),x[valid].max()]),np.array([y[valid].min(),y[valid].max()])


def _box(state,n):
    if not isinstance(n,(int,np.integer)) or n<2:raise ValueError('Border sampling count must be at least two')
    if state.rectbox=='circle':
        angle = np.linspace(0,2*np.pi,3*n)
        return state.params['rhomax']*np.cos(angle),state.params['rhomax']*np.sin(angle)
    a,b = _perimeter(state.xlims,state.ylims,n) if state.rectbox=='on' else _perimeter(state.lon,state.lat,n)
    return (a,b) if state.rectbox=='on' else _forward(state,a,b)


def _lllimits(state):
    x,y = _box(state,181)
    lon,lat = _inverse(state,x,y)
    valid = np.isfinite(lon)&np.isfinite(lat)
    if not valid.any():raise ValueError('Map border has no invertible points')
    longitude,latitude = np.array([lon[valid].min(),lon[valid].max()]),np.array([lat[valid].min(),lat[valid].max()])
    px,py,_ = _ll2xy(state,[state.params.get('lon0',0)]*2,[-90,90],'point')
    poles = np.isfinite(px)&np.isfinite(py)
    if poles.any():
        longitude = np.array([-180,180])+state.params.get('lon0',0)
        if poles[0]:latitude[0] = -90
        if poles[1]:latitude[1] = 90
    return longitude,latitude


def _initialize(family,name,opts):
    defaults = dict(lon=[-180,180],lat=[-85,85])
    if family=='mp_conic':defaults.update(lon=[-180,-50],lat=[10,85],rectbox='off',ellipsoid='normal')
    elif family=='mp_tmerc':defaults.update(lon=[-125,-122] if name=='Transverse Mercator' else [-180,180],lat=[47,51] if name=='Transverse Mercator' else [-80,80],rectbox='off')
    elif family=='mp_utm':defaults.update(lon=[-72,-68],lat=[40,44],rectbox='off',ellipsoid='normal',zone=0,hemisphere=-1)
    elif family=='mp_azim':defaults.update(lon=0.,lat=60.,radius=90.,rectbox='circle',rotation=0.)
    elif family=='mp_omerc':defaults.update(lon=[-132,-125],lat=[56,40],aspect=.5,direction='vertical',rectbox='on')
    defaults.update(opts)
    opts = defaults
    rectbox = str(opts.get('rectbox','off')).lower()
    if rectbox not in ('on','off','circle') or rectbox=='circle' and family!='mp_azim':raise ValueError('Unsupported rectbox value')
    lon = np.asarray(opts['lon'],float).ravel()
    lat = np.asarray(opts['lat'],float).ravel()
    p = {}
    if family=='mp_azim':
        if len(lon)!=1 or len(lat)!=1 or abs(lat[0])>90:raise ValueError('Azimuthal lon/lat are scalar centers')
        p.update(lon0=float(lon[0]),lat0=float(lat[0]),rotation=float(opts['rotation']),code=_FAMILIES[family].index(name))
        radius = np.asarray(opts['radius'],float).ravel()
        if name=='Satellite':
            altitude = float(opts.get('altitude',radius[0]))
            if not np.isfinite(altitude) or altitude<=0:raise ValueError('Satellite altitude fraction must be positive')
            p['altitude'] = altitude
            radius = np.array([np.rad2deg(np.arccos(1/(1+altitude)))*.98])
        if len(radius)==2:
            dl,ph,ph0 = np.deg2rad(radius[0]-p['lon0']),np.deg2rad(radius[1]),np.deg2rad(p['lat0'])
            radius = np.array([np.rad2deg(np.arccos(np.clip(np.sin(ph0)*np.sin(ph)+np.cos(ph0)*np.cos(ph)*np.cos(dl),-1,1)))])
        if len(radius)!=1 or not np.isfinite(radius[0]) or not 0<radius[0]<180:raise ValueError('Radius must be degrees in (0,180) or a boundary lon/lat pair')
        r = min(float(radius[0]),89.8 if name=='Orthographic' else 80 if name=='Gnomonic' else 179.999999)
        c = np.deg2rad(r)
        rho = (2*np.tan(c/2) if p['code']==0 else np.sin(c) if p['code']==1 else 2*np.sin(c/2)
               if p['code']==2 else c if p['code']==3 else np.tan(c) if p['code']==4 else np.sin(c)/(1+(1-np.cos(c))/p['altitude']))
        p.update(radius=r,rhomax=float(rho))
        state = _EngineState(family,name,opts,np.array([-180,180])+p['lon0'],np.array([-90,90]),np.array([-rho,rho]),np.array([-rho,rho]),'circle' if rectbox=='off' else rectbox,p)
        if state.rectbox=='on':state.xlims/=np.sqrt(2);state.ylims/=np.sqrt(2)
        state.lon,state.lat = _lllimits(state)
    elif family=='mp_omerc':
        if lon.size!=2 or lat.size!=2 or np.any(~np.isfinite(np.r_[lon,lat])) or np.any(abs(lat)>90):raise ValueError('Oblique Mercator needs two baseline points')
        la,lo = np.deg2rad(lat),np.deg2rad(lon)
        polelon = np.arctan2(np.cos(la[0])*np.sin(la[1])*np.cos(lo[0])-np.sin(la[0])*np.cos(la[1])*np.cos(lo[1]),np.sin(la[0])*np.cos(la[1])*np.sin(lo[1])-np.cos(la[0])*np.sin(la[1])*np.sin(lo[0]))
        polelat = np.arctan2(-np.cos(polelon-lo[0]),np.tan(la[0]))
        # atan, rather than atan2, selects the pole branch used by MATLAB.
        polelat = np.arctan(-np.cos(polelon-lo[0])/np.tan(la[0]))
        direction = str(opts['direction']).lower()
        if direction.startswith('hor'):direction='horizontal'
        elif direction.startswith('ver'):direction='vertical'
        else:raise ValueError('direction must be horizontal or vertical')
        if not np.isfinite(opts['aspect']) or opts['aspect']<=0:raise ValueError('aspect must be positive')
        p.update(lon0=float(np.mean(lon)),l0=polelon+np.pi/2,polelat=polelat,direction=direction)
        state = _EngineState(family,name,opts,np.sort(lon),np.sort(lat),np.zeros(2),np.zeros(2),'on',p)
        x,y = _forward(state,lon,lat)
        if direction=='horizontal':state.xlims=np.sort(x);state.ylims=np.array([-.5,.5])*np.ptp(x)*opts['aspect']
        else:state.ylims=np.sort(y);state.xlims=np.array([-.5,.5])*np.ptp(y)*opts['aspect']
        if np.ptp(state.xlims)<=0 or np.ptp(state.ylims)<=0:raise ValueError('Baseline points do not define a map')
        state.lon,state.lat = _lllimits(state)
    else:
        if len(lon)==1:lon=lon[0]+np.array([-180,180])
        if len(lat)==1:lat=abs(lat[0])*np.array([-1,1])
        lon,lat = _bounds(lon,'lon'),_bounds(lat,'lat')
        if np.ptp(lon)>360 or np.any(abs(lat)>90):raise ValueError('Invalid geographic limits')
        center = float(opts.get('clongitude',np.mean(lon)))
        p.update(lon0=center,lat0=float(np.mean(lat)))
        projector = None
        if family in ('mp_conic','mp_utm'):
            ellipsoids = mc_ellips()
            ellipsoid = str(opts['ellipsoid']).lower()
            if ellipsoid not in ellipsoids:raise ValueError('Unknown historical ellipsoid')
            a,f = ellipsoids[ellipsoid]
            projection = dict(a=a,b=a*(1-f))
            if family=='mp_conic':
                parallels = np.asarray(opts.get('parallels',[np.mean(lat)]*2),float).ravel()
                false = np.asarray(opts.get('false_origin',[center,np.mean(parallels)]),float).ravel()
                if parallels.size!=2 or false.size!=2 or np.any(~np.isfinite(np.r_[parallels,false])):raise ValueError('parallels and false_origin need two finite entries')
                projection.update(proj='aea' if name.startswith('Albers') else 'lcc',lon_0=float(false[0]),lat_0=float(false[1]),lat_1=float(parallels[0]),lat_2=float(parallels[1]))
                p.update(lon0=float(false[0]),parallels=parallels,false_origin=false)
            else:
                zone = opts['zone'] or int(np.floor(((center+180)%360)/6))+1
                hemisphere = opts['hemisphere']
                if isinstance(hemisphere,str):hemisphere={'north':0,'south':1}.get(hemisphere.lower(),-2)
                if hemisphere==-1:hemisphere=int(np.mean(lat)<0)
                if int(zone)!=zone or not 1<=zone<=60 or hemisphere not in (0,1):raise ValueError('UTM zone1..60; hemisphere0/north or1/south')
                projection.update(proj='tmerc',lon_0=6*int(zone)-183,lat_0=0,k_0=.9996,x_0=500000 if a>1 else 0,y_0=10000000 if a>1 and hemisphere else 0)
                p.update(lon0=6*int(zone)-183,zone=int(zone),hemisphere=int(hemisphere),units='metres' if a>1 else 'radius')
            projector = _pyproj().Proj(**projection)
        state = _EngineState(family,name,opts,lon,lat,np.zeros(2),np.zeros(2),rectbox,p,projector)
        state.xlims,state.ylims = _limits(state)
        if state.rectbox=='on':state.lon,state.lat = _lllimits(state)
    if np.any(~np.isfinite(np.r_[state.lon,state.lat,state.xlims,state.ylims])):raise ValueError('Projection parameters produce nonfinite bounds')
    return state


def _ticks(lo,hi,ticks):
    if not np.isfinite(lo+hi) or lo>=hi:raise ValueError('Tick limits must increase')
    ticks = np.asarray(ticks,float).ravel()
    if not len(ticks) or np.any(~np.isfinite(ticks)):raise ValueError('Provide finite ticks or a desired count')
    if len(ticks)>1:return ticks[(ticks>=lo)&(ticks<=hi)]
    if ticks[0]<=2:return np.array([lo,hi])
    nice = np.r_[.1,.2,.25,.5,1,2,3,4,5,6,10,12,15,20,30,60*np.array([1,2,3,4,5,6,8,9,10,12,15,18,20,25,30,40,50,60,100,120,180])]
    interval = nice[np.argmin(abs(nice-(hi-lo)/(ticks[0]-1)*60))]/60
    values = interval*np.arange(np.ceil(lo/interval),np.floor(hi/interval)+1)
    return values[(values>=lo)&(values<=hi)]


def _grid(state,command,limits,other,ticks,samples,label):
    limits,other = _bounds(limits,'grid limits'),_bounds(other,'other limits')
    if not isinstance(samples,(int,np.integer)) or samples<2:raise ValueError('Grid samples must be >=2')
    values = _ticks(*limits,ticks)
    rectangular = state.rectbox in ('on','circle')
    if rectangular:values=values[(values>limits[0])&(values<limits[1])]
    rows = 2*samples if rectangular else samples
    A,B = np.meshgrid(values,np.linspace(*other,rows))
    x,y,_ = _ll2xy(state,A if command=='xgrid' else B,B if command=='xgrid' else A,'on' if rectangular else 'off')
    if label in ('left','bottom','west','south'):label_index=0
    elif label=='middle':label_index=int(np.floor(rows/2))
    elif label in ('right','top','east','north'):label_index=rows-1
    else:raise ValueError('Unknown grid label position')
    return x,y,values,label_index


def mu_util(option,*args,state=None):
    """Source clip/tick/grid/border utilities with explicit returned limits.

    clip(mode,X,edge,outside,Y) retains the source column-polyline contract.
    xylimits and lllimits return (x_bounds,y_bounds)/(lon_bounds,lat_bounds)
    instead of mutating MATLAB globals. Grid label indices are zero based.
    Grid generation samples/clips curves; the old iterative endpoint-resampling
    heuristics are not reproduced. Geographic borders can be curved; circle
    borders use the source 3*n samples. No axes/figure globals are changed.
    """
    option = str(option).lower()
    if option=='clip':
        if len(args)!=5:raise TypeError('clip needs mode,X,edge,outside,Y')
        return _clip_edge(*args)
    if option=='axisticks':
        if len(args)!=3:raise TypeError('axisticks needs min,max,count/vector')
        return _ticks(*args)
    state = _state(state)
    if option=='xylimits':return state.xlims.copy(),state.ylims.copy()
    if option=='lllimits':return _lllimits(state)
    if option=='box':return _box(state,31 if not args else args[0])
    if option in ('xgrid','ygrid'):
        if len(args)!=5:raise TypeError('grid needs limits,other_limits,ticks,samples,label_position')
        return _grid(state,option,*args)
    raise NotImplementedError('Unsupported mu_util operation: '+option)


def _engine(family,command,*args,state=None,**kwargs):
    command = str(command).lower()
    if command=='name':return _FAMILIES[family]
    if command in ('set','usage'):
        return dict(projections=_FAMILIES[family],options=_OPTIONS[family],commands=('initialize','get','ll2xy','xy2ll','xgrid','ygrid','box'))
    if command=='initialize':
        name = _name(family,args[0]) if args else _FAMILIES[family][0]
        state = _initialize(family,name,_options(family,args[1:],kwargs))
        cache = dict(_saved.get() or {});cache[family]=state;_saved.set(cache);_active.set(state)
        return state
    state = _state(state,family)
    if command=='get':return {key:deepcopy(value) for key,value in vars(state).items() if key!='projector'}
    if command in ('ll2xy','xy2ll'):
        if len(args)<2:raise TypeError('Provide two coordinate arrays')
        options = dict(zip(args[2::2],args[3::2]));options.update(kwargs)
        if len(args[2:])%2 or set(options)-{'clip'}:raise TypeError('Only clip name/value option is accepted')
        if command=='ll2xy':return _ll2xy(state,args[0],args[1],options.get('clip','on'))
        x,y = np.broadcast_arrays(np.asarray(args[0],float),np.asarray(args[1],float))
        return _inverse(state,x,y)
    if command=='box':return _box(state,31 if not args else args[0])
    if command in ('xgrid','ygrid'):
        if len(args) not in (2,3):raise TypeError('grid needs ticks,label_position or ticks,samples,label_position')
        limits,other = (state.lon,state.lat) if command=='xgrid' else (state.lat,state.lon)
        return _grid(state,command,limits,other,args[0],3 if len(args)==2 else args[1],args[-1])
    raise NotImplementedError('Unsupported '+family+' operation: '+command)


def mp_azim(command,*args,**kwargs):
    """Azimuthal source projections, angular-radius clipping, rotation and grids."""
    return _engine('mp_azim',command,*args,**kwargs)


def mp_conic(command,*args,**kwargs):
    """Conic command engine with historical ellipsoid and false-origin parameters."""
    return _engine('mp_conic',command,*args,**kwargs)


def mp_cyl(command,*args,**kwargs):
    """Cylindrical source formulas, including its latitude-dependent eqc scale."""
    return _engine('mp_cyl',command,*args,**kwargs)


def mp_omerc(command,*args,**kwargs):
    """Two-point spherical oblique Mercator with source aspect/orientation geometry."""
    return _engine('mp_omerc',command,*args,**kwargs)


def mp_tmerc(command,*args,**kwargs):
    """Transverse/pseudocylindrical source formulas and invertible Robinson table.

    Fixes the source Robinson inverse's undefined ``clon`` field to ``clong``.
    """
    return _engine('mp_tmerc',command,*args,**kwargs)


def mp_utm(command,*args,**kwargs):
    """UTM with source unit-radius/no-false-offset or physical ellipsoid units."""
    return _engine('mp_utm',command,*args,**kwargs)


def mu_coast(option='default',*args,state=None,return_data=False,ax=None,**kwargs):
    """Coast extraction/drawing: d/default, u/file, v/vector, or c/l/i/h/f/file.

    return_data=True replaces MATLAB's three-output extraction form and returns
    (ncst,Area,k), with zero-based NaN delimiter indices k. Files are numeric
    MAT/NPZ or GSHHS2.x; old GSHHS1.x and speckle rendering are unsupported.
    Lines use the private engine's actual line intersections. Patch drawing
    requires polygons already inside the map: the old complex coastline
    topology repair/Antarctic seam heuristics are explicitly not claimed.
    """
    from ._datasets import _load_coast,_read_gshhs
    option = str(option).lower()
    if not option:raise ValueError('Provide a coastline source option')
    args = list(args)
    if option[0] in 'clihf':
        if not args:raise TypeError('GSHHS source requires an explicit filename')
        data = _read_gshhs(args.pop(0))
    elif option[0]=='u':
        if not args:raise TypeError('User coastline requires a filename')
        data = _load_coast(args.pop(0))
    elif option[0]=='v':
        if not args:raise TypeError('Vector coastline requires N by2 coordinates')
        data = {'ncst':np.asarray(args.pop(0),float)}
    elif option[0]=='d':data=_load_coast()
    else:raise ValueError('Unknown coastline source')
    ncst = np.asarray(data['ncst'],float)
    if ncst.ndim!=2 or ncst.shape[1]!=2 or np.any(np.isinf(ncst)):raise ValueError('Coast coordinates must be finite/NaN N by2')
    ncst = np.vstack((np.full((1,2),np.nan),ncst,np.full((1,2),np.nan)))
    k = np.flatnonzero(~np.isfinite(ncst).all(axis=1))
    segments = [ncst[a+1:b] for a,b in zip(k[:-1],k[1:]) if b>a+1]
    areas = np.asarray(data.get('Area',np.ones(len(segments)))).ravel()
    if len(areas)!=len(segments):raise ValueError('Area metadata does not match coastline segments')
    if return_data:return ncst.copy(),areas.copy(),k
    state = _state(state)
    import matplotlib.pyplot as plt
    ax = plt.gca() if ax is None else ax
    style = args.pop(0) if args and args[0] in ('line','patch','speckle') else 'line'
    if style=='speckle':raise NotImplementedError('Source speckle/topology repair is not implemented')
    if len(args)%2:raise ValueError('Coast properties must be name/value pairs')
    props = {str(key).lower():value for key,value in zip(args[::2],args[1::2])};props.update(kwargs)
    artists = []
    for index in np.argsort(abs(areas))[::-1]:
        segment = segments[index]
        x,y,mask = _ll2xy(state,segment[:,0],segment[:,1],'on' if style=='line' else 'point')
        if style=='line':
            seams = np.flatnonzero(abs(np.diff(segment[:,0]))>180)+1
            x[seams]=np.nan;y[seams]=np.nan
            artists.extend(ax.plot(x,y,**props))
        else:
            if mask.any():raise NotImplementedError('Patch polygons crossing map boundaries require prior polygon clipping')
            options = dict(props)
            if areas[index]<0:options['facecolor']=ax.get_facecolor()
            artists.extend(ax.fill(x,y,**options))
    return artists


def clabel(cs,*args,ax=None,manual=None,**kwargs):
    """Label a Matplotlib ContourSet or MATLAB's packed 2xN contour matrix.

    Native contour sets receive inline labels. Packed matrices receive source
    plus markers and numeric labels spaced by ``label`` points (default144).
    Numeric positional levels select a subset; source text name/value options
    and Python text properties are accepted. Each Text carries its numeric
    ``wafo_contour_level``. Interactive mouse placement/3D contour handles are
    unsupported; pass coordinate lists through manual= for native contour sets.
    """
    import matplotlib.pyplot as plt
    from matplotlib.contour import ContourSet
    args = list(args)
    if args and isinstance(args[0],ContourSet):
        cs = args.pop(0)
    native = isinstance(cs,ContourSet)
    levels = None
    if args and not isinstance(args[0],str):levels=np.asarray(args.pop(0),float).ravel()
    if args and args[0]=='manual':raise NotImplementedError('Interactive mouse placement is not supported')
    if manual is True:raise NotImplementedError('Provide explicit manual coordinates; no interactive input is opened')
    if len(args)%2:raise ValueError('Text options must be name/value pairs')
    props = {str(key).lower():value for key,value in zip(args[::2],args[1::2])};props.update(kwargs)
    spacing = props.pop('label',144.)
    if not np.isfinite(spacing) or spacing<=0:raise ValueError('label spacing must be positive points')
    if native:
        if spacing!=144:raise NotImplementedError('Native contour label density is controlled by Matplotlib; use packed data for label spacing')
        ax = cs.axes
        options = {key:props.pop(key) for key in ('fontsize','inline','fmt') if key in props}
        if 'color' in props:options['colors']=props.pop('color')
        if manual is not None:options['manual']=manual
        labels = list(ax.clabel(cs,levels=levels,**options))
        for text in labels:
            text.set(**props)
            try:text.wafo_contour_level=float(text.get_text())
            except ValueError:text.wafo_contour_level=None
        return labels
    if manual is not None:raise NotImplementedError('Packed-matrix labels currently use automatic arc-length placement')
    matrix = np.asarray(cs,float)
    if matrix.ndim!=2 or matrix.shape[0]!=2 or np.any(~np.isfinite(matrix)):raise ValueError('Packed contours must be a finite 2 byN array')
    segments = []
    column = 0
    while column<matrix.shape[1]:
        level,count = matrix[:,column]
        if int(count)!=count or count<2 or column+1+count>matrix.shape[1]:raise ValueError('Invalid packed contour header/vertex count')
        segments.append((level,matrix[:,column+1:column+1+int(count)].T))
        column += 1+int(count)
    available = np.array([level for level,_ in segments])
    if levels is not None and any(not np.any(np.isclose(level,available)) for level in levels):raise ValueError('Requested level is absent from packed contours')
    ax = plt.gca() if ax is None else ax
    ax.figure.canvas.draw()
    result = []
    for level,points in segments:
        if levels is not None and not np.any(np.isclose(level,levels)):continue
        display = ax.transData.transform(points)
        lengths = np.r_[0,np.cumsum(np.linalg.norm(np.diff(display,axis=0),axis=1))]
        if lengths[-1]<=0:continue
        distance = float(spacing)*ax.figure.dpi/72
        positions = np.arange(distance/2,lengths[-1],distance)
        if not len(positions):positions=np.array([lengths[-1]/2])
        for distance in positions:
            point = np.array([np.interp(distance,lengths,points[:,j]) for j in (0,1)])
            marker = ax.plot(*point,marker='+',linestyle='none',color=props.get('color','k'))[0]
            text = ax.annotate(f'{level:g}',point,xytext=(3,3),textcoords='offset points',**props)
            text.wafo_contour_level=float(level)
            result.extend((marker,text))
    return result
