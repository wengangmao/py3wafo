"""Bundled coast rendering and explicit external map-dataset readers.

Only the original m_coasts.mat numeric data are bundled. Global elevation,
GSHHS and political-boundary databases were not present in this checkout;
their readers require explicit user-supplied paths and never download data.
"""
from pathlib import Path
from importlib.resources import files
import struct
import numpy as np
from scipy.io import loadmat
from ._projection import _state,m_proj
from ._plotting import _ax,m_plot,m_patch,m_contour,m_contourf


def _load_coast(filename=None):
    if filename is None:
        with files('wafo.m_map').joinpath('data/m_coasts.npz').open('rb') as stream:
            with np.load(stream,allow_pickle=False) as data:
                return {key:data[key] for key in data.files}
    path=Path(filename)
    if not path.is_file():raise FileNotFoundError(path)
    if path.suffix.lower()=='.mat':
        data={k:v for k,v in loadmat(path).items() if not k.startswith('_')}
    elif path.suffix.lower()=='.npz':
        with np.load(path,allow_pickle=False) as archive:data={key:archive[key] for key in archive.files}
    else:
        return {'ncst':np.loadtxt(path)}
    if 'ncst' not in data:raise ValueError('Coast file needs an ncst longitude/latitude array')
    return data


def _draw_coast(data,args,ax,kwargs):
    ncst=np.asarray(data['ncst'],dtype=float)
    if ncst.ndim!=2 or ncst.shape[1]!=2:raise ValueError('Coast coordinates must be N by 2')
    style='line';args=list(args)
    if args and args[0] in ('line','patch'):
        style=args.pop(0)
    if style=='line':return m_plot(ncst[:,0],ncst[:,1],*args,ax=ax,**kwargs)
    # Preserve source polygon order. Polygon/lake hierarchy is handled when
    # signed Area metadata exists, rendering smallest regions last.
    starts=np.flatnonzero(~np.isfinite(ncst).all(axis=1))
    if not len(starts) or starts[0]!=0:starts=np.r_[-1,starts]
    ends=np.r_[starts[1:],len(ncst)]
    polygons=[ncst[a+1:b] for a,b in zip(starts,ends) if b-a>3]
    areas=np.asarray(data.get('Area',np.ones(len(polygons)))).ravel()
    if len(areas)!=len(polygons):areas=np.ones(len(polygons))
    artists=[]
    for index in np.argsort(abs(areas))[::-1]:
        options=dict(kwargs)
        if areas[index]<0:options['facecolor']=_ax(ax).get_facecolor()
        artists.extend(m_patch(polygons[index][:,0],polygons[index][:,1],*args,ax=ax,**options))
    return artists


def m_coast(*args,ax=None,**kwargs):
    """Draw the supplied 1/4-degree coast database as lines or signed-area patches."""
    return _draw_coast(_load_coast(),args,ax,kwargs)


def m_usercoast(filename,*args,ax=None,**kwargs):
    """Draw a numeric-only NPZ, MATLAB ncst file, or two-column text coastline."""
    return _draw_coast(_load_coast(filename),args,ax,kwargs)


def _read_gshhs(filename):
    """Read current 11-int GSHHS 2.x headers and microdegree coordinate records."""
    path=Path(filename)
    if not path.is_file():raise FileNotFoundError(path)
    parts=[];areas=[]
    with path.open('rb') as stream:
        while True:
            header=stream.read(44)
            if not header:break
            if len(header)!=44:raise ValueError('Truncated GSHHS header')
            fields=struct.unpack('>11i',header)
            identity,n,flag,west,east,south,north,area,area_full,container,ancestor=fields
            version=(flag>>8)&255
            if n<0 or n>50_000_000 or version<7 or version>30:
                raise ValueError('Expected a GSHHS 2.x big-endian 11-int header; older binary formats are unsupported')
            raw=stream.read(8*n)
            if len(raw)!=8*n:raise ValueError('Truncated GSHHS coordinate record')
            coordinates=np.frombuffer(raw,dtype='>i4').reshape(-1,2).astype(float)/1e6
            # Greenwich-crossing polygons store the western section above360.
            coordinates[:,0]=np.where(coordinates[:,0]>180,coordinates[:,0]-360,coordinates[:,0])
            parts.extend((np.full((1,2),np.nan),coordinates))
            level=flag&255
            areas.append(abs(area)*(-1 if level%2==0 else 1))
    return {'ncst':np.vstack(parts) if parts else np.empty((0,2)),'Area':np.asarray(areas)}


def m_gshhs(resolution='c',*args,filename=None,ax=None,**kwargs):
    """Draw a user-supplied GSHHS2.x database; no database is fabricated/downloaded.

    resolution selects metadata/expected file naming; filename is required.
    `save=path` stores numeric NPZ extraction for m_usercoast.
    """
    aliases={1:'c',2:'l',3:'i',4:'h',5:'f','crude':'c','low':'l','intermediate':'i','high':'h','full':'f'}
    resolution=aliases.get(resolution,resolution)
    if not isinstance(resolution,str) or resolution[0].lower() not in 'clihf' or len(resolution)>2 or len(resolution)==2 and resolution[1].lower() not in 'cbr':
        raise ValueError('Resolution must be c/l/i/h/f optionally followed by c/b/r')
    if filename is None:
        raise FileNotFoundError('The checkout contains no GSHHS database; pass filename= to a GSHHS2.x binary')
    data=_read_gshhs(filename)
    args=list(args)
    if args and args[0]=='save':
        if len(args)!=2:raise ValueError("Use m_gshhs(...,'save',path,filename=...)")
        with Path(args[1]).open('wb') as output:np.savez_compressed(output,**data)
        return data
    return _draw_coast(data,args,ax,kwargs)


def m_gshhs_c(*args,**kwargs):return m_gshhs('c',*args,**kwargs)
def m_gshhs_l(*args,**kwargs):return m_gshhs('l',*args,**kwargs)
def m_gshhs_i(*args,**kwargs):return m_gshhs('i',*args,**kwargs)
def m_gshhs_h(*args,**kwargs):return m_gshhs('h',*args,**kwargs)
def m_gshhs_f(*args,**kwargs):return m_gshhs('f',*args,**kwargs)


def m_plotbndry(name,*args,ax=None,return_data=False,**kwargs):
    """Read DCW text boundaries, MATLAB lon/lat arrays, or two-column text.

    DCW two-coordinate lines are latitude/longitude; segment ID and END lines
    delimit sections. No interactive file dialog or implicit file writes.
    """
    path=Path(name)
    if not path.is_file():
        path=Path(str(name)+'2pts.txt')
    if not path.is_file():raise FileNotFoundError(path)
    if path.suffix.lower()=='.mat':
        data=loadmat(path)
        longitude=np.asarray(data.get('lon',data.get('long'))).ravel()
        latitude=np.asarray(data['lat']).ravel()
    else:
        coordinates=[]
        for line in path.read_text(encoding='utf-8').splitlines():
            fields=line.replace(',',' ').split()
            try:values=[float(v) for v in fields]
            except ValueError:values=[]
            if len(values)==2:coordinates.append([values[1],values[0]])
            elif coordinates and not np.isnan(coordinates[-1][0]):coordinates.append([np.nan,np.nan])
        if not coordinates:raise ValueError('No latitude/longitude boundary pairs found')
        longitude,latitude=np.asarray(coordinates).T
    artists=m_plot(longitude,latitude,*args,ax=ax,**kwargs)
    return (longitude,latitude,artists) if return_data else artists


def _elevation(datafile,limits,shape=None,byteorder='<'):
    if datafile is None:
        raise FileNotFoundError('Global elevation data were absent from the checkout; pass datafile= explicitly')
    path=Path(datafile)
    if not path.is_file():raise FileNotFoundError(path)
    if path.suffix.lower()=='.npz':
        with np.load(path,allow_pickle=False) as data:
            lon=np.asarray(data['lon']);lat=np.asarray(data['lat']);elev=np.asarray(data['elevation'])
    elif path.suffix.lower()=='.mat':
        data=loadmat(path)
        if all(k in data for k in ('lon','lat','elevation')):
            lon=np.ravel(data['lon']);lat=np.ravel(data['lat']);elev=np.asarray(data['elevation'])
        elif 'topo' in data:
            elev=np.asarray(data['topo']);lat=np.linspace(-89.5,89.5,elev.shape[0]);lon=np.linspace(.5,359.5,elev.shape[1])
        else:raise ValueError('MAT elevation file needs topo or lon/lat/elevation')
    else:
        if shape is None:raise ValueError('Raw elevation grids need an explicit shape')
        expected=np.prod(shape)*2
        if path.stat().st_size!=expected:raise ValueError('Raw elevation file size does not match the specified grid')
        elev=np.memmap(path,dtype=byteorder+'i2',mode='r',shape=tuple(shape))
        lat=np.linspace(90-90/shape[0],-90+90/shape[0],shape[0])
        lon=np.linspace(-180+180/shape[1],180-180/shape[1],shape[1])
    if lon.ndim!=1 or lat.ndim!=1 or elev.shape!=(len(lat),len(lon)):
        raise ValueError('Elevation grid and lon/lat axes do not match')
    lon=(lon+180)%360-180
    xi=np.argsort(lon);yi=np.argsort(lat);lon=lon[xi];lat=lat[yi]
    if limits is None:
        state=_state();limits=[*state['lon'],*state['lat']]
    bounds=np.asarray(limits,dtype=float)
    if bounds.shape!=(4,) or bounds[0]>=bounds[1] or bounds[2]>=bounds[3]:raise ValueError('limits must be lon_min,lon_max,lat_min,lat_max')
    xselect=(lon>=bounds[0])&(lon<=bounds[1]);yselect=(lat>=bounds[2])&(lat<=bounds[3])
    if not xselect.any() or not yselect.any():raise ValueError('Requested bounds contain no elevation grid points')
    return np.asarray(elev[np.ix_(yi[yselect],xi[xselect])]),lon[xselect],lat[yselect]


def m_elev(limits=None,*,datafile=None,draw=None,levels=None,ax=None,**kwargs):
    """Extract explicit elevation data or draw contour/contourf output.

    datafile is a MATLAB topo/axes file or numeric NPZ with lon,lat,elevation.
    With draw=None returns (elevation,longitude,latitude), independent of plot.
    """
    elev,lon,lat=_elevation(datafile,limits)
    if draw is None:return elev,lon,lat
    if draw not in ('contour','contourf'):raise ValueError('draw must be contour,contourf or None')
    if levels is not None:kwargs['levels']=levels
    return (m_contour if draw=='contour' else m_contourf)(lon,lat,elev,ax=ax,**kwargs)


def m_tbase(limits=None,*,datafile=None,shape=(2160,4320),byteorder='<',draw=None,levels=None,ax=None,**kwargs):
    """Read raw TerrainBase int16 grids (source default2160x4320), optionally draw."""
    elev,lon,lat=_elevation(datafile,limits,shape,byteorder)
    if draw is None:return elev,lon,lat
    if draw not in ('contour','contourf'):raise ValueError('draw must be contour/contourf or None')
    if levels is not None:kwargs['levels']=levels
    return (m_contour if draw=='contour' else m_contourf)(lon,lat,elev,ax=ax,**kwargs)


def m_etopo2(limits=None,*,datafile=None,shape=(5400,10800),byteorder='>',draw=None,levels=None,ax=None,**kwargs):
    """Read user-supplied ETOPO2 int16 cells, using explicit endian/grid metadata."""
    return m_tbase(limits,datafile=datafile,shape=shape,byteorder=byteorder,draw=draw,levels=levels,ax=ax,**kwargs)


def m_tba2b(source='tbase',destination=None,*,byteorder='<'):
    """Stream whitespace ASCII TerrainBase values to checked int16 binary.

    Source/destination are explicit filenames; no fixed working-directory moves.
    Returns the number of converted samples.
    """
    source=Path(source)
    destination=source.with_suffix('.int') if destination is None else Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('Source and destination must differ')
    count=0
    with source.open('r',encoding='ascii') as inp,destination.open('wb') as out:
        for line in inp:
            fields=line.split()
            if not fields:continue
            values=np.asarray([int(v) for v in fields],dtype=np.int64)
            if np.any(values<-32768)|np.any(values>32767):raise ValueError('TerrainBase sample exceeds int16 range')
            out.write(values.astype(byteorder+'i2').tobytes());count+=len(values)
    return count
