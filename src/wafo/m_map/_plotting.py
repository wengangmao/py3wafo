"""Matplotlib rendering of source M_Map line/field/vector capabilities.

Functions return artists and never call show(). Property/value pairs accept
common source spellings, while unsupported MATLAB GUI properties raise.
"""
import numpy as np
from ._projection import _state,m_ll2xy,m_xy2ll,m_proj
from ._geodesy import EARTH_RADIUS,m_lldist,m_fdist,m_xydist


def _ax(ax=None):
    import matplotlib.pyplot as plt
    return plt.gca() if ax is None else ax


def _options(args,kwargs):
    args=list(args); positional=[]; options=dict(kwargs)
    if args and isinstance(args[0],str) and (len(args)%2 or args[0].startswith(('-',':','.')) or len(args[0])<=2):
        positional.append(args.pop(0))
    if len(args)%2:raise ValueError('Expected paired artist properties')
    options.update({str(k).lower():v for k,v in zip(args[::2],args[1::2])})
    mapping={'linewidth':'linewidth','linestyle':'linestyle','markersize':'markersize',
             'facecolor':'facecolor','edgecolor':'edgecolor','tag':'gid','fontsize':'fontsize',
             'horizontalalignment':'ha','verticalalignment':'va'}
    return positional,{mapping.get(k,k):v for k,v in options.items()}


def _limits(ax):
    state=_state()
    lon=np.linspace(*state['lon'],81);lat=np.linspace(*state['lat'],41)
    xx,yy=m_ll2xy(*np.meshgrid(lon,lat),clip='off')
    valid=np.isfinite(xx)&np.isfinite(yy)
    if not valid.any():raise ValueError('No finite projected area in these bounds')
    ax.set_xlim(xx[valid].min(),xx[valid].max());ax.set_ylim(yy[valid].min(),yy[valid].max())
    ax.set_aspect('equal',adjustable='box')


def m_plot(long,lat,*args,ax=None,**kwargs):
    """Projected line plot; standard Matplotlib linespec/properties accepted."""
    ax=_ax(ax);positional,options=_options(args,kwargs)
    x,y=m_ll2xy(long,lat,clip='off')
    # Separate world-seam jumps; NaN clipping keeps false cross-map chords away.
    if np.ndim(x)==1 and len(x)>1:
        center=np.mean(_state()['lon'])
        wrapped=(np.asarray(long)-center+180)%360+center-180
        jumps=np.r_[False,abs(np.diff(wrapped))>180]
        x=np.where(jumps,np.nan,x);y=np.where(jumps,np.nan,y)
    lines=ax.plot(x,y,*positional,**options)
    for line in lines:line.set_gid(options.get('gid','m_plot'))
    return lines


def m_line(long,lat,*args,ax=None,**kwargs):
    return m_plot(long,lat,*args,ax=ax,**kwargs)


def m_text(long,lat,strings,*args,ax=None,**kwargs):
    """Projected text labels; scalar or broadcast-compatible label arrays."""
    ax=_ax(ax);positional,options=_options(args,kwargs)
    if positional:raise ValueError('Text has no line-style argument')
    x,y=m_ll2xy(long,lat,clip='point');x,y,labels=np.broadcast_arrays(x,y,strings)
    artists=[]
    for xx,yy,label in zip(x.ravel(),y.ravel(),labels.ravel()):
        if np.isfinite(xx+yy):
            artist=ax.text(xx,yy,str(label),**options);artist.set_gid(options.get('gid','m_text'));artists.append(artist)
    return artists[0] if len(artists)==1 else artists


def _field(long,lat,data):
    lon,latitude=np.asarray(long),np.asarray(lat)
    if lon.ndim==latitude.ndim==1 and lon.shape!=np.shape(data):
        lon,latitude=np.meshgrid(lon,latitude)
    if lon.shape!=latitude.shape or lon.shape!=np.shape(data):
        raise ValueError('Field coordinates and data must have matching grids')
    return m_ll2xy(lon,latitude,clip='off')


def m_contour(long,lat,data,*args,ax=None,**kwargs):
    ax=_ax(ax);x,y=_field(long,lat,data)
    result=ax.contour(x,y,data,*args,**kwargs);result.set_gid('m_contour')
    return result


def m_contourf(long,lat,data,*args,ax=None,**kwargs):
    ax=_ax(ax);x,y=_field(long,lat,data)
    result=ax.contourf(x,y,data,*args,**kwargs);result.set_gid('m_contourf')
    return result


def m_pcolor(long,lat,data,*args,ax=None,**kwargs):
    ax=_ax(ax);x,y=_field(long,lat,data)
    kwargs.setdefault('shading','auto')
    result=ax.pcolormesh(x,y,data,*args,**kwargs);result.set_gid('m_pcolor')
    return result


def m_patch(long,lat,*args,ax=None,**kwargs):
    """Fill projected polygons; source geographic clipping differs at borders."""
    ax=_ax(ax);positional,options=_options(args,kwargs)
    if positional:
        if len(positional)!=1:raise ValueError('Expected at most one fill colour')
        options.setdefault('facecolor',positional[0])
    x,y=m_ll2xy(long,lat,clip='off')
    artists=[]
    for indices in np.split(np.arange(np.size(x)),np.flatnonzero(~np.isfinite(np.ravel(x)+np.ravel(y)))):
        indices=indices[np.isfinite(np.ravel(x)[indices]+np.ravel(y)[indices])]
        if len(indices)>=3:
            artists.extend(ax.fill(np.ravel(x)[indices],np.ravel(y)[indices],**options))
    for artist in artists:artist.set_gid(options.get('gid','m_patch'))
    return artists


def m_quiver(long,lat,u,v,*args,ax=None,**kwargs):
    """East/north vectors rotated by the local projection Jacobian.

    Vector magnitudes use normalized projection axes directions; quiver's
    autoscale controls display length as in standard Matplotlib. Singular
    projection points are masked rather than producing nonfinite vectors.
    """
    ax=_ax(ax)
    lon,latitude,u,v=np.broadcast_arrays(long,lat,u,v)
    x,y=m_ll2xy(lon,latitude,clip='off')
    delta=1e-5
    xe,ye=m_ll2xy(lon+delta,latitude,clip='off')
    xn,yn=m_ll2xy(lon,np.minimum(latitude+delta,89.99999),clip='off')
    east=np.hypot(xe-x,ye-y);north=np.hypot(xn-x,yn-y)
    uu=np.divide((xe-x)*u,east,out=np.zeros_like(x,dtype=float),where=east>0)+np.divide((xn-x)*v,north,out=np.zeros_like(x,dtype=float),where=north>0)
    vv=np.divide((ye-y)*u,east,out=np.zeros_like(y,dtype=float),where=east>0)+np.divide((yn-y)*v,north,out=np.zeros_like(y,dtype=float),where=north>0)
    result=ax.quiver(x,y,uu,vv,*args,**kwargs);result.set_gid('m_quiver')
    return result


def m_vec(scale,long,lat,u,v,*args,ax=None,**kwargs):
    """Vector arrows using explicit Matplotlib quiver display scale."""
    if not np.isfinite(scale) or scale<=0:raise ValueError('Vector scale must be positive')
    kwargs['scale']=scale
    return m_quiver(long,lat,u,v,*args,ax=ax,**kwargs)


def m_grid(*args,ax=None,**kwargs):
    """Draw labelled meridians/parallels; returns grid artists.

    Supports lon/lat tick arrays (xtick/ytick), colour/linestyle/linewidth and
    font size. Legacy box/axis-direction options have no silent substitutes.
    """
    ax=_ax(ax)
    if len(args)%2:raise ValueError('Expected grid name/value pairs')
    options={str(k).lower():v for k,v in zip(args[::2],args[1::2])};options.update(kwargs)
    legal={'xtick','ytick','color','linestyle','linewidth','fontsize','labels','box'}
    if set(options)-legal:raise TypeError('Unsupported grid options: '+', '.join(set(options)-legal))
    state=_state();_limits(ax)
    longitudes=np.asarray(options.pop('xtick',np.linspace(*state['lon'],7)))
    latitudes=np.asarray(options.pop('ytick',np.linspace(*state['lat'],7)))
    if longitudes.ndim==0:longitudes=np.linspace(*state['lon'],int(longitudes))
    if latitudes.ndim==0:latitudes=np.linspace(*state['lat'],int(latitudes))
    font=options.pop('fontsize',8);labels=options.pop('labels',True);box=options.pop('box',True)
    if box not in (True,False,'on','off'):raise ValueError('box must be boolean/on/off')
    style=dict(color=options.pop('color','0.65'),linestyle=options.pop('linestyle',':'),linewidth=options.pop('linewidth',.7))
    artists=[]
    for longitude in longitudes:
        line=ax.plot(*m_ll2xy(np.full(181,longitude),np.linspace(*state['lat'],181),clip='off'),**style)[0]
        line.set_gid('m_grid');artists.append(line)
        if labels:
            x,y=m_ll2xy(longitude,state['lat'][0],clip='off')
            if np.isfinite(x+y):
                text=ax.text(x,y,f'{longitude:g}°',fontsize=font,ha='center',va='top');text.set_gid('m_grid');artists.append(text)
    for latitude in latitudes:
        line=ax.plot(*m_ll2xy(np.linspace(*state['lon'],361),np.full(361,latitude),clip='off'),**style)[0]
        line.set_gid('m_grid');artists.append(line)
        if labels:
            x,y=m_ll2xy(state['lon'][0],latitude,clip='off')
            if np.isfinite(x+y):
                text=ax.text(x,y,f'{latitude:g}°',fontsize=font,ha='right',va='center');text.set_gid('m_grid');artists.append(text)
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(box not in (False,'off'))
    return artists


def m_ungrid(element='m_grid',*,ax=None):
    """Remove tagged map artists; default removes only grid elements."""
    ax=_ax(ax);removed=0
    for artist in list(ax.get_children()):
        tag=artist.get_gid()
        if tag and (element=='all' and tag.startswith('m_') or tag==element):
            artist.remove();removed+=1
    return removed


def m_legend(*args,ax=None,**kwargs):
    result=_ax(ax).legend(*args,**kwargs);result.set_gid('m_legend')
    return result


def m_track(long,lat,time=None,*args,ax=None,annotate=5,**kwargs):
    """Plot and label a track; annotate is the number of evenly spaced labels."""
    ax=_ax(ax);lines=m_plot(long,lat,*args,ax=ax,**kwargs)
    lon,latitude=np.broadcast_arrays(long,lat)
    values=np.arange(lon.size) if time is None else np.asarray(time).ravel()
    if len(values)!=lon.size or int(annotate)!=annotate or annotate<0:
        raise ValueError('Time must match track points and annotate must be nonnegative')
    selected=np.unique(np.linspace(0,lon.size-1,min(int(annotate),lon.size)).astype(int)) if annotate else []
    texts=[]
    for i in selected:texts.extend(np.atleast_1d(m_text(lon.ravel()[i],latitude.ravel()[i],values[i],ax=ax)).tolist())
    for artist in [*lines,*texts]:artist.set_gid('m_track')
    return lines,texts


def m_range_ring(long,lat,range,npts=72,*args,ax=None,**kwargs):
    """Spherical range rings: radii in km, retained source Earth radius."""
    if int(npts)!=npts or npts<4:raise ValueError('npts must be >=4')
    lon,latitude=np.broadcast_arrays(long,lat);ranges=np.atleast_1d(range).astype(float)
    if np.any(ranges<0) or not np.all(np.isfinite(ranges)):raise ValueError('Ranges must be finite and nonnegative')
    azimuth=np.linspace(0,360,int(npts)+1);artists=[]
    for lo,la in zip(lon.ravel(),latitude.ravel()):
        for radius in ranges:
            longitude,latitudes,_=m_fdist(lo,la,azimuth,np.full_like(azimuth,radius*1000),(EARTH_RADIUS,0))
            artists.extend(m_plot(longitude,latitudes,*args,ax=ax,**kwargs))
    for artist in artists:artist.set_gid('m_range_ring')
    return artists


def m_hatch(lon,lat,style='single',angle=45,step=5,*,ax=None,**kwargs):
    """Matplotlib polygon hatch fill; returns patch artists.

    Hatch angle/spacing are display approximations: Matplotlib's native hatch
    alphabet supplies horizontal/vertical/diagonal patterns. Outside speckling
    and numeric hatch-intersection output are explicitly unsupported.
    """
    if style not in ('single','cross','speckle'):
        raise ValueError('Supported hatch styles: single,cross,speckle')
    if not np.isfinite(angle) or not np.isfinite(step) or step<=0:raise ValueError('Invalid hatch geometry')
    symbol='.' if style=='speckle' else 'x' if style=='cross' else '/' if 0<=angle%180<90 else '\\'
    kwargs.setdefault('facecolor','none');kwargs.setdefault('edgecolor','k')
    kwargs['hatch']=symbol*max(1,min(12,int(round(10/step))))
    return m_patch(lon,lat,ax=ax,**kwargs)


def m_scale(scale_factor=None,*,ax=None):
    """Get/set approximate physical paper scale 1:factor; 'auto' restores limits.

    Figure/display dimensions determine the physical size, so this is an
    explicit rendering adaptation rather than MATLAB paperposition parity.
    """
    ax=_ax(ax);state=_state()
    bbox=ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    metres_per_unit=1 if state['units']=='metres' else EARTH_RADIUS/state['radius']
    if scale_factor is None:
        return max(np.ptp(ax.get_xlim())*metres_per_unit/(bbox.width*.0254),np.ptp(ax.get_ylim())*metres_per_unit/(bbox.height*.0254))
    if isinstance(scale_factor,str):
        if scale_factor!='auto':raise ValueError('Scale string must be auto')
        _limits(ax);return m_scale(ax=ax)
    if not np.isfinite(scale_factor) or scale_factor<=0:raise ValueError('Scale must be positive')
    xmid=np.mean(ax.get_xlim());ymid=np.mean(ax.get_ylim())
    width=bbox.width*.0254*scale_factor/metres_per_unit;height=bbox.height*.0254*scale_factor/metres_per_unit
    ax.set_xlim(xmid-width/2,xmid+width/2);ax.set_ylim(ymid-height/2,ymid+height/2)
    return float(scale_factor)


def m_ruler(posx,posy,nints=4,*,ax=None,color='k',fontsize=8,linewidth=2):
    """Draw a normalized-axis scale ruler with geodesic kilometre labels."""
    ax=_ax(ax);px,py=np.broadcast_arrays(np.atleast_1d(posx),np.atleast_1d(posy))
    if px.size!=2 or int(nints)!=nints or nints<1 or np.any(px<0) or np.any(px>1) or np.any(py<0) or np.any(py>1):
        raise ValueError('Provide two normalized endpoints and a positive interval count')
    fractions=np.linspace(0,1,int(nints)+1)
    normalized=np.column_stack((px[0]+fractions*(px[1]-px[0]),py[0]+fractions*(py[1]-py[0])))
    projected=ax.transData.inverted().transform(ax.transAxes.transform(normalized))
    distances=np.r_[0,np.cumsum(m_xydist(projected[:,0],projected[:,1]))]
    artists=ax.plot(projected[:,0],projected[:,1],color=color,linewidth=linewidth)
    for (x,y),distance in zip(projected,distances):
        artists.append(ax.text(x,y,f'{distance:.3g} km',color=color,fontsize=fontsize,va='bottom',ha='center'))
    for artist in artists:artist.set_gid('m_ruler')
    return artists
