"""M_Map cartographic capabilities for Python.

PROJ/pyproj replaces hand-written projections/ellipsoid iteration; Matplotlib
replaces MATLAB graphics. Optional dependencies are imported lazily. The
original M_Map was authored by Rich Pawlowicz; original notices remain in the
supplied wafo_matlab/m_map source. This is an adapted capability port, not
full MATLAB property/clipping/file-format parity.
"""
from ._projection import m_proj,m_coord,m_ll2xy,m_xy2ll
from ._geodesy import m_idist,m_fdist,m_geodesic,m_lldist,m_xydist,m_geo2mag,m_mag2geo
from ._plotting import (m_plot,m_line,m_text,m_contour,m_contourf,m_pcolor,
    m_patch,m_quiver,m_vec,m_grid,m_ungrid,m_legend,m_track,m_range_ring,
    m_hatch,m_scale,m_ruler)
from ._datasets import (m_coast,m_usercoast,m_gshhs,m_gshhs_c,m_gshhs_l,
    m_gshhs_i,m_gshhs_h,m_gshhs_f,m_plotbndry,m_elev,m_tbase,m_etopo2,m_tba2b)
from ._compatibility import mc_ellips,mc_coords
from ._private_engines import (clabel,mp_azim,mp_conic,mp_cyl,mp_omerc,mp_tmerc,
    mp_utm,mu_coast,mu_util)


def m_demo(example=1):
    """Return a rendered coastline demonstration figure (examples1–3).

    This replaces the interactive source demo browser with three deterministic
    headless-compatible examples. It does not reproduce all legacy demo pages.
    """
    import matplotlib.pyplot as plt
    if example not in (1,2,3):raise ValueError('Available Python demo examples:1,2,3')
    figure,ax=plt.subplots(figsize=(8,4))
    if example==1:m_proj('miller',lon=[-180,180],lat=[-80,80])
    elif example==2:m_proj('mercator',lon=[-15,40],lat=[30,70])
    else:m_proj('orthographic',lon=[-100,80],lat=[-80,80],lon0=-10,lat0=20)
    m_coast(ax=ax,color='0.15',linewidth=.7);m_grid(ax=ax,labels=example!=3)
    ax.set_title('Python M_Map coastline demonstration')
    return figure,ax


__all__=sorted(name for name in globals() if name.startswith(('m_','mp_','mu_','mc_')))+['clabel']
