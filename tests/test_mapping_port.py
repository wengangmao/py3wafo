"""Actual numerical/format/render checks for the adapted M_Map capability port."""
import json
from importlib.resources import files
from pathlib import Path
import struct
import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal
import matplotlib
matplotlib.use('Agg',force=True)
import matplotlib.pyplot as plt
from wafo import m_map as m


PROJECTIONS=['stereographic','orthographic','azimuthal equal-area',
    'azimuthal equidistant','gnomonic','satellite','albers equal-area conic',
    'lambert conformal conic','equidistant conic','mercator','miller cylindrical',
    'equidistant cylindrical','sinusoidal','gall-peters','mollweide','robinson',
    'hammer-aitoff','transverse mercator','utm','oblique mercator']


@pytest.fixture(autouse=True)
def reset_map_context():
    m.m_coord('geographic')
    yield
    plt.close('all')
    m.m_coord('geographic')


@pytest.mark.parametrize('projection',PROJECTIONS)
def test_every_projection_roundtrip_and_explicit_units(projection):
    options=dict(lon=[-40,40],lat=[10,70],lon0=0,lat0=40)
    if projection=='utm':options['zone']=31
    state=m.m_proj(projection,**options)
    lon=np.array([-4,0,5.]);lat=np.array([30,40,50.])
    x,y=m.m_ll2xy(lon,lat,clip='off')
    assert np.all(np.isfinite(x+y))
    back=m.m_xy2ll(x,y)
    assert_allclose(back[0],lon,atol=4e-6)
    assert_allclose(back[1],lat,atol=4e-6)
    assert state['units']==('metres' if projection=='utm' else 'radius')
    assert m.m_proj('get')['name']==projection


def test_projection_option_errors_point_clipping_and_mercator_identity():
    assert 'mercator' in m.m_proj('set')
    m.m_proj('mercator','lon',[-20,20],'lat',[-60,60])
    x,y,changed=m.m_ll2xy([0,10,30],[0,30,40],return_mask=True)
    assert_allclose(x[:2],np.deg2rad([0,10]))
    assert_allclose(y[:2],[0,np.log(np.sqrt(3))],atol=1e-14)
    assert np.isnan(x[2]) and changed[2]
    assert np.isfinite(m.m_ll2xy(30,40,'clip','off')[0])
    assert_allclose(m.m_ll2xy(30,40,clip='patch')[0],np.deg2rad(20))
    with pytest.raises(ValueError):m.m_proj('not-a-map')
    with pytest.raises(TypeError):m.m_proj('mercator',imaginary_option=2)
    with pytest.raises(ValueError):m.m_proj('utm',zone=61)


def test_ellipsoid_forward_inverse_and_antipodes():
    d,a,b=m.m_idist([0,0],[0,0],[1,180],[0,0])
    assert_allclose(d,[111319.49079327357,20003931.458625447],atol=1e-6)
    assert_allclose(a[0],90);assert_allclose(b[0],270)
    lon,lat,back=m.m_fdist(0,0,a,d)
    assert_allclose(lon,[1,180],atol=1e-10)
    assert_allclose(lat,[0,0],atol=1e-10)
    s,lo,la=m.m_geodesic([0,10],[0,50],[90,20],[0,55],17)
    assert s.shape==lo.shape==la.shape==(17,2)
    assert_allclose(lo[[0,-1]],[[0,10],[90,20]],atol=1e-10)
    assert_allclose(la[[0,-1]],[[0,50],[0,55]],atol=1e-10)


def test_source_ellipsoid_table_and_explicit_historical_geodesy():
    historical=m.mc_ellips()
    assert len(historical)==12
    assert_allclose(historical['normal'],[1,0])
    assert_allclose(historical['wgs84'],[6378137,1/298.257],rtol=1e-15)
    assert_allclose(historical['grs67'],[6378160,1/247.247],rtol=1e-15)
    assert_allclose(historical['clrk66'],[6378206.4,1/294.97870],rtol=1e-15)
    for pair in historical.values():
        distance,_,_=m.m_idist(0,0,1,0,pair)
        assert_allclose(distance,pair[0]*np.pi/180,rtol=2e-13)
    historical['sphere'][0]=1
    assert m.mc_ellips()['sphere'][0]==6370997


def test_coordinate_private_dispatcher_parameters_vectors_and_identity():
    assert m.mc_coords('name')['name']==('geographic','IGRF2000-geomagnetic')
    p=m.mc_coords('parameters','IGRF2000-geomagnetic')
    assert_allclose(p['lambda'],np.arctan(5186/-1728),rtol=1e-15)
    assert p['phi']>0 and p['name']=='IGRF2000-geomagnetic'
    assert m.mc_coords('parameters','geographic')['phi']==0
    lon=np.array([[-45.,0.],[50.,170.]])
    lat=np.array([[-60.,10.],[40.,80.]])
    east=np.full_like(lon,2.)
    north=np.full_like(lon,3.)
    out=m.mc_coords('geo2mag',lon,lat,east,north)
    back=m.mc_coords('mag2geo',*out)
    for actual,expected in zip(back,(lon,lat,east,north)):
        assert_allclose(actual,expected,atol=1e-12)
    assert_allclose(out[2]**2+out[3]**2,13,atol=1e-13)
    for actual,expected in zip(m.mc_coords('geo2mag',lon,lat,east,north,system='geographic'),(lon,lat,east,north)):
        assert_array_equal(actual,expected)
    with pytest.raises(TypeError):m.mc_coords('geo2mag',lon,lat,east)
    with pytest.raises(ValueError):m.mc_coords('parameters','unknown')
    with pytest.raises(ValueError):m.mc_coords('invalid')


def test_spherical_distances_paths_and_xy_distances():
    d,lon,lat=m.m_lldist([0,90,90],[0,0,0],8)
    assert_allclose(d,[6378.137*np.pi/2,0])
    assert lon.shape==lat.shape==(2,9)
    assert_allclose(lon[0,[0,-1]],[0,90],atol=1e-10)
    m.m_proj('mercator',lon=[-10,100],lat=[-30,30])
    x,y=m.m_ll2xy([0,90],[0,0])
    assert_allclose(m.m_xydist(x,y),d[:1])


def test_dipole_and_tangent_vector_roundtrips_and_coordinate_selection():
    lon,lat=np.array([0,45,-80,170]),np.array([-60,-20,30,70])
    east,north=np.array([1,2,3,4]),np.array([-2,1,3,-1])
    rotated=m.m_geo2mag(lon,lat,east,north)
    back=m.m_mag2geo(*rotated)
    for actual,expected in zip(back,(lon,lat,east,north)):
        assert_allclose(actual,expected,atol=2e-12)
    assert_allclose(rotated[2]**2+rotated[3]**2,east**2+north**2,atol=1e-12)
    assert m.m_coord('get')=='geographic'
    assert 'IGRF2000-geomagnetic' in m.m_coord('set')
    m.m_coord('geomagnetic');m.m_proj('mercator',lon=[-180,180],lat=[-80,80])
    first=m.m_ll2xy(*rotated[:2],clip='off')
    m.m_coord('geographic')
    second=m.m_ll2xy(lon,lat,clip='off')
    assert_allclose(first,second,atol=1e-12)


def test_all_plot_primitives_render_and_preserve_data(tmp_path):
    m.m_proj('mercator',lon=[-20,20],lat=[-20,20])
    fig,ax=plt.subplots(figsize=(8,6))
    line=m.m_plot([-10,10],[0,0],'r-',ax=ax,label='track')[0]
    assert np.isfinite(line.get_xdata()).all()  # two-point line must not be seam-masked
    assert len(m.m_line([-10,-5],[1,2],ax=ax))==1
    assert m.m_text(0,10,'point',ax=ax).get_text()=='point'
    x=np.linspace(-15,15,15);xx,yy=np.meshgrid(x,x);z=np.sin(xx/5)+np.cos(yy/5)
    assert m.m_contour(x,x,z,levels=4,ax=ax) is not None
    assert m.m_contourf(x,x,z,levels=4,alpha=.2,ax=ax) is not None
    assert m.m_pcolor(x,x,z,alpha=.2,ax=ax) is not None
    assert len(m.m_patch([-5,0,5],[-5,2,-5],ax=ax,facecolor='gold'))==1
    assert m.m_quiver([-10,10],[0,0],[1,0],[0,1],ax=ax) is not None
    assert m.m_vec(10,[0],[5],[1],[1],ax=ax) is not None
    lines,text=m.m_track([-10,0,10],[-10,-8,-10],['a','b','c'],ax=ax,annotate=3)
    assert len(text)==3
    assert len(m.m_range_ring(0,0,[100,300],24,ax=ax))==2
    assert len(m.m_hatch([-10,-5,-5,-10],[5,5,10,10],'cross',ax=ax))==1
    assert m.m_legend(ax=ax) is not None
    artists=m.m_grid(xtick=[-20,0,20],ytick=[-20,0,20],ax=ax)
    assert len(artists)>=6
    fig.canvas.draw()
    assert m.m_scale(ax=ax)>0
    assert m.m_scale(5_000_000,ax=ax)==5_000_000
    m.m_scale('auto',ax=ax)
    assert len(m.m_ruler([.2,.5],.1,3,ax=ax))==5
    assert m.m_ungrid(ax=ax)>=6
    m.m_grid(ax=ax,labels=False)
    output=tmp_path/'map.png';fig.savefig(output)
    assert output.stat().st_size>10000


def test_bundled_coast_and_external_user_coast(tmp_path):
    m.m_proj('miller',lon=[-180,180],lat=[-75,75])
    fig,ax=plt.subplots()
    assert len(m.m_coast(ax=ax,color='0.2'))>0
    assert len(m.m_coast('patch',ax=ax,facecolor='0.8',edgecolor='0.2'))>100
    file=tmp_path/'coast.npz'
    np.savez(file,ncst=[[np.nan,np.nan],[0,0],[3,0],[3,3],[0,0]],Area=[1])
    assert len(m.m_usercoast(file,ax=ax))==1
    assert len(m.m_usercoast(file,'patch',ax=ax,facecolor='b'))==1
    provenance=json.loads(files('wafo.m_map').joinpath('data/PROVENANCE.json').read_text())
    assert provenance['source']=='m_map/private/m_coasts.mat' and len(provenance['sha256'])==64
    output=tmp_path/'coast.png';fig.savefig(output)
    assert output.stat().st_size>10000


def _gshhs(path):
    coords=np.array([[0,0],[3_000_000,0],[3_000_000,3_000_000],[0,0]],dtype='>i4')
    header=struct.pack('>11i',1,4,(15<<8)|1,0,3_000_000,0,3_000_000,100,100,-1,-1)
    path.write_bytes(header+coords.tobytes())


def test_gshhs2_binary_reading_wrappers_and_numeric_save(tmp_path):
    m.m_proj('mercator',lon=[-10,10],lat=[-10,10])
    database=tmp_path/'gshhs_c.b';_gshhs(database)
    for function in (m.m_gshhs_c,m.m_gshhs_l,m.m_gshhs_i,m.m_gshhs_h,m.m_gshhs_f):
        assert len(function(filename=database))==1
    output=tmp_path/'subset.npz'
    saved=m.m_gshhs('c','save',output,filename=database)
    assert saved['ncst'].shape==(5,2)
    assert len(m.m_usercoast(output,'patch',facecolor='gray'))==1
    with pytest.raises(FileNotFoundError):m.m_gshhs('f')
    with pytest.raises(ValueError):m.m_gshhs('bad',filename=database)


def test_political_boundary_and_explicit_elevation_datasets(tmp_path):
    m.m_proj('mercator',lon=[-10,10],lat=[-10,10])
    boundary=tmp_path/'border2pts.txt'
    boundary.write_text('1\n0 1\n1 2\nEND\n2\n2 3\n3 4\nEND\nEND\n')
    lon,lat,artists=m.m_plotbndry(tmp_path/'border',return_data=True)
    assert_allclose(lon[np.isfinite(lon)],[1,2,3,4]);assert len(artists)==1
    file=tmp_path/'topo.npz';x=np.array([-5,0,5]);y=np.array([-5,0,5])
    z=np.arange(9).reshape(3,3);np.savez(file,lon=x,lat=y,elevation=z)
    elev,lo,la=m.m_elev([-5,5,-5,5],datafile=file)
    assert_array_equal(elev,z);assert_array_equal(lo,x);assert_array_equal(la,y)
    assert m.m_elev([-5,5,-5,5],datafile=file,draw='contour') is not None
    raw=tmp_path/'tbase.int';z=np.array([[1,2,3],[4,5,6]],dtype='<i2');raw.write_bytes(z.tobytes())
    tbase,lo,la=m.m_tbase([-180,180,-90,90],datafile=raw,shape=(2,3))
    assert_array_equal(tbase,z[::-1]);assert_array_equal(la,[-45,45])
    big=tmp_path/'etopo.bin';big.write_bytes(z.astype('>i2').tobytes())
    assert_array_equal(m.m_etopo2([-180,180,-90,90],datafile=big,shape=(2,3))[0],tbase)
    ascii=tmp_path/'tbase';ascii.write_text('1 2 -3\n4 5 -6\n')
    output=tmp_path/'converted.int'
    assert m.m_tba2b(ascii,output)==6
    assert_array_equal(np.fromfile(output,dtype='<i2'),[1,2,-3,4,5,-6])
    with pytest.raises(FileNotFoundError):m.m_elev()


@pytest.mark.parametrize('example',[1,2,3])
def test_map_demo_renders(example,tmp_path):
    fig,ax=m.m_demo(example)
    output=tmp_path/f'demo{example}.png';fig.savefig(output)
    assert output.stat().st_size>10000
