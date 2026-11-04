# All MATLAB source files and conversion status

`implemented` means a corresponding Python implementation exists; it does not certify numerical parity with MATLAB. `adapted` additionally marks known API/algorithm restrictions. Tests cover representative core workflows; many entries have no individual parity fixture.

| MATLAB file | Functionality | Status | Python API |
|---|---|---|---|
| Contents.m | WAFO Toolbox | documentation-or-script |  |
| cycles/cc2amp.m | CC2AMP Calculates the amplitudes from a cycle count. | implemented | wafo.cycles.cc2amp |
| cycles/cc2cmat.m | CC2CMAT Calculates the cycle count matrix from a cycle count. | adapted | wafo.cycles.cc2cmat |
| cycles/cc2dcc.m | CC2DCC Discretize a cycle count. | implemented | wafo.cycles.cc2dcc |
| cycles/cc2lc.m | CC2LC Calculates the number of upcrossings from a cycle count | implemented | wafo.cycles.cc2lc |
| cycles/ccplot.m | CCPLOT Plots a cycle count as a point process in the plane. | adapted | wafo.cycles.ccplot |
| cycles/cmat2amp.m | CMAT2AMP Calculates a histogram of amplitudes from a cycle matrix. | implemented | wafo.cycles.cmat2amp |
| cycles/cmat2extralc.m | CMAT2EXTRALC  Extrapolate level crossing spectrum | adapted | wafo.cycles.cmat2extralc |
| cycles/cmat2lc.m | CMAT2LC Calculates the level crossings from a cycle matrix. | implemented | wafo.cycles.cmat2lc |
| cycles/cmat2nt.m | CMAT2NT Calculates a counting distribution from a cycle matrix. | implemented | wafo.cycles.cmat2nt |
| cycles/cmat2rmcmat.m | CMAT2RMCMAT  Converts a cycle matrix from Range-Mean format to min-max format. | implemented | wafo.cycles.cmat2rmcmat |
| cycles/cmatcombine.m | CMATCOMBINE Combines two cycle matrices. | implemented | wafo.cycles.cmatcombine |
| cycles/cmatplot.m | CMATPLOT Plots a cycle matrix, e.g. a rainflow matrix. | adapted | wafo.cycles.cmatplot |
| cycles/cmatresamp.m | CMATRESAMP Resamples a cycle matrix. | implemented | wafo.cycles.cmatresamp |
| cycles/cocc.m | COCC Plots cycles as points together with isolines of a cycle matrix. | adapted | wafo.cycles.cocc |
| cycles/Contents.m | Module CYCLES in WAFO Toolbox. | documentation-or-script |  |
| cycles/dat2dtp.m | DAT2DTP The sequence of discretized turning points from a signal. | implemented | wafo.cycles.dat2dtp |
| cycles/dat2rfm.m | DAT2RFM  Calculates the rainflow matrix from a time signal. | implemented | wafo.cycles.dat2rfm |
| cycles/dcc2cmat.m | DCC2CMAT  Calculates the cycle matrix for a discrete cycle count. | implemented | wafo.cycles.dcc2cmat |
| cycles/dtp2arfm.m | DTP2ARFM Calculates asymmetric RFM from discrete turning points. | implemented | wafo.cycles.dtp2arfm |
| cycles/dtp2arfm4p.m | DTP2ARFM4P Calculates asymmetric RFM from discrete turning points (4-point). | implemented | wafo.cycles.dtp2arfm4p |
| cycles/dtp2arfm_sid.m | DTP2ARFM_SID  Asymmetric RFM from discrete TP with side information. | adapted | wafo.cycles.dtp2arfm_sid |
| cycles/dtp2rfm.m | DTP2RFM Calculates rainflow matrix from discrete turning points. | implemented | wafo.cycles.dtp2rfm |
| cycles/dtp2rfm_sid.m | DTP2RFM_SID  Rainflow matrix from discrete turning points with side information. | adapted | wafo.cycles.dtp2rfm_sid |
| cycles/extralc.m | EXTRALC  Extrapolate level crossing spectrum | adapted | wafo.cycles.extralc |
| cycles/lc2rfmextreme.m | LC2RFMEXTREME Compute extreme RFM from level crossings. | adapted | wafo.cycles.lc2rfmextreme |
| cycles/lsplot.m | LSPLOT  Plot load spectra. | adapted | wafo.cycles.lsplot |
| cycles/nt2cmat.m | NT2CMAT  Calculates a cycle matrix from a counting distribution. | implemented | wafo.cycles.nt2cmat |
| cycles/nt2lc.m | NT2LC Calculates the level crossings from a cycle matrix. | implemented | wafo.cycles.nt2lc |
| cycles/Readme.m | README Readme file for module CYCLES in WAFO Toolbox. | documentation-or-script |  |
| cycles/res2arfc.m | RES2ARFC Calculates asymmetric rainflow cycles for a residual. | adapted | wafo.cycles.res2arfc |
| cycles/rfcfilter.m | RFCFILTER Rainflow filter a signal. | implemented | wafo.cycles.rfcfilter |
| cycles/rfmextrapolate.m | RFMEXTRAPOLATE Extrapolates a rainflow matrix. | adapted | wafo.cycles.rfmextrapolate |
| cycles/rmcmat2cmat.m | RMCMAT2CMAT Converts a cycle matrix from Range-Mean format to min-max format. | implemented | wafo.cycles.rmcmat2cmat |
| cycles/smoothcmat.m | SMOOTHCMAT Smooth a cycle matrix using (adaptive) kernel smoothing | adapted | wafo.cycles.smoothcmat |
| cycles/smoothcmat_hnorm.m | SMOOTHCMAT_HNORM  Bandwidth selection for kernel smoothing of a cycle matrix. | adapted | wafo.cycles.smoothcmat_hnorm |
| cycles/test/test_cycles.m | TEST_CYCLES Quick test of the routines in module 'cycles' | documentation-or-script |  |
| cycles/tp2arfc.m | TP2ARFC Calculates asymmetric rainflow cycles from turning points. | implemented | wafo.cycles.tp2arfc |
| cycles/tp2arfc4p.m | TP2ARFC4P Calculates asymmetric rainflow cycles from turning points (4-point). | implemented | wafo.cycles.tp2arfc4p |
| cycles/tp2lc.m | TP2LC  Calculates the number of upcrossings from the turning points. | implemented | wafo.cycles.tp2lc |
| cycles/tp2mm.m | TP2MM Calculates min2Max and Max2min cycles from a sequence of turning points | implemented | wafo.cycles.tp2mm |
| cycles/tp2rfc.m | TP2RFC Finds the rainflow cycles from the sequence of turning points. | implemented | wafo.cycles.tp2rfc |
| cycles/tpextrapolate.m | TPEXTRAPOLATE Extrapolates a sequence of turning points. | adapted | wafo.cycles.tpextrapolate |
| cycles/wgpdfit_mld.m | WGPDFIT_MLD Parameter estimates for GPD data | adapted | wafo.cycles.wgpdfit_mld |
| damage/cc2dam.m | CC2DAM   Calculates the total Palmgren-Miner damage of a cycle count. | implemented | wafo.damage.cc2dam |
| damage/cmat2dam.m | CMAT2DAM Calculates the total Palmgren-Miner damage of a cycle matrix. | implemented | wafo.damage.cmat2dam |
| damage/cmat2dmat.m | CMAT2DMAT Computes the (Palmgren-Miner) damage matrix from a cycle matrix. | implemented | wafo.damage.cmat2dmat |
| damage/Contents.m | Module DAMAGE in WAFO Toolbox. | documentation-or-script |  |
| damage/damint.m | DAMINT Calculates damage intensity from counting distribution. | adapted | wafo.damage.damint |
| damage/down2cc.m | DOWN2CC Calculates the cycle count which has the highest damage | adapted | wafo.damage.down2cc |
| damage/ftf.m | FTF  Calculates the fatigue failure time distribution. | implemented | wafo.damage.ftf |
| damage/lc2dplus.m | LC2DPLUS Upper bound for total damage from level crossings. | adapted | wafo.damage.lc2dplus |
| damage/roadspec.m | ROADSPEC Spectral density (frequency) for a road | implemented | wafo.damage.roadspec |
| damage/snplot.m | SNPLOT Plots SN-data and estimates parameters | adapted | wafo.damage.snplot |
| damage/spec2dplus.m | SPEC2DPLUS Calculates an upper bound of the damage intensity explicitly | implemented | wafo.damage.spec2dplus |
| damage/sphdam.m | SPHDAM Calculates spherical damage for a 3-D load. | implemented | wafo.damage.sphdam |
| data/atlantic.m | ATLANTIC Significant wave-height data recorded in the Atlantic Ocean | documentation-or-script |  |
| data/Contents.m | Module DATA in WAFO Toolbox | documentation-or-script |  |
| data/gfaks89.m | GFAKS89  Surface elevation measured at Gullfaks C 24.12.1989 | documentation-or-script |  |
| data/gfaksr89.m | GFAKSR89  Reconstructed surface elevation measured at Gullfaks C 24.12.1989. | documentation-or-script |  |
| data/japansea.m | JAPANSEA  coastline map of The Japan Sea | documentation-or-script |  |
| data/northsea.m | NORTHSEA  coastline map of The Nortsea | documentation-or-script |  |
| data/readme.m | README Readme file for module DATA in  WAFO Toolbox. | documentation-or-script |  |
| data/sea.m | SEA Surface elevation dataset used in WAT version 1.1. | documentation-or-script |  |
| data/sfa89.m | SFA89 Wind measurements at Statfjord A 24.12.1989 | documentation-or-script |  |
| data/sn.m | SN Fatigue experiment, constant-amplitude loading. | documentation-or-script |  |
| data/yura87.m | YURA87  Surface elevation measured off the coast of Yura. | documentation-or-script |  |
| docs/aboutwafo.m | ABOUTWAFO  WAFO is a Matlab toolbox for analysis of random waves | documentation-or-script |  |
| docs/addfeatr.m | ADDFEATR  How To Add New Features to WAFO | documentation-or-script |  |
| docs/ampdef.m | AMPDEF wave heights and amplitude definitions and nomenclature | documentation-or-script |  |
| docs/bugreport.m | BUGREPORT Bug Report Form for WAFO | documentation-or-script |  |
| docs/Contents.m | DOCS MODULE in WAFO Toolbox | documentation-or-script |  |
| docs/crossdef.m | CROSSDEF level v crossing definitions and nomenclature | documentation-or-script |  |
| docs/datastructures.m | DATASTRUCTURES of spectrum, covariance function and density (pdf) in WAFO | documentation-or-script |  |
| docs/getstart.m | GETSTART  ideas for getting started with WAFO | documentation-or-script |  |
| docs/install.m | Installation description of WAFO Toolbox | documentation-or-script |  |
| docs/limitations.m | LIMITATIONS (+)  WAFO known limitations | documentation-or-script |  |
| docs/perioddef.m | PERIODDEF wave periods (lengths) definitions and nomenclature | documentation-or-script |  |
| docs/startup.m | STARTUP example startup file to put in your matlab path | documentation-or-script |  |
| docs/tpdef.m | TPDEF turning points definitions and numenclature | documentation-or-script |  |
| docs/wafohelp.m | WAFOHELP  Launch HTML help for the WAFO Toolbox. | adapted | wafo.docs.wafohelp |
| docs/wafomenu.m | WAFOMENU displays a user interface to the documentation files of WAFO | adapted | wafo.docs.wafomenu |
| docs/wafoweb.m | WAFOWEB  World Wide Web site of the WAFO Toolbox. | adapted | wafo.docs.wafoweb |
| docs/wavedef.m | WAVEDEF Wave definitions and nomenclature | documentation-or-script |  |
| exec/Contents.m | WAFO  Toolbox /exec | documentation-or-script |  |
| fatigue.m | Fatigue in WAFO (Wave Analysis for Fatigue and Oceanography). | documentation-or-script |  |
| initwafo.m | INITWAFO   Initiation of WAFO Toolbox. | adapted | wafo.root.initwafo |
| kdetools/bincount.m | BINCOUNT 1-dimensional Bin Count | adapted | wafo.kdetools.bincount |
| kdetools/chi2gof.m | CHI2GOF CHI Squared Goodness Of Fit test. | adapted | wafo.kdetools.chi2gof |
| kdetools/chi2gof2.m | CHI2GOF2 CHI Squared Goodness Of Fit test. | adapted | wafo.kdetools.chi2gof2 |
| kdetools/Contents.m | Module KDETOOLS in WAFO Toolbox. | documentation-or-script |  |
| kdetools/deriv.m | DERIV  4th, 6th, 8th and 10th derivatives of the kernel function. | implemented | wafo.kdetools.deriv |
| kdetools/deriv2.m | DERIV2  High order partial derivatives of the Gaussian kernel. | implemented | wafo.kdetools.deriv2 |
| kdetools/evalpdf.m | EVALPDF evaluates a PDF struct by interpolation | adapted | wafo.kdetools.evalpdf |
| kdetools/ffndgrid.m | FFNDGRID  Fast 'n' Furious N-D data gridding. | adapted | wafo.kdetools.ffndgrid |
| kdetools/fftce.m | FFTCE  Circulant Embedding of a vector or matrix | adapted | wafo.kdetools.fftce |
| kdetools/gridcount.m | GRIDCOUNT D-dimensional histogram using linear binning. | adapted | wafo.kdetools.gridcount |
| kdetools/hbcv.m | HBCV  Biased Cross-Validation estimate of smoothing parameter. | adapted | wafo.kdetools.hbcv |
| kdetools/hbcv2.m | HBCV2 Biased Cross-Validation smoothing parameter for 2D data. | adapted | wafo.kdetools.hbcv2 |
| kdetools/hboot.m | HBOOT  Bootstrap cross-validation estimate of smoothing parameter. | adapted | wafo.kdetools.hboot |
| kdetools/hldpi.m | HLDPI L-stage Direct Plug-In estimate of smoothing parameter. | adapted | wafo.kdetools.hldpi |
| kdetools/hldpi2.m | HLDPI2  L-stage DPI estimate of smoothing parameter for 2D data | adapted | wafo.kdetools.hldpi2 |
| kdetools/hldpi2fft.m | HLDPI2FFT L-stage DPI estimate of smoothing parameter for 2D data | adapted | wafo.kdetools.hldpi2fft |
| kdetools/hlscv.m | HLSCV  Least Squares Cross-Validation estimate of smoothing parameter | adapted | wafo.kdetools.hlscv |
| kdetools/hmns.m | HMNS Multivariate Normal Scale Estimate of Smoothing Parameter. | implemented | wafo.kdetools.hmns |
| kdetools/hns.m | HNS Normal Scale Estimate of Smoothing Parameter. | adapted | wafo.kdetools.hns |
| kdetools/hos.m | HOS Oversmoothing Parameter. | implemented | wafo.kdetools.hos |
| kdetools/hscv.m | HSCV Smoothed cross-validation estimate of smoothing parameter. | adapted | wafo.kdetools.hscv |
| kdetools/hste.m | HSTE 2-Stage Solve the Equation estimate of smoothing parameter. | adapted | wafo.kdetools.hste |
| kdetools/hstt.m | HSTT Scott-Tapia-Thompson estimate of smoothing parameter. | adapted | wafo.kdetools.hstt |
| kdetools/kde.m | KDE Kernel Density Estimator. | adapted | wafo.kdetools.kde |
| kdetools/kde1dgui.m | KDE1DGUI GUI to Kernel Density Estimator. | adapted | wafo.kdetools.kde1dgui |
| kdetools/kde2dgui.m | KDE2DGUI GUI to Kernel Density Estimator in two dimensions. | adapted | wafo.kdetools.kde2dgui |
| kdetools/kdebin.m | KDEBIN Binned Kernel Density Estimator. | adapted | wafo.kdetools.kdebin |
| kdetools/kdedemo1.m | KDEDEMO1 Demonstrate the smoothing parameter impact on KDE | adapted | wafo.kdetools.kdedemo1 |
| kdetools/kdedemo2.m | KDEDEMO2 Demonstrate the difference between transformation- and ordinary-KDE | adapted | wafo.kdetools.kdedemo2 |
| kdetools/kdefun.m | KDEFUN  Kernel Density Estimator. | adapted | wafo.kdetools.kdefun |
| kdetools/kdeoptset.m | KDEOPTSET Create or alter KDE OPTIONS structure. | adapted | wafo.kdetools.kdeoptset |
| kdetools/kernelstats.m | KERNELSTATS Return 2'nd order moment of kernel pdf | implemented | wafo.kdetools.kernelstats |
| kdetools/mkernel.m | MKERNEL Multivariate Kernel Function. | adapted | wafo.kdetools.mkernel |
| kdetools/mkernel2.m | MKERNEL2 Multivariate Kernel Function, alternative version. | adapted | wafo.kdetools.mkernel2 |
| kdetools/mkernelrnd.m | MKERNELRND Random numbers from the Multivariate Kernel Function. | adapted | wafo.kdetools.mkernelrnd |
| kdetools/qlevels.m | QLEVELS Calculates quantile levels which encloses P% of PDF | adapted | wafo.kdetools.qlevels |
| kdetools/qlevels2.m | QLEVELS2 Calculates quantile levels which encloses P% of data | adapted | wafo.kdetools.qlevels2 |
| kdetools/Readme.m | README Readme file for module KDETOOLS in WAFO Toolbox. | documentation-or-script |  |
| kdetools/sample.m | SAMPLE  Random sampling of points from a data-set | adapted | wafo.kdetools.sample |
| kdetools/ssample.m | SSAMPLE  Random sampling from a smoothed empirical distribution | adapted | wafo.kdetools.ssample |
| kdetools/vsph.m | VSPH calculates volume of  d-dimensional sphere with radius r0 | adapted | wafo.kdetools.vsph |
| m_map/Contents.m | M_Map - mapping toolbox (Author: rich@eos.ubc.ca) | documentation-or-script |  |
| m_map/m_coast.m | M_COAST Add a coastline to a given map. | adapted | wafo.m_map.m_coast |
| m_map/m_contour.m | M_CONTOUR Draws contour lines on a map | adapted | wafo.m_map.m_contour |
| m_map/m_contourf.m | M_CONTOURF Adds filled contours to a map | adapted | wafo.m_map.m_contourf |
| m_map/m_coord.m | M_COORD Initializes the coordinate system for varous conversions. | adapted | wafo.m_map.m_coord |
| m_map/m_demo.m | M_DEMO  Demonstration program showing various maps in M_Map package | implemented | wafo.m_map.m_demo |
| m_map/m_elev.m | M_ELEV Contour elevation onto a map using a 1-degree database | adapted | wafo.m_map.m_elev |
| m_map/m_etopo2.m | M_ETOPO2 Contour elevation onto a map using the 2-minute ETOPO2 database | adapted | wafo.m_map.m_etopo2 |
| m_map/m_fdist.m | M_FDIST - On an ellipsoidal earth, compute location of | adapted | wafo.m_map.m_fdist |
| m_map/m_geo2mag.m | M_GEO2MAG  Converts geographic to geomagnetic coordinates. | adapted | wafo.m_map.m_geo2mag |
| m_map/m_geodesic.m | M_GEODESIC - Compute points along geodesics of an ellipsoidal earth. | adapted | wafo.m_map.m_geodesic |
| m_map/m_grid.m | M_GRID make a grid on a map. | adapted | wafo.m_map.m_grid |
| m_map/m_gshhs.m | M_GSHHS Add a coastline to a given map using | adapted | wafo.m_map.m_gshhs |
| m_map/m_gshhs_c.m | M_GSHHS_C Add a coastline to a given map using the 'crude' resolution of | adapted | wafo.m_map.m_gshhs_c |
| m_map/m_gshhs_f.m | M_GSHHS_F Add a coastline to a given map using the 'full' resolution of | adapted | wafo.m_map.m_gshhs_f |
| m_map/m_gshhs_h.m | M_GSHHS_H Add a coastline to a given map using the 'high' resolution of | adapted | wafo.m_map.m_gshhs_h |
| m_map/m_gshhs_i.m | M_GSHHS_I Add a coastline to a given map using the 'intermediate' resolution of | adapted | wafo.m_map.m_gshhs_i |
| m_map/m_gshhs_l.m | M_GSHHS_L Add a coastline to a given map using the 'low' resolution of | adapted | wafo.m_map.m_gshhs_l |
| m_map/m_hatch.m | M_HATCH Draws hatched or speckled interiors to a patch | adapted | wafo.m_map.m_hatch |
| m_map/m_idist.m | M_IDIST - On an ellipsoidal earth, compute the distance between | adapted | wafo.m_map.m_idist |
| m_map/m_legend.m | M_LEGEND Add a legend to a map. This function does not have the | adapted | wafo.m_map.m_legend |
| m_map/m_line.m | M_LINE Create a line on a map | adapted | wafo.m_map.m_line |
| m_map/m_ll2xy.m | M_LL2XY Converts long,lat to X,Y coordinates using the current projection | adapted | wafo.m_map.m_ll2xy |
| m_map/m_lldist.m | M_LLDIST Spherical earth distance between points in long/lat coordinates. | adapted | wafo.m_map.m_lldist |
| m_map/m_mag2geo.m | M_MAG2GEO  Converts magnetic to geographic coordinates. | adapted | wafo.m_map.m_mag2geo |
| m_map/m_patch.m | M_PATCH Create patches on a map | adapted | wafo.m_map.m_patch |
| m_map/m_pcolor.m | M_PCOLOR Makes a pcolor image on a map. | adapted | wafo.m_map.m_pcolor |
| m_map/m_plot.m | M_PLOT Plot objects on an M_MAP plot.   All of the normal | adapted | wafo.m_map.m_plot |
| m_map/m_plotbndry.m | M_PLOTBNDRY plots text files of Lat,Lon for political boundaries. | adapted | wafo.m_map.m_plotbndry |
| m_map/m_proj.m | M_PROJ  Initializes map projections info, putting the result into a structure | adapted | wafo.m_map.m_proj |
| m_map/m_quiver.m | M_QUIVER Makes a quiverplot on a map (QUIVER-style) | adapted | wafo.m_map.m_quiver |
| m_map/m_range_ring.m | M_RANGE_RING Creates range rings on a map | adapted | wafo.m_map.m_range_ring |
| m_map/m_ruler.m | M_RULER Draws a distance scalebar for a map | adapted | wafo.m_map.m_ruler |
| m_map/m_scale.m | M_SCALE Draws the map at a specified scale | adapted | wafo.m_map.m_scale |
| m_map/m_tba2b.m | M_TBA2B Converts the ASCII TerrainBase 5-minute bathymetry database | adapted | wafo.m_map.m_tba2b |
| m_map/m_tbase.m | M_TBASE Contour elevation onto a map using the 5-minute TerrainBase database | adapted | wafo.m_map.m_tbase |
| m_map/m_text.m | M_TEXT Text Annotation | adapted | wafo.m_map.m_text |
| m_map/m_track.m | M_TRACK Draw a trackline on a map | adapted | wafo.m_map.m_track |
| m_map/m_ungrid.m | M_UNGRID Removes a grid; | adapted | wafo.m_map.m_ungrid |
| m_map/m_usercoast.m | M_USERCOAST Add a coastline using a user-specified coastline file. | adapted | wafo.m_map.m_usercoast |
| m_map/m_vec.m | M_VEC Draws fancy arrows/quiverplots on a map. | adapted | wafo.m_map.m_vec |
| m_map/m_xy2ll.m | M_XY2LL Converts X,Y to long,lat coordinates using the current projection | adapted | wafo.m_map.m_xy2ll |
| m_map/m_xydist.m | M_XYDIST Spherical earth distance between points in map projection coordinates. | adapted | wafo.m_map.m_xydist |
| m_map/private/clabel.m | CLABEL Contour plot elevation labels. | adapted | wafo.m_map.clabel |
| m_map/private/mc_coords.m | MP_COORD Converts between coordinate systems based on different | adapted | wafo.m_map.mc_coords |
| m_map/private/mc_ellips.m | Definitions of different ellipsoids | adapted | wafo.m_map.mc_ellips |
| m_map/private/mp_azim.m | MP_AZIM Azimuthal projections | adapted | wafo.m_map.mp_azim |
| m_map/private/mp_conic.m | MP_CONIC  Conic projections | adapted | wafo.m_map.mp_conic |
| m_map/private/mp_cyl.m | MP_CYL   Cylindrical projections | adapted | wafo.m_map.mp_cyl |
| m_map/private/mp_omerc.m | MP_TMERC   Oblique Mercator projection | adapted | wafo.m_map.mp_omerc |
| m_map/private/mp_tmerc.m | MP_TMERC   Transverse Mercator projection | adapted | wafo.m_map.mp_tmerc |
| m_map/private/mp_utm.m | MP_UTM   Universal Transverse Mercator projection | adapted | wafo.m_map.mp_utm |
| m_map/private/mu_coast.m | MU_COAST Add a coastline to a given map. | adapted | wafo.m_map.mu_coast |
| m_map/private/mu_util.m | MU_UTIL Various utility routines | adapted | wafo.m_map.mu_util |
| markov/arfm2mctp.m | ARFM2MCTP  Calculates the markov matrix given an asymmetric rainflow matrix. | adapted | wafo.markov.arfm2mctp |
| markov/chi2cmat.m | CHI2CMAT  Chi-square distance of cycle matrix. | implemented | wafo.markov.chi2cmat |
| markov/Contents.m | Module MARKOV in WAFO Toolbox. | documentation-or-script |  |
| markov/estmc.m | ESTMC  estimates transition matrix P from a time series of a Markov chain. | implemented | wafo.markov.estmc |
| markov/estsmctp.m | ESTSMCTP  Estimate SMCTP model from an observed rainflow matrix. | adapted | wafo.markov.estsmctp |
| markov/f_funm.m | F_FUNM  Calculate min-max matrix for Model-structure. | implemented | wafo.markov.f_funm |
| markov/f_smctp.m | F_SMCTP  Auxiliary function used by ESTSMCTP | adapted | wafo.markov.f_smctp |
| markov/hdcmat.m | HDCMAT  Hellinger distance of cycle matrix. | implemented | wafo.markov.hdcmat |
| markov/hmmplot.m | HMMPLOT  plots a Hidden Markov Model. | adapted | wafo.markov.hmmplot |
| markov/klcmat.m | KLCMAT  Kullback-Leibler distance of cycle matrix. | implemented | wafo.markov.klcmat |
| markov/loglcmat.m | LOGLCMAT  log-Likelihood of cycle matrix. | implemented | wafo.markov.loglcmat |
| markov/mat2tmat.m | MAT2TMAT  Converts a matrix to a transition matrix. | implemented | wafo.markov.mat2tmat |
| markov/mc2rfm.m | MC2RFM  Calculates the rainflow matrix/intensity for a Markov chain. | adapted | wafo.markov.mc2rfm |
| markov/mc2stat.m | MC2STAT  Calculates the stationary distribution for a Markov chain. | implemented | wafo.markov.mc2stat |
| markov/mctp2arfm.m | MCTP2ARFM  Calculates asymmetric rainflow matrix for a MCTP. | adapted | wafo.markov.mctp2arfm |
| markov/mctp2reverse.m | MCTP2REVERSE  Calculates the time-reversed MCTP for a SMCTP. | implemented | wafo.markov.mctp2reverse |
| markov/mctp2rfm.m | MCTP2RFM  Calculates the rainflow matrix for a MCTP. | implemented | wafo.markov.mctp2rfm |
| markov/mctp2stat.m | MCTP2STAT  Calculates the stationary distribution for a MCTP. | implemented | wafo.markov.mctp2stat |
| markov/mktestmat.m | MKTESTMAT   Makes test matrices for min-max (and max-min) matrices. | implemented | wafo.markov.mktestmat |
| markov/Readme.m | README Readme file for module MARKOV in WAFO Toolbox. | documentation-or-script |  |
| markov/scalemat.m | SCALEMAT Scale and translate a cycle matrix. | adapted | wafo.markov.scalemat |
| markov/smc2rfm.m | SMC2RFM Calculates the rainflow matrix/intensity for a switching Markov chain. | adapted | wafo.markov.smc2rfm |
| markov/smctp2arfm.m | SMCTP2ARFM  Calculates the asymmetric rainflow matrix for a SMCTP. | adapted | wafo.markov.smctp2arfm |
| markov/smctp2joint.m | SMCTP2JOINT  Calculates the joint MCTP for a SMCTP. | implemented | wafo.markov.smctp2joint |
| markov/smctp2rfm.m | SMCTP2RFM  Calculates the rainflow matrix for a SMCTP. | implemented | wafo.markov.smctp2rfm |
| markov/smctp2stat.m | SMCTP2STAT  Stationary distributions for a switching MCTP. | implemented | wafo.markov.smctp2stat |
| markov/test/test_markov.m | TEST_MARKOV Quick test of the routines in module 'markov' | documentation-or-script |  |
| markov/tr_m2x.m | TR_M2X Transform Model-structure to X-vector. | implemented | wafo.markov.tr_m2x |
| markov/tr_p2x.m | TR_P2X  Transform P-matrix to X-vector | implemented | wafo.markov.tr_p2x |
| markov/tr_x2m.m | TR_X2M Transform X-vector to Model-structure. | implemented | wafo.markov.tr_x2m |
| markov/tr_x2p.m | TR_X2P  Transforms a vector X to a transition matrix P. | implemented | wafo.markov.tr_x2p |
| misc/binom.m | BINOM calculates the binomial coefficient n!/((n-m)!*m!) | implemented | wafo.misc.binom |
| misc/ccquad.m | CCQUAD Numerical integration using a Clenshaw-Curtis quadrature. | adapted | wafo.misc.ccquad |
| misc/choices.m | CHOICES Create a list of choices with uicontrols and callbacks. | adapted | wafo.misc.choices |
| misc/choicex.m | CHOICEX Close a list of choices. | adapted | wafo.misc.choicex |
| misc/clevels.m | CLEVELS  Extract the contour levels from the contour matrix | adapted | wafo.misc.clevels |
| misc/cltext.m | CLTEXT Places contour level text in the current window | adapted | wafo.misc.cltext |
| misc/comnsize.m | COMNSIZE Check if all input arguments are either scalar or of common size. | adapted | wafo.misc.comnsize |
| misc/Contents.m | Module MISC in WAFO Toolbox. | documentation-or-script |  |
| misc/convlv.m | CONVLV Convolves real data set with a response function. | adapted | wafo.misc.convlv |
| misc/csort.m | CSORT Counting sorting | implemented | wafo.misc.csort |
| misc/fcolorbar.m | FCOLORBAR  Display colorbar with discrete color axis for filled contour plot | adapted | wafo.misc.fcolorbar |
| misc/figtext.m | FIGTEXT Places text in figure window. | adapted | wafo.misc.figtext |
| misc/findpeaks.m | FINDPEAKS find peaks of vector or matrix possibly rainflow filtered | adapted | wafo.misc.findpeaks |
| misc/fwaitbar.m | FWAITBAR Fast display of wait bar. | adapted | wafo.misc.fwaitbar |
| misc/gaussq.m | GAUSSQ Numerically evaluates a integral using a Gauss quadrature. | adapted | wafo.misc.gaussq |
| misc/gaussq2d.m | GAUSSQ2D Numerically evaluates a 2D integral using Gauss quadrature. | adapted | wafo.misc.gaussq2d |
| misc/genchol.m | GENCHOL Generalized Cholesky factorization | adapted | wafo.misc.genchol |
| misc/geth1line.m | GETH1LINE Extracts the first comment line (the H1 line) of a m-file | adapted | wafo.misc.geth1line |
| misc/gravity.m | GRAVITY  returns the constant acceleration of gravity | implemented | wafo.misc.gravity |
| misc/hypgf.m | HYPGF  Hypergeometric function F(a,b,c,x) | implemented | wafo.misc.hypgf |
| misc/levels.m | LEVELS Calculates discrete levels given the parameter matrix. | implemented | wafo.misc.levels |
| misc/loaddata.m | LOADDATA Loads a matrix from a text file. | implemented | wafo.misc.loaddata |
| misc/maximizefigs.m | MAXIMIZEFIGS  Maximize figure(s) window size | adapted | wafo.misc.maximizefigs |
| misc/mkcontents.m | MKCONTENTS Makes Contents file in current working directory. | adapted | wafo.misc.mkcontents |
| misc/parseoptions.m | PARSEOPTIONS Create or alter a OPTIONS structure. | adapted | wafo.misc.parseoptions |
| misc/qrule.m | QRULE compute abscissas and weight factors for Gaussian quadratures | implemented | wafo.misc.qrule |
| misc/qrule2d.m | QRULE2D compute abscissas and weight factors for Gaussian quadratures | adapted | wafo.misc.qrule2d |
| misc/readme.m | README Readme file for module MISC in WAFO Toolbox | documentation-or-script |  |
| misc/savgol.m | SAVGOL  Savitzky-Golay filter coefficients. | adapted | wafo.misc.savgol |
| misc/simpson.m | SIMPSON Numerical integration with the Simpson method | implemented | wafo.misc.simpson |
| misc/sinc.m | SINC Sin(pi*x)/(pi*x) function. | implemented | wafo.misc.sinc |
| misc/smooth.m | SMOOTH Calculates a smoothing spline. | adapted | wafo.misc.smooth |
| misc/wafostamp.m | WAFOSTAMP Prints a caption "made by WAFO" in current figure. | adapted | wafo.misc.wafostamp |
| misc/wafoversion.m | WAFOVERSION Wave Analysis for Fatigue and Oceanography version. | adapted | wafo.misc.wafoversion |
| misc/wdensity.m | WDENSITY  Returns the water density | implemented | wafo.misc.wdensity |
| multidim/Contents.m | Module MULTIDIM in WAFO Toolbox. | documentation-or-script |  |
| multidim/dat2dspec.m | DAT2DSPEC Estimates the directional wave spectrum from timeseries | adapted | wafo.multidim.dat2dspec |
| multidim/dspec2spreading.m | DSPEC2SPREADING Returns the directional spreading function D of S | implemented | wafo.multidim.dspec2spreading |
| multidim/fourier.m | FOURIER Returns Fourier coefficients. | implemented | wafo.multidim.fourier |
| multidim/private/bfsspec.m | BFSSPEC Estimate frequency spectrum for the surface elevation from the bfs timeseries | adapted | wafo.multidim.bfsspec |
| multidim/private/Contents.m | private Module to MULTIDIM in WAFO Toolbox | documentation-or-script |  |
| multidim/private/emem.m | EMEM  Extended Maximum Entropy Method | adapted | wafo.multidim.emem |
| multidim/private/getcrossspectra.m | GETCROSSSPECTRA Compute the cross spectra by integration | adapted | wafo.multidim.getcrossspectra |
| multidim/private/hwestimate.m | HWESTIMATE  Estimate absolute value of transfer function H(w) from sensor spectra | adapted | wafo.multidim.hwestimate |
| multidim/private/imlm.m | IMLM  Iterated maximum likelihood method for estimating the directional distribution | adapted | wafo.multidim.imlm |
| multidim/private/mem.m | MEM  maximum entropy method for estimating the directional distribution | adapted | wafo.multidim.mem |
| multidim/private/memfun.m | MEMFUN private function for MEM | adapted | wafo.multidim.memfun |
| multidim/private/mlm.m | MLM  maximum likelihood method for estimating the directional distribution | adapted | wafo.multidim.mlm |
| multidim/private/normspfn.m | NORMSPFN normalizes the spreading function | adapted | wafo.multidim.normspfn |
| multidim/ratio.m | RATIO compute ratio of hyperbolic functions | implemented | wafo.multidim.ratio |
| multidim/sensortype.m | SENSORTYPE Return sensortype name | implemented | wafo.multidim.sensortype |
| multidim/sensortypeid.m | SENSORTYPEID Return ID for sensortype name | implemented | wafo.multidim.sensortypeid |
| multidim/specoptset.m | SPECOPTSET Create or alter SPECTRUM OPTIONS structure. | adapted | wafo.multidim.specoptset |
| multidim/testbuoy.m | TESTBUOY creates a test case for a buoy measurement | adapted | wafo.multidim.testbuoy |
| multidim/testmeasurements.m | TESTMEASUREMENTS Creates a test case for measurement time series | adapted | wafo.multidim.testmeasurements |
| multidim/testsurf.m | TESTSURF creates a test case for a surface elevation measurement | adapted | wafo.multidim.testsurf |
| multidim/tran.m | TRAN Computes transfer functions based on linear wave theory | implemented | wafo.multidim.tran |
| onedim/Contents.m | Module ONEDIM in WAFO Toolbox. | documentation-or-script |  |
| onedim/covplot.m | COVPLOT Plots the auto covariance function (ACF) 1D or 2D. | adapted | wafo.onedim.covplot |
| onedim/dat2cor.m | DAT2COR Estimate auto correlation function from data. | adapted | wafo.onedim.dat2cor |
| onedim/dat2cov.m | DAT2COV Estimate auto covariance function from data. | adapted | wafo.onedim.dat2cov |
| onedim/dat2crossind.m | DAT2CROSSIND Finds indices to level v down and/or upcrossings from data | adapted | wafo.onedim.dat2crossind |
| onedim/dat2lc.m | DAT2LC Extracts level-crossing spectrum from data, | implemented | wafo.onedim.dat2lc |
| onedim/dat2midind.m | DAT2MIDIND Finds indices to midpoints between a min and Max and Max and min. | adapted | wafo.onedim.dat2midind |
| onedim/dat2spec.m | DAT2SPEC Estimate one-sided spectral density from data. | adapted | wafo.onedim.dat2spec |
| onedim/dat2spec2.m | DAT2SPEC2 Estimate one-sided spectral density, version 2. | adapted | wafo.onedim.dat2spec2 |
| onedim/dat2steep.m | DAT2STEEP Extracts waveheights and steepnesses from data. | adapted | wafo.onedim.dat2steep |
| onedim/dat2tc.m | DAT2TC Extracts troughs and crests from data. | adapted | wafo.onedim.dat2tc |
| onedim/dat2tp.m | DAT2TP Extracts turning points from data, | implemented | wafo.onedim.dat2tp |
| onedim/dat2wa.m | DAT2WA Extracts sequence of wavelengths from data. | adapted | wafo.onedim.dat2wa |
| onedim/detrendma.m | DETRENDMA Removes a trend from data using a moving average | adapted | wafo.onedim.detrendma |
| onedim/ecross.m | ECROSS Extracts exact level v crossings | implemented | wafo.onedim.ecross |
| onedim/findcross.m | FINDCROSS Finds indices to level v up and downcrossings of a vector | implemented | wafo.onedim.findcross |
| onedim/findextrema.m | FINDEXTREMA Finds indices to minima and maxima of data | implemented | wafo.onedim.findextrema |
| onedim/findoutliers.m | FINDOUTLIERS Finds the indices to spurious points in a timeseries | adapted | wafo.onedim.findoutliers |
| onedim/findrfc.m | FINDRFC Finds indices to rainflow cycles of a sequence of TP. | implemented | wafo.onedim.findrfc |
| onedim/hs2sign.m | HS2SIGN Calculates a ratio-significant value of a histogram. | adapted | wafo.onedim.hs2sign |
| onedim/lcplot.m | LCPLOT Plots level-crossing spectrum (lc) | adapted | wafo.onedim.lcplot |
| onedim/mm2lc.m | MM2LC Extracts level-crossing spectrum from min2Max cycles. | adapted | wafo.onedim.mm2lc |
| onedim/private/bingham.m | BINGHAM  returns the N-point Bingham window in a column vector. | adapted | wafo.onedim.bingham |
| onedim/private/hanning.m | HANNING returns the N-point Hanning window in a column vector. | adapted | wafo.onedim.hanning |
| onedim/private/parzen.m | PARZEN returns the N-point Parzen window in a column vector. | adapted | wafo.onedim.parzen |
| onedim/Readme.m | README Readme file for module ONEDIM in WAFO Toolbox. | documentation-or-script |  |
| onedim/reconstruct.m | RECONSTRUCT reconstruct the spurious/missing points of timeseries | adapted | wafo.onedim.reconstruct |
| onedim/spwaveplot.m | SPWAVEPLOT Plots specified waves from a timeseries | adapted | wafo.onedim.spwaveplot |
| onedim/vc2sign.m | VC2SIGN Calculates ratio-significant value of the input vector. | adapted | wafo.onedim.vc2sign |
| onedim/waveplot.m | WAVEPLOT Plots the surface elevation of timeseries. | adapted | wafo.onedim.waveplot |
| papers/Contents.m | Module PAPERS in WAFO Toolbox. | documentation-or-script |  |
| papers/papermenu.m | PAPERMENU displays a user interface to the paper scripts of WAFO | adapted | wafo.papers.papermenu |
| papers/rec/Contents.m | Paper REC in WAFO Toolbox. | documentation-or-script |  |
| papers/rec/recdemo.m | RECDEMO  Show Rec figures with point and click interface | adapted | wafo.papers.recdemo |
| papers/rec/recfig.m | RECFIG  callback implementing functions of RecDemo | adapted | wafo.papers.recfig |
| papers/rec/recfig1.m | RECFIG1  Location of Gullfaks C and Statfjord A platforms in The North Sea | adapted | wafo.papers.recfig1 |
| papers/rec/recfig10.m | RECFIG10 Joint distribution of  V and H: | adapted | wafo.papers.recfig10 |
| papers/rec/recfig11.m | RECFIG11 Probability of exceeding H: Model (dash); data (dots) | adapted | wafo.papers.recfig11 |
| papers/rec/recfig12.m | RECFIG12 Probability of exceeding V: Model (dash); data (dots) | adapted | wafo.papers.recfig12 |
| papers/rec/recfig13.m | RECFIG13 The conditional probability of exceeding V given H: Model (dash); data (dots) | adapted | wafo.papers.recfig13 |
| papers/rec/recfig2.m | RECFIG2 10 minutes mean values of wind  (dash) and direction (solid) | adapted | wafo.papers.recfig2 |
| papers/rec/recfig3.m | RECFIG3 Example of the reconstructed (solid) and original data set (pluses) | adapted | wafo.papers.recfig3 |
| papers/rec/recfig4.m | RECFIG4 Estimated spectral density (solid) and 95% confidence intervals (dots) | adapted | wafo.papers.recfig4 |
| papers/rec/recfig5.m | RECFIG5 Transfer function, g, versus the crossing level u | adapted | wafo.papers.recfig5 |
| papers/rec/recfig6.m | RECFIG6 Variability of simulated e(g(u)-u) (circles) | adapted | wafo.papers.recfig6 |
| papers/rec/recfig7.m | RECFIG7 Estimated Weibull scale parameter versus h=H/Hrms (circles) | adapted | wafo.papers.recfig7 |
| papers/rec/recfig8.m | RECFIG8 Estimated Weibull shape parameter versus h=H/Hrms (circles) | adapted | wafo.papers.recfig8 |
| papers/rec/recfig9.m | RECFIG9 Conditional mean (solid, circle) and standard deviation (dash,cross) of V given H: | adapted | wafo.papers.recfig9 |
| papers/rec/recinit.m | RECINIT  setup all global variables of the RECDEMO | adapted | wafo.papers.recinit |
| papers/rec/recintro.m | RECINTRO  Info about RECDEMO: A statistical procedure for reconstruction of 1D signals. | adapted | wafo.papers.recintro |
| papers/rec/recleanup.m | RECLEANUP Clears global variables defined and used by the RECDEMO | adapted | wafo.papers.recleanup |
| papers/tutorcom/Chapter1.m | CHAPTER1 demonstrates some applications of WAFO | documentation-or-script |  |
| papers/tutorcom/Chapter2.m | CHAPTER2 Modelling random loads and stochastic waves | documentation-or-script |  |
| papers/tutorcom/Chapter3.m | CHAPTER3  Demonstrates distributions of wave characteristics | documentation-or-script |  |
| papers/tutorcom/Chapter4.m | CHAPTER4 contains the commands used in Chapter 4 of the tutorial | documentation-or-script |  |
| papers/tutorcom/Chapter5.m | CHAPTER5 contains the commands used in Chapter 5 of the tutorial | documentation-or-script |  |
| papers/tutorcom/Contents.m | Paper TUTORCOM in WAFO Toolbox. | documentation-or-script |  |
| papers/wafodemo/Contents.m | Paper WAFODEMO in WAFO Toolbox. | documentation-or-script |  |
| papers/wafodemo/wafocleanup.m | WAFOCLEANUP Clears global variables defined and used by the WAFODEMO | adapted | wafo.papers.wafocleanup |
| papers/wafodemo/wafodemo.m | WAFODEMO  Show WAFO figures with point and click interface | adapted | wafo.papers.wafodemo |
| papers/wafodemo/wafofig.m | WAFOFIG  callback implementing functions of WAFODEMO | adapted | wafo.papers.wafofig |
| papers/wafodemo/wafofig1.m | WAFOFIG1  Linear simulation from a Torsethaugen spectrum. | adapted | wafo.papers.wafofig1 |
| papers/wafodemo/wafofig10.m | WAFOFIG10 Intensity of trough-crest cycles computed from  St | adapted | wafo.papers.wafofig10 |
| papers/wafodemo/wafofig2.m | WAFOFIG2  Spectral densities: | adapted | wafo.papers.wafofig2 |
| papers/wafodemo/wafofig3.m | WAFOFIG3  Probability density distributions (pdf) of wave period, Tt, | adapted | wafo.papers.wafofig3 |
| papers/wafodemo/wafofig4.m | WAFOFIG4  Directional spectra using Thorsethaugen and cos2s spreading | adapted | wafo.papers.wafofig4 |
| papers/wafodemo/wafofig5.m | WAFOFIG5  Joint distribution (pdf) of crest front velocity and wave height: | adapted | wafo.papers.wafofig5 |
| papers/wafodemo/wafofig6.m | WAFOFIG6  Joint distribution (pdf) of crest front period, Tcf, and crest amplitude, Ac | adapted | wafo.papers.wafofig6 |
| papers/wafodemo/wafofig7.m | WAFOFIG7  Joint distribution (pdf) of crest wavelength, Lc, and crest amplitude, Ac | adapted | wafo.papers.wafofig7 |
| papers/wafodemo/wafofig8.m | WAFOFIG8  Joint distribution (pdf) of crest wavelength, Lc, and crest amplitude, Ac for extremal waves | adapted | wafo.papers.wafofig8 |
| papers/wafodemo/wafofig9.m | WAFOFIG9  Intensity of rainflow cycles computed from St | adapted | wafo.papers.wafofig9 |
| papers/wafodemo/wafoinit.m | WAFOINIT  setup all global variables of the WAFODEMO | adapted | wafo.papers.wafoinit |
| papers/wafodemo/wafointro.m | WAFOINTRO  Info about WAFODEMO: WAFO - a Matlab toolbox for analysis of random waves and loads | adapted | wafo.papers.wafointro |
| parzen.m | PARZEN returns the N-point Parzen window in a column vector. | adapted | wafo.root.parzen |
| pathdef.m | PATHDEF Search path defaults. | adapted | wafo.root.pathdef |
| Readme.m | README Readme file for WAFO Toolbox - Wave Analysis for Fatigue and Oceanography | documentation-or-script |  |
| spec/Contents.m | Module SPEC in WAFO Toolbox. | documentation-or-script |  |
| spec/cov2spec.m | COV2SPEC Computes spectral density given the auto covariance function | adapted | wafo.spec.cov2spec |
| spec/covinterp.m | COVINTERP Interpolation of covariance function and derivatives | adapted | wafo.spec.covinterp |
| spec/createcov.m | CREATECOV Covariance class constructor | adapted | wafo.spec.createcov |
| spec/createspec.m | CREATESPEC Spectrum structure constructor | implemented | wafo.spec.createspec |
| spec/demospec.m | DEMOSPEC Loads a precreated spectrum of chosen type | adapted | wafo.spec.demospec |
| spec/dspec2char.m | DSPEC2CHAR Evaluates directional spectral characteristics | implemented | wafo.spec.dspec2char |
| spec/freqtype.m | FREQTYPE returns the frequency type of a Spectral density struct. | adapted | wafo.spec.freqtype |
| spec/getjonswappeakedness.m | GETJONSWAPPEAKEDNESS Peakedness factor Gamma given Hm0 and Tp for JONSWAP | implemented | wafo.spec.getjonswappeakedness |
| spec/jonswap.m | JONSWAP Calculates (and plots) a JONSWAP spectral density | implemented | wafo.spec.jonswap |
| spec/k2w.m | K2W Translates from wave number to frequency | implemented | wafo.spec.k2w |
| spec/lagtype.m | LAGTYPE Returns the lag type of a Covariance struct. | adapted | wafo.spec.lagtype |
| spec/mccormick.m | MCCORMICK Calculates (and plots) a McCormick spectral density. | implemented | wafo.spec.mccormick |
| spec/mkdspec.m | MKDSPEC Make a directional spectrum | adapted | wafo.spec.mkdspec |
| spec/ohspec.m | OHSPEC Calculates (and plots) a Ochi-Hubble spectral density. | implemented | wafo.spec.ohspec |
| spec/ohspec2.m | OHSPEC2 Calculates and plots a bimodal Ochi-Hubble spectral density | implemented | wafo.spec.ohspec2 |
| spec/ohspec3.m | OHSPEC3 Calculates Bimodal Ochi-Hubble spectral densities | implemented | wafo.spec.ohspec3 |
| spec/oscspec.m | OSCSPEC Spectral density for a harmonic oscillator | implemented | wafo.spec.oscspec |
| spec/phi1.m | PHI1 factor for transforming spectra to finite water depth spectra | adapted | wafo.spec.phi1 |
| spec/pmspec.m | PMSPEC Calculates (and plots) a Pierson-Moskowitz spectral density. | implemented | wafo.spec.pmspec |
| spec/polplot.m |  | documentation-or-script |  |
| spec/private/dir2enc.m | DIR2ENC Transform a dir. spectrum to an encountered. (Used in spec2spec) | adapted | wafo.spec.dir2enc |
| spec/private/dir2k1d.m | DIR2K1D Transforms directional spectrum to wavenumber spectrum. | adapted | wafo.spec.dir2k1d |
| spec/private/dspec2dcov.m | DSPEC2DCOV Return covariance function given a directional spectrum | adapted | wafo.spec.dspec2dcov |
| spec/private/spa2time.m | SPA2TIME Transform of spectrum from wave no. to frequency (used in spec2spec) | adapted | wafo.spec.spa2time |
| spec/private/time2spa.m | TIME2SPA Transform of spectrum from frequency to wave no. (used in spec2spec) | adapted | wafo.spec.time2spa |
| spec/Readme.m | README Readme file for module SPEC in WAFO Toolbox | documentation-or-script |  |
| spec/rotspec.m | ROTSPEC Rotate spectrum anti-clockwise around the origin. | adapted | wafo.spec.rotspec |
| spec/scalespec.m | SCALESPEC  Scale spectral density so that the moments equals m0,m2. | adapted | wafo.spec.scalespec |
| spec/spec2bw.m | SPEC2BW Evaluates some spectral bandwidth and irregularity factors | implemented | wafo.spec.spec2bw |
| spec/spec2char.m | SPEC2CHAR  Evaluates spectral characteristics and their covariance | adapted | wafo.spec.spec2char |
| spec/spec2cov.m | SPEC2COV Computes covariance function and its derivatives | adapted | wafo.spec.spec2cov |
| spec/spec2cov2.m | SPEC2COV2 Computes covariance function and its derivatives, alternative version | adapted | wafo.spec.spec2cov2 |
| spec/spec2dt.m | SPEC2DT  Computes sampling interval from Nyquist frequency in spectrum | implemented | wafo.spec.spec2dt |
| spec/spec2mom.m | SPEC2MOM Calculates spectral moments from spectrum | implemented | wafo.spec.spec2mom |
| spec/spec2spec.m | SPEC2SPEC Transforms between different types of spectra | adapted | wafo.spec.spec2spec |
| spec/specdiag.m | SPECDIAG  Diagonalize a 2 x 2 spectral matrix. | implemented | wafo.spec.specdiag |
| spec/specinterp.m | SPECINTERP Interpolation and zero-padding of spectrum | adapted | wafo.spec.specinterp |
| spec/spreading.m | SPREADING Directional spreading functions | adapted | wafo.spec.spreading |
| spec/tmaspec.m | TMASPEC Calculates a JONSWAP spectral density for finite water depth | adapted | wafo.spec.tmaspec |
| spec/torsethaugen.m | TORSETHAUGEN Calculates a double peaked (swell + wind) spectrum | implemented | wafo.spec.torsethaugen |
| spec/ttspec.m | TTSPEC Toggle Transform between angular frequency and frequency spectrum | adapted | wafo.spec.ttspec |
| spec/w2k.m | W2K Translates from frequency to wave number | implemented | wafo.spec.w2k |
| spec/wallop.m | WALLOP  Calculates (and plots) a Wallop spectral density. | implemented | wafo.spec.wallop |
| spec/wnormspec.m | WNORMSPEC Normalize a spectral density such that m0=m2=1 | adapted | wafo.spec.wnormspec |
| spec/wspecplot.m | WSPECPLOT Plot a spectral density | adapted | wafo.spec.wspecplot |
| trgauss/bvnormcdf.m | BVNORMCDF Bivariate Normal cumulative distribution function | adapted | wafo.trgauss.bvnormcdf |
| trgauss/bvnormprb.m | BVNORMPRB Bivariate Normal probability | adapted | wafo.trgauss.bvnormprb |
| trgauss/cdf2tr.m | CDF2TR Estimate transformation, g, from observed CDF. | adapted | wafo.trgauss.cdf2tr |
| trgauss/chitwo2lc_sorm.m | CHITWO2LC_SORM  SORM-approximation of the crossing intensity for the noncentral Chi^2 process | adapted | wafo.trgauss.chitwo2lc_sorm |
| trgauss/chitwo2lc_sp.m | CHITWO2LC_SP  Saddlepoint approximation of the crossing intensity for the noncentral Chi^2 process | adapted | wafo.trgauss.chitwo2lc_sp |
| trgauss/Contents.m | Module TRGAUSS in WAFO Toolbox. | documentation-or-script |  |
| trgauss/createpdf.m | CREATEPDF PDF class constructor | adapted | wafo.trgauss.createpdf |
| trgauss/dat2gaus.m | DAT2GAUS Transforms  x  using the transformation  g. | adapted | wafo.trgauss.dat2gaus |
| trgauss/dat2tr.m | DAT2TR Estimate transformation, g, from data. | adapted | wafo.trgauss.dat2tr |
| trgauss/dirsp2chitwo.m | DIRSP2CHITWO  gives parameters in non-central CHI-TWO process for directional Stokes waves. | adapted | wafo.trgauss.dirsp2chitwo |
| trgauss/gaus2dat.m | GAUS2DAT Transforms  xx  using the inverse of transformation  g. | adapted | wafo.trgauss.gaus2dat |
| trgauss/hermitetr.m | HERMITETR Calculate transformation, g, proposed by Winterstein | adapted | wafo.trgauss.hermitetr |
| trgauss/initoptions.m | INITOPTIONS Initializes RIND options according to speed. | adapted | wafo.trgauss.initoptions |
| trgauss/iter.m | ITER  Calculates a Markov matrix given a rainflow matrix | adapted | wafo.trgauss.iter |
| trgauss/iter_mc.m | ITER_MC  Calculates a kernel of a MC given a rainflow matrix | adapted | wafo.trgauss.iter_mc |
| trgauss/lc2tr.m | LC2TR Estimate transformation, g, from observed crossing intensity. | adapted | wafo.trgauss.lc2tr |
| trgauss/lc2tr2.m | LC2TR2 Estimate transformation, g, from observed crossing intensity, version2. | adapted | wafo.trgauss.lc2tr2 |
| trgauss/lomaxcdf.m | LOMAXCDF CDF for local maxima for a zero-mean Gaussian process | implemented | wafo.trgauss.lomaxcdf |
| trgauss/mc2rfc.m | MC2RFC  Calculates a rainflow matrix given a Markov chain with kernel f_xy; | adapted | wafo.trgauss.mc2rfc |
| trgauss/mctp2rfc.m | MCTP2RFC  Rainflow matrix given a Markov matrix of a Markov chain of turning points | adapted | wafo.trgauss.mctp2rfc |
| trgauss/mctp2tc.m | MCTP2TC  Frequencies of upcrossing troughs and crests using Markov chain of turning points. | adapted | wafo.trgauss.mctp2tc |
| trgauss/mctrtest.m | MCTRTEST Test if a stochastic process is Gaussian. | adapted | wafo.trgauss.mctrtest |
| trgauss/mvnormpcprb.m | MVNORMPCPRB Multivariate Normal probabilities with product correlation | adapted | wafo.trgauss.mvnormpcprb |
| trgauss/mvnormprb.m | MVNORMPRB Multivariate Normal probability by Genz' algorithm. | adapted | wafo.trgauss.mvnormprb |
| trgauss/mvnortpcprb.m | MVNORTPCPRB Multivariate normal or student T probability with product correlation structure. | adapted | wafo.trgauss.mvnortpcprb |
| trgauss/nt2fr.m | NT2FR  Calculates the frequency matrix given the counting distribution matrix. | adapted | wafo.trgauss.nt2fr |
| trgauss/ochitr.m | OCHITR  Calculates transformation, g, proposed by Ochi et al. | adapted | wafo.trgauss.ochitr |
| trgauss/pdfplot.m | PDFPLOT Plot contents of pdf structures | adapted | wafo.trgauss.pdfplot |
| trgauss/private/Contents.m | private Module to TRGAUSS in WAFO Toolbox | documentation-or-script |  |
| trgauss/private/ewwdir.m | EWWDIR Computes values of the quadratic transfer function E, for quadratic sea | adapted | wafo.trgauss.ewwdir |
| trgauss/private/hermitefun.m | HERMITEFUN Calculates the transformation by a Hermite polynomial. | adapted | wafo.trgauss.hermitefun |
| trgauss/private/holmberg1.m | HOLMBERG1 Computes moments for higher order reliability methods. | adapted | wafo.trgauss.holmberg1 |
| trgauss/private/holmberg2.m | HOLMBERG2 Computes moments for higher order reliability methods. | adapted | wafo.trgauss.holmberg2 |
| trgauss/private/holmberg3.m | HOLMBERG3 Computes moments for higher order reliability methods. | adapted | wafo.trgauss.holmberg3 |
| trgauss/private/holmquist1.m | HOLMQUIST1 Computes moments for higher order reliability methods. | adapted | wafo.trgauss.holmquist1 |
| trgauss/private/holmquist2.m | HOLMQUIST2 Computes moments for higher order reliability methods. | adapted | wafo.trgauss.holmquist2 |
| trgauss/private/holmquist3.m | HOLMQUIST3 Computes moments for higher order reliability methods. | adapted | wafo.trgauss.holmquist3 |
| trgauss/private/kchitwo.m | KCHITWO Computes the cumulant generating function formula for noncentral Chi2-process. | adapted | wafo.trgauss.kchitwo |
| trgauss/private/mexGenzMvnPrb.m | MEXGENZMVNPRB Computes multivariate normal probability by Genz' algorithm | adapted | wafo.trgauss.mexGenzMvnPrb |
| trgauss/private/mexmvnprb.m | MEXMVNPRB Computes multivariate normal probability by Genz' algorithm | adapted | wafo.trgauss.mexmvnprb |
| trgauss/private/mexmvnprb2.m | MEXMVNPRB2 Computes multivariate normal probability by Genz' algorithm | adapted | wafo.trgauss.mexmvnprb2 |
| trgauss/private/mindist.m | MINDIST Finds minimal distance to the origin on the surface b'*x+x'*diag(g)*x=u | adapted | wafo.trgauss.mindist |
| trgauss/private/mindist3.m | MINDIST3 Finds point of minimal distance to the origin on the surface b'*x+g'*x.^2=u. | adapted | wafo.trgauss.mindist3 |
| trgauss/private/mvnprodcorrprbmex.m | MVNPRODCORRPRBMEX Computes multivariate normal probability | documentation-or-script |  |
| trgauss/private/ochifun.m | OCHIFUN Calculates the transformation g proposed by Ochi et al. | adapted | wafo.trgauss.ochifun |
| trgauss/private/rfadd.m | RFADD Addition of two rational functions | adapted | wafo.trgauss.rfadd |
| trgauss/private/rqlf_asympt.m | RQLF_ASYMPT Gives first two terms in an asymptotic expansion of the | adapted | wafo.trgauss.rqlf_asympt |
| trgauss/private/wminmax.m | WMINMAX Calculates joint density of minimum and following maximum | adapted | wafo.trgauss.wminmax |
| trgauss/private/writecov.m | WRITECOV Calculates spline coefficients for the covariance and | adapted | wafo.trgauss.writecov |
| trgauss/Readme.m | README Readme file for module TRGAUSS in WAFO Toolbox. | documentation-or-script |  |
| trgauss/rind.m | RIND Computes multivariate normal expectations | adapted | wafo.trgauss.rind |
| trgauss/rindoptset.m | RINDOPTSET Create or alter RIND OPTIONS structure. | adapted | wafo.trgauss.rindoptset |
| trgauss/spec2AcAt.m | SPEC2ACAT   Evaluates survival function R(h1,h2)=P(Ac>h1,At>h2). | adapted | wafo.trgauss.spec2AcAt |
| trgauss/spec2Acdf.m | SPEC2ACDF  Evaluates cdf of crests P(Ac<=h) or troughs P(At<=h). | adapted | wafo.trgauss.spec2Acdf |
| trgauss/spec2cmat.m | SPEC2CMAT Joint intensity matrix for cycles (max,min)-, rainflow- and (crest,trough) | adapted | wafo.trgauss.spec2cmat |
| trgauss/spec2mmtpdf.m | SPEC2MMTPDF Calculates joint density of Maximum, minimum and period. | adapted | wafo.trgauss.spec2mmtpdf |
| trgauss/spec2skew.m | SPEC2SKEW Estimates the moments of 2'nd order non-linear waves | adapted | wafo.trgauss.spec2skew |
| trgauss/spec2tccpdf.m | SPEC2TCCPDF Evaluates densities of wave period Tcc, wave lenght Lcc. | adapted | wafo.trgauss.spec2tccpdf |
| trgauss/spec2thpdf.m | SPEC2THPDF Joint density of amplitude and period/wave-length characteristics | adapted | wafo.trgauss.spec2thpdf |
| trgauss/spec2tpdf.m | SPEC2TPDF Evaluates densities for crest-,trough-period, length. | adapted | wafo.trgauss.spec2tpdf |
| trgauss/spec2tpdf2.m | SPEC2TPDF2 Evaluates densities for various wave periods or wave lengths | adapted | wafo.trgauss.spec2tpdf2 |
| trgauss/specq2lc.m | SPECQ2LC  Saddlepoint approximation of the crossing intensity for the quadratic sea. | adapted | wafo.trgauss.specq2lc |
| trgauss/th2vhpdf.m | TH2VHPDF Transform joint T-H density to V-H density | adapted | wafo.trgauss.th2vhpdf |
| trgauss/trangood.m | TRANGOOD Makes a transformation that is suitable for efficient transforms. | adapted | wafo.trgauss.trangood |
| trgauss/tranproc.m | TRANPROC Transforms process X and up to four derivatives | adapted | wafo.trgauss.tranproc |
| trgauss/trmak.m | TRMAK Put together a transformation object. | adapted | wafo.trgauss.trmak |
| trgauss/troptset.m | TROPTSET Create or alter TRANSFORM OPTIONS structure. | adapted | wafo.trgauss.troptset |
| trgauss/trplot.m | TRPLOT Plots transformation, g, eg. estimated with dat2tr. | adapted | wafo.trgauss.trplot |
| trgauss/trunmak.m | TRUNMAK Split a transformation object into its pieces. | adapted | wafo.trgauss.trunmak |
| wafoexepath.m | WAFOEXEPATH Returns the path to executables for the WAFO Toolbox | adapted | wafo.root.wafoexepath |
| wafopath.m | WAFOPATH Adds or removes WAFO subdirectories from the search path. | adapted | wafo.root.wafopath |
| waforoot.m | WAFOROOT Root directory of WAFO installation. | adapted | wafo.root.waforoot |
| wavemodels/b04cdf.m | B04CDF  Brodtkorb (2004) joint (Scf,Hd) CDF of laboratory data. | implemented | wafo.wavemodels.b04cdf |
| wavemodels/b04jcdf.m | B04JCDF  Brodtkorb (2004) joint (Scf,Hd) CDF from Japan Sea. | implemented | wafo.wavemodels.b04jcdf |
| wavemodels/b04jpdf.m | B04JPDF Brodtkorb (2004) joint (Scf,Hd) PDF from Japan Sea. | implemented | wafo.wavemodels.b04jpdf |
| wavemodels/b04jpdf2.m | B04JPDF2 Brodtkorb  (2004) joint (Scf,Hd) PDF from Japan Sea. | implemented | wafo.wavemodels.b04jpdf2 |
| wavemodels/b04pdf.m | B04PDF Brodtkorb (2004) joint (Scf,Hd) PDF of laboratory data. | implemented | wafo.wavemodels.b04pdf |
| wavemodels/b04pdf2.m | B04PDF2 Brodtkorb  (2004) joint (Scf,Hd) PDF of laboratory data. | implemented | wafo.wavemodels.b04pdf2 |
| wavemodels/bmr00cdf.m | BMR00CDF  Brodtkorb et.al. (2000) joint (Scf,Hd) CDF from North Sea. | implemented | wafo.wavemodels.bmr00cdf |
| wavemodels/bmr00pdf.m | BMR00PDF Brodtkorb et.al (2000) joint (Scf,Hd) PDF from North Sea. | implemented | wafo.wavemodels.bmr00pdf |
| wavemodels/bmr00pdf2.m | BMR00PDF2 Brodtkorb et.al (2000) joint (Scf,Hd) PDF from North Sea. | implemented | wafo.wavemodels.bmr00pdf2 |
| wavemodels/braylcdf.m | BRAYLCDF Beta Rayleigh CDF of wave heights | implemented | wafo.wavemodels.braylcdf |
| wavemodels/braylfit.m | BRAYLFIT Parameter estimates for Beta-Rayleigh data. | adapted | wafo.wavemodels.braylfit |
| wavemodels/braylpdf.m | BRAYLPDF Beta Rayleigh PDF of wave heigth | implemented | wafo.wavemodels.braylpdf |
| wavemodels/cav76pdf.m | CAV76PDF Cavanie et al. (1976) approximation of the density  (Tc,Ac) | implemented | wafo.wavemodels.cav76pdf |
| wavemodels/Contents.m | Module WAVEMODELS in WAFO Toolbox. | documentation-or-script |  |
| wavemodels/jhnlwparfun.m | JHNLWPARFUN Wave height, Hd, distribution parameters for Stokes waves with Jonswap spectrum. | implemented | wafo.wavemodels.jhnlwparfun |
| wavemodels/jhscdf.m | JHSCDF Joint (Scf,Hd) CDF for linear waves with a JONSWAP spectrum. | adapted | wafo.wavemodels.jhscdf |
| wavemodels/jhsnlcdf.m | JHSNLCDF Joint (Scf,Hd) CDF for non-linear waves with JONSWAP spectrum. | adapted | wafo.wavemodels.jhsnlcdf |
| wavemodels/jhsnlpdf.m | JHSNLPDF Joint (Scf,Hd) PDF for nonlinear waves with a JONSWAP spectra. | adapted | wafo.wavemodels.jhsnlpdf |
| wavemodels/jhsnlpdf2.m | JHSNLPDF2 Joint (Scf,Hd) PDF for non-linear waves with a JONSWAP spectra. | adapted | wafo.wavemodels.jhsnlpdf2 |
| wavemodels/jhspdf.m | JHSPDF Joint (Scf,Hd) PDF for linear waves with JONSWAP spectra. | adapted | wafo.wavemodels.jhspdf |
| wavemodels/jhspdf2.m | JHSPDF2 Joint (Scf,Hd) PDF for linear waves with a JONSWAP spectrum. | adapted | wafo.wavemodels.jhspdf2 |
| wavemodels/jhvcdf.m | JHVCDF Joint (Vcf,Hd) CDF for linear waves with JONSWAP spectrum. | adapted | wafo.wavemodels.jhvcdf |
| wavemodels/jhvnlcdf.m | JHVNLCDF Joint (Vcf,Hd) CDF for non-linear waves with JONSWAP spectrum. | adapted | wafo.wavemodels.jhvnlcdf |
| wavemodels/jhvnlpdf.m | JHVNLPDF Joint (Vcf,Hd) PDF for linear waves with a JONSWAP spectrum. | adapted | wafo.wavemodels.jhvnlpdf |
| wavemodels/jhvnlpdf2.m | JHVNLPDF2 Joint (Vcf,Hd) PDF for non-linear waves with a JONSWAP spectrum. | adapted | wafo.wavemodels.jhvnlpdf2 |
| wavemodels/jhvpdf.m | JHVPDF Joint (Vcf,Hd) PDF for linear waves with a JONSWAP spectrum. | adapted | wafo.wavemodels.jhvpdf |
| wavemodels/jhvpdf2.m | JHVPDF2 Joint (Vcf,Hd) PDF for linear waves with a JONSWAP spectrum. | adapted | wafo.wavemodels.jhvpdf2 |
| wavemodels/jhwparfun.m | JHWPARFUN Wave height, Hd, distribution parameters for Jonswap spectrum. | implemented | wafo.wavemodels.jhwparfun |
| wavemodels/lh83pdf.m | LH83PDF Longuet-Higgins (1983) approximation of the density (Tc,Ac) | implemented | wafo.wavemodels.lh83pdf |
| wavemodels/ltwcpdf.m | LTWCPDF Long Term Wave Climate PDF of significant wave height and wave period | adapted | wafo.wavemodels.ltwcpdf |
| wavemodels/mk87cdf.m | MK87CDF  Myrhaug and Kjeldsen (1987) joint (Scf,Hd) CDF. | implemented | wafo.wavemodels.mk87cdf |
| wavemodels/mk87pdf.m | MK87PDF Myrhaug and Kjeldsen (1987) joint (Scf,Hd) PDF. | implemented | wafo.wavemodels.mk87pdf |
| wavemodels/mk87pdf2.m | MK87PDF2 Myrhaug and Kjeldsen (1987) joint (Scf,Hd) PDF. | implemented | wafo.wavemodels.mk87pdf2 |
| wavemodels/mk87rnd.m | MK87RND Random points from MK87 distribution of steepness and wave height. | adapted | wafo.wavemodels.mk87rnd |
| wavemodels/ochi98cdf.m | OCHI98CDF Ochi's (1998) CDF of peaks and troughs of non-gaussian processes | implemented | wafo.wavemodels.ochi98cdf |
| wavemodels/ochi98fit.m | OCHI98FIT Parameter estimates and confidence intervals for Ochi data. | adapted | wafo.wavemodels.ochi98fit |
| wavemodels/ochi98pdf.m | OCHI98PDF Ochi's (1998) PDF of peaks and troughs of non-gaussian processes | implemented | wafo.wavemodels.ochi98pdf |
| wavemodels/ohhcdf.m | OHHCDF Marginal wave height, Hd, CDF for Ochi-Hubble spectra. | adapted | wafo.wavemodels.ohhcdf |
| wavemodels/ohhgparfun.m | OHHGPARFUN Wave height, Hd, distribution parameters for Ochi-Hubble spectra. | adapted | wafo.wavemodels.ohhgparfun |
| wavemodels/ohhpdf.m | OHHPDF Marginal wave height, Hd, PDF for Bimodal Ochi-Hubble spectra. | adapted | wafo.wavemodels.ohhpdf |
| wavemodels/ohhscdf.m | OHHSCDF Joint (Scf,Hd) CDF for linear waves with Ochi-Hubble spectra. | adapted | wafo.wavemodels.ohhscdf |
| wavemodels/ohhspdf.m | OHHSPDF Joint (Scf,Hd) PDF for linear waves with Ochi-Hubble spectra. | adapted | wafo.wavemodels.ohhspdf |
| wavemodels/ohhspdf2.m | OHHSPDF2 Joint (Scf,Hd) PDF for linear waves with Ochi-Hubble spectra. | adapted | wafo.wavemodels.ohhspdf2 |
| wavemodels/ohhsscdf.m | OHHSSCDF Joint (Scf,Hd) CDF for linear waves in space with Ochi-Hubble spectra. | adapted | wafo.wavemodels.ohhsscdf |
| wavemodels/ohhsspdf.m | OHHSSPDF Joint (Scf,Hd) PDF linear waves in space with Ochi-Hubble spectra. | adapted | wafo.wavemodels.ohhsspdf |
| wavemodels/ohhsspdf2.m | OHHSSPDF2 Joint (Scf,Hd) PDF linear waves in space with Ochi-Hubble spectra. | adapted | wafo.wavemodels.ohhsspdf2 |
| wavemodels/ohhvpdf.m | OHHVPDF Joint (Vcf,Hd) PDF for lineare waves with Ochi-Hubble spectra. | adapted | wafo.wavemodels.ohhvpdf |
| wavemodels/Readme.m | README Readme file for module WAVEMODELS in WAFO Toolbox. | documentation-or-script |  |
| wavemodels/tay81cdf.m | TAY81CDF Tayfun (1981) CDF of  breaking limited wave heights | adapted | wafo.wavemodels.tay81cdf |
| wavemodels/tay81fun.m | TAY81FUN internal function to tay81pdf and tay81cdf. | implemented | wafo.wavemodels.tay81fun |
| wavemodels/tay81pdf.m | TAY81PDF  Tayfun (1981) PDF of  breaking limited wave heights | adapted | wafo.wavemodels.tay81pdf |
| wavemodels/tay90cdf.m | TAY90CDF Tayfun (1990) CDF of large wave heights | implemented | wafo.wavemodels.tay90cdf |
| wavemodels/tay90fun.m | TAY90FUN is an internal integrand function to tay90pdf | implemented | wafo.wavemodels.tay90fun |
| wavemodels/tay90pdf.m | TAY90PDF Tayfun (1990) PDF of large wave heights | implemented | wafo.wavemodels.tay90pdf |
| wavemodels/thcdf.m | THCDF Marginal wave height, Hd, CDF for Torsethaugen spectra. | adapted | wafo.wavemodels.thcdf |
| wavemodels/thgparfun.m | THGPARFUN Wave height, Hd, distribution parameters for Torsethaugen spectra. | adapted | wafo.wavemodels.thgparfun |
| wavemodels/thpdf.m | THPDF Marginal wave height, Hd, PDF for Torsethaugen spectra. | adapted | wafo.wavemodels.thpdf |
| wavemodels/thscdf.m | THSCDF Joint (Scf,Hd) CDF for linear waves with Torsethaugen spectra. | adapted | wafo.wavemodels.thscdf |
| wavemodels/thsnlcdf.m | THSNLCDF Joint (Scf,Hd) CDF for nonlinear waves with Torsethaugen spectra. | adapted | wafo.wavemodels.thsnlcdf |
| wavemodels/thsnlpdf.m | THSNLPDF Joint (Scf,Hd) PDF for nonlinear waves with Torsethaugen spectra. | adapted | wafo.wavemodels.thsnlpdf |
| wavemodels/thsnlpdf2.m | THSNLPDF2 Joint (Scf,Hd) PDF for nonlinear waves with Torsethaugen spectra. | adapted | wafo.wavemodels.thsnlpdf2 |
| wavemodels/thspdf.m | THSPDF Joint (Scf,Hd) PDF for linear waves with Torsethaugen spectra. | adapted | wafo.wavemodels.thspdf |
| wavemodels/thspdf2.m | THSPDF2 Joint (Scf,Hd) PDF for linear waves with Torsethaugen spectra. | adapted | wafo.wavemodels.thspdf2 |
| wavemodels/thsscdf.m | THSSCDF Joint (Scf,Hd) CDF for linear waves in space with Torsethaugen spectra. | adapted | wafo.wavemodels.thsscdf |
| wavemodels/thsspdf.m | THSSPDF Joint (Scf,Hd) PDF for linear waves in space with Torsethaugen spectra. | adapted | wafo.wavemodels.thsspdf |
| wavemodels/thsspdf2.m | THSSPDF2 Joint (Scf,Hd) PDF for linear waves in space with Torsethaugen spectra. | adapted | wafo.wavemodels.thsspdf2 |
| wavemodels/thvcdf.m | THVCDF Joint (Vcf,Hd) CDF for linear waves with Torsethaugen spectra. | adapted | wafo.wavemodels.thvcdf |
| wavemodels/thvpdf.m | THVPDF Joint (Vcf,Hd) PDF for linear waves with Torsethaugen spectra. | adapted | wafo.wavemodels.thvpdf |
| wavemodels/thvpdf2.m | THVPDF2 Joint (Vcf,Hd) PDF for linear waves with Torsethaugen spectra. | adapted | wafo.wavemodels.thvpdf2 |
| wavemodels/thwparfun.m | THWPARFUN Wave height, Hd, distribution parameters for Torsethaugen spectra. | adapted | wafo.wavemodels.thwparfun |
| wavemodels/trraylpdf.m | TRRAYLPDF Calculates transformed Rayleigh approximation for amplitudes | implemented | wafo.wavemodels.trraylpdf |
| wdemos/Contents.m | Module WDEMOS in WAFO Toolbox. | documentation-or-script |  |
| wdemos/democc.m | DEMOCC A program for visualization of cycle counts in random | documentation-or-script |  |
| wdemos/democc_markmax.m | DEMOCC_MARKMAX plots load and marks a maximum | adapted | wafo.wdemos.democc_markmax |
| wdemos/democc_mmdef.m | DEMOCC_MMDEF illustrates the definition of min-max cycles. | adapted | wafo.wdemos.democc_mmdef |
| wdemos/democc_plotmat.m | DEMOCC_PLOTMAT plots RFC and min-max counts. | adapted | wafo.wdemos.democc_plotmat |
| wdemos/democc_rfcdef.m | DEMOCC_RFCDEF illustrates the definition of rainflow cycles. | adapted | wafo.wdemos.democc_rfcdef |
| wdemos/democc_tpdef.m | DEMOCC_TPDEF illustrates the definition of turning points. | adapted | wafo.wdemos.democc_tpdef |
| wdemos/itmkurs/alevel.m | ALEVEL Slice virkler.mat at a given crack length. | adapted | wafo.wdemos.alevel |
| wdemos/itmkurs/Contents.m | Contents of itmkurs | documentation-or-script |  |
| wdemos/itmkurs/itmkurs_lab1.m | ITMKURS_LAB1 Script to computer exercises 1 | documentation-or-script |  |
| wdemos/itmkurs/itmkurs_lab2.m | ITMKURS_LAB2 Script to computer exercises 2 | documentation-or-script |  |
| wdemos/itmkurs/itmkurs_lab3.m | ITMKURS_LAB3 Script to computer exercises 3 | documentation-or-script |  |
| wdemos/itmkurs/itmkurs_lab4.m | ITMKURS_LAB4 Script to computer exercises 4 | documentation-or-script |  |
| wdemos/itmkurs/load2ud.m | LOAD2UD Calculates the number of up/downcrossings of a load. | adapted | wafo.wdemos.load2ud |
| wdemos/itmkurs/nlevel.m | NLEVEL Slice virkler.mat at a given number of cycles. | adapted | wafo.wdemos.nlevel |
| wdemos/itmkurs/splitload.m | SPLITLOAD Split a switching load (e.g. switchingload.mat) | adapted | wafo.wdemos.splitload |
| wdemos/itmkurs.m | ITMKURS Initiate paths for Demo Load and Fatigue Analysis | documentation-or-script |  |
| wdemos/rfcdemo1/armacov.m | ARMACOV   Computes the covariance function for an AR- or ARMA-model. | adapted | wafo.wdemos.armacov |
| wdemos/rfcdemo1/armaspec.m | ARMASPEC   Computes the spectral density for an AR- or ARMA-model. | adapted | wafo.wdemos.armaspec |
| wdemos/rfcdemo1/discar.m | DISCAR   Discretizes an AR(1) process. | adapted | wafo.wdemos.discar |
| wdemos/rfcdemo1/f_ar.m | F_AR  Auxiliary function used by RFCDEMO1 | adapted | wafo.wdemos.f_ar |
| wdemos/rfcdemo1.m | RFCDEMO1  Demo for switching AR(1)-processes. | adapted | wafo.wdemos.rfcdemo1 |
| wdemos/rfcdemo2.m | RFCDEMO2 Rainflow matrix for Switching Markov Chains of Turning Points. | adapted | wafo.wdemos.rfcdemo2 |
| wsim/Contents.m | Module WSIM in WAFO Toolbox. | documentation-or-script |  |
| wsim/cov2csdat.m | COV2CSDAT generates conditionally simulated values | adapted | wafo.wsim.cov2csdat |
| wsim/cov2sdat.m | COV2SDAT Simulates a Gaussian process and its derivative | adapted | wafo.wsim.cov2sdat |
| wsim/disufq.m | DISUFQ  Return difference- and sum-frequency effects. | implemented | wafo.wsim.disufq |
| wsim/duffsim.m | DUFFSIM Generates a sample path of a harmonic oscillator | adapted | wafo.wsim.duffsim |
| wsim/lc2sdat.m | LC2SDAT Simulates process with given irregularity factor and crossing spectrum | adapted | wafo.wsim.lc2sdat |
| wsim/mcsim.m | MCSIM   Simulates a Markov chain. | implemented | wafo.wsim.mcsim |
| wsim/mctpsim.m | MCTPSIM  Simulates a Markov chain of turning points | adapted | wafo.wsim.mctpsim |
| wsim/private/comb2pro.m | COMB2PRO Probability vector of placing a cycle on a specific place given all combinations | adapted | wafo.wsim.comb2pro |
| wsim/private/fr2comb.m | FR2COMB Combination matrix for placing out cycles given the frequency matrix | adapted | wafo.wsim.fr2comb |
| wsim/private/fr2nt.m | FR2NT  Calculates the counting distribution given the frequency matrix. | adapted | wafo.wsim.fr2nt |
| wsim/private/fr2res.m | FR2RES Generates a stationary residual from the frequency matrix. | adapted | wafo.wsim.fr2res |
| wsim/private/getulcc.m | GETULCC Finds the upper left element > 0 in a matrix. | adapted | wafo.wsim.getulcc |
| wsim/private/rfc2load_fat.m | RFC2LOAD_FAT  Recontructs a load process given the frequency matrix (and residual). | adapted | wafo.wsim.rfc2load_fat |
| wsim/readme.m | README Readme file for module WSIM in WAFO Toolbox | documentation-or-script |  |
| wsim/rfm2dtp.m | RFM2DTP  Reconstructs a sequence of turning points from a rainflow matrix. | adapted | wafo.wsim.rfm2dtp |
| wsim/sarmasim.m | SARMASIM   Simulates a switching ARMA-process. | adapted | wafo.wsim.sarmasim |
| wsim/seamovie.m | SEAMOVIE Makes a movie of a 2D (x,t) or 3D (x,y,t) simulated sea | adapted | wafo.wsim.seamovie |
| wsim/seasim.m | SEASIM Spectral simulation of a Gaussian sea, 2D (x,t) or 3D (x,y,t) | adapted | wafo.wsim.seasim |
| wsim/smcsim.m | SMCSIM  Simulates a Switching Markov chain with state space. | adapted | wafo.wsim.smcsim |
| wsim/smctpsim.m | SMCTPSIM  Simulates a switching Markov chain of turning points, | adapted | wafo.wsim.smctpsim |
| wsim/spec2linspec.m | SPEC2LINSPEC  Separates the linear component of the Spectrum | adapted | wafo.wsim.spec2linspec |
| wsim/spec2nlsdat.m | SPEC2NLSDAT Simulates a Randomized 2nd order non-linear wave X(t) | adapted | wafo.wsim.spec2nlsdat |
| wsim/spec2sdat.m | SPEC2SDAT Simulates a Gaussian process and its derivative from spectrum | adapted | wafo.wsim.spec2sdat |
| wstats/alias.m | ALIAS Alias structure of a fractional design. | implemented | wafo.wstats.alias |
| wstats/cdr.m | CDR Complete Defining Relation | implemented | wafo.wstats.cdr |
| wstats/cempdistr.m | CEMPDISTR Computes and plots the conditional empirical CDF | implemented | wafo.wstats.cempdistr |
| wstats/cl2cnr.m | CL2CNR Column Label to Column Number | implemented | wafo.wstats.cl2cnr |
| wstats/cnr2cl.m | CNR2CL Column Number to Column Label. | implemented | wafo.wstats.cnr2cl |
| wstats/comnsize.m | COMNSIZE Check if all input arguments are either scalar or of common size. | adapted | wafo.wstats.comnsize |
| wstats/Contents.m | Module WSTATS in WAFO Toolbox. | documentation-or-script |  |
| wstats/cplot.m | CPLOT Cubic plot of responses | adapted | wafo.wstats.cplot |
| wstats/dist2dcdf.m | DIST2DCDF Joint 2D CDF computed as int F(X1<v/X2=x2).*f(x2)dx2 | adapted | wafo.wstats.dist2dcdf |
| wstats/dist2dcdfplot.m | DIST2DCDFPLOT Plot conditional empirical CDF of X1 given X2=x2 | adapted | wafo.wstats.dist2dcdfplot |
| wstats/dist2dfit.m | DIST2DFIT Parameter estimates for DIST2D data. | adapted | wafo.wstats.dist2dfit |
| wstats/dist2dfun.m | DIST2DFUN is an internal function to dist2dcdf dist2dprb. | adapted | wafo.wstats.dist2dfun |
| wstats/dist2dparamplot.m | DIST2DPARAMPLOT Plot parameters of the conditional distribution | adapted | wafo.wstats.dist2dparamplot |
| wstats/dist2dpdf.m | DIST2DPDF Joint 2D PDF computed as f(x1/X2=x2)*f(x2) | adapted | wafo.wstats.dist2dpdf |
| wstats/dist2dpdf2.m | DIST2DPDF2 Joint 2D PDF computed as f(x1/X2=x2)*f(x2) | adapted | wafo.wstats.dist2dpdf2 |
| wstats/dist2dprb.m | DIST2DPRB returns the probability for rectangular regions. | adapted | wafo.wstats.dist2dprb |
| wstats/dist2drnd.m | DIST2DRND  Random points from a bivariate DIST2D distribution | adapted | wafo.wstats.dist2drnd |
| wstats/dist2dsmfun.m | DIST2DSMFUN Smooths the conditional DIST2D distribution parameters. | adapted | wafo.wstats.dist2dsmfun |
| wstats/dist2dsmfun2.m | DIST2DSMFUN2 Smooths the conditional DIST2D distribution parameters. | adapted | wafo.wstats.dist2dsmfun2 |
| wstats/dist2dstat.m | DIST2DSTAT  Mean and variance for the DIST2D distribution | adapted | wafo.wstats.dist2dstat |
| wstats/dist2dstatplot.m | DIST2DSTATPLOT Computes and plots the conditional mean and standard deviation | adapted | wafo.wstats.dist2dstatplot |
| wstats/distplot.m | DISTPLOT Displays a 1D distribution probability plot. | adapted | wafo.wstats.distplot |
| wstats/empdistr.m | EMPDISTR Computes and plots the empirical CDF | implemented | wafo.wstats.empdistr |
| wstats/ffd.m | FFD Two-level Fractional Factorial Design | implemented | wafo.wstats.ffd |
| wstats/fitmodel.m | FITMODEL  Fits response by polynomial | adapted | wafo.wstats.fitmodel |
| wstats/getmodel.m | GETMODEL Return the model parameters | adapted | wafo.wstats.getmodel |
| wstats/identify.m | IDENTIFY Identify points on a plot by clicking with the mouse. | adapted | wafo.wstats.identify |
| wstats/iqr.m | IQR Computes the Inter Quartile Range | implemented | wafo.wstats.iqr |
| wstats/kdeplot.m | KDEPLOT Computes and plots a kernel density estimate of PDF | adapted | wafo.wstats.kdeplot |
| wstats/loglike.m | LOGLIKE Log-likelihood function. | implemented | wafo.wstats.loglike |
| wstats/mdist2dcdf.m | MDIST2DCDF Joint 2D CDF due to Plackett | implemented | wafo.wstats.mdist2dcdf |
| wstats/mdist2dcdfplot.m | MDIST2DCDFPLOT Plot conditional empirical CDF of X1 given X2=x2 | adapted | wafo.wstats.mdist2dcdfplot |
| wstats/mdist2dcinv.m | MDIST2DCINV Inverse of the conditional cdf of X2 given X1. | implemented | wafo.wstats.mdist2dcinv |
| wstats/mdist2dfit.m | MDIST2DFIT Parameter estimates for MDIST2D data. | adapted | wafo.wstats.mdist2dfit |
| wstats/mdist2dlike.m | MDIST2DLIKE MDIST log-likelihood function. | implemented | wafo.wstats.mdist2dlike |
| wstats/mdist2dpdf.m | MDIST2DPDF Joint 2D PDF due to Plackett given as  f{x1}*f{x2}*G(x1,x2;Psi). | implemented | wafo.wstats.mdist2dpdf |
| wstats/mdist2dpdf2.m | MDIST2DPDF2 Joint 2D PDF due to Plackett given as  f{x1}*f{x2}*G(x1,x2;Psi). | implemented | wafo.wstats.mdist2dpdf2 |
| wstats/mdist2drnd.m | MDIST2DRND  Random points from a bivariate MDIST2D distribution | adapted | wafo.wstats.mdist2drnd |
| wstats/mdist2dstat.m | MDIST2DSTAT  Mean and variance for the MDIST2D distribution. | adapted | wafo.wstats.mdist2dstat |
| wstats/mdist2dstatplot.m | MDIST2DSTATPLOT Computes and plots the conditional mean and standard deviation. | adapted | wafo.wstats.mdist2dstatplot |
| wstats/mvnormpdf.m | MVNORMPDF Multivariate Normal probability density function. | implemented | wafo.wstats.mvnormpdf |
| wstats/nplot.m | NPLOT Normal probability plot of effects | adapted | wafo.wstats.nplot |
| wstats/pairs.m | PAIRS  Pairwise scatter plots. | adapted | wafo.wstats.pairs |
| wstats/range.m | RANGE  Calculates the difference between the maximum and minimum values. | implemented | wafo.wstats.range |
| wstats/Readme.m | README Readme file for module WSTATS in WAFO Toolbox. | documentation-or-script |  |
| wstats/ryates.m | RYATES Reverse Yates' algorithm to give estimated responses | implemented | wafo.wstats.ryates |
| wstats/sudg.m | SUDG Some Useful Design Generators | implemented | wafo.wstats.sudg |
| wstats/var.m | VAR  Variance | implemented | wafo.wstats.var |
| wstats/walpharnd.m | WALPHARND Random matrices from a symmetric alpha-stable distribution | adapted | wafo.wstats.walpharnd |
| wstats/wbetacdf.m | WBETACDF   Beta cumulative distribution function | implemented | wafo.wstats.wbetacdf |
| wstats/wbetafit.m | WBETAFIT Parameter estimates for Beta data. | adapted | wafo.wstats.wbetafit |
| wstats/wbetainv.m | WBETAINV  Inverse of the Beta distribution function | implemented | wafo.wstats.wbetainv |
| wstats/wbetapdf.m | WBETAPDF   Beta probability density function | implemented | wafo.wstats.wbetapdf |
| wstats/wbetarnd.m | WBETARND  Random matrices from a Beta distribution | adapted | wafo.wstats.wbetarnd |
| wstats/wbetastat.m | WBETASTAT Mean and variance for the Beta distribution. | implemented | wafo.wstats.wbetastat |
| wstats/wchi2cdf.m | WCHI2CDF Chi squared cumulative distribution function | implemented | wafo.wstats.wchi2cdf |
| wstats/wchi2fit.m | WCHI2FIT Parameter estimates for Chi squared data. | adapted | wafo.wstats.wchi2fit |
| wstats/wchi2inv.m | WCHI2INV Inverse of the Chi squared distribution function | implemented | wafo.wstats.wchi2inv |
| wstats/wchi2pdf.m | WCHI2PDF Chi squared probability density function | implemented | wafo.wstats.wchi2pdf |
| wstats/wchi2rnd.m | WCHI2RND Random matrices from a Chi squared distribution. | adapted | wafo.wstats.wchi2rnd |
| wstats/wchi2stat.m | WCHI2STAT Mean and variance for the Chi squared distribution. | implemented | wafo.wstats.wchi2stat |
| wstats/weib2dcdf.m | WEIB2DCDF Joint 2D Weibull cumulative distribution function | implemented | wafo.wstats.weib2dcdf |
| wstats/weib2dcdfplot.m | WEIB2DCDFPLOT Plot conditional empirical CDF of X1 given X2=x2 | adapted | wafo.wstats.weib2dcdfplot |
| wstats/weib2dcinv.m | WEIB2DCINV Inverse of the conditional 2D weibull cdf of X2 given X1. | implemented | wafo.wstats.weib2dcinv |
| wstats/weib2dfit.m | WEIB2DFIT Parameter estimates for 2D Weibull data. | adapted | wafo.wstats.weib2dfit |
| wstats/weib2dlike.m | WEIB2DLIKE 2D Weibull log-likelihood function. | implemented | wafo.wstats.weib2dlike |
| wstats/weib2dpdf.m | WEIB2DPDF 2D Weibull probability density function (pdf). | implemented | wafo.wstats.weib2dpdf |
| wstats/weib2dpdf2.m | WEIB2DPDF2 Joint 2D Weibull probability density function | implemented | wafo.wstats.weib2dpdf2 |
| wstats/weib2dprb.m | WEIB2DPRB returns the probability for rectangular regions. | implemented | wafo.wstats.weib2dprb |
| wstats/weib2drnd.m | WEIB2DRND Random numbers from the 2D Weibull distribution. | adapted | wafo.wstats.weib2drnd |
| wstats/weib2dstat.m | WEIB2DSTAT  Mean and variance for the 2D Weibull distribution | implemented | wafo.wstats.weib2dstat |
| wstats/weib2dstatplot.m | WEIB2DSTATPLOT Computes and plots the conditional mean and standard deviation | adapted | wafo.wstats.weib2dstatplot |
| wstats/wexpcdf.m | WEXPCDF Exponential cumulative distribution function | implemented | wafo.wstats.wexpcdf |
| wstats/wexpfit.m | WEXPFIT Parameter estimates for Exponential data. | adapted | wafo.wstats.wexpfit |
| wstats/wexpinv.m | WEXPINV Inverse of the Exponential distribution function | implemented | wafo.wstats.wexpinv |
| wstats/wexppdf.m | WEXPPDF Exponential probability density function | implemented | wafo.wstats.wexppdf |
| wstats/wexpplot.m | WEXPPLOT Plots data on a Exponential distribution paper | adapted | wafo.wstats.wexpplot |
| wstats/wexprnd.m | WEXPRND Random matrices from an Exponential distribution | adapted | wafo.wstats.wexprnd |
| wstats/wexpstat.m | WEXPSTAT Mean and variance for the Exponential distribution. | implemented | wafo.wstats.wexpstat |
| wstats/wfcdf.m | WFCDF  Snedecor's F cumulative distribution function | implemented | wafo.wstats.wfcdf |
| wstats/wfinv.m | WFINV  Inverse of the Snedecor's F distribution function | implemented | wafo.wstats.wfinv |
| wstats/wfpdf.m | WFPDF Snedecor's F probability density function | implemented | wafo.wstats.wfpdf |
| wstats/wfrechcdf.m | WFRECHCDF Frechet cumulative distribution function | implemented | wafo.wstats.wfrechcdf |
| wstats/wfrechinv.m | WFRECHINV Inverse of the Frechet distribution function | implemented | wafo.wstats.wfrechinv |
| wstats/wfrechpdf.m | WFRECHPDF Frechet probability density function | implemented | wafo.wstats.wfrechpdf |
| wstats/wfrechstat.m | WFRECHSTAT Mean and variance for the Frechet distribution. | implemented | wafo.wstats.wfrechstat |
| wstats/wfrnd.m | WFRND Random matrices from the Snedecor's F distribution | adapted | wafo.wstats.wfrnd |
| wstats/wfstat.m | WFSTAT Mean and variance for the Snedecor's F distribution. | implemented | wafo.wstats.wfstat |
| wstats/wgamafit.m | WGAMAFIT Is an internal routine for wgamfit | adapted | wafo.wstats.wgamafit |
| wstats/wgamcdf.m | WGAMCDF Gamma cumulative distribution function | implemented | wafo.wstats.wgamcdf |
| wstats/wgamfit.m | WGAMFIT Parameter estimates for Gamma data. | adapted | wafo.wstats.wgamfit |
| wstats/wgaminv.m | WGAMINV Inverse of the Gamma distribution function | implemented | wafo.wstats.wgaminv |
| wstats/wgampdf.m | WGAMPDF Gamma probability density function | implemented | wafo.wstats.wgampdf |
| wstats/wgamrnd.m | WGAMRND Random matrices from a Gamma distribution. | adapted | wafo.wstats.wgamrnd |
| wstats/wgamstat.m | WGAMSTAT Mean and variance for the Gamma distribution. | implemented | wafo.wstats.wgamstat |
| wstats/wgevcdf.m | WGEVCDF Generalized Extreme Value cumulative distribution function | implemented | wafo.wstats.wgevcdf |
| wstats/wgevfit.m | WGEVFIT Parameter estimates for GEV data | adapted | wafo.wstats.wgevfit |
| wstats/wgevinv.m | WGEVINV Inverse of the Generalized Extreme Value distribution function | implemented | wafo.wstats.wgevinv |
| wstats/wgevlike.m | WGEVLIKE Is an internal routine for wgevfit | implemented | wafo.wstats.wgevlike |
| wstats/wgevpdf.m | WGEVPDF Generalized Extreme Value probability density function | implemented | wafo.wstats.wgevpdf |
| wstats/wgevrnd.m | WGEVRND Random matrices from a Generalized Extreme Value distribution | adapted | wafo.wstats.wgevrnd |
| wstats/wgevstat.m | WGEVSTAT Mean and variance for the GEV distribution. | implemented | wafo.wstats.wgevstat |
| wstats/wggambfit.m | WGGAMBFIT Is an internal routine for wggamfit | adapted | wafo.wstats.wggambfit |
| wstats/wggamcdf.m | WGGAMCDF Generalized Gamma cumulative distribution function | implemented | wafo.wstats.wggamcdf |
| wstats/wggamfit.m | WGGAMFIT Parameter estimates for Generalized Gamma data. | adapted | wafo.wstats.wggamfit |
| wstats/wggaminv.m | WGGAMINV Inverse of the Generalized Gamma distribution function | implemented | wafo.wstats.wggaminv |
| wstats/wggampdf.m | WGGAMPDF Generalized Gamma probability density function | implemented | wafo.wstats.wggampdf |
| wstats/wggamrnd.m | WGGAMRND Random matrices from a Generalized Gamma distribution. | adapted | wafo.wstats.wggamrnd |
| wstats/wggamstat.m | WGGAMSTAT Mean and variance for the Generalized Gamma distribution. | implemented | wafo.wstats.wggamstat |
| wstats/wgpdcdf.m | WGPDCDF Generalized Pareto cumulative distribution function | implemented | wafo.wstats.wgpdcdf |
| wstats/wgpdfit.m | WGPDFIT Parameter estimates for Generalized Pareto data | adapted | wafo.wstats.wgpdfit |
| wstats/wgpdfit_ml.m | WGPDFIT_ML Internal routine for wgpdfit (ML estimates for GPD data) | implemented | wafo.wstats.wgpdfit_ml |
| wstats/wgpdinv.m | WGPDINV Inverse of the Generalized Pareto distribution function | implemented | wafo.wstats.wgpdinv |
| wstats/wgpdpdf.m | WGPDPDF Generalized Pareto probability density function | implemented | wafo.wstats.wgpdpdf |
| wstats/wgpdrnd.m | WGPDRND Random matrices from a Generalized Pareto Distribution | adapted | wafo.wstats.wgpdrnd |
| wstats/wgpdstat.m | WGPDSTAT Mean and variance for the Generalized Pareto distribution. | implemented | wafo.wstats.wgpdstat |
| wstats/wgumbafit.m | WGUMBAFIT Is an internal routine for wgumbfit | adapted | wafo.wstats.wgumbafit |
| wstats/wgumbcdf.m | WGUMBCDF Gumbel cumulative distribution function. | implemented | wafo.wstats.wgumbcdf |
| wstats/wgumbfit.m | WGUMBFIT Parameter estimates for Gumbel data. | adapted | wafo.wstats.wgumbfit |
| wstats/wgumbinv.m | WGUMBINV Inverse of the Gumbel distribution function. | implemented | wafo.wstats.wgumbinv |
| wstats/wgumbpdf.m | WGUMBPDF Gumbel probability density function. | implemented | wafo.wstats.wgumbpdf |
| wstats/wgumbplot.m | WGUMBPLOT Plots data on a Gumbel distribution paper. | adapted | wafo.wstats.wgumbplot |
| wstats/wgumbrnd.m | WGUMBRND Random matrices from a Gumbel distribution. | adapted | wafo.wstats.wgumbrnd |
| wstats/wgumbstat.m | WGUMBSTAT Mean and variance for the Gumbel distribution. | implemented | wafo.wstats.wgumbstat |
| wstats/wgumbtest.m | WGUMBTEST Tests whether the shape parameter in a GEV is equal to zero | adapted | wafo.wstats.wgumbtest |
| wstats/whisto.m | WHISTO Plots a histogram | adapted | wafo.wstats.whisto |
| wstats/winvgcdf.m | WINVGCDF Inverse Gaussian cumulative distribution function | implemented | wafo.wstats.winvgcdf |
| wstats/winvgfit.m | WINVGFIT Parameter estimates for Inverse Gaussian data. | adapted | wafo.wstats.winvgfit |
| wstats/winvginv.m | WINVGINV Inverse of the Inverse Gaussian distribution function | implemented | wafo.wstats.winvginv |
| wstats/winvgpdf.m | WINVGPDF Inverse Gaussian probability density function | implemented | wafo.wstats.winvgpdf |
| wstats/winvgrnd.m | WINVGRND Random matrices from a Inverse Gaussian distribution. | adapted | wafo.wstats.winvgrnd |
| wstats/winvgstat.m | WINVGSTAT Mean and variance for the Inverse Gaussian distribution. | implemented | wafo.wstats.winvgstat |
| wstats/wkurtosis.m | WKURTOSIS Computes sample kurtosis | implemented | wafo.wstats.wkurtosis |
| wstats/wlogncdf.m | WLOGNCDF Lognormal cumulative distribution function | implemented | wafo.wstats.wlogncdf |
| wstats/wlognfit.m | WLOGNFIT Parameter estimates for Lognormal data. | adapted | wafo.wstats.wlognfit |
| wstats/wlogninv.m | WLOGNINV Inverse of the Lognormal distribution function | implemented | wafo.wstats.wlogninv |
| wstats/wlognpdf.m | WLOGNPDF Lognormal probability density function | implemented | wafo.wstats.wlognpdf |
| wstats/wlognrnd.m | WLOGNRND Random matrices from a Lognormal distribution. | adapted | wafo.wstats.wlognrnd |
| wstats/wlognstat.m | WLOGNSTAT Mean and variance for the Lognormal distribution. | implemented | wafo.wstats.wlognstat |
| wstats/wmnormrnd.m | WMNORMRND Random vectors from a multivariate Normal distribution | adapted | wafo.wstats.wmnormrnd |
| wstats/wnormcdf.m | WNORMCDF Normal cumulative distribution function | implemented | wafo.wstats.wnormcdf |
| wstats/wnormfit.m | WNORMFIT Parameter estimates for Normal data. | adapted | wafo.wstats.wnormfit |
| wstats/wnorminv.m | WNORMINV Inverse of the Normal distribution function | implemented | wafo.wstats.wnorminv |
| wstats/wnormndpdf.m | WNORMNDPDF N-Dimensional Normal probability density function. | implemented | wafo.wstats.wnormndpdf |
| wstats/wnormpdf.m | WNORMPDF Normal probability density function | implemented | wafo.wstats.wnormpdf |
| wstats/wnormplot.m | WNORMPLOT Plots data on a Normal distribution paper | adapted | wafo.wstats.wnormplot |
| wstats/wnormrnd.m | WNORMRND Random matrices from a Normal distribution. | adapted | wafo.wstats.wnormrnd |
| wstats/wnormstat.m | WNORMSTAT Mean and variance for the Normal distribution. | implemented | wafo.wstats.wnormstat |
| wstats/wqqplot.m | WQQPLOT   Plot empirical quantile of X vs empirical quantile of Y | adapted | wafo.wstats.wqqplot |
| wstats/wquantile.m | WQUANTILE Empirical quantile (percentile). | implemented | wafo.wstats.wquantile |
| wstats/wraylcdf.m | WRAYLCDF Rayleigh cumulative distribution function | implemented | wafo.wstats.wraylcdf |
| wstats/wraylfit.m | WRAYLFIT Parameter estimates for Rayleigh data. | adapted | wafo.wstats.wraylfit |
| wstats/wraylinv.m | WRAYLINV Inverse of the Rayleigh distribution function | implemented | wafo.wstats.wraylinv |
| wstats/wraylpdf.m | WRAYLPDF Rayleigh probability density function | implemented | wafo.wstats.wraylpdf |
| wstats/wraylplot.m | WRAYLPLOT Plots data on a Rayleigh distribution paper | adapted | wafo.wstats.wraylplot |
| wstats/wraylrnd.m | WRAYLRND Random matrices from a Rayleigh distribution | adapted | wafo.wstats.wraylrnd |
| wstats/wraylstat.m | WRAYLSTAT Mean and variance for the Rayleigh distribution. | implemented | wafo.wstats.wraylstat |
| wstats/wskewness.m | WSKEWNESS Computes sample skewness | implemented | wafo.wstats.wskewness |
| wstats/wtcdf.m | WTCDF  Student's T  cumulative distribution function | implemented | wafo.wstats.wtcdf |
| wstats/wtfit.m | WTFIT Parameter estimates for Student's T data. | adapted | wafo.wstats.wtfit |
| wstats/wtinv.m | WTINV Inverse of the Student's T distribution function | implemented | wafo.wstats.wtinv |
| wstats/wtpdf.m | WTPDF  Student's T probability density function | implemented | wafo.wstats.wtpdf |
| wstats/wtraylcdf.m | WTRAYLCDF Truncated Rayleigh cumulative distribution function | implemented | wafo.wstats.wtraylcdf |
| wstats/wtraylfit.m | WTRAYLFIT Parameter estimates for Truncated Rayleigh data. | adapted | wafo.wstats.wtraylfit |
| wstats/wtraylpdf.m | WTRAYLPDF Truncated Rayleigh probability density function | implemented | wafo.wstats.wtraylpdf |
| wstats/wtrnd.m | WTRND  Random matrices from a Student's T distribution | adapted | wafo.wstats.wtrnd |
| wstats/wtstat.m | WTSTAT Mean and variance for the Student's T  distribution. | implemented | wafo.wstats.wtstat |
| wstats/wtweibcdf.m | WTWEIBCDF Truncated Weibull cumulative distribution function | implemented | wafo.wstats.wtweibcdf |
| wstats/wtweibfit.m | WTWEIBFIT Parameter estimates for truncated Weibull data. | adapted | wafo.wstats.wtweibfit |
| wstats/wtweibfun.m | WTWEIBFUN Is an internal routine for wtweibfit | implemented | wafo.wstats.wtweibfun |
| wstats/wtweibpdf.m | WTWEIBPDF Truncated Weibull probability density function | implemented | wafo.wstats.wtweibpdf |
| wstats/wweibcdf.m | WWEIBCDF Weibull cumulative distribution function | implemented | wafo.wstats.wweibcdf |
| wstats/wweibcfit.m | WWEIBCFIT Is an internal routine for wweibfit | adapted | wafo.wstats.wweibcfit |
| wstats/wweibfit.m | WWEIBFIT Parameter estimates for Weibull data. | adapted | wafo.wstats.wweibfit |
| wstats/wweibinv.m | WWEIBINV Inverse of the Weibull distribution function | implemented | wafo.wstats.wweibinv |
| wstats/wweibpdf.m | WWEIBPDF Weibull probability density function | implemented | wafo.wstats.wweibpdf |
| wstats/wweibplot.m | WWEIBPLOT Plots data on a Weibull distribution paper | adapted | wafo.wstats.wweibplot |
| wstats/wweibrnd.m | WWEIBRND Random matrices a the Weibull distribution. | adapted | wafo.wstats.wweibrnd |
| wstats/wweibstat.m | WWEIBSTAT Mean and variance for the Weibull  distribution. | implemented | wafo.wstats.wweibstat |
| wstats/yates.m | YATES Calculates main and interaction effects using Yates' algorithm. | implemented | wafo.wstats.yates |
