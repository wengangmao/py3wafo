# Supplied MATLAB toolbox structure

## cycles (43 MATLAB files)

```matlab
% Module CYCLES in WAFO Toolbox. 
% Version 2.1.1   06-Sep-2005 
%
%
% This module contains routines for cycle counting, discretization, 
% and crossings.
% 
% The routines are described in the contents of 'Fatigue in WAFO'.
% Try 'help fatigue'.
```

## damage (12 MATLAB files)

```matlab
% Module DAMAGE in WAFO Toolbox. 
% Version 2.1.1  07-Sep-2005 
% 
%
% This module contains routines connected to fatigue damage calculations.
% 
% The routines are described in the contents of 'Fatigue in WAFO'.
% Try 'help fatigue'.
```

## data (11 MATLAB files)

```matlab
% Module DATA in WAFO Toolbox 
% Version 2.1.1   07-Sep-2005 
% 
% 
% atlantic - Significant wave-height data recorded in the Atlantic Ocean 
% gfaks89  - Surface elevation measured at Gullfaks C 24.12.1989 
% gfaksr89 - Reconstructed surface elevation measured at Gullfaks C 24.12.1989. 
% japansea - Coastline map of The Japan Sea. 
% northsea - Coastline map of The Northsea. 
% sea      - Surface elevation dataset used in WAT version 1.1. 
% sfa89    - Wind measurements at Statfjord A 24.12.1989 
% sn       - Fatigue experiment, constant-amplitude loading. 
% yura87   - Surface elevation measured off the coast of Yura. 
```

## docs (17 MATLAB files)

```matlab
% DOCS MODULE in WAFO Toolbox 
% Version 2.1.1   15-Sep-2005 
% 
% 
% ampdef        - Wave heights and amplitude definitions and nomenclature. 
% crossdef      - Level v crossing definitions and nomenclature. 
% perioddef     - Wave periods (lengths) definitions and nomenclature. 
% tpdef         - Turning points definitions and numenclature. 
% wavedef       - Wave definitions and nomenclature. 
% startup       - Example startup file to put in your matlab path. 
% addfeatr      - How To Add New Features to WAFO. 
% bugreport     - Bug Report Form for WAFO. 
% getstart      - Ideas for getting started with WAFO. 
% install       - Installaion description of WAFO Toolbox 
% limitations   - WAFO known limitations 
% datastructures - Of spectrum, covariance function and density (pdf) in WAFO 
% aboutwafo     - WAFO is a Matlab toolbox for analysis of random waves 
% wafohelp      - Launch HTML help for the WAFO Toolbox.
% wafoweb       - World Wide Web site of the WAFO Toolbox.
% wafomenu      - Displays a user interface to the documentation files of WAFO 


% TODO % Clean up the documentation
```

## exec (1 MATLAB files)

```matlab
% WAFO  Toolbox /exec   
% 
% This directory shall contain a subdirectory with executable files for the
% computer type at hand (pcwin, sol2, alpha). 
%
% When downloading the toolbox from the Internet, this directory is empty. 
% The executable files shall be placed in this directory; hence, the 
% structure should look like for example:
%
% .../wafo/exec/pcwin/minmax.exe

```

## kdetools (41 MATLAB files)

```matlab
% Module KDETOOLS in WAFO Toolbox. 
% Version 2.1.1   15-sep-2005 
% 
% readme     - Readme file for module KDETOOLS in WAFO Toolbox.
% 
% Misc
%   chi2gof    - CHI Squared Goodness Of Fit test. 
%   chi2gof2   - CHI Squared Goodness Of Fit test. 
%   evalpdf    - Evaluates a PDF struct by interpolation. 
%   fftce      - Circulant Embedding of a vector or matrix. 
%   kdedemo1   - Demonstrate smoothing parameter impact on KDE.
%   kdedemo2   - Demonstrate difference between transformation- and ordinary-KDE.
%   kdeoptset  - Create or alter KDE OPTIONS structure.
%   qlevels    - Calculates quantile levels which encloses P% of PDF. 
%   qlevels2   - Calculates quantile levels which encloses P% of data. 
%   vsph       - Calculates volume of d-dimensional sphere with radius r0. 
%
% Bandwidth Selectors
%   hbcv       - Biased Cross-Validation estimate of smoothing parameter. 
%   hbcv2      - Biased Cross-Validation smoothing parameter for 2D data. 
%   hboot      - Bootstrap cross-validation estimate of smoothing parameter. 
%   hldpi      - L-stage Direct Plug-In estimate of smoothing parameter. 
%   hldpi2     - L-stage DPI estimate of smoothing parameter for 2D data. 
%   hldpi2fft  - L-stage DPI estimate of smoothing parameter for 2D data. 
%   hlscv      - LSCV estimate of smoothing parameter. 
%   hmns       - Multivariate Normal Scale Estimate of Smoothing Parameter. 
%   hns        - Normal Scale Estimate of Smoothing Parameter. 
%   hos        - Oversmoothing Parameter. 
%   hscv       - Smoothed cross-validation estimate of smoothing parameter. 
%   hste       - 2-Stage Solve the Equation estimate of smoothing parameter. 
%   hstt       - Scott-Tapia-Thompson estimate of smoothing parameter. 
%
% Kernel density estimators and data-gridders
%   bincount   - 1-dimensional Bin count.
%   ffndgrid   - Fast 'n' Furious N-D data gridding.
%   gridcount  - D-dimensional histogram using linear binning.
%   kde        - Kernel Density Estimator. 
%   kdebin     - Binned Kernel Density Estimator. 
%   kdefun     - Kernel Density Estimator. 
%   kde1dgui   - GUI to Kernel Density Estimator
%   kde2dgui   - GUI to Kernel Density Estimator in two dimensions.
%
% Kernel functions
%   deriv      - 4th, 6th, 8th and 10th derivatives of the kernel function. 
%   deriv2     - High order partial derivatives of the Gaussian kernel. 
%   kernelstats- Return 2'nd order moment of kernel pdf.
%   mkernel    - Multivariate Kernel Function. 
%   mkernel2   - Multivariate Kernel Function, alternative version. 
%
% Random number generators
%   mkernelrnd - Random numbers from the Multivariate Kernel Function. 
%   sample     - Random sampling of points from a data-set. 
%   ssample    - Random sampling from a smoothed empirical distribution. 

```

## m_map (54 MATLAB files)

```matlab
% M_Map - mapping toolbox (Author: rich@eos.ubc.ca)
% Version 1.4d 1 Oct 2008
%
% You have collected your data, loaded it into Matlab, analyzed 
% everything to death, and now you want to make a simple map showing 
% how it relates to the world. 
%
% But you can't. 
%
% Instead you have to figure out how to save all your data, and 
% then read it into a mapping program, and then spend all that extra 
% time figuring out why the mapping program doesn't give you what
% you expected it would...
%
% No more! 
%
%                            Announcing M_Map v1.4! 
%
% M_Map is a set of mapping tools written for Matlab v5. These include: 
%
%    1. Routines to project data in 18 different spherical 
%       projections (and determine inverse mappings) 
%    2. A grid generation routine to make nice axes with 
%       limits either in long/lat terms or in planar
%       X/Y terms. 
%    3. A coastline database (with 1/4 degree resolution) 
%    4. A global elevation database (1 degree resolution) 
%    5. Hooks into freely available high-resolution coastlines and
%       bathymetry/topography.
%
% M_Map v1.4 is available via the web at 
%
%       http://www.eos.ubc.ca/~rich/
%
%
% Toolbox contents
%
%    Contents.m    - This file
%    m_demo.m      - demonstrates a few different maps.
%
%  User-callable functions
%
%    m_proj.m      - initializes projections
%    m_coord.m     - converts between geomagnetic and geographic coords.
%
%    m_grid.m      - draws grids 
%    m_scale       - forces map to a given scale.
%    m_ruler       - draw a scale ruler
%
%    m_ungrid.m    - erases map elements (if you want to change parameters)
%
%    m_coast.m     - draws a coastline
%    m_elev.m      - draws elevation data from 1 degree database
%
%    m_tbase.m     - draws elevation data from 5-minute TerrainBase database
%    m_gshhs.m     - draws coastline from GSHHS with specified resolution
%    m_gshhs_c.m   - draws coastline from GSHHS crude database
%    m_gshhs_l.m   - draws coastline from GSHHS low-resolution database
%    m_gshhs_i.m   - draws coastline from GSHHS intermediate-resolution database
%    m_gshhs_h.m   - draws coastline from GSHHS high-resolution database
```

## markov (31 MATLAB files)

```matlab
% Module MARKOV in WAFO Toolbox. 
% Version 2.1.1   07-Sep-2005 
% 
%
% This module contains routines for Markov loads, 
% switchiching Markov loads, and their connection to rainflow cycles.
% 
% The routines are described in the contents of 'Fatigue in WAFO'.
% Try 'help fatigue'.
```

## misc (35 MATLAB files)

```matlab
% Module MISC in WAFO Toolbox. 
% Version 2.1.1   15-Sep-2005 
% 
% readme      - Readme file for module MISC in WAFO Toolbox 
% 
% Functions
% binom       - Calculates the binomial coefficient n!/((n-m)!*m!) 
% csort       - Counting sorting
% findpeaks   - Find peaks of vector or matrix possibly rainflow filtered 
% genchol     - Generalized Cholesky factorization
% hypgf       - Hypergeometric function F(a,b,c,z) 
% sinc        - Sin(pi*x)/(pi*x) function. 
% smooth      - Calculates a smoothing spline. 
%
% Oceanographic constants
% gravity     - Returns the constant acceleration of gravity 
% wdensity    - Returns the water density 
%
% Integration
% ccquad      - Numerical integration using a Clenshaw-Curtis quadrature. 
% gaussq      - Numerically evaluates a integral using a Gauss quadrature. 
% gaussq2d    - Numerically evaluates a 2D integral using Gauss quadrature. 
% qrule       - Compute abscissas and weight factors for Gaussian quadratures 
% qrule2d     - Compute abscissas and weight factors for Gaussian quadratures 
% simpson     - Numerical integration with the Simpson method 
%
% Plotting utilities
% clevels     - Extract the contour levels from the contour matrix
% cltext      - Places contour level text in the current window 
% fcolorbar   - Display color bar with discrete color axis for filled contour plot.
% fwaitbar    - Fast display of wait bar.
% figtext     - Places text in figure window. 
% maximizefigs- Maximize figure(s) window size
% wafostamp   - Prints a caption "made by WAFO" in current figure. 
%
% Documentation utilities
% geth1line   - Extracts the first comment line (the H1 line) of a m-file 
% mkcontents  - Makes Contents file in current working directory. 
% wafoversion - Wave Analysis for Fatigue and Oceanography version.
%
% Misc
% choices     - Create a list of choices with uicontrols and callbacks.
% choicex     - Close a list of choices.
% comnsize    - Check if all input arguments are either scalar or of common size.
% levels      - Calculates discrete levels given the parameter matrix. 
% loaddata    - Loads a matrix from a text file. 
% parseoptions- Create or alter a OPTIONS structure


```

## multidim (22 MATLAB files)

```matlab
% Module MULTIDIM in WAFO Toolbox. 
% Version 2.1.1   07-Sep-2005 
% 
% 
% dat2dspec        - Estimates directional wave spectrum from data.
% dspec2spreading  - Returns directional spreading function D of S 
% fourier          - Returns Fourier coefficients.
% ratio            - Compute ratio of hyperbolic functions 
% sensortypeid     - Return ID for sensortype name
% sensortype       - Return sensortype name
% specoptset       - Create or alter SPECTRUM OPTIONS structure. 
% testbuoy         - Creates a test case for a buoy measurement 
% testmeasurements - Creates a test case for measurement time series 
% testsurf         - Creates a test case for surface elevation measurement 
% tran             - Computes transfer functions based on linear wave theory 
```

## onedim (30 MATLAB files)

```matlab
% Module ONEDIM in WAFO Toolbox. 
% Version 2.1.1  07-Sep-2005 
% 
% 
% Readme       - Readme file for module ONEDIM in WAFO Toolbox. 
%
% Misc
% detrendma    - Removes a trend from data using a moving average 
% findoutliers - Finds the indices to spurious points in a timeseries
% reconstruct  - Reconstruct the spurious/missing points of timeseries
%
% Plotting
% covplot      - Plots auto covariance function (ACF) 1D or 2D. 
% lcplot       - Plots level-crossing spectrum (lc) 
% spwaveplot   - Plots specified waves from a timeseries  
% waveplot     - Plots surface elevation of timeseries. 
%
% Autocovariance and spectrum estimation
% dat2cor      - Estimate auto correlation function from data.
% dat2cov      - Estimate auto covariance function from data.
% dat2spec     - Estimate one-sided spectral density from data
% dat2spec2    - Estimate one-sided spectral density, version 2. 
%
% Data extraction
% dat2crossind - Finds indices of down and/or upcrossings from data.
% dat2lc       - Extracts level-crossing spectrum from data.
% dat2steep    - Extracts waveheights and steepnesses from data. 
% dat2tc       - Extracts troughs and crests from data. 
% dat2tp       - Extracts turning points from data. 
% dat2wa       - Extracts sequence of wavelengths from data. 
% ecross       - Extracts exact level v crossings.
% findcross    - Finds indices to level v up and downcrossings of a vector 
% findextrema  - Finds indices to minima and maxima of data.
% findrfc      - Finds indices to rainflow cycles from sequence of TP. 
% mm2lc        - Extracts level-crossing spectrum from min2Max cycles.




 

```

## papers (43 MATLAB files)

```matlab
% Module PAPERS in WAFO Toolbox. 
% Version 2.1.1   15-Sep-2005 
% 
% papermenu - displays a user interface to the paper scripts of WAFO
%
% List of articles in the subdirectory papers in WAFO. The scripts are found
% in the directories: (e.g try 'help wafodemo') 
% 
% tutorcom
%      The WAFO Group (2000): WAFO - Tutorial
%
% wafodemo
%      Brodtkorb, P.A., Johannesson, P., Lindgren, G., Rychlik, I., Rydén, J.,
%      and Sjö, E. (2000). WAFO - a Matlab toolbox for analysis of random
%      waves and loads. Proceedings of the 10th International Offshore and
%      Polar Engineering conference, Seattle, Vol III, pp. 343-350. 
%
% rec
%      Brodtkorb, P.A., Myrhaug, D. and Rue, H (1999). Joint distribution of
%      wave height and wave crest velocity from reconstructed data.
%      Proceedings of 9th ISOPE Conference, Brest, Vol III, pp. 66-73 

```

## source (0 MATLAB files)

```matlab

```

## spec (46 MATLAB files)

```matlab
%Module SPEC in WAFO Toolbox. 
%Version 2.1.1   07-Sep-2005 
% 
%   readme - Readme file for module SPEC in WAFO Toolbox 
%
%Model spectra
% demospec     - Loads a precreated spectrum of chosen type 
% getjonswappeakedness - Peakedness factor Gamma given Hm0 and Tp for JONSWAP
% jonswap      - Calculates (and plots) a JONSWAP spectral density 
% mccormick    - Calculates (and plots) a McCormick spectral density. 
% ohspec       - Calculates (and plots) a Ochi-Hubble spectral density. 
% ohspec2      - Calculates and plots a bimodal Ochi-Hubble spectral density 
% ohspec3      - Calculates Bimodal Ochi-Hubble spectral densities 
% oscspec      - Spectral density for a harmonic oscillator 
% phi1         - Factor for transforming spectra to finite water depth spectra 
% pmspec       - Calculates (and plots) a Pierson-Moskowitz spectral density. 
% tmaspec      - Calculates a JONSWAP spectral density for finite water depth 
% torsethaugen - Calculates a double peaked (swell + wind) spectrum 
% wallop       - Calculates (and plots) a Wallop spectral density. 
%
%Directional spectra and spreading functions
% mkdspec      - Make a directional spectrum 
% spreading    - Directional spreading functions 
%
%Spectral characteristics
% dspec2char   - Evaluates directional spectral characteristics 
% spec2bw      - Evaluates some spectral bandwidth and irregularity factors 
% spec2char    - Evaluates spectral characteristics and their covariance 
% spec2mom     - Calculates spectral moments from spectrum 
%
%Spectrum and covariance functions
% cov2spec     - Computes spectral density given the auto covariance function 
% covinterp    - Interpolation of covariance function and derivatives 
% createcov    - Covariance class constructor 
% createspec   - Spectrum structure constructor 
% freqtype     - Returns the frequency type of a Spectral density struct. 
% lagtype      - Returns the lag type of a Covariance struct.
% rotspec      - Rotate spectrum anti-clockwise around the origin. 
% scalespec    - Scale spectral density so that the moments equals m0,m2. 
% spec2cov     - Computes covariance function and its derivatives 
% spec2cov2    - Computes covariance function and its derivatives, alternative version
% spec2dt      - Computes sampling interval from Nyquist frequency in spectrum 
% spec2spec    - Transforms between different types of spectra 
% specinterp   - Interpolaton and zero-padding of spectrum 
% ttspec       - Toggle Transform between angular frequency and frequency spectrum 
% wnormspec    - Normalize a spectral density such that m0=m2=1 (private) 
% wspecplot    - Plot a spectral density 
%
%Dispersion relation
% k2w          - Translates from wave number to frequency 
% w2k          - Translates from frequency to wave number 




```

## trgauss (69 MATLAB files)

```matlab
% Module TRGAUSS in WAFO Toolbox. 
% Version 2.1.1   07-Sep-2005
% 
% 
% Readme        - Readme file for module TRGAUSS in WAFO Toolbox. 
%
% Misc
% createpdf     - PDF class constructor 
% pdfplot       - Plot contents of pdf structures 
% trplot        - Plot transformation, g, eg.  estimated with dat2tr.
%
%Transforms and non-linearities
% dat2gaus      - Transforms  x  using the transformation  g. 
% gaus2dat      - Transforms  xx  using the inverse of transformation  g. 
% mctrtest      - Test if stochastic process is Gaussian.
% spec2skew     - Estimate moments of 2'nd order waves due to Marthinsen and Winterstein 
% trangood      - Make transformation suitable for efficient transforms. 
% tranproc      - Transforms process X and up to four derivatives 
% trmak         - Put together a transformation object. 
% troptset      - Create or alter TRANSFORM OPTIONS structure. 
% trunmak       - Split a transformation object into its pieces.
%
% Transformed Gaussian model estimation
% cdf2tr        - Estimate transformation, g, from observed CDF. 
% dat2tr        - Estimate transformation, g, from data. 
% hermitetr     - Calculate transformation, g, proposed by Winterstein 
% ochitr        - Calculate transformation, g, proposed by Ochi et al. 
% lc2tr         - Estimate transformation, g, from observed crossing intensity. 
% lc2tr2        - Estimate transformation, g, from observed crossing intensity, version 2. 
%
%Gaussian probabilities and expectations
% bvnormcdf     - Bivariate Normal cumulative distribution function
% bvnormprb     - Bivariate Normal probability  
% mvnormpcprb   - Multivariate Normal probability with product correlation
% mvnormprb     - Multivariate Normal probability by Genz' algorithm.
% mvnortpcprb   - Multivariate Normal or student T probability with product correlation.
% rind          - Multivariate normal expectations
%
%Probability density functions (pdf) or intensity matrices
% chitwo2lc_sorm- SORM-approximation of crossing intensity for noncentral Chi^2 process
% chitwo2lc_sp  - Saddlepoint approximation of crossing intensity for noncentral Chi^2 process 
% dirsp2chitwo  - Parameters in non-central CHI-TWO process for directional Stokes waves.
% iter          - Calculates a Markov matrix fmM  given a rainflow matrix frfc; 
% iter_mc       - Calculates a kernel f_xy of a MC given a rainflow matrix 
% mc2rfc        - Calculates a rainflow matrix given a Markov chain with kernel f_xy; 
% mctp2rfc      - Calculates a rainflow matrix given a Markov matrix f_mM of a Markov chain 
% mctp2tc       - Calculates frequencies for the upcrossing troughs and crests 
% nt2fr         - Calculates the frequency matrix given the counting distribution matrix. 
% spec2cmat     - Joint intensity matrix for cycles 
% spec2mmtpdf   - Calculates joint density of Maximum, minimum and period. 
% spec2tccpdf   - Evaluates densities of wave period Tcc, wave lenght Lcc. 
% spec2thpdf    - Joint density of amplitude and period/wave-length characteristics 
% spec2tpdf     - Evaluates densities for crest-,trough-period, length. 
% spec2tpdf2    - Evaluates densities for various wave periods or wave lengths 
% specq2lc      - Saddlepoint approximation of crossing intensity for quadratic sea. 
% th2vhpdf      - Transform joint T-H density to V-H density 
%
%Cumulative distribution functions (cdf)
% lomaxcdf      - CDF for local maxima for a zero-mean Gaussian process 
% spec2acat     - Evaluates survival function R(h1,h2)=P(Ac>h1,At>h2). 
```

## wavemodels (71 MATLAB files)

```matlab
% Module WAVEMODELS in WAFO Toolbox. 
% Version 2.1.1   07-Sep-2005 
%
% readme     - Readme file for module WAVEMODELS in WAFO Toolbox.  
%
%Probability density functions (PDF)
% b04jpdf    - Brodtkorb (2004) joint (Scf,Hd) PDF from Japan Sea.
% b04jpdf2   - Brodtkorb (2004) joint (Scf,Hd) PDF from Japan Sea.
% b04pdf     - Brodtkorb (2004) joint (Scf,Hd) PDF of laboratory data.
% b04pdf2    - Brodtkorb (2004) joint (Scf,Hd) PDF of laboratory data.
% bmr00pdf   - Brodtkorb et.al (2000) joint (Scf,Hd) PDF from North Sea.
% bmr00pdf2  - Brodtkorb et.al (2000) joint (Scf,Hd) PDF from North Sea.
% braylpdf   - Beta Rayleigh PDF of wave heigth 
% cav76pdf   - Cavanie et al. (1976) approximation of the density  (Tc,Ac) 
% jhspdf     - Joint (Scf,Hd) PDF for linear waves with a JONSWAP spectrum.
% jhspdf2    - Joint (Scf,Hd) PDF for linear waves with a JONSWAP spectrum.
% jhsnlpdf   - Joint (Scf,Hd) PDF for non-linear waves with a JONSWAP spectrum.
% jhsnlpdf2  - Joint (Scf,Hd) PDF for non-linear waves with a JONSWAP spectrum.
% jhvnlpdf   - Joint (Vcf,Hd) PDF for non-linear waves with a JONSWAP spectrum.
% jhvnlpdf2  - Joint (Vcf,Hd) PDF for non-linear waves with a JONSWAP spectrum.
% jhvpdf     - Joint (Vcf,Hd) PDF for linear waves with a JONSWAP spectrum. 
% jhvpdf2    - Joint (Vcf,Hd) PDF for linear waves with a JONSWAP spectrum. 
% lh83pdf    - Longuet-Higgins (1983) approximation of the density (Tc,Ac) 
% ltwcpdf    - Long Term Wave Climate PDF of Hm0 and wave period 
% mk87pdf    - Myrhaug and Kjeldsen (1987) joint (Scf,Hd) PDF. 
% mk87pdf2   - Myrhaug and Kjeldsen (1987) joint (Scf,Hd) PDF. 
% ochi98pdf  - Ochi's (1998) PDF of peaks and troughs of non-gaussian processes 
% ohhpdf     - Marginal wave height, Hd, PDF for Bimodal Ochi-Hubble spectra. 
% ohhspdf    - Joint (Scf,Hd) PDF for linear waves with Ochi-Hubble spectra. 
% ohhspdf2   - Joint (Scf,Hd) PDF for linear waves with Ochi-Hubble spectra. 
% ohhsspdf   - Joint (Scf,Hd) PDF for linear waves in space with Ochi-Hubble spectra. 
% ohhsspdf2  - Joint (Scf,Hd) PDF for linear waves in space with Ochi-Hubble spectra. 
% ohhvpdf    - Joint (Vcf,Hd) PDF for linear waves with Ochi-Hubble spectra. 
% tay81pdf   - Tayfun (1981) PDF of  breaking limited wave heights 
% tay90pdf   - Tayfun (1990) PDF of large wave heights 
% thpdf      - Marginal wave height, Hd, pdf for Torsethaugen spectra. 
% thspdf     - Joint (Scf,Hd) PDF for linear waves with Torsethaugen spectra. 
% thspdf2    - Joint (Scf,Hd) PDF for linear waves with Torsethaugen spectra. 
% thsspdf    - Joint (Scf,Hd) PDF for linear waves in space with Torsethaugen spectra. 
% thsspdf2   - Joint (Scf,Hd) PDF for linear waves in space with Torsethaugen spectra. 
% thvpdf     - Joint (Vcf,Hd) PDF for linear waves with Torsethaugen spectra. 
% thvpdf2    - Joint (Vcf,Hd) PDF for linear waves with Torsethaugen spectra. 
% trraylpdf  - Transformed Rayleigh approximation for amplitudes 
%
%Cumulative distribution functions (cdf)
% b04cdf     - Brodtkorb (2004) joint (Scf,Hd) CDF of laboratory data.
% b04jcdf    - Brodtkorb (2004) joint (Scf,Hd) CDF from Japan Sea.
% bmr00cdf   - Brodtkorb et.al (2000) joint (Scf,Hd) CDF from North Sea.
% braylcdf   - Beta Rayleigh CDF of wave heights 
% jhscdf     - Joint (Scf,Hd) CDF for linear waves with a JONSWAP spectrum.
% jhsnlcdf   - Joint (Scf,Hd) CDF for non-linear waves with a JONSWAP spectrum.
% jhvcdf     - Joint (Vcf,Hd) CDF for linear waves with a JONSWAP spectrum. 
% jhvnlcdf   - Joint (Vcf,Hd) CDF for non-linear waves with a JONSWAP spectrum. 
% mk87cdf    - Myrhaug and Kjeldsen (1987) joint (Scf,Hd) CDF. 
% ochi98cdf  - Ochi's (1998) CDF of peaks and troughs of non-gaussian processes 
% ohhcdf     - Marginal wave height, Hd, CDF for Ochi-Hubble spectra.
% ohhscdf    - Joint (Scf,Hd) CDF for linear waves with Ochi-Hubble spectra.
% ohhsscdf   - Joint (Scf,Hd) CDF for linear waves in space with Ochi-Hubble spectra.
% tay81cdf   - Tayfun (1981) CDF of  breaking limited wave heights 
% tay90cdf   - Tayfun (1990) CDF of large wave heights 
```

## wdemos (23 MATLAB files)

```matlab
% Module WDEMOS in WAFO Toolbox. 
% Version 2.1.1   07-Sep-2005 
% 
% 
% democc         - A program for visualization of cycle counts in random 
% democc_markmax - Plots load and marks a maximum 
% democc_mmdef   - Illustrates the definition of min-max cycles. 
% democc_plotmat - Plots RFC and min-max counts. 
% democc_rfcdef  - Illustrates the definition of rainflow cycles. 
% democc_tpdef   - Illustrates the definition of turning points. 
% itmkurs        - Initiate paths for Demo Load and Fatigue Analysis 
% rfcdemo1       - Demo for switching AR(1)-processes. 
% rfcdemo2       - Rainflow matrix for Switching Markov Chains of Turning Points.
%
% See also  papers/tutorcom, papers/wafodemo
% For initiation of module PAPERS type
%   initwafo('full');
```

## wsim (24 MATLAB files)

```matlab
% Module WSIM in WAFO Toolbox. 
% Version 2.1.1   07-Sep-2005 
% 
% 
% cov2csdat       - Generates conditionally simulated values 
% cov2sdat        - Simulates a Gaussian process and its derivative 
% disufq          - Return difference- and sum-frequency effects.
% duffsim         - Generates a sample path of a harmonic oscillator 
% lc2sdat         - Generates process with given irreg. factor and cross. spectrum 
% mcsim           - Simulates a Markov chain. 
% mctpsim         - Simulates a Markov chain of turning points 
% rfm2dtp         - Reconstructs a sequence of turning points from a rainflow matrix
% sarmasim        - Simulates a switching ARMA-process. 
% seamovie        - Makes a movie of a 2D (x,t) or 3D (x,y,t) simulated sea 
% seasim          - Spectral simulation of a Gaussian sea, 2D (x,t) or 3D (x,y,t) 
% smcsim          - Simulates a Switching Markov chain with state space. 
% smctpsim        - Simulates a switching Markov chain of turning points, 
% spec2linspec    - Separates the linear component of the Spectrum 
% spec2nlsdat     - Simulates a Randomized 2nd order non-linear wave X(t) 
% spec2sdat       - Simulates a Gaussian process and its derivative from spectrum 
```

## wstats (180 MATLAB files)

```matlab
% Module WSTATS in WAFO Toolbox. 
% Version 2.1.1  07-Sep-2005
%
% What's new
%   Readme     - New features, bug fixes, and changes in WSTATS. 
%
% Parameter estimation
%   loglike    - Log-likelihood function
%   wbetafit   - Parameter estimates for Beta data
%   wchi2fit   - Parameter estimates for Chi squared data
%   wexpfit    - Parameter estimates for Exponential data
%   wgamfit    - Parameter estimates for Gamma data
%   wgamafit   - Is an internal routine for wgamfit
%   wgevfit    - Parameter estimates for GEV data
%   wgevlike   - Is an internal routine for wgevfit
%   wggamfit   - Parameter estimates for Generalized Gamma data
%   wggambfit  - Is an internal routine for wggamfit
%   wgpdfit    - Parameter estimates for GPD data
%   wgpdfit_ml - Internal routine for wgpdfit (ML estimates for GPD data)
%   wgumbfit   - Parameter estimates for Gumbel data
%   wgumbafit  - Is an internal routine for wgumbfit
%   winvgfit   - Parameter estimates for Inverse Gaussian data
%   wlognfit   - Parameter estimates for Lognormal data
%   wnormfit   - Parameter estimates for Normal data
%   wraylfit   - Parameter estimates for Rayleigh data
%   wtraylfit  - Parameter estimates for Truncated Rayleigh data
%   wtfit      - Parameter estimates for Student's T data
%   wweibfit   - Parameter estimates for Weibull data
%   wweibcfit  - Is an internal routine for wweibfit
%   wtweibfit  - Parameter estimates for Truncated Weibull data
%   wtweibfun  - Is an internal routine for wtweibfit
%   weib2dfit  - Parameter estimates for 2D Weibull data
%   weib2dlike - 2D Weibull log-likelihood function
%   dist2dfit  - Parameter estimates for DIST2D data
%   dist2dsmfun  - Smooths the conditional DIST2D distribution parameters
%   dist2dsmfun2 - Smooths the conditional DIST2D distribution parameters
%   mdist2dfit   - Parameter estimates for MDIST2D data
%   mdist2dlike  - MDIST log-likelihood function
%
% Probability density functions (pdf)
%   wbetapdf   - Beta probability density function 
%   wchi2pdf   - Chi squared probability density function
%   wexppdf    - Exponential probability density function
%   wfpdf      - Snedecor's F probability density function 
%   wfrechpdf  - Frechet probability density function
%   wgampdf    - Gamma probability density function
%   wggampdf   - Generalized Gamma probability density function
%   wgevpdf    - Generalized Extreme Value probability density function
%   wgpdpdf    - Generalized Pareto probability density function
%   wgumbpdf   - Gumbel probability density function
%   winvgpdf   - Inverse Gaussian probability density function
%   wlognpdf   - Lognormal probability density function
%   wnormpdf   - Normal probability density function
%   mvnormpdf  - Multivariate Normal probability density function
%   wraylpdf   - Rayleigh probability density function
%   wtraylpdf  - Truncated Rayleigh probability density function
%   wtpdf      - Student's T probability density function 
%   wweibpdf   - Weibull probability density function
%   wtweibpdf  - Truncated Weibull probability density function
%   weib2dpdf  - 2D Weibull probability density function (pdf)
```
