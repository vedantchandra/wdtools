import warnings
import numpy as np
from bisect import bisect_left
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import scipy
from PyAstronomy.pyasl import crosscorrRV, quadExtreme, dopplerShift
from scipy.interpolate import interp1d
import lmfit
from lmfit.models import LinearModel, VoigtModel, GaussianModel
import numpy.polynomial.polynomial as poly

speed_light = 299792458 #m/s

plt.rcParams.update({'font.size': 18})
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

class SpecTools():

    '''

    Spectrum processing tools and functions. 

    '''

    def __init__(self, plot_continuum = False, smoothing = 1e-15, filter_skylines = True, crop = True):
        self.plot_continuum = plot_continuum
        self.smoothing = smoothing
        self.filter_skylines = filter_skylines
        self.crop = crop
        self.halpha = 6564.61
        self.hbeta = 4862.68
        self.hgamma = 4341.68
        self.hdelta = 4102.89
        linear_model = LinearModel(prefix = 'l_')
        self.params = linear_model.make_params()
        voigt_model = VoigtModel(prefix = 'v_')
        self.params.update(voigt_model.make_params())
        self.cm = linear_model - voigt_model
        self.params['v_amplitude'].set(value = 150)
        self.params['v_sigma'].set(value = 5)
        self.params['l_intercept'].set(value = 25)
        self.params['l_slope'].set(value = 0)


    def continuum_normalize(self, wl, fl, ivar = None, croprange = (0, 2)):
        '''
        Continuum-normalization with smoothing splines that avoid a pre-made list of absorption lines for DA and DB spectra. To normalize spectra that only have Balmer lines (DA), we recommend using the `normalize_balmer` function instead. Also crops the spectrum to the 3700 - 7000 Angstrom range. 
        
        Parameters
        ---------
        wl : array
            Wavelength array of spectrum
        fl : array
            Flux array of spectrum
        ivar : array, optional
            Inverse variance array. If `None`, will return only the normalized wavelength and flux. 

        Returns
        -------
            tuple
                Tuple of cropped wavelength, cropped and normalized flux, and (if ivar is not None) cropped and normalized inverse variance array. 

        '''


        cmask = (
            ((wl > 6900) * (wl < 7500)) +\
            ((wl > 6200) * (wl < 6450)) +\
            ((wl > 5150) * (wl < 5650)) +\
            ((wl > 4600) * (wl < 4670)) +\
            ((wl > 4200) * (wl < 4300)) +\
            ((wl > 3910) * (wl < 3950)) +\
            ((wl > 3750) * (wl < 3775)) + ((wl > np.min(wl)) * (wl < np.min(wl) + 50)) + (wl > 7500)
            )

        spl = scipy.interpolate.splrep(wl[cmask], fl[cmask], k = 1 , s = 1000)
        # plt.figure()
        # plt.plot(wl, fl)
        # plt.plot(wl,scipy.interpolate.splev(wl, spl))
        # plt.show()
        norm = fl/scipy.interpolate.splev(wl, spl)

        valid = (norm > croprange[0]) * (norm < croprange[1])

        if ivar is not None:
            ivar_norm = ivar * scipy.interpolate.splev(wl, spl)**2
            return wl[valid], norm[valid], ivar_norm[valid]
        elif ivar is None:
            return wl[valid], norm[valid]

    def normalize_line(self, wl, fl, ivar, centroid, distance, make_plot = False):

        '''
        Continuum-normalization of a single absorption line by fitting a linear model added to a Voigt profile to the spectrum, and dividing out the linear model. 
        
        Parameters
        ---------
        wl : array
            Wavelength array of spectrum
        fl : array
            Flux array of spectrum
        ivar : array, optional
            Inverse variance array. If `None`, will return only the normalized wavelength and flux. 
        centroid : float
            The theoretical centroid of the absorption line that is being fitted, in wavelength units.
        distance : float
            Distance in Angstroms away from the line centroid to include in the fit. Should include the entire absorption line wings with minimal continum. 
        make_plot : bool, optional
            Whether to plot the linear + Voigt fit. Use for debugging. 

        Returns
        -------
            tuple
                Tuple of cropped wavelength, cropped and normalized flux, and (if ivar is not None) cropped and normalized inverse variance array. 

        '''

        self.params['v_center'].set(value = centroid)

        crop1 = bisect_left(wl, centroid - distance)
        crop2 = bisect_left(wl, centroid + distance)

        cropped_wl = wl[crop1:crop2]
        cropped_fl = fl[crop1:crop2]

        cropped_fl = cropped_fl / np.nanmax(cropped_fl)

        try:
            res = self.cm.fit(cropped_fl, self.params, x = cropped_wl, nan_policy = 'omit')
        except TypeError:
            print('profile fit failed. ensure all lines passed to normalize_balmer are present on the spectrum!')
            raise

        if res.message != 'Fit succeeded.':
            print('the line fit was ill-constrained. visually inspect the fit quality with make_plot = True')
        slope = res.params['l_slope']
        intercept = res.params['l_intercept']
        
        if make_plot:
            plt.plot(cropped_wl, cropped_fl)
            #plt.plot(cropped_wl, self.cm.eval(params, x=cropped_wl))
            plt.plot(cropped_wl, res.eval(res.params, x=cropped_wl))
            plt.plot(cropped_wl, cropped_wl*slope + intercept)
            plt.show()
        
        continuum = (slope * cropped_wl + intercept)
        
        fl_normalized = cropped_fl / continuum
        
        if ivar is not None:
            cropped_ivar = ivar[crop1:crop2]
            ivar_normalized = cropped_ivar * continuum**2
            return cropped_wl, fl_normalized, ivar_normalized
        else:
            return cropped_wl, fl_normalized

    def normalize_balmer(self, wl, fl, ivar = None, lines = ['alpha', 'beta', 'gamma', 'delta'], \
                         skylines = False, make_plot = False, make_subplot = False, make_stackedplot = False, \
                             centroid_dict = dict(alpha = 6564.61, beta = 4862.68, gamma = 4341.68, delta = 4102.89, eps = 3971.20, h8 = 3890.12),
                                distance_dict = dict(alpha = 300, beta = 200, gamma = 120, delta = 75, eps = 50, h8 = 25), sky_fill = np.nan):
        

        '''
        Continuum-normalization of any spectrum by fitting each line individually. 

        Fits every absorption line by fitting a linear model added to a Voigt profile to the spectrum, and dividing out the linear model. 
        All normalized lines are concatenated and returned. For statistical and plotting purposes, two adjacent lines should not have overlapping regions (governed by the `distance_dict`). 
        
        Parameters
        ---------
        wl : array
            Wavelength array of spectrum
        fl : array
            Flux array of spectrum
        ivar : array, optional
            Inverse variance array. If `None`, will return only the normalized wavelength and flux. 
        lines : array-like, optional
            Array of which Balmer lines to include in the fit. Can be any combination of ['alpha', 'beta', 'gamma', 'delta', 'eps', 'h8']
        skylines : bool, optional
            If True, masks out pre-selected telluric features and replace them with `np.nan`. 
        make_plot : bool, optional
            Whether to plot the continuum-normalized spectrum.
        make_subplot : bool, optional
            Whether to plot each individual fit of the linear + Voigt profiles. Use for debugging. 
        make_stackedplot : bool, optional
            Plot continuum-normalized lines stacked with a common centroid, vertically displaced for clarity. 
        centroid_dict : dict, optional
            Dictionary of centroid names and theoretical wavelengths. Change this if your wavelength calibration is different from SDSS. 
        distance_dict : dict, optional
            Dictionary of centroid names and distances from the centroid to include in the normalization process. Should include the entire wings of each line and minimal continuum. No 
            two adjacent lines should have overlapping regions. 

        Returns
        -------
            tuple
                Tuple of cropped wavelength, cropped and normalized flux, and (if ivar is not None) cropped and normalized inverse variance array. 

        '''


        fl_normalized = [];
        wl_normalized = [];
        ivar_normalized = [];
        ct = 0;
        
        
        for line in lines:
            if ivar is not None:
                wl_segment, fl_segment, ivar_segment = self.normalize_line(wl, fl, ivar, centroid_dict[line], distance_dict[line], make_plot = make_subplot)
                fl_normalized = np.append(fl_segment, fl_normalized)
                wl_normalized = np.append(wl_segment, wl_normalized)
                ivar_normalized = np.append(ivar_segment, ivar_normalized)
                
            else:
                wl_segment, fl_segment = self.normalize_line(wl, fl, None, centroid_dict[line],\
                                                        distance_dict[line], make_plot = make_subplot)
                plt.show()
                fl_normalized = np.append(fl_segment, fl_normalized)
                wl_normalized = np.append(wl_segment, wl_normalized)
                
        if skylines:
            skylinemask = (wl_normalized > 5578.5 - 10)*(wl_normalized < 5578.5 + 10) + (wl_normalized > 5894.6 - 10)\
            *(wl_normalized < 5894.6 + 10) + (wl_normalized > 6301.7 - 10)*(wl_normalized < 6301.7 + 10) + \
            (wl_normalized > 7246.0 - 10)*(wl_normalized < 7246.0 + 10)
            fl_normalized[skylinemask] = sky_fill
        
        if make_plot:
            plt.plot(wl_normalized, fl_normalized, 'k')

        if make_stackedplot:
            breakpoints = np.nonzero(np.diff(wl_normalized) > 5)[0]
            breakpoints = np.concatenate(([0], breakpoints, [None]))
            plt.figure(figsize = (5,8))
            for kk in range(len(breakpoints) - 1):
                wl_seg = wl_normalized[breakpoints[kk] + 1:breakpoints[kk+1]]
                fl_seg = fl_normalized[breakpoints[kk] + 1:breakpoints[kk+1]]
                peak = int(len(wl_seg)/2)
                delta_wl = wl_seg - wl_seg[peak]
                plt.plot(delta_wl, 1 + fl_seg - 0.35 * kk, 'k')
            plt.show()

        if ivar is not None:
            return wl_normalized, fl_normalized, ivar_normalized
        else:
            return wl_normalized, fl_normalized

    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def rv_corr(self, wl, normflux):
        alphaline = self.find_nearest(wl,self.halpha)
        betaline = self.find_nearest(wl,self.hbeta)
        gammaline = self.find_nearest(wl,self.hgamma)
        deltaline = self.find_nearest(wl,self.hdelta)
        cores = (wl == alphaline) + (wl == betaline) + (wl == gammaline) + (wl == deltaline)
        cores = cores.astype(int)
        xcorrs = crosscorrRV(wl,1-normflux,wl,cores, -1250, 1250, 50, 'doppler', skipedge = 25)
        corrs = xcorrs[1]
        rvs = xcorrs[0]
        # plt.plot(rvs,corrs)
        try:
            p = np.polyfit(rvs, corrs, 2)
            specrv = - p[1] / (2*p[0])
            # plt.plot(rvs,np.polyval(p,rvs))
            # plt.axvline(specrv)
            # print(specrv)
        except:
            print('find rv failed. defaulting to zero')
            specrv = 0
        shift_flux,shift_wl = dopplerShift(wl,normflux,specrv, edgeHandling = 'fillValue', fillValue = 1)
        return shift_wl,shift_flux

    def interpolate(self, wl, flux, target_wl = np.arange(4000,8000)):
        func = interp1d(wl, flux, kind='linear', assume_sorted = True, fill_value = 'extrapolate')
        interpflux = func(target_wl)[1]
        return target_wl,interpflux

    def linear(self, wl, p1, p2):

        ''' Linear polynomial of degree 1 '''

        return p1 + p2*wl

    def chisquare(self, residual):
        
        ''' Chi^2 statistics from residual

        Unscaled chi^2 statistic from an array of residuals (does not account for uncertainties).
        '''

        return np.sum(residual**2)

    def find_centroid(self, wl, flux, centroid, half_window = 25, window_step = 2, n_fit = 12, make_plot = False, \
                 pltname = ''):

        '''
        Statistical inference of spectral redshift by iteratively fitting Voigt profiles to cropped windows around the line centroid. 

        TODO: Add ivar

        Parameters
        ---------
        wl : array
            Wavelength array of spectrum
        flux : array
            Flux array of spectrum
        centroid : float
            Theoretical wavelength of line centroid
        half_window : float, optional
            Distance in Angstroms from the theoretical centroid to include in the fit
        window_step : float, optional
            Step size in Angstroms to reduce the half-window size after each fitting iteration
        n_fit : int, optional
            Number of iterated fits to perform
        make_plot : bool, optional
            Whether to plot the absorption line with all fits overlaid.
        pltname : str, optional
            If not '', saves the plot to the supplied path with whatever extension you specify. 

        Returns
        -------
            tuple
                Tuple of 3 values: the mean fitted centroid across iterations, the propagated uncertainty reported by the fitting routine, and the standard deviation
                of the centroid across all iterations. We find the latter is a good estimator of statistical uncertainty in the fitted centroid. s

        '''
    
        window_step = -window_step
        in1 = bisect_left(wl,centroid-60)
        in2 = bisect_left(wl,centroid+60)
        cropped_wl = wl[in1:in2]
        cflux = flux[in1:in2]
        
        cmask = (cropped_wl < centroid - 50)+(cropped_wl > centroid + 50)

        p,cov = curve_fit(self.linear,cropped_wl[cmask][~np.isnan(cflux[cmask])],cflux[cmask][~np.isnan(cflux[cmask])])

        contcorr = cflux / self.linear(cropped_wl, *p)
        linemodel = lmfit.models.GaussianModel()
        params = linemodel.make_params()
        params['amplitude'].set(value = 2)
        params['center'].set(value = centroid)
        params['sigma'].set(value=2)
        centres = [];
        errors = [];
        
        if make_plot:
            plt.figure(figsize = (10, 5))
            #plt.title(str(centroid)+"$\AA$")
            plt.plot(cropped_wl,contcorr,'k')
            #plt.plot(cropped_wl, 1-linemodel.eval(params, x = cropped_wl))
        
        crop1 = bisect_left(cropped_wl,centroid - 15)
        crop2 = bisect_left(cropped_wl,centroid + 15)
        init_result = linemodel.fit(1-contcorr[crop1:crop2],params,x = cropped_wl[crop1:crop2],\
                            nan_policy = 'omit',method='lm')
        
        adaptive_centroid = init_result.params['center'].value
        
        for ii in range(n_fit):
            
            crop1 = bisect_left(cropped_wl, adaptive_centroid - half_window - ii*window_step)
            crop2 = bisect_left(cropped_wl, adaptive_centroid + half_window + ii*window_step)
            
            try:
                result = linemodel.fit(1-contcorr[crop1:crop2],params,x = cropped_wl[crop1:crop2],\
                            nan_policy = 'omit')
                if np.abs(result.params['center'].value - adaptive_centroid) > 5:
                    continue
            except ValueError:
    #             print('one fit failed. skipping...')
                continue
                
            if ii != 0:
                centres.append(result.params['center'].value)
                errors.append(result.params['center'].stderr)
            
            adaptive_centroid = result.params['center'].value
            
    #         print(len(cropped_wl[crop1:crop2]))
            if make_plot:
                xgrid = np.linspace(cropped_wl[crop1:crop2][0], cropped_wl[crop1:crop2][-1], 1000)
                
                plt.plot(xgrid,1-linemodel.eval(result.params, x = xgrid),\
                         'r', linewidth = 1, alpha = 0.7)
    #            plt.plot(cropped_wl[crop1:crop2],1-linemodel.eval(params,x=cropped_wl[crop1:crop2]),'k--')
        
        mean_centre = np.mean(centres)
        sigma_sample = np.std(centres)
        sigma_propagated = np.median(errors)
        total_sigma = np.sqrt(sigma_propagated**2 + sigma_sample**2) 
        
        if make_plot:
    #         gap = (50*1e-5)*centroid
    #         ticks = np.arange(centroid - gap*4, centroid + gap*4, gap)
    #         rvticks = ((ticks - centroid) / centroid)*3e5
    #         plt.xticks(ticks, np.round(rvticks).astype(int))
            plt.xlabel('Wavelength ($\mathrm{\AA}$)')
            plt.ylabel('Flux (Normalized)')
            plt.xlim(centroid - 35,centroid + 35)
            plt.axvline(centroid, color = 'k', linestyle = '--')
            plt.axvline(mean_centre, color = 'r', linestyle = '--')
            plt.tick_params(bottom=True, top=True, left=True, right=True)
            plt.text(0.65, 0.1, '$\mathrm{v_r}$ = %i ± %i km/s'\
                     %(speed_light*1e-3*(mean_centre - centroid)/centroid, speed_light*1e-3*(sigma_sample/centroid)),\
                    fontsize = 22, transform = plt.gca().transAxes)
            #plt.xlim(adaptive_centroid - 10, adaptive_centroid + 10)
            plt.minorticks_on()
            plt.tick_params(which='major', length=10, width=1, direction='in', top = True, right = True)
            plt.tick_params(which='minor', length=5, width=1, direction='in', top = True, right = True)
            plt.tight_layout()
        if None in errors or np.nan in errors:
            return mean_centre, np.nan, sigma_sample
            #print(np.isnan(np.array(errors)))
            
        return mean_centre, sigma_propagated, sigma_sample