import warnings
import numpy as np
from bisect import bisect_left
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pickle
import os
import scipy
import lmfit
from lmfit.models import LinearModel, VoigtModel, GaussianModel
import numpy.polynomial.polynomial as poly
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from scipy.interpolate import splev,splrep,LSQUnivariateSpline

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


    def continuum_normalize(self, wl, fl, ivar = None):
        '''
        Continuum-normalization with smoothing splines that avoid a pre-made list of absorption 
        lines for DA and DB spectra. To normalize spectra that only have Balmer lines (DA),
         we recommend using the `normalize_balmer` function instead. Also crops the spectrum 
         to the 3700 - 7000 Angstrom range. 
        
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
                Tuple of cropped wavelength, cropped and normalized flux, and (if ivar is not None)
                 cropped and normalized inverse variance array. 

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

        if ivar is not None:
            ivar_norm = ivar * scipy.interpolate.splev(wl, spl)**2
            return wl, norm, ivar_norm
        elif ivar is None:
            return wl, norm

    def normalize_line(self, wl, fl, ivar, centroid, distance, make_plot = False, return_centre = False):

        '''
        Continuum-normalization of a single absorption line by fitting a linear model added 
        to a Voigt profile to the spectrum, and dividing out the linear model. 
        
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
            Distance in Angstroms away from the line centroid to include in the fit. Should include 
            the entire absorption line wings with minimal continum. 
        make_plot : bool, optional
            Whether to plot the linear + Voigt fit. Use for debugging. 

        Returns
        -------
            tuple
                Tuple of cropped wavelength, cropped and normalized flux, and (if ivar is not None) 
                cropped and normalized inverse variance array. 

        '''

        self.params['v_center'].set(value = centroid)

        crop1 = bisect_left(wl, centroid - distance)
        crop2 = bisect_left(wl, centroid + distance)

        cropped_wl = wl[crop1:crop2]
        cropped_fl = fl[crop1:crop2]

        #cropped_fl = cropped_fl / np.nanmax(cropped_fl)


        try:
            res = self.cm.fit(cropped_fl, self.params, x = cropped_wl, nan_policy = 'omit')
        except TypeError:
            print('profile fit failed. ensure all lines passed to normalize_balmer are present on the spectrum!')
            raise Exception('profile fit failed. ensure all lines passed to normalize_balmer are present on the spectrum!')

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
        elif return_centre:
            return cropped_wl, fl_normalized, res.params['v_center']
        else:
            return cropped_wl, fl_normalized

    def normalize_balmer(self, wl, fl, ivar = None, lines = ['alpha', 'beta', 'gamma', 'delta'], \
                         skylines = False, make_plot = False, make_subplot = False, make_stackedplot = False, \
            centroid_dict = dict(alpha = 6564.61, beta = 4862.68, gamma = 4341.68, delta = 4102.89, eps = 3971.20, h8 = 3890.12),
            distance_dict = dict(alpha = 300, beta = 200, gamma = 120, delta = 75, eps = 50, h8 = 25), sky_fill = np.nan):
        

        '''
        Continuum-normalization of any spectrum by fitting each line individually. 

        Fits every absorption line by fitting a linear model added to a Voigt profile to 
        the spectrum, and dividing out the linear model. 
        All normalized lines are concatenated and returned. For statistical and plotting 
        purposes, two adjacent lines should not have overlapping regions (governed by the `distance_dict`). 
        
        Parameters
        ---------
        wl : array
            Wavelength array of spectrum
        fl : array
            Flux array of spectrum
        ivar : array, optional
            Inverse variance array. If `None`, will return only the normalized wavelength and flux. 
        lines : array-like, optional
            Array of which Balmer lines to include in the fit. Can be any combination of 
            ['alpha', 'beta', 'gamma', 'delta', 'eps', 'h8']
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
        sky_fill : float
            What value to replace the telluric features with on the normalized spectrum. Defaults to np.nan. 

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
                if make_subplot:
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

            plt.xlabel(r'$\mathrm{\Delta \lambda}\ (\mathrm{\AA})$')
            plt.ylabel('Normalized Flux')
            plt.show()

        if ivar is not None:
            return wl_normalized, fl_normalized, ivar_normalized
        else:
            return wl_normalized, fl_normalized

    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]


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
                 pltname = '', debug = False, normalize = True):

        '''
        Statistical inference of spectral redshift by iteratively fitting Voigt profiles to cropped windows around the line centroid. 

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
                of the centroid across all iterations. We find the latter is a good estimator of statistical uncertainty in the fitted centroid.

        '''
    
        window_step = -window_step
        in1 = bisect_left(wl,centroid-100)
        in2 = bisect_left(wl,centroid+100)
        cropped_wl = wl[in1:in2]
        cflux = flux[in1:in2]

        if normalize:
        
            cmask = (cropped_wl < centroid - 50)+(cropped_wl > centroid + 50)

            p,cov = curve_fit(self.linear,cropped_wl[cmask][~np.isnan(cflux[cmask])],cflux[cmask][~np.isnan(cflux[cmask])])

            contcorr = cflux / self.linear(cropped_wl, *p)
        else:
            contcorr = cflux

        linemodel = lmfit.models.GaussianModel()
        params = linemodel.make_params()
        params['amplitude'].set(value = 2)
        params['center'].set(value = centroid)
        params['sigma'].set(value=5)
        centres = [];
        errors = [];
        
        if make_plot:
            plt.figure(figsize = (10, 5))
            #plt.title(str(centroid)+"$\AA$")
            plt.plot(cropped_wl,contcorr,'k')
            #plt.plot(cropped_wl, 1-linemodel.eval(params, x = cropped_wl))
        
        crop1 = bisect_left(cropped_wl,centroid - half_window)
        crop2 = bisect_left(cropped_wl,centroid + half_window)
        init_result = linemodel.fit(1-contcorr[crop1:crop2],params,x = cropped_wl[crop1:crop2],\
                            nan_policy = 'omit', method='leastsq')

        if debug:
            plt.figure()
            plt.plot(cropped_wl[crop1:crop2], 1-contcorr[crop1:crop2])
            plt.plot(cropped_wl[crop1:crop2], linemodel.eval(params, x = cropped_wl[crop1:crop2]))
            plt.plot(cropped_wl[crop1:crop2], linemodel.eval(init_result.params, x = cropped_wl[crop1:crop2]))
            plt.show()

        #plt.plot(cropped_wl, init_result.eval(init_result.params, x = cropped_wl))
        
        adaptive_centroid = init_result.params['center'].value
        
        for ii in range(n_fit):
            
            crop1 = bisect_left(cropped_wl, adaptive_centroid - half_window - ii*window_step)
            crop2 = bisect_left(cropped_wl, adaptive_centroid + half_window + ii*window_step)
            try:

                result = linemodel.fit(1-contcorr[crop1:crop2],params,x = cropped_wl[crop1:crop2],\
                            nan_policy = 'omit', method = 'leastsq')
                if np.abs(result.params['center'].value - adaptive_centroid) > 5:
                    continue
            except ValueError:
                print('one fit failed. skipping...')
                continue
                
            if ii != 0:
                centres.append(result.params['center'].value)
                errors.append(result.params['center'].stderr)
            
            adaptive_centroid = result.params['center'].value
            
    #        print(len(cropped_wl[crop1:crop2]))
            if make_plot:
                xgrid = np.linspace(cropped_wl[crop1:crop2][0], cropped_wl[crop1:crop2][-1], 1000)
                
                plt.plot(xgrid,1-linemodel.eval(result.params, x = xgrid),\
                         'r', linewidth = 1, alpha = 0.7)
    #            plt.plot(cropped_wl[crop1:crop2],1-linemodel.eval(params,x=cropped_wl[crop1:crop2]),'k--')
        mean_centre = np.mean(centres)
        sigma_sample = np.std(centres)
        if len(centres) == 0:
            centres = [np.nan]
            errors = [np.nan]
            print('caution, none of the iterated fits were succesful')
        final_centre = centres[-1]

        if None in errors or np.nan in errors:
            errors = [np.nan]

        sigma_propagated = np.nanmedian(errors)
        sigma_final_centre = errors[-1]
        total_sigma = np.sqrt(sigma_propagated**2 + sigma_sample**2) 

        
        if make_plot:
    #         gap = (50*1e-5)*centroid
    #         ticks = np.arange(centroid - gap*4, centroid + gap*4, gap)
    #         rvticks = ((ticks - centroid) / centroid)*3e5
    #         plt.xticks(ticks, np.round(rvticks).astype(int))
            plt.xlabel('Wavelength ($\mathrm{\AA}$)')
            plt.ylabel('Flux (Normalized)')
            plt.xlim(centroid - 35,centroid + 35)
            #plt.axvline(centroid, color = 'k', linestyle = '--')
            plt.axvline(mean_centre, color = 'r', linestyle = '--')
            plt.tick_params(bottom=True, top=True, left=True, right=True)
            plt.minorticks_on()
            plt.tick_params(which='major', length=10, width=1, direction='in', top = True, right = True)
            plt.tick_params(which='minor', length=5, width=1, direction='in', top = True, right = True)
            plt.xlabel('Wavelength ($\mathrm{\AA}$)')
            plt.ylabel('Normalized Flux')
            plt.tight_layout()
            #print(np.isnan(np.array(errors)))
            
        return mean_centre, final_centre, sigma_final_centre, sigma_propagated, sigma_sample

    def doppler_shift(self, wl, fl, dv):
        c = 2.99792458e5
        df = np.sqrt((1 - dv/c)/(1 + dv/c)) 
        new_wl = wl * df
        new_fl = np.interp(new_wl, wl, fl)
        return new_fl

    def xcorr_rv(self, wl, fl, temp_wl, temp_fl, init_rv = 0, rv_range = 500, npoint = None):
        if npoint is None:
            npoint = int(2 * rv_range)
        rvgrid = np.linspace(init_rv - rv_range, init_rv + rv_range, npoint)
        cc = np.zeros(npoint)
        for ii,rv in enumerate(rvgrid):
            shift_model = self.doppler_shift(temp_wl, temp_fl, rv)
            corr = np.corrcoef(fl, shift_model)[1, 0]
            #corr = -np.sum((fl - shift_model)**2) # MINIMIZE LSQ DIFF. MAYBE PROPAGATE IVAR HERE?
            cc[ii] = corr
        return rvgrid, cc

    def quad_max(self, rv, cc):
        maxpt = np.argmax(cc)
        max_rv = rv[maxpt]
        # in1 = maxpt - 5
        # in2 = maxpt + 5
        # rv,cc = rv[in1:in2], cc[in1:in2]
        # pfit = np.polynomial.polynomial.polyfit(rv, cc, 2)
        # max_rv = - pfit[1] / (2 * pfit[2])

        # plt.plot(rv, cc)
        # plt.axvline(max_rv)
        # plt.show()
        return max_rv

    def get_one_rv(self, wl, fl, temp_wl, temp_fl, r1 = 1000, p1 = 100, r2 = 100, p2 = 100, plot = False): # IMPLEMENT UNCERTAINTIES AT SPECTRUM LEVEL
        rv, cc = self.xcorr_rv(wl, fl, temp_wl, temp_fl, init_rv = 0, rv_range = r1, npoint = p1)

        # if plot:
        #     plt.plot(rv, cc, color = 'k', alpha = 0.1)

        rv_guess = self.quad_max(rv, cc)
        rv, cc = self.xcorr_rv(wl, fl, temp_wl, temp_fl, init_rv = rv_guess, rv_range = r2, npoint = p2)
        if plot:
            plt.plot(rv, cc, color = 'k', alpha = 0.1)
        return self.quad_max(rv, cc)

    def get_rv(self, wl, fl, ivar, temp_wl, temp_fl, N = 100, kwargs = {}):

        nans = np.isnan(fl) + np.isnan(ivar) + np.isnan(temp_fl)

        if np.sum(nans) > 0:
            print("NaNs detected in RV routine... removing them...")
            wl = wl[~nans]
            fl = fl[~nans]
            ivar = ivar[~nans]
            temp_wl = temp_wl[~nans]
            temp_fl = temp_fl[~nans]

        rv = self.get_one_rv(wl, fl, temp_wl, temp_fl, **kwargs)

        rvs = [];
        for ii in range(N):
            fl_i = fl + np.sqrt(1/(ivar + 1e-10)) * np.random.normal(size = len(fl))
            rvs.append(self.get_one_rv(wl, fl_i, temp_wl, temp_fl, **kwargs))
        return rv, (np.quantile(rvs, 0.84) - np.quantile(rvs, 0.16)) / 2

    def spline_norm(self, wl, fl, ivar, exclude_wl, sfac = 1, k = 3, plot = False, niter = 0):
        
        fl_norm = fl / np.nanmedian(fl)
        nivar = ivar * np.nanmedian(fl)**2
        x = (wl - np.min(wl)) / (np.max(wl) - np.min(wl))
        
        cont_mask = np.ones(len(wl))

        for ii in range(len(exclude_wl) - 1):
            if ii % 2 != 0:
                continue
            c1 = bisect_left(wl, exclude_wl[ii])
            c2 = bisect_left(wl, exclude_wl[ii + 1])

            cont_mask[c1:c2] = 0
        cont_mask = cont_mask.astype(bool)
        
        s = (len(x) - np.sqrt(2 * len(x))) * sfac # SFAC to scale rule of thumb smoothing
            
        spline = splev(x, splrep(x[cont_mask], fl_norm[cont_mask], k = k, s = s, w = np.sqrt(nivar[cont_mask])))

        # t = [];

        # for ii,wl in enumerate(exclude_wl):
        #     if ii % 2 == 0:
        #         t.append(wl - 5)
        #     else:
        #         t.append(wl  5)

        # spline = LSQUnivariateSpline(x[cont_mask], fl_norm[cont_mask], t = t, k = 3)(x)

        fl_prev = fl_norm
        fl_norm = fl_norm / spline
        nivar = nivar * spline**2
        
        for n in range(niter): # Repeat spline fit with reduced smoothing. don't use without testing
            fl_prev = fl_norm
            spline = splev(x, splrep(x[cont_mask], fl_norm[cont_mask], k = k, s = s - 0.1 * n * s, 
                                     w = np.sqrt(nivar[cont_mask])))
            fl_norm = fl_norm / spline
            nivar = nivar * spline**2

        
        if plot:
            plt.figure(figsize = (12, 10))
            plt.subplot(211)
            plt.plot(wl, fl_prev, color = 'k')
            plt.plot(wl, spline, color = 'r')
            plt.title('Continuum Fit (iteration %i/%i)' % (niter + 1, niter + 1))
            up = np.quantile(fl_prev, 0.7) + 1
            low = np.quantile(fl_prev, 0.2) - 1
            plt.ylim(low,)

            plt.subplot(212)
            plt.plot(wl, fl_norm, color = 'k')
            plt.vlines(exclude_wl, ymin = fl_norm.min(), ymax = fl_norm.max(), color = 'k', 
                       linestyle = '--', lw = 1, zorder = 10)
            plt.title('Normalized Spectrum')
            plt.ylim(0, 1.5)
            plt.show()
            
        return fl_norm, nivar


    def get_line_rv(self, wl, fl, ivar, centroid, template = None, return_template = False, distance = 50, edge = 10, nmodel = 2, plot = False, rv_kwargs = {},
                        init_width = 20, init_amp = 5):

        c1 = bisect_left(wl, centroid - distance)
        c2 = bisect_left(wl, centroid + distance)
        lower = centroid - distance + edge
        upper = centroid + distance - edge

        cwl, cfl, civar = wl[c1:c2], fl[c1:c2], ivar[c1:c2]

        edgemask = (cwl < lower) + (cwl > upper)

        line = np.polyval(np.polyfit(cwl[edgemask], cfl[edgemask], 1), cwl)

        nfl = 1 - cfl / line
        nivar = civar * line**2

        vel = 299792458 * 1e-3 * (cwl - centroid) / centroid

        if plot:
            plt.plot(vel, 1-nfl, 'k')

        if template is None:

            for ii in range(nmodel):
                if ii == 0:
                    model = VoigtModel(prefix = 'g' + str(ii) + '_')
                else:
                    model += VoigtModel(prefix = 'g' + str(ii) + '_')

            params = model.make_params()

            init_center = centroid

            #print(init_center)

            for ii in range(nmodel):
                params['g' + str(ii) + '_center'].set(value = init_center, vary = False, expr = 'g0_center')
                params['g' + str(ii) + '_sigma'].set(value = init_width, vary = True)
                params['g' + str(ii) + '_amplitude'].set(value = init_amp/nmodel, vary = True)

            params['g0_center'].set(value = init_center, vary = True, expr = None)

            res = model.fit(nfl, params, x = cwl, method = 'nelder')

            res.params['g0_center'].set(value = centroid)

            template = model.eval(res.params, x = cwl)

            if plot:
                plt.plot(vel, 1-model.eval(params, x = cwl), 'b')

        rv, e_rv = self.get_rv(cwl, nfl, nivar, cwl, template, **rv_kwargs)

        if plot:
            plt.plot(vel, 1 - self.doppler_shift(cwl, template, rv), 'r')
            plt.xlabel('Relative Velocity')
            plt.ylabel('Normalized Flux')
            plt.title('RV = %.1f ± %.1f km/s' % (rv, e_rv))
            plt.axvline(0, color = 'k', linestyle = '--')
            plt.axvline(rv, color = 'r', linestyle = '--')

        if return_template:
            return rv, e_rv, template
        else:
            return rv, e_rv