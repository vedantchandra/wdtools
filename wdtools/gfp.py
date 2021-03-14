import numpy as np
import astropy
import scipy
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import glob
import pickle
from astropy import table 
import sys
from tqdm import tqdm
from scipy import interpolate
import os
import emcee
import corner
from scipy import optimize as opt
from bisect import bisect_left
import warnings
import lmfit

import tensorflow as tf
from tensorflow.python.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

from numpy.polynomial.polynomial import polyfit, polyval
from numpy.polynomial.chebyshev import chebfit, chebval
from scipy.interpolate import splev, splrep


path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
sys.path.append(dir_path)

from .spectrum import SpecTools

interp1d = interpolate.interp1d
Table = table.Table

halpha = 6564.61
hbeta = 4862.68
hgamma = 4341.68
hdelta = 4102.89
planck_h = 6.62607004e-34
speed_light = 299792458
k_B = 1.38064852e-23

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class GFP:

    """ Generative Fitting Pipeline. 

    """

    def __init__(self, resolution = 3, specclass = 'DA'):

        '''
        Initializes class. 

        Parameters
        ---------
        resolution : float
            Spectral resolution of the observed spectrum, in Angstroms (sigma). The synthetic spectra are convolved with this Gaussian kernel before fitting. 
        specclass : str ['DA', 'DB']
            Specifies whether to fit hydrogen-rich (DA) or helium-rich (DB) atmospheric models. DB atmospheric models are not publicly available at this time. 
        '''


        self.res_ang = resolution
        self.resolution = {};
        self.model = {};
        self.lamgrid = {};

        self.H_DA = 128
        self.lamgrid_DA = np.loadtxt(dir_path + '/models/neural_gen/DA_lamgrid.txt')
        self.model_DA = self.generator(self.H_DA, len(self.lamgrid_DA))
        self.model_DA.load_weights(dir_path + '/models/neural_gen/DA_normNN.h5')
        self.spec_min, self.spec_max = np.loadtxt(dir_path + '/models/neural_gen/DA_specsc.txt')
        pix_per_a = len(self.lamgrid_DA) / (self.lamgrid_DA[-1] - self.lamgrid_DA[0])
        self.resolution['DA'] = resolution * pix_per_a
        self.model['DA'] = self.model_DA
        self.lamgrid['DA'] = self.lamgrid_DA
        self.exclude_wl_default = np.array([3790, 3810, 3819, 3855,3863, 3920, 3930 , 4020 , 4040, 4180, 4215,
                       4490, 4662.68, 5062.68, 6314.61, 6814.61]);
        self.exclude_wl = self.exclude_wl_default

        self.cont_fixed = False
        self.rv_fixed = False
        self.rv = 0


        self.centroid_dict = dict(alpha = 6564.61, beta = 4862.68, gamma = 4341.68, delta = 4102.89, eps = 3971.20, h8 = 3890.12)
        self.distance_dict = dict(alpha = 250, beta = 250, gamma = 85, delta = 70, eps = 45, h8 = 30)

        if specclass == 'DB':
            raise Exception('DB models unfortunately under restricted access right now and unavailable in this package.')

        # try:
        #     self.H_DB = 128
        #     self.lamgrid_DB = pickle.load(open(dir_path + '/models/neural_gen/DB_lamgrid.p', 'rb'))
        #     self.model_DB = self.generator(self.H_DB, len(self.lamgrid_DB))
        #     self.model_DB.load_weights(dir_path + '/models/neural_gen/DB_normNN.h5')
        #     pix_per_a = len(self.lamgrid_DB) / (self.lamgrid_DB[-1] - self.lamgrid_DB[0])
        #     self.resolution['DB'] = resolution * pix_per_a
        #     self.model['DB'] = self.model_DB
        #     self.lamgrid['DB'] = self.lamgrid_DB
        # except:
        #     print('DB models not loaded.')

        if '+' not in specclass:
            self.isbinary = False;
            self.specclass = specclass;
        elif '+' in specclass:
            classes = specclass.split('+')
            self.specclass = [classes[0], classes[1]]
            self.isbinary = True
        
        self.sp = SpecTools()


    def label_sc(self, label_array):

        """
        Label scaler to transform Teff and logg to [0,1] interval based on preset bounds. 

        Parameters
        ---------
        label_array : array
            Unscaled array with Teff in the first column and logg in the second column
        Returns
        -------
            array
                Scaled array
        """
        teffs = label_array[:, 0];
        loggs = label_array[:, 1];
        teffs = (teffs - 5500) / (40000 - 5500)
        loggs = (loggs - 6.5) / (9.5 - 6.5)
        return np.vstack((teffs, loggs)).T

    def inv_label_sc(self, label_array):
        """
        Inverse label scaler to transform Teff and logg from [0,1] to original scale based on preset bounds. 

        Parameters
        ---------
        label_array : array
            Scaled array with Teff in the first column and logg in the second column
        Returns
        -------
            array
                Unscaled array
        """
        teffs = label_array[:, 0];
        loggs = label_array[:, 1];
        teffs = (teffs * (40000 - 5500)) + 5500
        loggs = (loggs * (9.5 - 6.5)) + 6.5
        return np.vstack((teffs, loggs)).T

    def spec_sc(self, spec):
        return (spec - self.spec_min) / (self.spec_max - self.spec_min)

    def inv_spec_sc(self, spec):
        return spec * (self.spec_max - self.spec_min) + self.spec_min

    def generator(self, H, n_pix):
        x = Input(shape=(2,))
        y = Dense(H,activation='relu',trainable = True)(x)
        y = Dense(H,activation='relu',trainable = True)(y)
        y = Dense(H,activation='relu',trainable = True)(y)
        out = Dense(n_pix,activation='linear',trainable = True)(y)
        
        model = Model(inputs = x, outputs = out)
        model.compile(optimizer = Adamax(), loss = 'mse', \
                      metrics = ['mae'])
        return model

    def synth_spectrum_sampler(self, wl, teff, logg, rv, specclass = None):
        """
        Generates synthetic spectra from labels using the neural network, translated by some radial velocity. These are _not_ interpolated onto the requested wavelength grid;
        The interpolation is performed only one time after the Gaussian convolution with the instrument resolution in `GFP.spectrum_sampler`. Use `GFP.spectrum_sampler` in most cases. 
        
        Parameters
        ----------
        wl : array
            Array of spectral wavelengths (included for completeness, not used by this function)
        teff : float
            Effective surface temperature of sampled spectrum
        logg : float
            log surface gravity of sampled spectrum (cgs)
        rv : float
            Radial velocity (redshift) of sampled spectrum in km/s
        specclass : str ['DA', 'DB']
            Whether to use hydrogen-rich (DA) or helium-rich (DB) atmospheric models. If None, uses default.  

        Returns
        -------
            array
                Synthetic spectrum with desired parameters, interpolated onto the supplied wavelength grid. 
        """

        if specclass is None:
            specclass = self.specclass;

        label = self.label_sc(np.asarray(np.stack((teff,logg)).reshape(1,-1)))
        synth = self.model[specclass].predict(label)[0]
        synth = 10**self.inv_spec_sc(synth)
        synth = self.sp.doppler_shift(self.lamgrid[specclass], synth, rv)
        synth =  (np.ravel(synth).astype('float64'))

        return synth

    def spectrum_sampler(self, wl, teff, logg, *polyargs, specclass = None):
        """
        Wrapper function that talks to the generative neural network in scaled units, and also performs the Gaussian convolution to instrument resolution. 
        
        Parameters
        ----------
        wl : array
            Array of spectral wavelengths on which to generate the synthetic spectrum
        teff : float
            Effective surface temperature of sampled spectrum
        logg : float
            log surface gravity of sampled spectrum (cgs)
        rv : float
            radial velocity (redshift) of sampled spectrum in km/s
        specclass : str ['DA', 'DB']
            Whether to use hydrogen-rich (DA) or helium-rich (DB) atmospheric models. If none, reverts to default. 
        Returns
        -------
            array
                Synthetic spectrum with desired parameters, interpolated onto the supplied wavelength grid and convolved with the instrument resolution. 
        """

        if self.rv_fixed:
            rv = self.rv
        else:
            rv = 0

        if specclass is None:
            specclass = self.specclass;
        synth = self.synth_spectrum_sampler(self.lamgrid[specclass], teff, logg, rv, specclass)
        synth = scipy.ndimage.gaussian_filter1d(synth, self.resolution[specclass])
        func = interp1d(self.lamgrid[specclass], synth, fill_value = np.nan, bounds_error = False)
        synth =  func(wl)

        # print(polyargs)

        if self.cont_fixed:

            # spline_pars = scipy.interpolate.splrep(wl[self.contmask], synth[self.contmask] / np.median(synth[self.contmask]),
            #                                     k = 3, s = self.smooth)
            # mod_smooth_cont = scipy.interpolate.splev(wl, spline_pars) * np.median(synth[self.contmask])
            # synth = synth / mod_smooth_cont

            dummy_ivar = 1 / np.repeat(0.001, len(wl))**2
            nanwhere = np.isnan(synth)
            dummy_ivar[nanwhere] = 0
            synth[nanwhere] = 0
            synth,_ = self.spline_norm_DA(wl, synth, dummy_ivar) # Use default KW from function
            synth[nanwhere] = np.nan

        if len(polyargs) > 0:
            synth = synth * chebval(2 * (wl - wl.min()) / (wl.max() - wl.min()) - 1, polyargs)

        return synth

    # def normalize_DA(self, wl, fl, ivar = None, cont_polyorder = 3, plot = False,  +++++++++ DO NOT USE, VERY SLOW +++++++
    #                         lines = ['alpha', 'beta', 'gamma', 'delta', 'eps', 'h8'],
    #                         maxfev = 500, smooth = 0, crop = True, return_soln = False):

    #     '''
    #     Continuum-normalization of a DA white dwarf spectrum by fitting a combination of the 
    #     model spectra and Chebyshev polynomials. 
        
    #     Parameters
    #     ---------
    #     wl : array
    #         Wavelength array of spectrum
    #     fl : array
    #         Flux array of spectrum
    #     ivar : array, optional
    #         Inverse variance array. If `None`, will return only the normalized wavelength and flux. 
    #     centroid : float
    #         The theoretical centroid of the absorption line that is being fitted, in wavelength units.
    #     distance : float
    #         Distance in Angstroms away from the line centroid to include in the fit. Should include 
    #         the entire absorption line wings with minimal continum. 
    #     make_plot : bool, optional
    #         Whether to plot the linear + Voigt fit. Use for debugging. 

    #     Returns
    #     -------
    #         tuple
    #             Tuple of cropped wavelength, cropped and normalized flux, and (if ivar is not None) 
    #             cropped and normalized inverse variance array. 

    #     '''

    #     self.smooth = smooth # Spline smoothing factor

    #     ## RESTRICT TO NN DOMAIN ##########

    #     wlbounds = self.lamgrid['DA'].min() + 5, self.lamgrid['DA'].max() - 5
    #     in1,in2 = bisect_left(wl, wlbounds[0]), bisect_left(wl, wlbounds[1])

    #     wl = wl[in1:in2]
    #     fl = fl[in1:in2]
    #     if ivar is not None: ivar = ivar[in1:in2]

    #     ##################################

    #     edges = [];
    #     breakpoints = [];

    #     mask = np.zeros(len(wl))
    #     for line in lines:
    #         c1 = self.centroid_dict[line] - self.distance_dict[line]
    #         c2 = self.centroid_dict[line] + self.distance_dict[line]
    #         mask += (wl > c1)*\
    #                 (wl < c2)
    #         edges.extend([c1, c2])
    #         breakpoints.extend([bisect_left(wl, c2), bisect_left(wl, c1)])

    #     self.breakpoints = breakpoints
    #     self.contmask = ~(mask > 0) # Select continuum pixels
    #     self.edges = edges
        
    #     init_prms = [12000, 8]
    #     bounds = [(6500, 39000), (6.5, 9.5)]
    #     init_prms.extend(chebfit(2*(wl - wl.min() / (wl.max())) - 1.0, fl, cont_polyorder))
    #     bounds.extend([(-np.Inf, np.Inf) for jj in range(cont_polyorder + 1)])

    #     self.mask = np.ones(len(wl)) > 0 ## FIT FULL SPECTRUM FOR CONTINUUM

    #     def residual(params):

    #         params = np.array(params)

    #         if params[0] < 6500 or params[0] > 39000 or params[1] < 6.5 or params[1] > 9.5:
    #             resid = 1e10
    #         else:
    #             resid = (fl - self.spectrum_sampler(wl, *params))[self.contmask]
    #         if ivar is None: 
    #             chi2 = (resid**2)
    #         else: 
    #             chi2 = (resid**2 * ivar[self.contmask])

    #         print(chi2.sum() / (len(chi2) - len(params)))
    #         return chi2

    #     # soln = scipy.optimize.minimize(residual, init_prms, method = 'lm', options = dict(maxfev = maxfev),
    #     #                     )

    #     params = lmfit.Parameters()
    #     params.add('teff', value = 12000, min = 6500, max = 40000)
    #     params.add('logg', value = 8, min = 6.5, max = 9.5)
    #     for ii in range(cont_polyorder + 1):
    #         params.add('c_' + str(ii), value = init_prms[ii + 2])

    #     print(params)
        
    #     res = lmfit.minimize(residual, params, method = 'leastsq', xtol = 1e-15, ftol = 1e-15)
    #     soln = np.array(res.params)

    #     self.mask = ~self.contmask

    #     model = self.spectrum_sampler(wl, *soln)
    #     spline_pars = scipy.interpolate.splrep(wl[self.contmask], model[self.contmask] / np.median(model[self.contmask]),
    #                                             k = 3, s = self.smooth)

    #     smooth_cont = scipy.interpolate.splev(wl, spline_pars) * np.median(model[self.contmask])

    #     if plot:
    #         plt.figure(figsize = (8, 6))
    #         plt.plot(wl, fl, 'k', label = 'Data')
    #         plt.plot(wl, model, 'r', label = 'Star + Continuum')
    #         plt.plot(wl[self.contmask], smooth_cont[self.contmask], 'go', label = 'Continuum')
    #         plt.xlabel('Wavelength')
    #         plt.ylabel('Flux')
    #         plt.legend()
    #         plt.show()

    #     fl = fl / smooth_cont

    #     if crop:
    #         wl = wl[(breakpoints[-1] - 5):(breakpoints[0] + 5)]
    #         fl = fl[(breakpoints[-1] - 5):(breakpoints[0] + 5)]
    #         self.contmask = self.contmask[(breakpoints[-1] - 5):(breakpoints[0] + 5)]
    #         self.mask = ~self.contmask

    #     ret = [wl, fl]

    #     if ivar is not None:
    #         ivar = ivar * smooth_cont**2
    #         if crop: ivar = ivar[(breakpoints[-1] - 5):(breakpoints[0] + 5)]
    #         ret.append(ivar)
        
    #     if return_soln:

    #         if soln[0] < 6500 or soln[0] > 39000 or soln[1] < 6.5 or soln[1] > 9.5:
    #             print('continuum stellar solution out of bounds of NN, do not trust') 
    #         ret.append(soln)

    #     return ret

    def spline_norm_DA(self, wl, fl, ivar, kwargs = dict(k = 3, sfac = 1, niter = 3), crop = None): # SETS DEFAULT KW

        if crop is not None:
            c1 = bisect_left(wl, crop[0])
            c2 = bisect_left(wl, crop[1])

            wl = wl[c1:c2]
            fl = fl[c1:c2]
            ivar = ivar[c1:c2]

        try:
            fl_norm, ivar_norm = self.sp.spline_norm(wl, fl, ivar, self.exclude_wl, **kwargs)
        except:
            print('spline normalization failed... returning NaNs')
            fl_norm, ivar_norm = np.nan*fl, np.nan*ivar

        if crop is not None:
            return wl, fl_norm, ivar_norm

        else:
            return fl_norm, ivar_norm

    def fit_spectrum(self, wl, fl, ivar = None, nwalkers = 25, burn = 25, ndraws = 25, make_plot = True, threads = 1, \
                    plot_trace = False, prior_teff = None, savename = None, isbinary = None, mask_threshold = 100,
                    DA = True, progress = True,
                    polyorder = 0, plot_init = False, plot_corner = False, plot_corner_full = False, verbose = True,
                    norm_kw = {}, mcmc = False,
                    lines = ['alpha', 'beta', 'gamma', 'delta', 'eps', 'h8'], maxfev = 1000, crop = (3600, 7500),
                    lmfit_kw = dict(method = 'leastsq', epsfcn = 0.1), rv_kw = dict(plot = False, distance = 50, nmodel = 2, edge = 10),
                            nteff = 3, fullspec = False, rv_line = 'alpha'):

        """
        Main fitting routine, takes a continuum-normalized spectrum and fits it with MCMC to recover steller labels. 
        
        Parameters
        ----------
        wl : array
            Array of observed spectral wavelengths
        fl : array
            Array of observed spectral fluxes, continuum-normalized. We recommend using the included `normalize_balmer` function from `wdtools.spectrum` to normalize DA spectra, 
            and the generic `continuum_normalize` function for DB spectra. 
        ivar : array
            Array of observed inverse-variance for uncertainty estimation. If this is not available, use `ivar = None` to infer a constant inverse variance mask using a second-order
            beta-sigma algorithm. In this case, since the errors are approximated, the chi-square likelihood may be inexact - treat returned uncertainties with caution. 
        nwalkers : int, optional
            Number of independent MCMC 'walkers' that will explore the parameter space
        burn : int, optional
            Number of steps to run and discard at the start of sampling to 'burn-in' the posterior parameter distribution. If intitializing from 
            a high-probability point, keep this value high to avoid under-estimating uncertainties. 
        ndraws : int, optional
            Number of 'production' steps after the burn-in. The final number of posterior samples will be nwalkers * ndraws.
        make_plot: bool, optional
            If True, produces a plot of the best-fit synthetic spectrum over the observed spectrum, as well as a corner plot of the fitted parameters. 
        threads : int, optional
            Number of threads for distributed sampling. 
        plot_trace: bool, optiomal
            If True, plots the trace of posterior samples of each parameter for the production steps. Can be used to visually determine the quality of mixing of
            the chains, and ascertain if a longer burn-in is required. 
        init : str, optional {'de', 'nm', 'unif', 'mle'}
            If 'de', the differential evolution algorithm is used to maximize the likelihood before MCMC sampling. It tries both hot and cold solutions, and choosing the one with the lowest chi^2.
            If 'unif', walkers are initialized uniformly in parameter space before the burn-in phase. If 'mle', there is a pre-burn phase with walkers initialized uniformly in 
            parameter space. The highest probability (lowest chi square) parameter set is taken as the MLE, and the main burn-in is initialized in a tight n-ball around this high
            probablity region. For most applications, we recommend using 'de'.
        prior_teff : tuple, optional
            Tuple of (mean, sigma) to define a Gaussian prior on the effective temperature parameter. This is especially useful if there is strong prior knowledge of temperature 
            from photometry. If not provided, a flat prior is used. 
        mleburn : int, optional
            Number of steps for the pre-burn phase to estimate the MLE.
        savename : str, optional
            If provided, the corner plot and best-fit plot will be saved as PDFs in the working folder. 
        normalize_DA : bool, optional
            If True, normalizes the Balmer lines on a DA spectrum before fitting. We recommend doing the normalization seperately, to ensure it's accurate. 
        lines : array, optional
            List of Balmer lines to normalize if `normalize_DA = True`. Defaults to all from H-alpha to H8. 
            
        Returns
        -------
            array
                Returns the fitted stellar labels along with a reduced chi-square statistic with the format: [(Teff, e_teff), (logg, e_logg), redchi]
        """

        self.cont_fixed = False
        self.rv_fixed = False

        nans = np.isnan(fl)

        if np.sum(nans) > 0:

            print('NaN detected in input... removing them...')

            wl = wl[~nans]
            fl = fl[~nans]
            ivar = ivar[~nans]

        if isbinary is None:
            isbinary == self.isbinary

        if ivar is None: # REPLACE THIS WITH YOUR OWN FUNCTION, REMOVE PYASL DEPENDENCE
            print('Please provide an IVAR array')
            # print('no inverse variance array provided, inferring ivar using the beta-sigma method. the chi-square likelihood will not be exact; treat returned uncertainties with caution!')
            # beq = pyasl.BSEqSamp()
            # std, _ = beq.betaSigma(fl, 1, 1)
            # ivar = np.repeat(1 / std**2, len(fl))

        prior_lows = [6500, 6.5]

        prior_highs = [40000, 9.5]

        nstarparams = 2

        def lnlike(prms):

            model = self.spectrum_sampler(wl, *prms)

            diff = (model - fl)**2 * ivar
            diff = diff[self.mask]
            chisq = np.sum(diff)

            if np.isnan(chisq):
                return -np.Inf
            lnlike = -0.5 * chisq
            #print(chisq / (np.sum(self.mask) - len(prms)))
            return lnlike

        def lnprior(prms):
            for jj in range(nstarparams):
                if prms[jj] < prior_lows[jj] or prms[jj] > prior_highs[jj]:
                    return -np.Inf

            if prior_teff is not None:
                mu,sigma = prior_teff
                return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(prms[0]-mu)**2/sigma**2
            else:
                return 0

        def lnprob(prms):
            lp = lnprior(prms)
            if not np.isfinite(lp):
                return -np.Inf
            return lp + lnlike(prms)


        param_names = ['$T_{eff}$', '$\log{g}$']
        param_names.extend(['$c_%i$' % ii for ii in range(polyorder + 1)])

        if verbose:
            print('fitting radial velocity...')
        
        self.rv, e_rv = self.sp.get_line_rv(wl, fl, ivar, self.centroid_dict[rv_line], **rv_kw)

        if verbose: 
            print('fitting continuum...')

        norm_kw['plot'] = plot_init

        outwl = (self.exclude_wl_default < np.min(wl)) & (self.exclude_wl_default > np.max(wl))
        self.exclude_wl = self.exclude_wl_default[~outwl]
        if len(self.exclude_wl) % 2 != 0:
            print('self.exclude_wl should have an even number of elements!')

        if DA:
            wl, fl, ivar = self.spline_norm_DA(wl, fl, ivar, kwargs = norm_kw, crop = crop)

        self.cont_fixed = True

        edges = [];
        mask = np.zeros(len(wl))

        for line in lines:
            wl1 = self.centroid_dict[line] - self.distance_dict[line]
            wl2 = self.centroid_dict[line] + self.distance_dict[line]
            c1 = bisect_left(wl, wl1)
            c2 = bisect_left(wl, wl2)

            edges.extend([wl2, wl1])

            mask[c1:c2] = 1

        self.mask = mask.astype(bool)
        edges = np.flip(edges)
        self.edges = edges
        if fullspec:
            self.mask = np.ones(len(fl)).astype(bool)

        ## ++++ TO DO +++++ IMPLEMENT LMFIT HERE, WITH GLOBAL OPTIMIZATION

        ######## LMFIT

        tscale = 10000
        lscale = 8

        params = lmfit.Parameters()
        params.add('teff', value = 12000 / tscale, min = 6500 / tscale, max = 40000 / tscale)
        params.add('logg', value = 8/lscale, min = 6.5/lscale, max = 9.5/lscale)

        for ii in range(polyorder):
            params.add('c_' + str(ii), value = 0, min = -1, max = 1)
            if ii == 0:
                params['c_0'].set(value = 1)


        def residual(params):
            params = np.array(params)
            params[0] = params[0] * tscale
            params[1] = params[1] * lscale
            model = self.spectrum_sampler(wl, *params)
            resid = fl - model
            chi = resid * np.sqrt(ivar)

            #print(np.sum(chi**2) / (np.sum(self.mask) - len(params)))

            return chi[self.mask]

        # res = lmfit.minimize(residual, params, **lmfit_kw)
        # params_arr = np.array(res.params)
        # teff = res.params['teff'].value * tscale
        # e_teff = res.params['teff'].stderr * tscale
        # logg = res.params['logg'].value * lscale
        # e_logg = res.params['logg'].stderr * lscale
        # print(teff, e_teff)
        # print(logg, e_logg)



        #self.rv, e_rv = self.sp.get_rv(wl, fl, ivar, wl, template)
        star_rv = self.rv
        print('Radial Velocity = %i ± %i km/s' % (self.rv, e_rv))
        self.rv_fixed = True

        if verbose:
            print('final optimization...')


        teffgrid = np.linspace(8000, 35000, nteff)

        chimin = 1e50

        for teff in teffgrid:
            print('initializing at teff = %i K' % teff)
            params['teff'].set(value = teff / tscale)
            res_i = lmfit.minimize(residual, params, **lmfit_kw)
            chi = np.sum(res_i.residual**2)
            if chi < chimin:
                res = res_i
                chimin = chi

        param_arr = np.array(res.params)
        teff = res.params['teff'].value * tscale
        logg = res.params['logg'].value * lscale
        if polyorder > 0:
            cheb_coef = np.array(res.params)[2:]
        redchi = np.sum(res.residual**2) / (np.sum(self.mask) - (2 + polyorder))

        have_stderr = False

        try:
            e_teff = res.params['teff'].stderr * tscale
            e_logg = res.params['logg'].stderr * lscale
            have_stderr = True
        except:
            e_teff = np.nan
            e_logg = np.nan
            print('no errors from lmfit...')

        try:
            e_coefs = [res.params['c_' + str(ii)].stderr for ii in range(polyorder)]
        except:
            e_coefs = 1e-2 * np.array(cheb_coef)

        # print(teff, e_teff)
        # print(logg, e_logg)


        #######################


        # if verbose:
        #     print('fitting cool solution...')
        # init_prms = [9000, 8]
        # if polyorder > 0:
        #     init_prms.extend(np.zeros(polyorder))
        #     #init_prms[nstarparams] = 1
        # nll = lambda *args: -lnprob(*args)
        # cool_soln = scipy.optimize.minimize(nll, init_prms, method = 'Nelder-Mead', options = dict(maxfev = maxfev))
        # cool_chi = -2 * lnprob(cool_soln.x) / (np.sum(self.mask) - 2)
        # if verbose:
        #     print('cool solution: T = %i K, logg = %.1f dex, redchi = %.2f' % (cool_soln.x[0], cool_soln.x[1], cool_chi))

        # if verbose:
        #     print('fitting warm solution...')
        # init_prms = [17000, 8]
        # if polyorder > 0:
        #     init_prms.extend(np.zeros(polyorder))
        #     #init_prms[nstarparams] = 1
        # nll = lambda *args: -lnprob(*args) 
        # warm_soln = scipy.optimize.minimize(nll, init_prms, method = 'Nelder-Mead', options = dict(maxfev = maxfev))
        # warm_chi = -2 * lnprob(warm_soln.x) / (np.sum(self.mask) - 2)
        # if verbose:
        #     print('warm solution: T = %i K, logg = %.1f dex, redchi = %.2f' % (warm_soln.x[0], warm_soln.x[1], warm_chi))

        # if cool_chi < warm_chi:
        #     soln = cool_soln.x
        #     chi = cool_chi
        #     tstr = 'cool'
        # else:
        #     soln = warm_soln.x
        #     chi = warm_chi
        #     tstr = 'warm'


        # if verbose:
        #     print('fitting radial velocity...')
        # template = self.spectrum_sampler(wl, *soln[0:2]) ## FIX THIS!! ----------------------------------------
        # self.rv, e_rv = self.sp.get_rv(wl, fl, ivar, wl, template)
        # star_rv = self.rv
        # print('Radial Velocity = %i ± %i km/s' % (self.rv, e_rv))
        # self.rv_fixed = True

        # if verbose:
        #     print('final optimization...')

        # nll = lambda *args: -lnprob(*args) 
        # soln = scipy.optimize.minimize(nll, soln, method = 'Nelder-Mead', options = dict(maxfev = maxfev))

        mle = [teff, logg]
        stds = [e_teff, e_logg]

        if polyorder > 0:
            mle.extend(cheb_coef)
            stds.extend(e_coefs)

        if mcmc:

            ndim = len(mle)
            
            sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob, threads = threads)

            pos0 = np.zeros((nwalkers,ndim))

            if have_stderr and polyorder == 0: # do not trust covariances when polyorder > 0
                sigmas = stds # USE ERR FROM LMFIT
            else:
                sigmas = np.abs(1e-2 * np.array(mle)) # USE 1% ERROR

            init = mle

            print(mle)
            print(sigmas)

            for jj in range(ndim):
                    pos0[:,jj] = (init[jj] + sigmas[jj]*np.random.normal(size = nwalkers))

            if verbose:
                print('burning in chains...')

            b = sampler.run_mcmc(pos0, burn, progress = progress)

            sampler.reset()

            if verbose:
                print('sampling posterior...')
            b = sampler.run_mcmc(b.coords, ndraws, progress = progress)

            if plot_trace:
                f, axs = plt.subplots(ndim, 1, figsize = (10, 6))
                for jj in range(ndim):
                    axs[jj].plot(sampler.chain[:,:,jj].T, alpha = 0.3, color = 'k');
                    plt.ylabel(param_names[jj])
                plt.xlabel('steps')
                plt.show()

            lnprobs = sampler.get_log_prob(flat = True)
            medians = np.median(sampler.flatchain, 0)
            mle = sampler.flatchain[np.argmax(lnprobs)]
            redchi = -2 * np.max(lnprobs) / (len(wl) - ndim)
            stds = np.std(sampler.flatchain, 0)
            self.flatchain = sampler.flatchain

            
            if mle[0] < 7000 or mle[0] > 38000:
                print('temperature is near bound of the model grid! exercise caution with this result')
            if mle[1] < 6.7 or mle[1] > 9.3:
                print('logg is near bound of the model grid! exercise caution with this result')

            if plot_corner:
                f = corner.corner(sampler.flatchain[:, :nstarparams], labels = param_names[:nstarparams], \
                         label_kwargs = dict(fontsize =  12), quantiles = (0.16, 0.5, 0.84),
                         show_titles = True, title_kwargs = dict(fontsize = 12))

                for ax in f.get_axes(): 
                  ax.tick_params(axis='both', labelsize=12)
                if savename is not None:
                    plt.savefig(savename + '_corner.jpg', bbox_inches = 'tight', dpi = 100)
                plt.show()

            if plot_corner_full:

                f = corner.corner(sampler.flatchain, labels = param_names, \
                         label_kwargs = dict(fontsize =  12), quantiles = (0.16, 0.5, 0.84),
                         show_titles = False)

                for ax in f.get_axes(): 
                  ax.tick_params(axis='both', labelsize=12)


        fit_fl = self.spectrum_sampler(wl, *mle)


        if make_plot:
            #fig,ax = plt.subplots(ndim, ndim, figsize = (15,15))

            if fullspec:
                plt.figure(figsize = (10, 8))
                plt.plot(wl, fl, 'k')
                plt.plot(wl, fit_fl, 'r')
                plt.ylabel('Normalized Flux')
                plt.xlabel('Wavelength')

                plt.ylim(0, 1.5)

                plt.text(0.97, 0.25, '$T_{\mathrm{eff}} = %.0f \pm %.0f\ K$' % (mle[0], stds[0]),
                 transform = plt.gca().transAxes, fontsize = 15, ha = 'right')
        
                plt.text(0.97, 0.15, '$\log{g} = %.2f \pm %.2f $' % (mle[1], stds[1]),
                         transform = plt.gca().transAxes, fontsize = 15, ha = 'right')
                 
                plt.text(0.97, 0.05, '$\chi_r^2$ = %.2f' % (redchi),
                         transform = plt.gca().transAxes, fontsize = 15, ha = 'right')


            else:
                plt.figure(figsize = (10, 10))
                breakpoints = [];
                for kk in range(len(self.edges)):
                    if (kk + 1)%2 == 0:
                        continue
                    breakpoints.append(bisect_left(wl, self.edges[kk]))
                    breakpoints.append(bisect_left(wl, self.edges[kk+1]))
                # print(breakpoints)
                for kk in range(len(breakpoints)):
                    if (kk + 1)%2 == 0:
                        continue
                    wl_seg = wl[breakpoints[kk]:breakpoints[kk+1]]
                    fl_seg = fl[breakpoints[kk]:breakpoints[kk+1]]
                    fit_fl_seg = fit_fl[breakpoints[kk]:breakpoints[kk+1]]
                    peak = int(len(wl_seg)/2)
                    delta_wl = wl_seg - wl_seg[peak]
                    plt.plot(delta_wl, 1 + fl_seg - 0.25 * kk, 'k')
                    plt.plot(delta_wl, 1 + fit_fl_seg - 0.25 * kk, 'r')
                plt.xlabel(r'$\mathrm{\Delta \lambda}\ (\mathrm{\AA})$')
                plt.ylabel('Normalized Flux')

                plt.text(0.97, 0.8, '$T_{\mathrm{eff}} = %.0f \pm %.0f\ K$' % (mle[0], stds[0]),
                 transform = plt.gca().transAxes, fontsize = 15, ha = 'right')
        
                plt.text(0.97, 0.7, '$\log{g} = %.2f \pm %.2f $' % (mle[1], stds[1]),
                         transform = plt.gca().transAxes, fontsize = 15, ha = 'right')
                 
                plt.text(0.97, 0.6, '$\chi_r^2$ = %.2f' % (redchi),
                         transform = plt.gca().transAxes, fontsize = 15, ha = 'right')

            #     if savename is not None:
            #         plt.savefig(savename + '_fit.pdf', bbox_inches = 'tight')

            if savename is not None:
                plt.savefig(savename + '_fit.jpg', bbox_inches = 'tight', dpi = 100)
            plt.show()

        self.exclude_wl = self.exclude_wl_default
        self.cont_fixed = False
        self.rv = 0 # RESET THESE PARAMETERS

        mle = mle#[0:2]
        stds = stds#[0:2]

        mle = np.append(mle, star_rv)
        stds = np.append(stds, e_rv)

        return mle, stds, redchi

    def blackbody(self, wl, teff):
        wl = wl * 1e-10
        num = 2 * planck_h * speed_light**2
        denom = wl**5 * (np.exp((planck_h * speed_light) / (wl * k_B * teff)) - 1)
        return num/denom
    
if __name__ == '__main__':
    
    gfp = GFP(resolution = 3)
    wl = np.linspace(4000, 8000, 4000)
    fl = gfp.spectrum_sampler(wl, 6500, 6.58, 0)
    
    plt.plot(wl, fl)
    
    result = gfp.fit_spectrum(wl, fl,  burn = 1, ndraws = 1, 
                              )
