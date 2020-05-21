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
from PyAstronomy import pyasl
import emcee
import corner
import keras
from scipy import optimize as opt
from bisect import bisect_left
import warnings
import lmfit

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

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
plt.rcParams.update({'font.size': 16})

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
            Spectral resolution of the observed spectrum, in Angstroms. The synthetic spectra are convolved with this Gaussian kernel before fitting. 
        specclass : str ['DA', 'DB']
            Specifies whether to fit hydrogen-rich (DA) or helium-rich (DB) atmospheric models. 
        '''


        self.res_ang = resolution
        self.resolution = {};
        self.model = {};
        self.lamgrid = {};

        self.H_DA = 128
        self.lamgrid_DA = pickle.load(open(dir_path + '/models/neural_gen/DA_lamgrid.p', 'rb'))
        self.model_DA = self.generator(self.H_DA, len(self.lamgrid_DA))
        self.model_DA.load_weights(dir_path + '/models/neural_gen/DA_normNN.h5')
        pix_per_a = len(self.lamgrid_DA) / (self.lamgrid_DA[-1] - self.lamgrid_DA[0])
        self.resolution['DA'] = resolution * pix_per_a
        self.model['DA'] = self.model_DA
        self.lamgrid['DA'] = self.lamgrid_DA

        if specclass == 'DB':
            print('DB models unfortunately under restricted access right now and unavailable in this package.')

        try:
            self.H_DB = 128
            self.lamgrid_DB = pickle.load(open(dir_path + '/models/neural_gen/DB_lamgrid.p', 'rb'))
            self.model_DB = self.generator(self.H_DB, len(self.lamgrid_DB))
            self.model_DB.load_weights(dir_path + '/models/neural_gen/DB_normNN.h5')
            pix_per_a = len(self.lamgrid_DB) / (self.lamgrid_DB[-1] - self.lamgrid_DB[0])
            self.resolution['DB'] = resolution * pix_per_a
            self.model['DB'] = self.model_DB
            self.lamgrid['DB'] = self.lamgrid_DB
        except:
            print('DB models not loaded.')

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
        teffs = (teffs - 2500) / (100000 - 2500)
        loggs = (loggs - 5) / (10 - 5)
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
        teffs = (teffs * (100000 - 2500)) + 2500
        loggs = (loggs * (10 - 5)) + 5
        return np.vstack((teffs, loggs)).T

    def generator(self, H, n_pix):
        """
        Basic 2-layer neural network to generate synthetic spectra.  
        
        Parameters
        ----------
        H : int
            Number of neurons in each hidden layer
        n_pix : int
            Number of pixels on the synthetic spectrum = number of neurons on the last layer. 
        Returns
        -------
            keras `Model`
                Keras neural network model instance
        """
        x = keras.layers.Input(shape=(2,))
        y = keras.layers.Dense(H,activation='relu',trainable = True)(x)
        y = keras.layers.Dense(H,activation='relu',trainable = True)(y)
        y = keras.layers.Dense(H,activation='relu',trainable = True)(y)
        out = keras.layers.Dense(n_pix,activation='linear',trainable = True)(y)
        
        model = keras.models.Model(inputs = x, outputs = out)
        model.compile(optimizer = keras.optimizers.Adamax(), loss = 'mse', \
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
            Whether to use hydrogen-rich (DA) or helium-rich (DB) atmospheric models.

        Returns
        -------
            array
                Synthetic spectrum with desired parameters, interpolated onto the supplied wavelength grid. 
        """

        if specclass is None:
            specclass = self.specclass;

        label = self.label_sc(np.asarray(np.stack((teff,logg)).reshape(1,-1)))
        synth = pyasl.dopplerShift(self.lamgrid[specclass],np.ravel(
                        (
                                self.model[specclass].predict(label))[0]
                        ), rv
                    )[0]
        synth =  (np.ravel(synth).astype('float64'))

        return synth

    def spectrum_sampler(self, wl, teff, logg, rv, specclass = None):
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
            Whether to use hydrogen-rich (DA) or helium-rich (DB) atmospheric models.
        Returns
        -------
            array
                Synthetic spectrum with desired parameters, interpolated onto the supplied wavelength grid and convolved with the instrument resolution. 
        """

        if specclass is None:
            specclass = self.specclass;
        synth = self.synth_spectrum_sampler(self.lamgrid[specclass], teff, logg, rv, specclass)
        synth = scipy.ndimage.gaussian_filter1d(synth, self.resolution[specclass])
        func = interp1d(self.lamgrid[specclass], synth, fill_value = np.nan, bounds_error = False)
        return func(wl)

    def binary_sampler(self, wl, teff_1, logg_1, rv_1, teff_2, logg_2, rv_2, lf = 1, specclass = None):

        if specclass is None:
            specclass = self.specclass;

        if isinstance(specclass,str):
            specclass = [specclass, specclass]

        bin_lamgrid = np.linspace(3500, 7000, 14000)

        normfl_1 = self.synth_spectrum_sampler(self.lamgrid[specclass[0]], teff_1, logg_1, rv_1, specclass[0])
        func1 = interp1d(self.lamgrid[specclass[0]], normfl_1, fill_value = 1, bounds_error = False)
        normfl_1 = func1(bin_lamgrid)

        normfl_2 = self.synth_spectrum_sampler(self.lamgrid[specclass[1]], teff_2, logg_2, rv_2, specclass[1])
        func2 = interp1d(self.lamgrid[specclass[1]], normfl_2, fill_value = 1, bounds_error = False)
        normfl_2 = func2(bin_lamgrid)

        continuum_1 = self.blackbody(bin_lamgrid, teff_1) * 1e-14
        continuum_2 = self.blackbody(bin_lamgrid, teff_2) * 1e-14

        fullspec_1 = normfl_1 * continuum_1 
        fullspec_2 = normfl_2 * continuum_2

        summed_spectrum = (fullspec_1 + lf * fullspec_2) # FL RATIO

        # _,finalspec = self.sp.normalize_balmer(self.lamgrid[specclass], summed_spectrum,
        #                     lines = ['alpha', 'beta', 'gamma', 'delta', 'eps','h8'],
        #                                  skylines = False, make_subplot = False)

        bin_lamgrid, finalspec = self.sp.continuum_normalize(bin_lamgrid, summed_spectrum)
        
        resolution = self.res_ang * (bin_lamgrid[1] - bin_lamgrid[0])

        synth = scipy.ndimage.gaussian_filter1d(finalspec, resolution)
        func = interp1d(bin_lamgrid, synth, fill_value = np.nan, bounds_error = False)
        
        return func(wl)

    def fit_spectrum(self, wl, fl, ivar = None, nwalkers = 100, burn = 100, ndraws = 50, make_plot = True, threads = 1, \
                    plot_trace = False, init = 'unif', prior_teff = None, mleburn = 50, savename = None, isbinary = None, mask_threshold = 100):

        """
        Main fitting routine, takes a continuum-normalized spectrum and fits it with MCMC to recover steller labels. 
        
        Parameters
        ----------
        wl : array
            Array of observed spectral wavelengths
        fl : array
            Array of observed spectral fluxes, continuum-normalized. We recommend using the included `normalize_balmer` function from `wdtools.spectrum` to normalize DA spectra, 
            and the generic `continuum_normalize` function for DB spectra. 
        ivar : array, str ['infer']
            Array of observed inverse-variance for uncertainty estimation. If this is not available, use `ivar = 'infer'` to infer a constant inverse variance mask using a second-order
            beta-sigma algorithm. In this case, since the errors are approximated, the chi-square likelihood may be inexact - treat returned uncertainties with caution. Spectra without
            good noise information may be better suited for `gfp.fit_spectrum_abc`, which performs likelihood-free inference.
        init : str, optional
            If 'unif', walkers are initialized uniformly in parameter space before the burn-in phase. If 'mle', there is a pre-burn phase with walkers initialized uniformly in 
            parameter space. The highest probability (lowest chi square) parameter set is taken as the MLE, and the main burn-in is initialized in a tight n-ball around this high
            probablity region. If 'opt', the MLE is estimated in one shot using Nelder-Mead optimization and the burn-in is initialized in a tight n-ball around that value. If 'de', differential 
            evolution is used instead of Nelder-Mead. Direct optimization is susceptible to local minima and starting conditions. We recommend first using 'unif' to identify any multi-modality, 
            and then 'mle' for the final fit.
        prior_teff : tuple, optional
            Tuple of (mean, sigma) to define a Gaussian prior on the effective temperature parameter. This is especially useful if there is strong prior knowledge of temperature 
            from photometry. If not provided, a flat prior is used.
        nwalkers : int, optional
            Number of independent MCMC 'walkers' that will explore the parameter space
        burn : int, optional
            Number of steps to run and discard at the start of sampling to 'burn-in' the posterior parameter distribution. If intitializing from 
            a high-probability point, keep this value high to avoid under-estimating uncertainties. 
        ndraws : int, optional
            Number of 'production' steps after the burn-in. The final number of posterior samples will be nwalkers * ndraws. 
        mleburn : int, optional
            Number of steps for the pre-burn phase to estimate the MLE. 
        threads : int, optional
            Number of threads for distributed sampling. 
        make_plot: bool, optional
            If True, produces a plot of the best-fit synthetic spectrum over the observed spectrum, as well as a corner plot of the fitted parameters. 
        plot_trace: bool, optiomal
            If True, plots the trace of posterior samples of each parameter for the production steps. Can be used to visually determine the quality of mixing of
            the chains, and ascertain if a longer burn-in is required. 
        savename: str, optional
            If provided, the corner plot and best-fit plot will be saved as PDFs. 


        Returns
        -------
            `sampler`
                emcee `sampler` object, from which posterior samples can be obtained using `sampler.flatchain`. 
        """

        if isbinary is None:
            isbinary == self.isbinary

        if ivar is None:
            print('no inverse variance array provided, inferring ivar using the beta-sigma method. the chi-square likelihood will not be exact; treat returned uncertainties with caution!')
            beq = pyasl.BSEqSamp()
            std, _ = beq.betaSigma(fl, 1, 1)
            ivar = np.repeat(1 / std**2, len(fl))

        prior_lows = [6500, 6.6, -1000, 6500, 6.6, -1000, 0]

        prior_highs = [40000, 9.4, 1000, 40000, 9.4, 1000, 1]

        if not isbinary:

            def lnlike(prms):

                model = self.spectrum_sampler(wl,*prms)

                nonan = (~np.isnan(model)) * (~np.isnan(fl)) * (~np.isnan(ivar))
                diff = model[nonan] - fl[nonan]
                chisq = np.sum(diff**2 * ivar[nonan])
                if np.isnan(chisq):
                    return -np.Inf
                lnlike = -0.5 * chisq
                return lnlike

        elif isbinary:
            def lnlike(prms):

                model = self.binary_sampler(wl,*prms)

                nonan = (~np.isnan(model)) * (~np.isnan(fl)) * (~np.isnan(ivar))
                diff = model[nonan] - fl[nonan]
                chisq = np.sum(diff**2 * ivar[nonan])
                if np.isnan(chisq):
                    return -np.Inf
                lnlike = -0.5 * chisq
                return lnlike

        def lnprior(prms):
            for jj in range(len(prms)):
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

        if isbinary:
            ndim = 7
            init_prms = [12000, 8, 0, 12000, 8, 0, 1]
            param_names = ['$T_{eff, 1}$', '$\log{g}_1$', '$RV_1$', '$T_{eff, 2}$', '$\log{g}_2$', '$RV_2$', '$f_{2,1}$']
        elif not isbinary:
            ndim = 3
            init_prms = [12000, 8, 0]
            param_names = ['$T_{eff}$', '$\log{g}$', '$RV$']

        pos0 = np.zeros((nwalkers,ndim))

        sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,threads = threads)

        if init == 'opt':
            print('finding optimal starting point...')
            nll = lambda *args: -lnprob(*args)
            result = opt.minimize(nll, init_prms, method = 'lm')

            for jj in range(ndim):
                pos0[:,jj] = (result.x[jj] + 0.001*np.random.normal(size = nwalkers))

        elif init == 'de':

            print('finding optimal starting point with differential evolution...')
            
            def residual(parvals):
                model = self.spectrum_sampler(wl, parvals['teff'], parvals['logg'], parvals['rv'])
                nonan = (~np.isnan(model)) * (~np.isnan(fl)) * (~np.isnan(ivar))
                diff = model[nonan] - fl[nonan]
                chisq = np.sum(diff**2 * ivar[nonan])
                return chisq

            params = lmfit.Parameters()
            params.add('teff', value = 12000, min = 6000, max = 40000)
            params.add('logg', value = 8, min = 6.5, max = 9.5)
            params.add('rv', value = 0, min = -1000, max = 1000)
            fitter = lmfit.Minimizer(residual, params, nan_policy = 'omit')
            res = fitter.minimize(method = 'differential_evolution')
            result = [res.params['teff'].value, res.params['logg'].value, res.params['rv'].value]

            for jj in range(ndim):
                pos0[:,jj] = (result[jj] + 0.001*np.random.normal(size = nwalkers))

        elif init == 'unif':
            for jj in range(ndim):
                pos0[:,jj] = np.random.uniform(prior_lows[jj], prior_highs[jj], nwalkers)
        elif init == 'mle':
            for jj in range(ndim):
                pos0[:,jj] = np.random.uniform(prior_lows[jj], prior_highs[jj], nwalkers)

            b = sampler.run_mcmc(pos0, mleburn, progress = True)
            lnprobs = sampler.get_log_prob(flat = True)
            mle = sampler.flatchain[np.argmax(lnprobs)]

            for jj in range(ndim):
                pos0[:,jj] = (mle[jj] + 0.001*np.random.normal(size = nwalkers))

            sampler.reset()


        #Initialize sampler
        b = sampler.run_mcmc(pos0,burn, progress = True)

        sampler.reset()

        b = sampler.run_mcmc(b.coords, ndraws, progress = True)

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

        if isbinary:
            fit_fl = self.binary_sampler(wl, *mle)
        elif not isbinary:
            fit_fl = self.spectrum_sampler(wl, *mle)

        if make_plot:
            #fig,ax = plt.subplots(ndim, ndim, figsize = (15,15))
            plt.rcParams.update({'font.size': 12})
            f = corner.corner(sampler.flatchain, labels = param_names, \
                     label_kwargs = dict(fontsize =  16), quantiles = (0.16, 0.5, 0.84),
                     )
            plt.tight_layout()
            if savename is not None:
                plt.savefig(savename + '_corner.pdf', bbox_inches = 'tight')
            plt.show()

            if self.specclass == 'DA':
                plt.figure(figsize = (8,5))
                breakpoints = np.nonzero(np.diff(wl) > 5)[0]
                breakpoints = np.concatenate(([0], breakpoints, [None]))

                for kk in range(len(breakpoints) - 1):
                    wl_seg = wl[breakpoints[kk] + 1:breakpoints[kk+1]]
                    fl_seg = fl[breakpoints[kk] + 1:breakpoints[kk+1]]
                    fit_fl_seg = fit_fl[breakpoints[kk] + 1:breakpoints[kk+1]]
                    peak = int(len(wl_seg)/2)
                    delta_wl = wl_seg - wl_seg[peak]
                    plt.plot(delta_wl, 1 + fl_seg - 0.35 * kk, 'k')
                    plt.plot(delta_wl, 1 + fit_fl_seg - 0.35 * kk, 'r')
                plt.xlabel(r'$\mathrm{\Delta \lambda}\ (\mathrm{\AA})$')
                plt.ylabel('Normalized Flux')
                if savename is not None:
                    plt.savefig(savename + '_fit.pdf', bbox_inches = 'tight')
                plt.show()
            else:
                plt.figure(figsize = (10,5))
                plt.plot(wl, fl, 'k')
                plt.plot(wl, fit_fl, 'r')
                plt.ylabel('Normalized Flux')
                plt.xlabel('Wavelength ($\mathrm{\AA}$)')
                plt.minorticks_on()
                plt.tick_params(which='major', length=10, width=1, direction='in', top = True, right = True)
                plt.tick_params(which='minor', length=5, width=1, direction='in', top = True, right = True)
                if savename is not None:
                    plt.savefig(savename + '_fit.pdf', bbox_inches = 'tight')
                plt.show()

        return sampler


    def blackbody(self, wl, teff):
        wl = wl * 1e-10
        num = 2 * planck_h * speed_light**2
        denom = wl**5 * (np.exp((planck_h * speed_light) / (wl * k_B * teff)) - 1)
        return num/denom