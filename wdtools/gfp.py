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

        self.cont_fixed = False

        self.centroid_dict = dict(alpha = 6564.61, beta = 4862.68, gamma = 4341.68, delta = 4102.89, eps = 3971.20, h8 = 3890.12)
        self.distance_dict = dict(alpha = 250, beta = 250, gamma = 120, delta = 75, eps = 50, h8 = 25)

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

    def doppler_shift(self, wl, fl, dv):
        c = 2.99792458e5
        df = np.sqrt((1 - dv/c)/(1 + dv/c)) 
        new_wl = wl * df
        new_fl = np.interp(new_wl, wl, fl)
        return new_fl


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
        synth = self.doppler_shift(self.lamgrid[specclass], synth, rv)
        synth =  (np.ravel(synth).astype('float64'))
        return synth

    def spectrum_sampler(self, wl, teff, logg, rv, *polyargs, specclass = None):
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

        if specclass is None:
            specclass = self.specclass;
        synth = self.synth_spectrum_sampler(self.lamgrid[specclass], teff, logg, rv, specclass)
        synth = scipy.ndimage.gaussian_filter1d(synth, self.resolution[specclass])
        func = interp1d(self.lamgrid[specclass], synth, fill_value = np.nan, bounds_error = False)
        synth =  func(wl)

        # print(polyargs)

        if self.cont_fixed:
            mod_smooth_cont = scipy.interpolate.interp1d(wl[self.contmask],  synth[self.contmask])(wl)
            synth = synth / mod_smooth_cont

        if len(polyargs) > 0:
            synth = synth * chebval(2*(wl - wl.min() / (wl.max())) - 1.0, polyargs)

        return synth

    def binary_sampler(self, wl, teff_1, logg_1, rv_1, teff_2, logg_2, rv_2, lf = 1, specclass = None):
        
        """
        Under development, do not use.
        """

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
                    plot_trace = False, init = 'de', prior_teff = None, mleburn = 50, savename = None, isbinary = None, mask_threshold = 100,
                    mask_DA = False, lines = ['alpha', 'beta', 'gamma', 'delta', 'eps', 'h8'], progress = True,
                    polyorder = 4, plot_init = False, plot_corner = False, plot_corner_full = False,
                    cont_polyorder = 6):

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

        if isbinary is None:
            isbinary == self.isbinary

        wlbounds = self.lamgrid['DA'].min() + 5, self.lamgrid['DA'].max() - 5
        in1,in2 = bisect_left(wl, wlbounds[0]), bisect_left(wl, wlbounds[1])

        wl = wl[in1:in2]
        fl = fl[in1:in2]
        ivar = ivar[in1:in2]

        # fl = fl / np.median(fl)
        # ivar = ivar * np.median(fl)**2

        # if normalize_DA == True:
        #     sp = SpecTools()
        #     if ivar is None:
        #         wl, fl = sp.normalize_balmer(wl, fl, ivar = None, lines = lines)
        #     else:
        #         wl, fl, ivar = sp.normalize_balmer(wl, fl, ivar, lines = lines)

        if ivar is None:
            print('no inverse variance array provided, inferring ivar using the beta-sigma method. the chi-square likelihood will not be exact; treat returned uncertainties with caution!')
            beq = pyasl.BSEqSamp()
            std, _ = beq.betaSigma(fl, 1, 1)
            ivar = np.repeat(1 / std**2, len(fl))

        prior_lows = [6500, 6.5, -1000, 6500, 6.5, -1000, 0]

        prior_highs = [40000, 9.5, 1000, 40000, 9.5, 1000, 1]

        edges = [];
        breakpoints = [];

        if mask_DA:
            mask = np.zeros(len(wl))
            for line in lines:
                c1 = self.centroid_dict[line] - self.distance_dict[line]
                c2 = self.centroid_dict[line] + self.distance_dict[line]
                mask += (wl > c1)*\
                        (wl < c2)
                edges.extend([c1, c2])
                breakpoints.extend([bisect_left(wl, c2), bisect_left(wl, c1)])

            self.contmask = ~(mask > 0)

        #smooth_cont = scipy.interpolate.interp1d(wl[contmask], fl[contmask])(wl)
        #fl = fl / smooth_cont
        #ivar = ivar * smooth_cont**2

        self.mask = np.ones(len(wl)) > 0

        if not isbinary:

            nstarparams = 3

            def lnlike(prms):

                model = self.spectrum_sampler(wl,*prms)



                diff = (model - fl)**2 * ivar

                ### TEST MASKING

                diff = diff[self.mask]

                ################

                chisq = np.sum(diff)

                if np.isnan(chisq):
                    return -np.Inf
                lnlike = -0.5 * chisq
                return lnlike

        # elif isbinary:
        #     def lnlike(prms):

        #         model = self.binary_sampler(wl,*prms)

        #         nonan = (~np.isnan(model)) * (~np.isnan(fl)) * (~np.isnan(ivar))
        #         diff = model[nonan] - fl[nonan]
        #         chisq = np.sum(diff**2 * ivar[nonan])
        #         if np.isnan(chisq):
        #             return -np.Inf
        #         lnlike = -0.5 * chisq
        #         return lnlike

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

        # if isbinary:
        #     ndim = 7
        #     init_prms = [12000, 8, 0, 12000, 8, 0, 1]
        #     param_names = ['$T_{eff, 1}$', '$\log{g}_1$', '$RV_1$', '$T_{eff, 2}$', '$\log{g}_2$', '$RV_2$', '$f_{2,1}$']
        if not isbinary:
            init_prms = [15000, 8, 0]
            init_prms.extend(chebfit(2*(wl - wl.min() / (wl.max())) - 1.0, fl, cont_polyorder))
            param_names = ['$T_{eff}$', '$\log{g}$', '$RV$']
            param_names.extend(['$c_%i$' % ii for ii in range(polyorder + 1)])


        # bounds = [];
        # for jj in range(nstarparams + polyorder + 1):
        #     if jj < nstarparams:
        #         bounds.append([prior_lows[jj], prior_highs[jj]])
        #     else:
        #         bounds.append([-np.inf, np.inf])

        # bounds_lo = prior_lows[:nstarparams]
        # bounds_hi = prior_highs[:nstarparams]

        # for jj in range(polyorder + 1):
        #     bounds_lo.append(-np.inf)
        #     bounds_hi.append(np.inf)

        # bounds = (bounds_lo, bounds_hi)

        # print(init_prms)
        # print(bounds)

        self.mask = np.ones(len(wl)) > 0
        nll = lambda *args: -lnprob(*args)
        soln = scipy.optimize.minimize(nll, init_prms, method = 'Nelder-Mead', options = dict(maxfev = 500))
        print(soln.x)

        # soln = scipy.optimize.curve_fit(self.spectrum_sampler, xdata = wl[mask], ydata = fl[mask],
        #             p0 = init_prms, sigma = np.sqrt(1/ivar[mask]), absolute_sigma = True,
        #                 bounds = bounds, method = 'trf', xtol = 1e-10, ftol = 1e-10)

        # fl = fl / polyval(wl/1000, soln.x[nstarparams:])
        # ivar = ivar * polyval(wl/1000, soln.x[nstarparams:])**2

        self.mask = mask > 0
        contmask = ~self.mask

        smooth_cont = scipy.interpolate.interp1d(wl[contmask], self.spectrum_sampler(wl, *soln.x)[contmask])(wl)
        self.poly_arg = soln.x[nstarparams:]

        if plot_init:
            plt.figure(figsize = (10, 8))
            plt.plot(wl, fl, 'k', label = 'Data')
            plt.plot(wl, self.spectrum_sampler(wl, *init_prms), label = 'Initial Guess')
            plt.plot(wl, self.spectrum_sampler(wl, *soln.x), 'r', label = 'Continuum Fit')
            plt.plot(wl, smooth_cont, 'g', label = 'Smooth Continuum')
            plt.legend()
            plt.show()


        fl = fl / smooth_cont
        ivar = ivar * smooth_cont**2

        self.cont_fixed = True
        self.smooth_cont = smooth_cont

        # plt.plot(wl, fl)
        # plt.plot(wl, self.spectrum_sampler(wl, *[17000, 7.9, 10]))
        # plt.plot(wl, self.spectrum_sampler(wl, *[8000, 8.5, 120]))
        # plt.show()


        # plt.plot(wl, 1 / np.sqrt(ivar))
        # plt.show()


        #### CHANGE BELOW TO CURVE FIT TO GET COV MATRIX AND PLUG INTO EMCEE

        init_prms = [9000, 8, soln.x[2]]
        init_prms.extend(np.zeros(polyorder))
        init_prms[nstarparams] = 1
        #print(init_prms)
        nll = lambda *args: -lnprob(*args)
        cool_soln = scipy.optimize.minimize(nll, init_prms, method = 'Nelder-Mead', options = dict(maxfev = 500))
        cool_chi = -2 * lnprob(cool_soln.x)
        #print(cool_soln.x)

        init_prms = [18000, 8, soln.x[2]]
        init_prms.extend(np.zeros(polyorder))
        init_prms[nstarparams] = 1
        nll = lambda *args: -lnprob(*args)
        warm_soln = scipy.optimize.minimize(nll, init_prms, method = 'Nelder-Mead', options = dict(maxfev = 500))
        warm_chi = -2 * lnprob(warm_soln.x)
        #print(warm_soln.x)

        #print(cool_chi, warm_chi)

        if cool_chi < warm_chi:
            soln.x = cool_soln.x
        else:
            soln.x = warm_soln.x


        ndim = len(soln.x)
        
        sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob, threads = threads)

        pos0 = np.zeros((nwalkers,ndim))


        for jj in range(ndim):
                pos0[:,jj] = (soln.x[jj] + 1e-5*soln.x[jj]*np.random.normal(size = nwalkers))

        #print(pos0.shape)

        b = sampler.run_mcmc(pos0, burn, progress = progress)

        sampler.reset()

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
        redchi = -2 * np.max(lnprobs) / (len(wl) - 3)
        stds = np.std(sampler.flatchain, 0)
        
        if mle[0] < 7000 or mle[0] > 38000:
            print('temperature is near bound of the model grid! exercise caution with this result')
        if mle[1] < 6.7 or mle[1] > 9.3:
            print('logg is near bound of the model grid! exercise caution with this result')

        if isbinary:
            fit_fl = self.binary_sampler(wl, *mle)
        elif not isbinary:
            fit_fl = self.spectrum_sampler(wl, *mle)

        if plot_corner:

            plt.rcParams.update({'font.size': 12})

            f = corner.corner(sampler.flatchain[:, :nstarparams], labels = param_names[:nstarparams], \
                     label_kwargs = dict(fontsize =  12), quantiles = (0.16, 0.5, 0.84),
                     show_titles = True, title_kwargs = dict(fontsize = 12))
            #plt.tight_layout()
            if savename is not None:
                plt.savefig(savename + '_corner.pdf', bbox_inches = 'tight')
            plt.show()

        if plot_corner_full:

            f = corner.corner(sampler.flatchain, labels = param_names, \
                     label_kwargs = dict(fontsize =  12), quantiles = (0.16, 0.5, 0.84),
                     show_titles = False)

        if make_plot:
            #fig,ax = plt.subplots(ndim, ndim, figsize = (15,15))

            if self.specclass == 'DA':
                plt.figure(figsize = (8,7))
                #breakpoints = np.concatenate(([0], breakpoints, [None]))
                breakpoints = np.flip(breakpoints)
                self.breakpoints = breakpoints
                print(breakpoints)
                for kk in range(len(breakpoints)):
                    if (kk + 1)%2 == 0:
                        continue
                    wl_seg = wl[breakpoints[kk]:breakpoints[kk+1]]
                    fl_seg = fl[breakpoints[kk]:breakpoints[kk+1]]
                    fit_fl_seg = fit_fl[breakpoints[kk]:breakpoints[kk+1]]
                    peak = int(len(wl_seg)/2)
                    delta_wl = wl_seg - wl_seg[peak]
                    plt.plot(delta_wl, 1 + fl_seg - 0.2 * kk, 'k')
                    plt.plot(delta_wl, 1 + fit_fl_seg - 0.2 * kk, 'r')
                plt.xlabel(r'$\mathrm{\Delta \lambda}\ (\mathrm{\AA})$')
                plt.ylabel('Normalized Flux')

                plt.text(0.05, 0.85, '$T_{\mathrm{eff}} = %i \pm %i\ K$' % (mle[0], stds[0]),
                 transform = plt.gca().transAxes, fontsize = 16)
        
                plt.text(0.65, 0.85, '$\log{g} = %.2f \pm %.2f $' % (mle[1], stds[1]),
                         transform = plt.gca().transAxes, fontsize = 16)
                
                plt.text(0.79, 0.75, '$\chi_r^2$ = %.2f' % (redchi),
                         transform = plt.gca().transAxes, fontsize = 16)

            #     if savename is not None:
            #         plt.savefig(savename + '_fit.pdf', bbox_inches = 'tight')

            plt.figure(figsize = (10,5))
            plt.plot(wl, fl, 'k')
            randidx = np.random.choice(len(sampler.flatchain), size = 10)
            plt.plot(wl, fit_fl, 'r')

            for idx in randidx:
                label = sampler.flatchain[idx]
                plt.plot(wl, self.spectrum_sampler(wl, *label), 'r', alpha = 0.25, lw = 0.5)
            
            plt.ylabel('Normalized Flux')
            plt.xlabel('Wavelength ($\mathrm{\AA}$)')
            plt.minorticks_on()
            plt.tick_params(which='major', length=10, width=1, direction='in', top = True, right = True)
            plt.tick_params(which='minor', length=5, width=1, direction='in', top = True, right = True)

            for edge in edges:
                plt.axvline(edge, linestyle = '--', color = 'k', lw = 0.5)

            if savename is not None:
                plt.savefig(savename + '_fit.pdf', bbox_inches = 'tight')
            plt.show()

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
    
    result = gfp.fit_spectrum(wl, fl, init = 'de', burn = 1, ndraws = 1, 
                              normalize_DA = True)
