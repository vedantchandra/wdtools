import numpy as np
import astropy
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import glob
import pickle
from astropy.table import Table
import sys
from tqdm import tqdm
from scipy.interpolate import interp1d
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
import os
from PyAstronomy.pyasl import dopplerShift
import emcee
import scipy
import corner
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import scipy.optimize as opt
import pyabc
halpha = 6564.61
hbeta = 4862.68
hgamma = 4341.68
hdelta = 4102.89
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
plt.rcParams.update({'font.size': 16})

class MCFit:

    def __init__(self, resolution = 3):
        self.H = 256
        self.reg = 0.0001
        self.lamgrid = pickle.load(open(dir_path + '/models/neural_gen/lamgrid.p', 'rb'))
        self.model = self.generator()
        self.model.load_weights(dir_path + '/models/neural_gen/new_normNN.h5')
        self.resolution = resolution

    def label_sc(self, label_array):
        teffs = label_array[:, 0];
        loggs = label_array[:, 1];
        teffs = (teffs - 2500) / (100000 - 2500)
        loggs = (loggs - 5) / (10 - 5)
        return np.vstack((teffs, loggs)).T
    def inv_label_sc(self, label_array):
        teffs = label_array[:, 0];
        loggs = label_array[:, 1];
        teffs = (teffs * (100000 - 2500)) + 2500
        loggs = (loggs * (10 - 5)) + 5
        return np.vstack((teffs, loggs)).T

    def generator(self):
        x = Input(shape=(2,))
        y = Dense(self.H,activation='relu',trainable = True)(x)
        y = Dense(self.H,activation='relu',trainable = True)(y)
        out = Dense(len(self.lamgrid),activation='linear',trainable = True)(y)
        
        model = Model(inputs = x, outputs = out)
        model.compile(optimizer = Adamax(), loss = 'mse', \
                      metrics = ['mae'])
        return model

    def spectrum_sampler(self, wl, teff, logg, rv):
        label = self.label_sc(np.asarray(np.stack((teff,logg)).reshape(1,-1)))
        synth = dopplerShift(self.lamgrid,np.ravel(
                        (
                                self.model.predict(label))[0]
                        ), rv
                    )[0]
        synth =  (np.ravel(synth).astype('float64'))
        synth = scipy.ndimage.gaussian_filter1d(synth, self.resolution)
        func = interp1d(self.lamgrid, synth, fill_value = 1, bounds_error = False)
        return func(wl)

    def fit_spectrum(self, wl, fl, ivar, nwalkers = 250, burn = 100, n_draws = 250, make_plot = False, threads = 1, \
                    plot_trace = False, init = 'unif', prior_teff = None, mleburn = 50):

        def lnlike(prms):
            model = self.spectrum_sampler(wl,prms[0],prms[1],prms[2])

            nonan = (~np.isnan(model)) * (~np.isnan(fl)) * (~np.isnan(ivar))
            diff = model[nonan] - fl[nonan]
            chisq = np.sum(diff**2 * ivar[nonan])
            if np.isnan(chisq):
                return -np.Inf
            lnlike = -0.5 * chisq
            return lnlike

        def lnprior(prms):
            if prms[0] < 6000 or prms[0] > 80000:
                return -np.Inf
            elif prms[1] < 6.5 or prms[1] > 9.5:
                return -np.Inf
            elif prms[2] < -300 or prms[2] > 300:
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
        ndim = 3
        pos0 = np.zeros((nwalkers,ndim))

        lows = [6500,6.6,-300]
        highs = [80000,9.4,300]

        sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,threads = threads)

        if init == 'opt':
            nll = lambda *args: -lnprob(*args)
            result = opt.minimize(nll, [12000, 8, 0], method = 'Nelder-Mead')

            for jj in range(ndim):
                pos0[:,jj] = (result.x[jj] + 0.001*np.random.normal(size = nwalkers))

        elif init == 'unif':
            for jj in range(ndim):
                pos0[:,jj] = np.random.uniform(lows[jj], highs[jj], nwalkers)
        elif init == 'mle':
            for jj in range(ndim):
                pos0[:,jj] = np.random.uniform(lows[jj], highs[jj], nwalkers)

            b = sampler.run_mcmc(pos0, mleburn, progress = True)
            lnprobs = sampler.get_log_prob(flat = True)
            mle = sampler.flatchain[np.argmax(lnprobs)]

            for jj in range(ndim):
                pos0[:,jj] = (mle[jj] + 0.001*np.random.normal(size = nwalkers))

            sampler.reset()


        #Initialize sampler
        b = sampler.run_mcmc(pos0,burn, progress = True)

        sampler.reset()

        b = sampler.run_mcmc(b.coords, n_draws, progress = True)

        if plot_trace:
            f, axs = plt.subplots(3, 1, figsize = (10, 6))
            for jj in range(ndim):
                axs[jj].plot(sampler.chain[:,:,jj].T, alpha = 0.3, color = 'k');
            plt.show()

        lnprobs = sampler.get_log_prob(flat = True)
        medians = np.median(sampler.flatchain, 0)
        mle = sampler.flatchain[np.argmax(lnprobs)]
        fit_fl = self.spectrum_sampler(wl, mle[0], mle[1], mle[2])

        if make_plot:
            fig,ax = plt.subplots(3,3, figsize = (10,10))
            f = corner.corner(sampler.flatchain, labels = ['$T_{eff}$', '$\log{g}$', 'RV'], \
                  fig = fig, show_titles = True, title_kwargs = dict(fontsize = 16),\
                     label_kwargs = dict(fontsize = 16), quantiles = (0.16, 0.5, 0.84))
            plt.tight_layout()
            plt.show()

            plt.figure(figsize = (8,5))

            breakpoints = np.nonzero(np.diff(wl) > 5)[0]
            breakpoints = np.concatenate(([0], breakpoints, [None]))

            for kk in range(len(breakpoints) - 1):
                wl_seg = wl[breakpoints[kk] + 1:breakpoints[kk+1]]
                fl_seg = fl[breakpoints[kk] + 1:breakpoints[kk+1]]
                fit_fl_seg = fit_fl[breakpoints[kk] + 1:breakpoints[kk+1]]
                peak = np.argmin(fl_seg)
                delta_wl = wl_seg - wl_seg[peak]
                plt.plot(delta_wl, 1 + fl_seg - 0.35 * kk, 'k')
                plt.plot(delta_wl, 1 + fit_fl_seg - 0.35 * kk, 'r')
            plt.xlabel(r'$\mathrm{\Delta \lambda}\ (\mathrm{\AA})$')
            plt.ylabel('Normalized Flux')
            plt.show()

            plt.figure(figsize = (10,5))
            plt.plot(wl, fl, 'k')
            plt.plot(wl, fit_fl, 'r')
            plt.show()
            return sampler


    def fit_spectrum_abc(self, wl, fl, ivar = None, make_plot = False, popsize = 100, max_pops = 25):
        if ivar is None:
            ivar = 1

        obs = dict(spec = fl)

        def sim(params):
            fl_sample = self.spectrum_sampler(wl, params['teff'], params['logg'], params['rv'])
            return dict(spec = fl_sample)

        def distance(sim1, sim2):
            resid = sim1['spec'] - sim2['spec']
            chisq = np.nansum(resid**2 * ivar)
            return chisq

        priors = pyabc.Distribution(teff = pyabc.RV("uniform", 6000, 74000),
                                       logg = pyabc.RV("uniform", 6.5, 3),
                                       rv = pyabc.RV("uniform", -300, 600))

        abc = pyabc.ABCSMC(sim, priors,\
                           distance, sampler = pyabc.sampler.SingleCoreSampler(),\
                           population_size = popsize, \
                           eps = pyabc.epsilon.QuantileEpsilon(alpha = 0.5))

        db = ("sqlite:///mcfit.db")
        abc.new(db, obs)

        history = abc.run(min_acceptance_rate = 0.01, max_nr_populations = max_pops)

        if make_plot:
            pyabc.visualization.plot_kde_matrix_highlevel(history, height = 4)
            plt.show()

            df, w = history.get_distribution()
            plt.figure(figsize = (8,5))
            medians = df.median()
            fit_fl = self.spectrum_sampler(wl, medians['teff'], medians['logg'], medians['rv'])
            breakpoints = np.nonzero(np.diff(wl) > 5)[0]
            breakpoints = np.concatenate(([0], breakpoints, [None]))

            for kk in range(len(breakpoints) - 1):
                wl_seg = wl[breakpoints[kk] + 1:breakpoints[kk+1]]
                fl_seg = fl[breakpoints[kk] + 1:breakpoints[kk+1]]
                fit_fl_seg = fit_fl[breakpoints[kk] + 1:breakpoints[kk+1]]
                peak = np.argmin(fl_seg)
                delta_wl = wl_seg - wl_seg[peak]
                plt.plot(delta_wl, 1 + fl_seg - 0.35 * kk, 'k')
                plt.plot(delta_wl, 1 + fit_fl_seg - 0.35 * kk, 'r')
            plt.xlabel(r'$\mathrm{\Delta \lambda}\ (\mathrm{\AA})$')
            plt.ylabel('Normalized Flux')
            plt.show()

        return history