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
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
plt.rcParams.update({'font.size': 16})

class MCFit:

	def __init__(self, resolution = 3):
		self.H = 32
		self.reg = 0.0001
		self.lamgrid = np.arange(4000,8000)
		self.sc = pickle.load(open(dir_path + '/models/neural_gen/normNN_sc.p', 'rb'))
		self.msc = pickle.load(open(dir_path + '/models/neural_gen/normNN_msc.p', 'rb'))
		self.model = self.generator()
		self.model.load_weights(dir_path + '/models/neural_gen/normNN.h5')
		self.resolution = resolution


	def generator(self):
		x = Input(shape=(2,))
		y = Dense(self.H,activation='sigmoid',trainable = True)(x)
		y = Dense(self.H,activation='sigmoid',trainable = True)(y)
		out = Dense(4000,activation='linear',trainable = True)(y)
		
		model = Model(inputs = x, outputs = out)
		model.compile(optimizer = Adamax(), loss = 'mse', \
					  metrics = ['mae'])
		return model

	def spectrum_sampler(self, wl, teff, logg, rv):
		label = self.sc.transform(np.asarray(np.stack((teff,logg)).reshape(1,-1)))
		synth = dopplerShift(self.lamgrid,np.ravel(
						self.msc.inverse_transform(
								self.model.predict(label))[0]
						), rv
					)[0]
		synth =  (np.ravel(synth).astype('float64'))
		synth = scipy.ndimage.gaussian_filter1d(synth, self.resolution)
		func = interp1d(self.lamgrid, synth, fill_value = 1, bounds_error = False)
		return func(wl)

	def fit_spectrum(self, wl, fl, ivar, nwalkers = 100, burn = 100, n_draws = 250, make_plot = False):

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
			return 0 #np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(prms[0]-mu)**2/sigma**2

		def lnprob(prms):
			lp = lnprior(prms)
			if not np.isfinite(lp):
				return -np.Inf
			return lp + lnlike(prms)

		ndim = 3
		pos0 = np.zeros((nwalkers,ndim))

		lows = [6000,6.6,-300]
		highs = [80000,9.4,300]

		for jj in range(ndim):
			pos0[:,jj] = ( np.random.uniform(lows[jj], highs[jj], nwalkers) )

		sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,threads = 1)

		#Initialize sampler
		print('Initial burn...')
		b = sampler.run_mcmc(pos0,burn)

		sampler.reset()

		b = sampler.run_mcmc(b.coords, n_draws)

		medians = np.median(sampler.flatchain, 0)

		if make_plot:
			fig,ax = plt.subplots(3,3, figsize = (10,10))
			f = corner.corner(sampler.flatchain, labels = ['$T_{eff}$', '$\log{g}$', 'RV'], \
                  fig = fig, show_titles = True, title_kwargs = dict(fontsize = 16), range = (0.99, 0.99, 0.99),\
                     label_kwargs = dict(fontsize = 16), quantiles = (0.16, 0.5, 0.84))
			plt.tight_layout()
			plt.show()

			plt.figure(figsize = (10,5))

			plt.plot(wl, fl, 'k')
			plt.plot(wl, self.spectrum_sampler(wl, medians[0], medians[1], medians[2]), 'r')

		return sampler.flatchain
