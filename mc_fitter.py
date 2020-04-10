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
import corner
from keras.models import *
from keras.layers import *
from keras.optimizers import *

class MCFit:

	def __init__():
		self.H = 32
		self.reg = 0.0001


	def generator(self):
	    x = Input(shape=(2,))
	    y = Dense(self.H,activation='sigmoid',trainable = True)(x)
	    y = Dense(self.H,activation='sigmoid',trainable = True)(y)
	    out = Dense(4000,activation='linear',trainable = True)(y)
	    
	    model = Model(inputs = x, outputs = out)
	    model.compile(optimizer = Adamax(), loss = 'mse', \
	                  metrics = ['mae'])
	    return model

	def spectrum_sampler(self, wl,teff,logg,rv):
    	label = sc.transform(np.asarray(np.stack((teff,logg)).reshape(1,-1)))
    	synth = dopplerShift(wl,np.ravel(
                        msc.inverse_transform(
                                model.predict(label))[0]
                        ), rv
                    )[0]
    return (np.ravel(synth).astype('float64'))

   	def fit_spectrum(self, wl, fl, nwalkers = 100, n_draws = 250, make_plot = False):

   		def lnlike(prms):
		    ivar = 1 / 0.1**2
		    model = self.spectrum_sampler(lamgrid,prms[0],prms[1],prms[2])
		    diff = model - spectrum
		    diff = diff[~np.isnan(model)]
		    chisq = np.sum(diff**2 * ivar)
		    lnlike = -0.5 * chisq
		#     plt.plot(model)
		#     plt.plot(spectrum)
		    return lnlike

		def lnprior(prms):
		    if prms[0] < 6000 or prms[0] > 80000:
		        return -np.Inf
		    elif prms[1] < 6.5 or prms[1] > 9.5:
		        return -np.Inf
		    elif prms[2] < -300 or prms[2] > 300:
		        return -np.Inf
		    return 0#np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(prms[0]-mu)**2/sigma**2

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
		b = sampler.run_mcmc(pos0,50)
		print('Burning in posterior...')

		if make_plot:
			plt.plot(sampler.chain[:,:,0].T, alpha = 0.3, color = 'k');
			plt.figure()
			plt.plot(sampler.chain[:,:,1].T, alpha = 0.3, color = 'k');
			plt.show()

		sampler.reset()

		b = sampler.run_mcmc(b.coords, n_draws)

		return sampler.flatchains
