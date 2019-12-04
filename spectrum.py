import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from bisect import bisect_left
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
plt.rcParams.update({'font.size': 18})
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
import pwlf
import scipy
import csaps
from PyAstronomy.pyasl import crosscorrRV, quadExtreme, dopplerShift
from scipy.interpolate import interp1d
import lmfit

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

	def continuum_normalize(self, wl, flux):
		pixels = wl - 4000
		continuum = (pixels > 2900) + (pixels < 2200)*(pixels > 1050) + (pixels < 650)*(pixels > 530) \
		+ (pixels < 220)*(pixels > 200) + (pixels < 35)*(pixels > 20) + (pixels < -200)*(pixels > -450)

		notnan = (~np.isnan(flux))+(~np.isnan(wl))
		skylinemask = (wl > 5578.5 - 10)*(wl < 5578.5 + 10) + (wl > 5894.6 - 10)*(wl < 5894.6 + 10)\
		+ (wl > 6301.7 - 10)*(wl < 6301.7 + 10) + (wl > 7246.0 - 10)*(wl < 7246.0 + 10)

		#spl = scipy.interpolate.UnivariateSpline(wl[continuum*notnan], flux[continuum*notnan], k = 1, s = self.smoothing)

		spl = csaps.UnivariateCubicSmoothingSpline(wl[continuum*notnan], flux[continuum*notnan], smooth = self.smoothing)

		cont = spl(wl)
		contcorr = flux / cont

		if self.filter_skylines:
			contcorr[skylinemask] = 1

		if self.crop:
			where = (contcorr > 1.5) + (contcorr < 0)
			contcorr[where] = 1

		if self.plot_continuum == True:
			plt.figure()
			plt.plot(wl,flux)
			plt.plot(wl,cont)

		return contcorr

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
		plt.plot(rvs,corrs)
		try:
			p = np.polyfit(rvs, corrs, 2)
			specrv = - p[1] / (2*p[0])
			plt.plot(rvs,np.polyval(p,rvs))
			plt.axvline(specrv)
			print(specrv)
		except:
			print('find rv failed. defaulting to zero')
			specrv = 0
		shift_flux,shift_wl = dopplerShift(wl,normflux,specrv, edgeHandling = 'fillValue', fillValue = 1)
		return shift_wl,shift_flux

	def interpolate(self, wl, flux, target_wl = np.arange(4000,8000)):
		func = interp1d(wl, flux, kind='linear', assume_sorted = True, fill_value = 'extrapolate')
		interpflux = func(target_wl)
		return target_wl,interpflux

class RVTools():

	def __init__(self):
		pass

	def linear(self, wl, p1, p2):

		''' Linear polynomial of degree 1 '''

		return p1 + p2*wl

	def chisquare(self, residual):
		
		''' Chi^2 statistics from residual

		Unscaled chi^2 statistic from an array of residuals (does not account for uncertainties).
		'''

		return np.sum(residual**2)

	def find_centroid(self, wl, flux, centroid, half_window = 25, window_step = 25, n_fit = 4, make_plot = False):

		in1 = bisect_left(wl,centroid-150)
		in2 = bisect_left(wl,centroid+150)
		cropped_wl = wl[in1:in2]
		cflux = flux[in1:in2]

		cmask = (cropped_wl < centroid - 75)+(cropped_wl > centroid + 75)

		p,cov = curve_fit(self.linear,cropped_wl[cmask],cflux[cmask])

		contcorr = cflux / self.linear(cropped_wl, p[0], p[1])


		linemodel = lmfit.models.VoigtModel()
		params = linemodel.make_params()
		params['amplitude'].set(value = 10)
		params['center'].set(value = centroid)
		params['sigma'].set(value=5)
		centres = [];
		errors = [];

		if make_plot:
			plt.figure(figsize=(10,5))
			plt.title(str(centroid)+"$\AA$")
			plt.plot(cropped_wl,contcorr,'k')

		for ii in range(n_fit):
		
			crop1 = bisect_left(cropped_wl,centroid - half_window - ii*window_step)
			crop2 = bisect_left(cropped_wl,centroid + half_window + ii*window_step)
			try:
				result = linemodel.fit(1-contcorr[crop1:crop2],params,x = cropped_wl[crop1:crop2],\
				nan_policy = 'omit',method='dual_annealing',fit_kws={'reduce_fcn':self.chisquare})
			except ValueError:
				print('fit failed! returning NaN')
				return np.nan,np.nan

			centres.append(result.params['center'].value)
			errors.append(result.params['center'].stderr)

			if make_plot:
				plt.plot(cropped_wl[crop1:crop2],1-linemodel.eval(result.params,x=cropped_wl[crop1:crop2]),'r', linewidth = 1)
				#plt.plot(cropped_wl[crop1:crop2],1-linemodel.eval(params,x=cropped_wl[crop1:crop2]),'k--')

		mean_centre = np.mean(centres)
		sigma_sample = np.std(centres)
		#print(np.isnan(np.array(errors)))
		if None in errors or np.nan in errors:
			return mean_centre, np.nan

		sigma_propagated = np.linalg.norm(errors) / n_fit

		total_sigma = np.sqrt(sigma_propagated**2 + sigma_sample**2)

		if make_plot:
			plt.xlabel('Wavelength ($\AA$)')
			plt.ylabel('Normalized Flux')
			plt.show()

		return mean_centre, total_sigma