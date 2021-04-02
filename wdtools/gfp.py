import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import stats
import glob
import pickle
import sys
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
from .corr3d import *

interp1d = interpolate.interp1d

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
		polyargs : float, optional
			All subsequent positional arguments are assumed to be coefficients for the additive Chebyshev polynomial. If none are provided,
			no polynomial is added to the model spectrum. 
		specclass : str, optional
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

		if self.cont_fixed:

			dummy_ivar = 1 / np.repeat(0.001, len(wl))**2
			nanwhere = np.isnan(synth)
			dummy_ivar[nanwhere] = 0
			synth[nanwhere] = 0
			synth,_ = self.spline_norm_DA(wl, synth, dummy_ivar, kwargs = self.norm_kw) # Use default KW from function
			synth[nanwhere] = np.nan

		if len(polyargs) > 0:
			synth = synth * chebval(2 * (wl - wl.min()) / (wl.max() - wl.min()) - 1, polyargs)

		return synth


	def spline_norm_DA(self, wl, fl, ivar, kwargs = dict(k = 3, sfac = 1, niter = 3), crop = None): # SETS DEFAULT KW
		"""
		Masks out Balmer lines, fits a smoothing spline to the continuum, and returns a continuum-normalized spectrum
		
		Parameters
		----------
		wl : array
			Array of observed spectral wavelengths.
		fl : array
			Array of observed spectral fluxes.
		ivar : array
			Array of observed inverse-variance. 
		kwargs : dict, optional
			Keyword arguments that are passed to the spline normalization function
		crop : tuple, optional
			Defines a start and end wavelength to crop the spectrum to before continuum-normalization. 

		Returns
		-------
			tuple
				If crop is None, returns a 2-tuple of (normalized_flux, normalized_ivar). If a crop region is provided, 
				then returns a 3-tuple of (cropped_wavelength, cropped_normalized_flux, cropped_normalized_ivar). 
		"""


		if crop is not None:
			c1 = bisect_left(wl, crop[0])
			c2 = bisect_left(wl, crop[1])

			wl = wl[c1:c2]
			fl = fl[c1:c2]
			ivar = ivar[c1:c2]

		# linear = np.polyval(np.polyfit(wl, fl, 2), wl)
		# fl = fl / linear
		# ivar = ivar * linear**2 # Initial divide by quadratic continuum

		try:
			fl_norm, ivar_norm = self.sp.spline_norm(wl, fl, ivar, self.exclude_wl, **kwargs)
		except:
			print('spline normalization failed... returning NaNs')
			fl_norm, ivar_norm = np.nan*fl, np.nan*ivar
			raise

		if crop is not None:
			return wl, fl_norm, ivar_norm

		else:
			return fl_norm, ivar_norm

	def fit_spectrum(self, wl, fl, ivar = None, prior_teff = None, mcmc = False, fullspec = False, polyorder = 0, 
						norm_kw = dict(k = 1, sfac = 0.5, niter = 0), 
						nwalkers = 25, burn = 25, ndraws = 25, threads = 1, progress = True,
						plot_init = False, make_plot = True, plot_corner = False, plot_corner_full = False, plot_trace = False,  savename = None, 
						DA = True, crop = (3600, 7500),
						verbose = True,
						lines = ['alpha', 'beta', 'gamma', 'delta', 'eps', 'h8'], lmfit_kw = dict(method = 'leastsq', epsfcn = 0.1), 
						rv_kw = dict(plot = False, distance = 100, nmodel = 2, edge = 15),
						nteff = 3,  rv_line = 'alpha', corr_3d = False):

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
		prior_teff : tuple, optional
			Tuple of (mean, sigma) to define a Gaussian prior on the effective temperature parameter. This is especially useful if there is strong prior knowledge of temperature 
			from photometry. If not provided, a flat prior is used. 
		mcmc : bool, optional
			Whether to run MCMC, or simply return the errors estimated by LMFIT
		fullspec : bool, optional
			Whether to fit the entire continuum-normalized spectrum, or only the Balmer lines. 
		polyorder : int, optional
			Order of additive Chebyshev polynomial during the fitting process. Can usually leave this to zero unless the normalization is really bad. 
		norm_kw : dict, optional
			Dictionary of keyword arguments that are passed to the spline normalization routine. 
		nwalkers : int, optional
			Number of independent MCMC 'walkers' that will explore the parameter space
		burn : int, optional
			Number of steps to run and discard at the start of sampling to 'burn-in' the posterior parameter distribution. If intitializing from 
			a high-probability point, keep this value high to avoid under-estimating uncertainties. 
		ndraws : int, optional
			Number of 'production' steps after the burn-in. The final number of posterior samples will be nwalkers * ndraws.
		threads : int, optional
			Number of threads for distributed sampling. 
		progress : bool, optional
			Whether to show a progress bar during the MCMC sampling. 
		plot_init : bool, optional
			Whether to plot the continuum-normalization routine
		make_plot: bool, optional
			If True, produces a plot of the best-fit synthetic spectrum over the observed spectrum. 
		plot_corner : bool, optional
			Makes a corner plot of the fitted stellar labels
		plot_corner_full : bool, optional
			Makes a corner plot of all sampled parameters, the stellar labels plus any Chebyshev coefficients if polyorder > 0
		plot_trace: bool, optiomal
			If True, plots the trace of posterior samples of each parameter for the production steps. Can be used to visually determine the quality of mixing of
			the chains, and ascertain if a longer burn-in is required. 
		savename : str, optional
			If provided, the corner plot and best-fit plot will be saved as PDFs in the working folder. 
		DA : bool, optional
			Whether the star is a DA white dwarf or not. As of now, this must be set to True. 
		crop : tuple, optional
			The region to crop the supplied spectrum before proceeding with the fit. Can be used to exclude low-SN regions at the edge of the spectrum.
		verbose : bool, optional
			If True, the routine prints several progress statements to the terminal. 
		lines : array, optional
			List of Balmer lines to utilize in the fit. Defaults to all from H-alpha to H8.
		lmfit_kw : dict, optional
			Dictionary of keyword arguments to the LMFIT solver
		rv_kw : dict, optional
			Dictionary of keyword arguments to the RV fitting routine
		nteff : int, optional
			Number of equidistant temperatures to try as initialization points for the minimization routine. 
		rv_line : str, optional
			Which Balmer line to use for the radial velocity fit. We recommend 'alpha'. 
		corr_3d : bool, optional
			If True, applies 3D corrections from Tremblay et al. (2013) to stellar parameters before returning them. 

		Returns
		-------
			array
				Returns the fitted stellar labels along with a reduced chi-square statistic with the format: [[labels], [e_labels], redchi]. If polyorder > 0,
				then the returned arrays include the Chebyshev coefficients. The radial velocity (and RV error) are always the last elements in the array, so if
				polyorder > 0, the label array will have temperature, surface gravity, the Chebyshev coefficients, and then RV. 
		"""

		self.cont_fixed = False
		self.rv_fixed = False

		nans = np.isnan(fl)

		if np.sum(nans) > 0:

			print('NaN detected in input... removing them...')

			wl = wl[~nans]
			fl = fl[~nans]
			ivar = ivar[~nans]

		if ivar is None: # REPLACE THIS WITH YOUR OWN FUNCTION TO ESTIMATE VARIANCE
			print('Please provide an IVAR array')
			raise


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


		param_names = [r'$T_{eff}$', r'$\log{g}$']
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

		self.norm_kw = norm_kw

		if DA:
			wl, fl, ivar = self.spline_norm_DA(wl, fl, ivar, kwargs = norm_kw, crop = crop)

		self.cont_fixed = True
		self.norm_kw['plot'] = False # Set to True to see how the models are normalized

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

		star_rv = self.rv
		if verbose:
			print('Radial Velocity = %i ± %i km/s' % (self.rv, e_rv))
		self.rv_fixed = True

		if verbose:
			print('final optimization...')


		teffgrid = np.linspace(8000, 35000, nteff)

		chimin = 1e50

		for teff in teffgrid:
			if verbose:
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
				f = corner.corner(sampler.flatchain[:, :nstarparams], labels = param_names[:nstarparams],
						 label_kwargs = dict(fontsize =  12), quantiles = (0.16, 0.5, 0.84),
						 show_titles = True, title_kwargs = dict(fontsize = 12))

				for ax in f.get_axes(): 
				  ax.tick_params(axis='both', labelsize=12)
				if savename is not None:
					plt.savefig(savename + '_corner.jpg', bbox_inches = 'tight', dpi = 100)
				plt.show()

			if plot_corner_full:

				f = corner.corner(sampler.flatchain, labels = param_names, 
						 label_kwargs = dict(fontsize =  12), quantiles = (0.16, 0.5, 0.84),
						 show_titles = False)

				for ax in f.get_axes(): 
				  ax.tick_params(axis='both', labelsize=12)


		fit_fl = self.spectrum_sampler(wl, *mle)

		if corr_3d and mle[0] < 15000:
			if verbose:
				print('applying 3D corrections...')
			corr = corr3d(mle[0], mle[1])
			mle[0] = corr[0]
			mle[1] = corr[1]

		if make_plot:
			#fig,ax = plt.subplots(ndim, ndim, figsize = (15,15))

			if fullspec:
				plt.figure(figsize = (10, 8))
				plt.plot(wl, fl, 'k')
				plt.plot(wl, fit_fl, 'r')
				plt.ylabel('Normalized Flux')
				plt.xlabel('Wavelength')

				plt.ylim(0, 1.5)

				plt.text(0.97, 0.25, r'$T_{\mathrm{eff}} = %.0f \pm %.0f\ K$' % (mle[0], stds[0]),
				 transform = plt.gca().transAxes, fontsize = 15, ha = 'right')
		
				plt.text(0.97, 0.15, r'$\log{g} = %.2f \pm %.2f $' % (mle[1], stds[1]),
						 transform = plt.gca().transAxes, fontsize = 15, ha = 'right')
				 
				plt.text(0.97, 0.05, r'$\chi_r^2$ = %.2f' % (redchi),
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

				plt.text(0.97, 0.8, r'$T_{\mathrm{eff}} = %.0f \pm %.0f\ K$' % (mle[0], stds[0]),
				 transform = plt.gca().transAxes, fontsize = 15, ha = 'right')
		
				plt.text(0.97, 0.7, r'$\log{g} = %.2f \pm %.2f $' % (mle[1], stds[1]),
						 transform = plt.gca().transAxes, fontsize = 15, ha = 'right')
				 
				plt.text(0.97, 0.6, r'$\chi_r^2$ = %.2f' % (redchi),
						 transform = plt.gca().transAxes, fontsize = 15, ha = 'right')



			if savename is not None:
				plt.savefig(savename + '_fit.jpg', bbox_inches = 'tight', dpi = 100)
			plt.show()

		self.exclude_wl = self.exclude_wl_default
		self.cont_fixed = False
		self.rv = 0 # RESET THESE PARAMETERS

		mle = mle
		stds = stds

		mle = np.append(mle, star_rv)
		stds = np.append(stds, e_rv)

		return mle, stds, redchi
	
if __name__ == '__main__':
	
	gfp = GFP(resolution = 3)
	wl = np.linspace(4000, 8000, 4000)
	fl = gfp.spectrum_sampler(wl, 6500, 6.58)
	
	plt.plot(wl, fl)
	
	result = gfp.fit_spectrum(wl, fl,  mcmc = False)
