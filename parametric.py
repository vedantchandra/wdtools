import numpy as np
from bisect import bisect_left
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from lmfit.models import VoigtModel
import pandas as pd
import pickle
import os
plt.rcParams.update({'font.size': 18})
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
# from .neural import ANN
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample

class LineProfiles:
	
	'''
	General class to fit Voigt profiles to the first 3 Balmer absorption lines, and then infer stellar labels.
	Uses a 25-tree random forest model by default, trained on 5326 spectra from the Sloan Digital Sky Survey. 
	Probabilistic prediction uses 100 boostrapped random forest models with 10 trees each, also SDSS-trained. 
	Ground truth labels are taken from Tremblay et al. (2019)
	Line profiles are fit using the LMFIT package via chi^2 minimization. 

	TODO:

	write radial velocity fitter based on Barstow (repeated fits etc.)

	To get started, use LineProfiles.infer_labels(wavelength, flux)
	For more details, run help(LineProfiles)
	'''

	def __init__(self, verbose = False, plot_profiles = False, modelname = 'bootstrap', n_trees = 25, n_bootstrap = 25):

		self.verbose = verbose
		self.optimizer = 'lm'
		self.halpha = 6564.61
		self.hbeta = 4862.68
		self.hgamma = 4341.68
		self.hdelta = 4102.89
		self.plot_profiles = plot_profiles
		self.n_trees = n_trees
		self.n_bootstrap = n_bootstrap
		self.modelname = modelname

		print('### Line Profiles Tool for White Dwarf Spectra ### \n')
		print('Verbose: '+str(verbose)+', plot_profiles: '+str(plot_profiles))
		print(LineProfiles.__doc__)

		if self.modelname == 'rf':
			self.model = RandomForestRegressor(n_estimators = self.n_trees)
		elif self.modelname == 'bootstrap':
			self.bootstrap_models = [];


	def linear(self, wl, p1, p2):
		
		''' Linear polynomial of degree 1 '''

		return p1 + p2*wl

	def chisquare(self, residual):
		
		''' Chi^2 statistics from residual

		Unscaled chi^2 statistic from an array of residuals (does not account for uncertainties).
		'''

		return np.sum(residual**2)

	def fit_line(self, wl, flux, centroid, window = 400, edges = 200):
		''' Fit Single Line
		
		Base function to fit Voigt profile around a specified centroid on the spectrum. 
		The continuum is normalized at each absorption line via a simple linear polynimial through the edges.
		Window size and edge size can be modified. 
		
		'''

		in1 = bisect_left(wl,centroid-window)
		in2 = bisect_left(wl,centroid+window)
		cropped_wl = wl[in1:in2]
		cropped_flux = flux[in1:in2]

		cmask = (cropped_wl < centroid - edges)+(cropped_wl > centroid + edges)

		p,cov = curve_fit(self.linear,cropped_wl[cmask],cropped_flux[cmask])

		continuum_normalized = 1 - (cropped_flux / self.linear(cropped_wl, p[0], p[1]))
		
		voigtfitter = VoigtModel()
		params = voigtfitter.make_params()
		params['amplitude'].set(min = 0,max = 100,value = 25)
		params['center'].set(value = centroid, max = centroid + 25, min = centroid - 25)
		params['sigma'].set(min = 0, max=200, value=10, vary = True)
		params['gamma'].set(value=10, min = 0, max=200, vary = True)

		result = voigtfitter.fit(continuum_normalized, params, x = cropped_wl, nan_policy = 'omit', method=self.optimizer, fit_kws={'reduce_fcn':self.chisquare})

		if self.plot_profiles == True:
			plt.figure(figsize = (7,5), )
			plt.plot(cropped_wl,1-continuum_normalized, 'k')
			plt.plot(cropped_wl,1-voigtfitter.eval(result.params, x = cropped_wl),'r')
			plt.xlabel('Wavelength ($\mathrm{\AA}$)')
			plt.ylabel('Normalized Flux')
			if centroid == self.halpha:
				plt.title(r'H-$\alpha$')
			elif centroid == self.hbeta:
				plt.title(r'H-$\beta$')
			elif centroid == self.hgamma:
				plt.title(r'H-$\gamma$')
			plt.show()

		return result

	def fit_balmer(self, wl, flux, return_centroids = True):
		''' Fit 3 Balmer Lines
		Input: spectrum wavelength, spectrum flux
		Output: 15 Balmer parameters in array

		Wrapper that runs fit_line on all 3 Balmer lines and returns 5x3 = 15 line parameters from the spectrum. 
		'''
		try:
			alpha_parameters = self.fit_line(wl, flux, self.halpha).params
			beta_parameters = self.fit_line(wl, flux, self.hbeta).params
			gamma_parameters = self.fit_line(wl, flux, self.hgamma, window = 150, edges = 75).params # Modified window for H-Gamma due to the nearby H-Delta line
		except KeyboardInterrupt:
			raise
		except:
			raise
			print('profile fit failed! returning NaN...')
			if return_centroids == True:
				return np.repeat(np.nan, 18)
			else:
				return np.repeat(np.nan, 15)

		balmer_parameters = np.concatenate((alpha_parameters, beta_parameters, gamma_parameters))

		if return_centroids == False:
			balmer_parameters = np.delete(balmer_parameters, [1,7,13]) # Drop line centroids since they aren't used in the model. 

		return balmer_parameters


	def train(self, x_data, y_data):
		'''
		train random forest model
		'''
		if self.modelname == 'rf':
			self.model.fit(x_data, y_data)

			print('random forest trained!')
			return None

		elif self.modelname == 'bootstrap':
			
			for i in range(self.n_bootstrap):
				X_sample, t_sample = resample(x_data, y_data, replace = True, n_samples = int(len(x_data)*0.67))
				rf = RandomForestRegressor(n_estimators = self.n_trees)
				rf.fit(X_sample,t_sample)
				self.bootstrap_models.append(rf)

			print('bootstrap ensemble of random forests is trained!')

			return None

	def labels_from_parameters(self, balmer_parameters):
		''' Predict Labels from Balmer Parameters
		Input: 15 Balmer parameters in array or list
		Output: (Teff (kelvin), Log(g) (log cm/s^2))

		Returns inferred stellar labels from 15 Balmer line parameters. 
		Pass the output of fit_balmer to this. 
		'''
		if balmer_parameters.shape[0] == 18:
			balmer_parameters = np.delete(balmer_parameters, [1,7,13]).reshape(1,-1) # Drop line centroids since they aren't used in the model. 
		else:
			balmer_parameters = balmer_parameters.reshape(1,-1)
		
		if np.isnan(balmer_parameters).any():
			print('NaNs detected! Aborting...')
			return np.repeat(np.nan, 2)
		
		if self.modelname == 'rf':

			predictions = self.model.predict(balmer_parameters)[0] # Deploy instantiated model. Defaults to random forest. 

			return predictions

		elif self.modelname == 'bootstrap':
			predictions = [];
			for bootstrap_model in self.bootstrap_models:
				prediction = bootstrap_model.predict(balmer_parameters)[0]
				predictions.append(prediction)
			predictions = np.asarray(predictions)
			mean_prediction = np.mean(predictions,0)
			std_prediction = np.std(predictions,0)
			labels = np.asarray([mean_prediction[0], std_prediction[0],mean_prediction[1], std_prediction[1]])
			
			return labels

	def save(self, modelname = 'wd'):
		pickle.dump(self.bootstrap_models, open(dir_path+'/models/'+modelname+'.p', 'wb'))
		print('model saved!')

	def load(self, modelname = 'wd'):
		self.bootstrap_models = pickle.load(open(dir_path+'/models/'+modelname+'.p', 'rb'))

	def labels_from_spectrum(self, wl, flux):
		''' Predict Labels from Spectrum
		Input: wavelengths, fluxes
		Output: (Teff (kelvin), Log(g) (log cm/s^2))

		All-in-one function that returns inferred stellar labels from
		a given spectrum by fitting the first 3 Balmer lines
		and deploying a single model of choice to make an inference.
		'''

		balmer_parameters = self.fit_balmer(wl,flux, return_centroids = False) 
		predictions = self.labels_from_parameters(balmer_parameters) # Deploy instantiated model. Defaults to random forest. 


		return predictions


	def pretty_infer_labels(self, wl, flux):
		''' Same as infer_labels, but prints output as well'''
		labels = self.infer_labels(wl,flux)
		print(r'Teff = %i ± %i K, log(g) = %.2f ± %0.2f dex'%(labels[0],labels[1],labels[2],labels[3]))
		return labels