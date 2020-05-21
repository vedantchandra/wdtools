import numpy as np
from bisect import bisect_left
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from lmfit.models import VoigtModel
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
import scipy

plt.rcParams.update({'font.size': 18})
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

class LineProfiles:
	
	'''
	Class to fit Voigt profiles to the first 3 Balmer absorption lines, and then infer stellar labels.

	Probabilistic prediction uses 100 boostrapped random forest models with 25 trees each, trained on 5326 spectra from the Sloan Digital Sky Survey. 
	Ground truth labels are taken from Tremblay et al. (2019)
	Line profiles are fit using the LMFIT package via chi^2 minimization. 
	'''

	def __init__(self, verbose = False, plot_profiles = False, n_trees = 25, n_bootstrap = 25, lines = ['alpha', 'beta', 'gamma']):

		self.verbose = verbose
		self.optimizer = 'lm'
		self.halpha = 6564.61
		self.hbeta = 4862.68
		self.hgamma = 4341.68
		self.hdelta = 4102.89
		self.plot_profiles = plot_profiles
		self.n_trees = n_trees
		self.n_bootstrap = n_bootstrap
		self.lines = lines
		self.linedict = dict(alpha = self.halpha, beta = self.hbeta, gamma = self.hgamma, delta = self.hdelta)
		self.window_dict = dict(alpha = 400, beta = 400, gamma = 150, delta = 75)
		self.edge_dict = dict(alpha = 200, beta = 200, gamma = 75, delta = 65)

		self.features = [];
		self.line_ident = '_'
		self.fit_params = ['amp', 'center', 'sigma', 'gamma', 'fwhm', 'height']
		for linename in lines:
			self.features.append(linename[0] + '_fwhm')
			self.features.append(linename[0] + '_height')
			self.line_ident = self.line_ident + linename[0]

		self.modelname = 'bootstrap'
		self.bootstrap_models = [];

		try:
			self.load('rf_model' + self.line_ident)
		except:
			print('no saved model found for this combination of lines. performing one-time initialization, training and saving model with parameters from 5326 SDSS spectra...')
			self.initialize();


	def linear(self, wl, p1, p2):
		return p1 + p2*wl

	def chisquare(self, residual):
		return np.sum(residual**2)

	def initialize(self):
		pass
		df = pd.read_csv(dir_path + '/models/sdss_parameters.csv')

		targets = ['teff', 'logg']

		clean = (
		    (df['a_fwhm'] < 250)&
		    (df['g_fwhm'] < 250)&
		    (df['b_fwhm'] < 250)&
		    (df['d_fwhm'] < 250)&
		    (df['d_height'] < 1)&
		    (df['g_height'] < 1)&
		    (df['a_height'] < 1)&
		    (df['b_height'] < 1)
		)

		X_train = np.asarray(df[clean][self.features])
		y_train = np.asarray(df[clean][targets])

		self.train(X_train, y_train)
		self.save('rf_model' + self.line_ident)

	def fit_line(self, wl, flux, centroid, window = 400, edges = 200, make_plot = False):
		'''
		Fit a Voigt profile around a specified centroid on the spectrum. 

		The continuum is normalized at each absorption line via a simple linear polynimial through the edges.
		Window size and edge size can be modified. 
		
		Parameters
        ---------
        wl : array
            Wavelength array of spectrum
        flux : array
        	Flux array of spectrum
        centroid : float
        	The theoretical centroid of the absorption line that is being fitted, in wavelength units. 
        window : float, optional
        	How many Angstroms away from the line centroid are included in the fit (in both directions). This should be large enough to include the absorption line as well as 
        	some continuum on either side.
        edges : float, optional
        	What distance in Angstroms around each line (measured from the line center outwards) to exclude from the continuum-fitting step. This should be large enough to cover most of the 
        	absorption line whilst leaving some continuum intact on either side. 
        Returns
        -------
            lmfit `result` object
                A `result` instance from the `lmfit` package, from which fitted parameters and fit statistics can be extracted. 

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

		try:

			result = voigtfitter.fit(continuum_normalized, params, x = cropped_wl, nan_policy = 'omit', method=self.optimizer, fit_kws={'reduce_fcn':self.chisquare})
		except:
			print('line profile fit failed! make sure the selected line is present on the provided spectrum')
			raise

		if make_plot:
			plt.figure(figsize = (6,3), )
			plt.plot(cropped_wl,1-continuum_normalized, 'k')
			plt.plot(cropped_wl,1-voigtfitter.eval(result.params, x = cropped_wl),'r')
			plt.xlabel('Wavelength ($\mathrm{\AA}$)')
			plt.ylabel('Normalized Flux')
			if centroid == self.halpha:
				plt.title(r'H$\alpha$')
			elif centroid == self.hbeta:
				plt.title(r'H$\beta$')
			elif centroid == self.hgamma:
				plt.title(r'H$\gamma$')
			elif centroid == self.hdelta:
				plt.title(r'H$\delta$')
			plt.show()

		return result

	def fit_balmer(self, wl, flux, make_plot = False):

		'''
		Fits Voigt profiles to the first three Balmer lines (H-alpha, H-beta, and H-gamma). Returns all 18 fitted parameters. 
		
		Parameters
        ---------
        wl : array
            Wavelength array of spectrum
        flux : array
        	Flux array of spectrum

        Returns
        -------
            array
                Array of 18 Balmer parameters, 6 for each line. If the profile fit fails, returns array of 18 `np.nan` values. 

		'''
		colnames = [];
		parameters = [];
		try:
			for linename in self.lines:
				line_parameters = np.asarray(self.fit_line(wl, flux, self.linedict[linename], self.window_dict[linename], self.edge_dict[linename], make_plot = make_plot).params)
				colnames.extend([linename[0] + '_' + fparam for fparam in self.fit_params])
				parameters.extend(line_parameters)
		except KeyboardInterrupt:
			raise
		except:
			print('profile fit failed! returning NaN...')
			return np.repeat(np.nan, 18)

		balmer_parameters = pd.DataFrame([parameters], columns = colnames)

		return balmer_parameters

	def train(self, x_data, y_data):
		'''
		Trains ensemble of random forests on the provided data. Does not require scaling.
		
		Parameters
        ---------
        x_data : array
            Input data, independent variables
        y_data : array
        	Output data, dependent variables

		'''

		self.bootstrap_models = [];
		kernel = scipy.stats.gaussian_kde(y_data.T)
		probs = kernel.pdf(y_data.T)
		weights = 1 / probs
		weights = weights / np.nansum(weights)
		
		for i in range(self.n_bootstrap):
			idxarray = np.arange(len(x_data))
			sampleidx = np.random.choice(idxarray, size = int(len(idxarray)*0.67), replace = True, p = weights)
			X_sample, t_sample = x_data[sampleidx], y_data[sampleidx]
			rf = RandomForestRegressor(n_estimators = self.n_trees)
			rf.fit(X_sample,t_sample)
			self.bootstrap_models.append(rf)

		print('bootstrap ensemble of random forests is trained!')

		return None

	def labels_from_parameters(self, balmer_parameters):
		'''
		Predicts stellar labels from Balmer line parameters.
		
		Parameters
        ---------
        balmer_parameters : array
            Array of fitted Balmer parameters from the `fit_balmer` function. 

        Returns
        -------
            array
                Array of predicted stellar labels with the following format: [Teff, e_Teff, logg, e_logg]. 

		'''

		df = balmer_parameters

		balmer_parameters = np.asarray(df[self.features])

		balmer_parameters = balmer_parameters.reshape(1,-1)
		
		if np.isnan(balmer_parameters).any():
			print('NaNs detected! Aborting...')
			return np.repeat(np.nan, 2)
	

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

	def labels_from_spectrum(self, wl, flux, make_plot = False):
		'''
		Wrapper function that directly predicts stellar labels from a provided spectrum. Performs continuum-normalization, fits Balmer profiles, and uses the bootstrap ensemble of random forests to infer labels. 
		
		Parameters
        ---------
        wl : array
            Array of spectrum wavelengths.
        fl : array
            Array of spectrum fluxes. Can be normalized or un-normalized. 

        Returns
        -------
            array
                Array of predicted stellar labels with the following format: [Teff, e_Teff, logg, e_logg]. 
		'''

		balmer_parameters = self.fit_balmer(wl,flux, make_plot = make_plot) 

		predictions = self.labels_from_parameters(balmer_parameters) # Deploy instantiated model. Defaults to random forest. 


		return predictions