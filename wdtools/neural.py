import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from bisect import bisect_left
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from lmfit.models import VoigtModel
import pandas as pd
import pickle
import os
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
import inspect
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adamax
from scipy.interpolate import interp1d
#from astroNN.nn import layers as annlayers

from .spectrum import SpecTools

class CNN:
	'''

	Base class for a simple convolutional neural network. Work in progress, do not use.

	'''

	def __init__(self, n_input = 4000, n_output = 2, n_hidden = 2, neurons = 32, n_conv = 2, n_filters = 4, filter_size = 8, pool_size = 4, activation = 'relu',\
		output_activation = 'linear', regularization = 0, loss = 'mse', bayesian = False, dropout = 0.1,\
		model = 'bayesnn'):

		'''
		
		Args:
			n_input (int): number of inputs
			n_output (int): number of outputs
			n_hidden (int): number of hiddden dense layers
			neurons (int): number of neurons per hidden layers
			n_conv (int): number of convolutional layers
			n_filters (int): number of filters per layer
			filter_size (int): width of each filter
			activation (str): hidden layer activation function (keras format)
			output_activation (str): output activation function (keras format)
			regularization (float): l2 regularization hyperparameter
			loss (str): loss function (keras format)
			bayesian (bool): enable dropout variational inference
			dropout (float): dropout fraction for bayesian inference
			input_bounds (list): list of tuple bounds for each input
			output_bounds (list): list of tuple bounds for each output
			model (str): use a built-in preloaded model
			
		'''


		self.is_initialized = False
		self.n_input = n_input
		self.n_output = n_output
		self.n_hidden = n_hidden
		self.activation = activation
		self.output_activation = output_activation
		self.reg = regularization
		self.neurons = neurons
		self.loss = loss
		self.dropout = dropout
		self.bayesian = bayesian
		self.scaler_isfit = False
		self.n_conv = n_conv
		self.n_filters = n_filters
		self.filter_size = filter_size
		self.pool_size = pool_size
		self.lamgrid = np.linspace(4000, 7000, 3000)

		if not isinstance(self.neurons,list):
			self.neurons = np.repeat(self.neurons, self.n_hidden)

		if not isinstance(self.n_filters,list):
			self.n_filters = np.repeat(self.n_filters, self.n_conv)

		if model == 'bayesnn':
			self.n_input = 3000
			self.n_output = 2
			self.n_hidden = 2
			self.neurons = 64
			self.n_filters = [8,4]
			self.n_conv = 2
			self.activation = 'relu'
			self.dropout = 0
			self.reg = 0.00
			self.pool_size = 3
			self.filter_size = 3
			self.bayesian = True
			self.output_activation = 'linear'
			self.loss = 'mse'
			self.model = self.nn()

	def nn(self):

		x = Input(batch_shape=(None, self.n_input, 1), name = 'Input')

		y = Conv1D(self.n_filters[0], (self.filter_size), padding = 'same', activation = self.activation, kernel_regularizer = l2(self.reg))(x)
		#y = Dropout(self.dropout)(y, training = True)
		#y = Conv1D(self.n_filters[ii+1], (self.filter_size), padding = 'same', activation = self.activation, kernel_regularizer = l2(self.reg), name = 'Conv_'+str(ii+2))(y)

		y = MaxPooling1D(self.pool_size) (y)
		y = Flatten()(y)

		#y = Dropout(self.dropout)(y, training = True)
		y = Dense(self.neurons, activation = self.activation, kernel_regularizer = l2(self.reg))(y)
		y = Dense(self.neurons, activation = self.activation, kernel_regularizer = l2(self.reg))(y)

		#y = Dropout(self.dropout)(y, training = True)
		
		out = Dense(self.n_output, activation = self.output_activation, name = 'Output')(y)

		network = Model(inputs = x, outputs = out)
		network.compile(optimizer = Adamax(), loss = self.loss)
		return network

	def train(self, x_data, y_data, model = 'default', n_epochs = 100, batchsize = 64, verbose = 0):
		

		x_data = x_data.reshape(len(x_data), self.n_input, 1)
		
		y_data = self.label_sc(y_data)
		
		h = self.model.fit(x_data, y_data, epochs = n_epochs, verbose = verbose, batch_size = batchsize)

		return h

	def eval_data(self, x_data, model = 'default', n_bootstrap = 100):


		x_data = x_data.reshape(len(x_data), self.n_input, 1)
		
		if self.bayesian:
			predictions = np.asarray([self.inv_label_sc(self.model.predict(x_data)) for i in range(n_bootstrap)])
			means = np.nanmean(predictions,0)
			stds = np.nanstd(predictions,0)
			results = np.empty((means.shape[0], means.shape[1] + stds.shape[1]), dtype = means.dtype)
			results[:,0::2] = means
			results[:,1::2] = stds
			return results

		elif not self.bayesian:
			return self.inv_label_sc(self.model.predict(x_data))


	def labels_from_spectrum(self, wl, flux):
		func = interp1d(wl, flux)
		interp_flux = func(self.lamgrid)
		return self.eval_data(interp_flux.reshape(1,-1))[0]

	def save(self, modelname):
		self.model.save_weights(dir_path + '/models/' + modelname + '.h5')
		print('model saved!')

	def load(self, modelname):
		self.model.load_weights(dir_path + '/models/' + modelname + '.h5')
		print('model loaded!')

	def args(self):
		print(self.__init__.__doc__)
		return None

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