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
plt.rcParams.update({'font.size': 18})
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
import inspect
from keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adamax
from scipy.interpolate import interp1d
#from astroNN.nn import layers as annlayers
from sklearn.preprocessing import MinMaxScaler
from .spectrum import SpecTools

class CNN:
	'''

	Base class for a simple convolutional neural network. Work in progress, do not use.

	'''

	def __init__(self, n_input = 4000, n_output = 2, n_hidden = 2, neurons = 32, n_conv = 2, n_filters = 4, filter_size = 8, pool_size = 4, activation = 'relu',\
		output_activation = 'linear', regularization = 0, loss = 'mse', bayesian = False, dropout = 0.1,\
		input_bounds = 'none', output_bounds = 'none', model = 'default'):

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
		self.input_bounds = input_bounds
		self.output_bounds = output_bounds
		self.scalewarningflag = 0
		self.scale_input = False
		self.scale_output = False
		self.input_scaler = MinMaxScaler()
		self.output_scaler = MinMaxScaler()

		if not isinstance(self.neurons,list):
			self.neurons = np.repeat(self.neurons, self.n_hidden)

		if not isinstance(self.n_filters,list):
			self.n_filters = np.repeat(self.n_filters, self.n_conv)

		if model == 'default':
			self.model = self.nn()
			self.is_initialized = True

		if model == 'bayesnn':
			self.n_input = 3000
			self.n_output = 2
			self.n_hidden = 2
			self.neurons = [196,86]
			self.n_filters = [2,4]
			self.n_conv = 2

			self.pool_size = 4
			self.filter_size = 8
			self.bayesian = True
			self.output_bounds = [[2500, 100000], [5, 10]];
			self.scale_output = True
			self.scale_input = False
			self.output_scaler.fit(np.asarray(self.output_bounds).T)
			self.model = self.nn()
			self.is_initialized = True
			self.scalewarningflag = False
	def nn(self):

		if len(self.neurons) != self.n_hidden:
			print('neuron list inconsistent with number of layers')
			raise

		x = Input(batch_shape=(None, self.n_input, 1), name = 'Input')

		y = Conv1D(self.n_filters[0], (self.filter_size), padding = 'same', activation = self.activation, kernel_regularizer = l2(self.reg), name = 'Conv_1')(x)
		for ii in range(self.n_conv - 1):
			if self.bayesian:
				y = Dropout(self.dropout)(y, training = True)
			y = Conv1D(self.n_filters[ii+1], (self.filter_size), padding = 'same', activation = self.activation, kernel_regularizer = l2(self.reg), name = 'Conv_'+str(ii+2))(y)

		y = MaxPooling1D(self.pool_size) (y)
		y = Flatten()(y)

		for ii in range(self.n_hidden):
			if self.bayesian:
				y = Dropout(self.dropout)(y, training = True)
			y = Dense(self.neurons[ii], activation = self.activation, kernel_regularizer = l2(self.reg), name = 'Dense_'+str(ii+1))(y)

		if self.bayesian:
			y = Dropout(self.dropout)(y, training = True)
		
		out = Dense(self.n_output, activation = self.output_activation, name = 'Output')(y)

		network = Model(inputs = x, outputs = out)
		network.compile(optimizer = Adamax(), loss = self.loss)
		return network

	def train(self, x_data, y_data, model = 'default', n_epochs = 100, batchsize = 64, verbose = 0):
		
		if not self.scaler_isfit and self.scalewarningflag == 0:
			print('Warning! Assuming data is scaled. If not, use fit_scaler(X,Y), then train.')
			self.scalewarningflag = 1

		if not self.scale_input:
			x_data = x_data.reshape(len(x_data), self.n_input, 1)
		elif self.scale_input:
			x_data = self.input_scaler.transform(x_data).reshape(len(x_data), self.n_input, 1)
		
		if self.scale_output:
			y_data = self.output_scaler.transform(y_data)
		
		h = self.model.fit(x_data, y_data, epochs = n_epochs, verbose = verbose, batch_size = batchsize)
		return h

	def eval_data(self, x_data, model = 'default', n_bootstrap = 100):

		if model == 'default':
			try:
				model = self.model
			except:
				print('model not trained! use train() or explicitly pass a model to eval()')
				raise
		if self.scale_input:
			x_data = self.input_scaler.transform(x_data).reshape(len(x_data), self.n_input, 1)
		elif not self.scale_input:
			x_data = x_data.reshape(len(x_data), self.n_input, 1)
		
		if self.bayesian:
			if self.scale_output:
				predictions = np.asarray([self.output_scaler.inverse_transform(self.model.predict(x_data)) for i in range(n_bootstrap)])
			elif not self.scale_output:
				predictions = np.asarray([self.model.predict(x_data) for i in range(n_bootstrap)])
			means = np.nanmean(predictions,0)
			stds = np.nanstd(predictions,0)
			results = np.empty((means.shape[0], means.shape[1] + stds.shape[1]), dtype = means.dtype)
			results[:,0::2] = means
			results[:,1::2] = stds
			return results

		elif not self.bayesian:
			if self.scale_output:
				return self.output_scaler.inverse_transform(self.model.predict(x_data))
			elif not self.scale_output:
				return self.model.predict(x_data)

	def labels_from_spectrum(self, wl, flux):
		func = interp1d(wl, flux)
		interp_flux = func(self.lamgrid)
		return self.eval_data(interp_flux.reshape(1,-1))[0]

	def fit_scaler(self, x_data, y_data):
		self.input_bounds = np.asarray([np.min(x_data,0),np.max(x_data,0)]).T
		self.output_bounds = np.asarray([np.min(y_data,0),np.max(y_data,0)]).T

		self.input_scaler.fit(self.input_bounds.T)
		self.output_scaler.fit(self.output_bounds.T)
		print('new bounds established!')
		self.scaler_isfit = True
		self.scale_input = True
		self.scale_output = True
		return None

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