Generative Fitting Pipeline
===================================

Introduction
#############

This module includes tools to generate synthetic white dwarf spectra using theoretical models and a speedy generative neural network.
These can be used to fit observed spectra in an MCMC framework to recover parameters like effective temperature, surface gravity, and radial velocity. 
The synthetic spectra are generated from models developed by :cite:`Koester2010`, and we use the ``emcee`` fitting routine from :cite:`DFM2013`. 

Say we have an un-normalized DA spectrum from the Sloan Digital Sky Survey (SDSS), defined by wavelengths ``wl``, fluxes ``fl``, and an inverse variance mask ``ivar``. 

.. code-block:: python

   import wdtools

   sp = wdtools.SpecTools()
   mcfit = wdtools.GFP(resolution = 3, specclass = 'DA')

   wl_norm, flux_norm, ivar_norm = sp.normalize_balmer(wl, flux, ivar = ivar)
   sampler = mcfit.fit_spectrum(wl_norm, flux_norm, ivar_norm, init = 'mle', make_plot = True)

The returned `sampler` object is an emcee sampler instance, from which posterior samples can be obtained using ``sampler.flatchain``, and the chi squared value of each posterior sample can be obtained with ``chis = - 2 * sampler.get_log_probs()``. 

.. bibliography:: bib.bib
   :style: plain

API
###

.. autoclass:: wdtools.GFP
   :special-members:
   :members: