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
   gfp = wdtools.GFP(resolution = 3, specclass = 'DA')

   wl_norm, flux_norm, ivar_norm = sp.normalize_balmer(wl, flux, ivar = ivar)
   labels, e_labels, redchi = gfp.fit_spectrum(wl_norm, flux_norm, ivar_norm, init = 'de', make_plot = True)

``label`` and ``e_labels`` respectively are 3-arrays of the fitted effective temperature, log surface gravity, and radial velocity along with respective uncertainties. ``redchi`` is the reduced chi-square statistic, which can be used as a rough estimate of the goodness-of-fit. More details are in our paper. 

.. bibliography:: bib.bib
   :style: plain

API
###

.. autoclass:: wdtools.GFP
   :members:
   :noindex: