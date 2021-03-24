Generative Fitting Pipeline
===================================

Introduction
#############

This module includes tools to generate synthetic white dwarf spectra using theoretical models and a speedy generative neural network.
These can be used to fit observed spectra in an MCMC framework to recover parameters like effective temperature, surface gravity, and radial velocity. 
The synthetic spectra are generated from models developed by :cite:`Koester2010`, and we use the ``emcee`` fitting routine from :cite:`DFM2013`. 

Say we have an un-normalized DA spectrum from the Sloan Digital Sky Survey (SDSS), defined by wavelengths ``wl``, fluxes ``fl``, and an inverse variance array ``ivar``. 

.. code-block:: python

   import wdtools

   gfp = wdtools.GFP(resolution = 3) # Spectral dispersion in Angstroms

   labels, e_labels, redchi = gfp.fit_spectrum(wl, flux, ivar,
                                mcmc = True, nwalkers = 50, burn = 100, ndraws = 100,
                                make_plot = True, plot_corner = True
                                )

``label`` and ``e_labels`` respectively are arrays of the fitted parameters. ``redchi`` is the reduced chi-square statistic, which can be used as a rough estimate of the goodness-of-fit. More details are in our paper, and a more complete example can be found in the tutorial. 

.. bibliography:: bib.bib
   :style: plain

API
###

.. autoclass:: wdtools.GFP
   :members:
   :noindex:
