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
                                fullspec = True, mcmc = True, nwalkers = 50, burn = 100, ndraws = 100,
                                make_plot = True, plot_corner = True
                                )

The GFP first estimates the radial velocity of the provided spectrum using the H-alpha absorption line. The synthetic models are shifted to this RV during the fitting process–hence there is no interpolation or re-binning of the observed spectrum, preventing correlated errors. The spectrum is then spline-normalized, during which the strong Balmer lines are masked out. 

If the ``fullspec`` argument is ``True``, then the entire spectrum is used during the fitting process. Otherwise, only the Balmer lines are used in the likelihood. You can select which Balmer lines to include in the fit with the ``lines`` argument (defaults to all lines from 'alpha' to 'h8'). 

Then, the ``lmfit`` least squares routine is used to fit the stellar parameters. The fit is initialized at several equidistant initial temperatures governed by the ``nteff`` keyword. The fit with the lowest chi-square is selected and returned. 

If the ``polyorder`` argument is greater than zero, then the continuum-normalized synthetic spectra have a Chebyshev polynomial of order ``polyorder`` added to them during the fitting process. These coefficients are also solved for and returned by the GFP. If you use this option, it's recommended to also run ``mcmc`` so that these coefficients are properly marginalized over. Also, it's only recommended to use ``polyorder`` if ``fullspec`` is ``True``.  

If the ``mcmc`` argument is ``True``, then the ``lmfit`` estimates are used as a starting point for a full MCMC run, governed by the ``nwalkers``, ``burn``, and ``ndraws`` arguements. If ``mcmc`` is ``False``, then the returned uncertainties are estimated from the covariance matrix returned by ``lmfit``. Turning off ``mcmc`` results in much quicker results, but the error estimates might not be robust. 

``label`` and ``e_labels`` respectively are arrays of the fitted parameters. The radial velocity (and RV error) are always the last elements in the array, so if ``polyorder`` > 0, the label array will have temperature, surface gravity, the Chebyshev coefficients, and then RV. ``redchi`` is the reduced chi-square statistic, which can be used as a rough estimate of the goodness-of-fit. More details are in our paper, and a more complete example can be found in the tutorial. 

.. bibliography:: bib.bib
   :style: plain

API
###

.. autoclass:: wdtools.GFP
   :members:
   :noindex:
