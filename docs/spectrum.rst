Spectrum Tools
===================================

Introduction
#############

This module contains general tools to process white dwarf spectra. 

Continuum-Normalizing Balmer Lines
####################################

Continuum-normalizing white dwarf spectra is a challenging task due to the strength of the Balmer absorption lines. :cite:`Bergeron1995` and subsequent studies propose normalizing
each Balmer line individually using the sum of Gaussian functions. We adapt this approach to fit the sum of a linear function and Voigt function to the spectrum around each Balmer line,
and then divide out the linear component to obtain the continuum-normalized spectrum for each absorption line. We concatenate these normalized lines into a discontinuous spectrum that can be
fitted with the ``wdtools.GFP`` routines. 

.. code-block:: python

   import wdtools

   sp = wdtools.SpecTools()

   wl_norm, flux_norm, ivar_norm = sp.normalize_balmer(wl, flux, ivar = ivar, 
   								lines = ['alpha', 'beta', 'gamma', 'delta', 'eps', 'h8'])

Alternatively, you can normalize the entire DA or DB spectrum with ``spectrum.continuum_normalize``. This function uses a pre-selected set of wavelength intervals to fit a smoothing splines continuum
model that avoids known hydrogen and helium absorption features. This is useful for any spectrum with helium components, but if only the hydrogen Balmer lines are present we recommend using ``normalize_balmer`` instead. 

.. code-block:: python

   wl_norm, flux_norm, ivar_norm = sp.continuum_normalize(wl, flux, ivar = ivar)

API
###

.. autoclass:: wdtools.SpecTools
   :members:
   :noindex: