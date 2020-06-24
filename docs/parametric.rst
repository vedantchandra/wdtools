Random Forest Regression
===================================

Introduction
#############

This method fits white dwarf Balmer lines with parametric Voigt profiles, deriving their full-width at half-max (FWHM) and line amplitudes. The lines parameters of width and breadth are used with a random forest regression model to predict the stellar labels of effective temperature and surface gravity. Currently, this model uses the first four Balmer lines (or any subset therein), and ships pre-trained on 5000 spectra from the Sloan Digital Sky Survey with stellar labels calculated by :cite:`Tremblay2019`.

.. bibliography:: bib.bib
   :style: plain

API
###

.. autoclass:: wdtools.LineProfiles
   :members:
   :noindex:
