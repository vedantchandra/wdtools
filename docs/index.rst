wdtools
===========

.. image:: https://img.shields.io/github/v/release/vedantchandra/wdtools?include_prereleases   
   :alt: GitHub release (latest by date including pre-releases)
   
.. image:: https://readthedocs.org/projects/wdtools/badge/?version=latest
   :target: https://wdtools.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
   
.. image:: https://img.shields.io/github/license/vedantchandra/wdtools   
   :alt: GitHub
   
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3828686.svg
   :target: https://doi.org/10.5281/zenodo.3828686

wdtools is a collection of techniques to characterize the atmospheric parameters of white dwarfs using spectroscopic data. The flagship class is the generative fitting pipeline (GFP), which fits ab-initio theoretical models to observed spectra in a Bayesian framework, using high-speed neural networks to interpolate synthetic spectra.

If you use this package, kindly cite the following references:

.. code-block:: none

   @ARTICLE{Chandra2020,
   author = {{Chandra}, Vedant and {Hwang}, Hsiang-Chih and {Zakamska}, Nadia L. and {Budav{\'a}ri}, Tam{\'a}s},
   title = "{Computational tools for the spectroscopic analysis of white dwarfs}",
   journal = {MNRAS},
   year = 2020,
   month = jul,
   volume = {497},
   number = {3},
   pages = {2688-2698},
   doi = {10.1093/mnras/staa2165},
   }
   
   @MISC{wdtools, 
   title={wdtools: Computational Tools for the Spectroscopic Analysis of White Dwarfs}, 
   DOI={10.5281/zenodo.3828686}, 
   publisher={Zenodo}, 
   author={Vedant Chandra}, 
   year={2020}}

Please don't hesitate to reach out or `raise an issue <https://github.com/vedantchandra/wdtools/issues>`_ for any bug reports, feature requests, or general feedback!

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   self

   install

   gfp
   
   parametric

   examples/1_fitting_wd_spectra

   spectrum
