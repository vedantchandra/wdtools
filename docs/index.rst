wdtools: Computational Tools for the Spectroscopic Analysis of White Dwarfs
=========================================================================================

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

If you use this package, kindly cite the following reference:

.. code-block:: none

   @ARTICLE{wdtools,
   author = {{Chandra}, Vedant and {Hwang}, Hsiang-Chih and
   {Zakamska}, Nadia and {Budavari}, Tamas},
   title = "{Computational Techniques for the Spectroscopic Analysis of White Dwarfs}",
   journal = {MNRAS (submitted)},
   year = "2020"}

Please don't hesitate to reach out or `raise an issue <https://github.com/vedantchandra/wdtools/issues>`_ for any bug reports, feature requests, or general feedback!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install

   gfp

   spectrum

   parametric

   examples/1_fitting_wd_spectra
