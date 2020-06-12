wdtools: Computational Techniques for the Spectroscopic Analysis of White Dwarf Stars
=========================================================================================

.. image:: https://readthedocs.org/projects/wdtools/badge/?version=latest
   :target: https://wdtools.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

A collection of methods to characterize the atmospheric parameters of white dwarfs using spectroscopic data. The flagship class is the generative fitting pipeline (GFP), 
which fits ab-initio theoretical models to observed spectra in a Bayesian framework, using high-speed neural networks to interpolate the models.

If you use this package, kindly cite the following reference:

.. code-block:: none

   @ARTICLE{wdtools,
   author = {{Chandra}, Vedant and {Hwang}, Hsiang-Chih and
   {Zakamska}, Nadia and {Budavari}, Tamas},
   title = "{Computational Techniques for the Spectroscopic Analysis of White Dwarfs}",
   journal = {MNRAS (submitted)},
   year = "2020"}

Please don't hesitate to reach out for any bug reports, feature requests, or general feedback!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install

   gfp

   spectrum

   parametric

   examples/1_fitting_wd_spectra.ipynb