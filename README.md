# wdtools
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/vedantchandra/wdtools?include_prereleases)
![GitHub](https://img.shields.io/github/license/vedantchandra/wdtools)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3828007.svg)](https://doi.org/10.5281/zenodo.3828007)
[![Documentation Status](https://readthedocs.org/projects/wdtools/badge/?version=latest)](https://wdtools.readthedocs.io/en/latest/?badge=latest)

<p align="center">
  <a href="https://wdtools.readthedocs.io/en/latest/">Documentation</a> | <a href="https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.2688C/abstract">Read the Paper</a>
</p>

## Installation

Simply clone the repository to your local machine and run the following code from the base directory:

``` bash
python setup.py install
```

This will install the code into your current Python environment, and you should be able to simply `import wdtools` into your work. If you use conda, ensure that you have activated the environment within which you wish you use this code before running `setup.py`. This code is writen and tested in Python 3.7 and 3.8 only. 

## Example Usage

A full demo is presented in this [Jupyter Notebook](https://wdtools.readthedocs.io/en/latest/examples/1_fitting_wd_spectra.html).

## Citing This Work

If you use wdtools for your research, we would appreciate if you cite the Zenodo repository linked above, as well as our paper describing the package. A BibTeX reference is reproduced below for convenience. 

```
@ARTICLE{Chandra2020,
  author = {{Chandra}, Vedant and {Hwang}, Hsiang-Chih and {Zakamska}, Nadia L. and {Budav{\'a}ri}, Tam{\'a}s},
  title = "{Computational tools for the spectroscopic analysis of white dwarfs}",
  journal = {\mnras},
  year = 2020,
  month = jul,
  volume = {497},
  number = {3},
  pages = {2688-2698},
  doi = {10.1093/mnras/staa2165}
}


@MISC{wdtools, 
  title={wdtools: Computational Tools for the Spectroscopic Analysis of White Dwarfs}, 
  DOI={10.5281/zenodo.3828007}, 
  publisher={Zenodo}, 
  author={Vedant Chandra}, 
  year={2020}
}
```

## Other References

You may also be interested in https://github.com/SihaoCheng/WD_models, which provides tools to simulate and fit white dwarf spectral energy distributions (SEDs) and photometry using synthetic color tables. Another relevant tool is https://github.com/gnarayan/WDmodel, which provides the ability to fit non-LTE white dwarf models (for higher temperatures) to spectroscopy and photometry, and has a more sophisticated treatment of extinction. 

If using the pre-trained generative neural network for white dwarf model atmospheres, kindly cite the original paper that describes these models: 

[Koester (2010)](https://ui.adsabs.harvard.edu/abs/2010MmSAI..81..921K/abstract)

These models also incorporate physics from the following papers (this list is not exhaustive):

[Fontaine (2001)](https://ui.adsabs.harvard.edu/abs/2001PASP..113..409F/abstract)

[Tremblay & Bergeron (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJ...696.1755T/abstract)

[Tassoul et al. (1990)](https://ui.adsabs.harvard.edu/abs/1990ApJS...72..335T/abstract)

The random forest regression model is trained using labelled spectra from the Sloan Digital Sky Survey and [Tremblay et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.5222T/abstract)


