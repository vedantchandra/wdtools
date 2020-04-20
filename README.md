# wdtools (WIP)
Package of tools to analyze white dwarf spectra. This is a work in progress, kindly contact the author for details regarding the current stage of development. 

To get started, clone the repository to your working folder. Each module can be imported into Python. Full documentation is WIP, refer to inbuilt docstrings with help() for a list of functions with descriptions for each class. 

## Spectrum 

Spectral processing tools like continuum-normalization, line fitting, and absorption line centroid determination (for radial velocity).

```python
from wdtools.spectrum import SpecTools
spectools = SpecTools()

wl_norm, fl_norm = spectools.normalize_balmer(wl, fl)
```

## Parametric

Functions to infer effective temperature and surface gravity from parametric measures of the Balmer absorption lines. Includes routines to fit the absorption lines and subsequently make inferences with a random forest regression model. Also includes a bootstrapped random forest regression model to produce labels with uncertainties. 

## Neural Network

Contains flexible base classes for artificial neural netoworks and convolutional neural networks. Some preloaded models include bloodhound (binary classification of spectra with convolutional neural networks) and bayesnn (stellar label inference straight from SDSS spectra).
