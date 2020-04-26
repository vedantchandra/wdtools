# wdtools (WIP)
Package of tools to analyze white dwarf spectra. This is a work in progress, kindly contact the author for details regarding the current stage of development. 

To get started, clone the repository to your working folder. Each module can be imported into Python. Full documentation is available here: https://wdtools.readthedocs.io/en/latest/. 

## spectrum.py

Include spectral processing tools like continuum-normalization, line fitting, and absorption line centroid fitting. The `normalize_balmer` function performs a specialized continuum normalization around each hydrogen Balmer absorption line and concatenates them together, adapting the technique of Bergeron et al. (1995). It is not suitable for non-DA white dwarfs. 

```python
from wdtools.spectrum import SpecTools
spectools = SpecTools()

wl_norm, fl_norm = spectools.normalize_balmer(wl, fl)
```

## mcfit.py

Fitting tool that uses theoretical synthetic spectra to infer temperature and surface gravity from spectroscopic observations of hydrogen-rich white dwarfs. The synthetic spectra are generated with a pre-trained neural network that interpolates the theoretical models of D. Koester (2010, reference below). The fitting procedure uses Markov-Chain Monte Carlo sampling to derive the posterior parameter distributions from the observed spectrum. 

```python
from wdtools.mcfit import MCFit
mcfit = MCFit(resolution = 3)

sampler = mcfit.fit_spectrum(wl_norm, fl_norm, ivar_norm, prior_teff = None, make_plot = True)
```

`fit_spectrum` returns an `emcee` sampler object. In the above example, `sampler.flatchain` returns an array with posterior samples of effective temperature, log surface gravity, and radial velocity, which can then be described with statistics like the median and interquartile range.

## parametric.py

Tool to infer effective temperature and surface gravity from parametric measures of the Balmer absorption lines. Includes routines to fit the absorption lines and subsequently make inferences with a random forest regression model. Continuua are automatically normalized during the profile fitting step, and the fitting is invariant under radial velocity transformations. 

The default model is pre-trained on 5000 white dwarfs from the Sloan Digital Sky Survey with spectroscopic labels from Tremblay et al. (2015). Our ensemble model consists of 100 random forests trained on 67% bootstrapped sub-samples of the parent dataset, enabling us to estimate the uncertainties of the predicted labels. 

```python
from wdtools.parametric import LineProfiles()
lp = LineProfiles(modelname = 'bootstrap')

labels = lp.labels_from_spectrum(wl, fl)
```

The returned `labels` is an array with 4 elements - the fitted temperature, its uncertainty, fitted log surface gravity and its uncertainty. 

## neural.py

Contains flexible base classes for artificial neural netoworks and convolutional neural networks. Some preloaded models include `bloodhound` (binary classification of spectra with convolutional neural networks) and `bayesnn` (stellar label inference straight from SDSS spectra).

```python
from wdtools.neural import CNN
from wdtools.spectrum import SpecTools
sp = SpecTools()
cnn = CNN(model = 'wd_bayesnn')

wl_norm, fl_norm = sp.normalize_balmer(wl, fl)
labels = cnn.eval(fl_norm)
```

The `wd_bayesnn` neural network is pre-trained on 5000 SDSS white dwarfs with labels from Tremblay et al. (2019). The returned `labels` is an array with 4 elements - the fitted temperature, its uncertainty, fitted log surface gravity and its uncertainty. 

## References

If using the pre-trained generative neural network for white dwarf model atmospheres, kindly cite the original paper that describes these models: 

Koester 2010 [[ADS](https://ui.adsabs.harvard.edu/abs/2010MmSAI..81..921K/abstract)]

These models also incorporate physics from the following papers (this list is not exhaustive):

Fontaine 2001 [[ADS](https://ui.adsabs.harvard.edu/abs/2001PASP..113..409F/abstract)]

Tremblay & Bergeron 2009 [[ADS](https://ui.adsabs.harvard.edu/abs/2009ApJ...696.1755T/abstract)]

Tassoul et al. 1990 [[ADS](https://ui.adsabs.harvard.edu/abs/1990ApJS...72..335T/abstract)]
