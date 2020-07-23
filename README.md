# wdtools
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/vedantchandra/wdtools?include_prereleases)
[![Documentation Status](https://readthedocs.org/projects/wdtools/badge/?version=latest)](https://wdtools.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/vedantchandra/wdtools)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3828007.svg)](https://doi.org/10.5281/zenodo.3828007)

**Full Documentation:** https://wdtools.readthedocs.io/en/latest/. 

To get started, clone the repository to your working folder and add this repository to your Python path. We recommend installing dependencies using ``conda`` and the provided ``environment.yaml`` file. 

Example usage of wdtools to infer the temperature and surface gravity of a white dwarf from its spectrum:

``` python
import sys
import numpy as np
from scipy import stats
sys.path.append('~/GitHub/wdtools/')

import wdtools

gfp = wdtools.GFP(resolution = 3)

labels, e_labels, redchi = gfp.fit_spectrum(wl_norm, flux_norm, ivar_norm, init = 'de', make_plot = True)

teff, e_teff = labels[0], e_labels[0]
logg, e_logg = labels[1], e_labels[1]
rv, e_rv = labels[2], e_labels[2]
```

You may also be interested in https://github.com/SihaoCheng/WD_models, which provides tools to simulate and fit white dwarf spectral energy distributions (SEDs) and photometry using synthetic color tables. Another relevant tool is https://github.com/gnarayan/WDmodel, which provides the ability to fit non-LTE white dwarf models (for higher temperatures) to spectroscopy and photometry, and has a more sophisticated treatment of extinction. 

## References

If you use wdtools for your research, we would appreciate if you cite the Zenodo repository linked above, as well as our paper describing the package. A BibTeX reference is reproduced below for convenience. 

```
@ARTICLE{Chandra2020,
author = {{Chandra}, Vedant and {Hwang}, Hsiang-Chih and
{Zakamska}, Nadia and {Budavari}, Tamas},
title = "{Computational Tools for the Spectroscopic Analysis of White Dwarfs}",
journal = {MNRAS (in press)},
year = {2020}
}

@MISC{wdtools, 
title={wdtools: Computational Tools for the Spectroscopic Analysis of White Dwarfs}, 
DOI={10.5281/zenodo.3828007}, 
publisher={Zenodo}, 
author={Vedant Chandra}, 
year={2020}
}
```


If using the pre-trained generative neural network for white dwarf model atmospheres, kindly cite the original paper that describes these models: 

Koester (2010) [[ADS](https://ui.adsabs.harvard.edu/abs/2010MmSAI..81..921K/abstract)]

These models also incorporate physics from the following papers (this list is not exhaustive):

Fontaine (2001) [[ADS](https://ui.adsabs.harvard.edu/abs/2001PASP..113..409F/abstract)]

Tremblay & Bergeron (2009) [[ADS](https://ui.adsabs.harvard.edu/abs/2009ApJ...696.1755T/abstract)]

Tassoul et al. (1990) [[ADS](https://ui.adsabs.harvard.edu/abs/1990ApJS...72..335T/abstract)]

The random forest regression model is trained using labelled spectra from the Sloan Digital Sky Survey and Tremblay et al. (2019):

Tremblay et al. (2019) [[ADS](https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.5222T/abstract)]


