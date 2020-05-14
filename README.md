# wdtools
Package of tools to analyze white dwarf spectra. This is a work in progress, kindly contact the author for details regarding the current stage of development. 

To get started, clone the repository to your working folder and add this repository to your Python path. For example,

``` python
import sys
sys.path.append('/User/yourname/GitHub/wdtools/')

import wdtools

gfp = wdtools.GFP(resolution = 3)

result = gfp.fit_spectrum(wl_norm, flux_norm, ivar_norm, make_plot = True)
```

**Documentation:** https://wdtools.readthedocs.io/en/latest/. 

## References

If using the pre-trained generative neural network for white dwarf model atmospheres, kindly cite the original paper that describes these models: 

Koester (2010) [[ADS](https://ui.adsabs.harvard.edu/abs/2010MmSAI..81..921K/abstract)]

These models also incorporate physics from the following papers (this list is not exhaustive):

Fontaine (2001) [[ADS](https://ui.adsabs.harvard.edu/abs/2001PASP..113..409F/abstract)]

Tremblay & Bergeron (2009) [[ADS](https://ui.adsabs.harvard.edu/abs/2009ApJ...696.1755T/abstract)]

Tassoul et al. (1990) [[ADS](https://ui.adsabs.harvard.edu/abs/1990ApJS...72..335T/abstract)]
