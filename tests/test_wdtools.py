import sys
sys.path.append('../wdtools/')
import wdtools
import numpy as np

def test_gfp():

	gfp = wdtools.GFP(resolution = 1)
	wl = np.linspace(3800, 7000, 7000 - 3800)

	teff = 12000
	logg = 8
	rv = 0

	fl = gfp.spectrum_sampler(wl, teff, logg)
	sigma = 0.005
	fl += sigma * np.random.normal(size = len(fl))
	ivar = np.repeat(1 / sigma**2, len(fl))

	labels, e_labels, redchi = gfp.fit_spectrum(wl, fl, ivar, mcmc = False, 
												fullspec = False, polyorder = 0,
												lines = ['beta', 'gamma', 'delta', 'eps', 'h8'],
												make_plot = False)

	print(labels)
	print(e_labels)
	print(redchi)

	assert labels[0] < teff + 500 and labels[0] > teff - 500
	assert labels[1] < logg + 0.1 and labels[1] > logg - 0.1