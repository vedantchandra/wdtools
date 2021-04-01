import sys
sys.path.append('../wdtools/')
import wdtools
import numpy as np

def test_gfp():

	gfp = wdtools.GFP(resolution = 1)
	wl = np.linspace(3800, 7000, 7000 - 3800)

	teffs = np.linspace(7000, 35000, 5)
	loggs = np.linspace(7, 9, 3)
	rv = 0

	SN = 100

	for teff in teffs:
		for logg in loggs:
			fl = gfp.spectrum_sampler(wl, teff, logg)
			sigma = fl / SN
			fl += sigma * np.random.normal(size = len(fl))
			ivar = 1 / sigma**2 #np.repeat(1 / sigma**2, len(fl))

			labels, e_labels, redchi = gfp.fit_spectrum(wl, fl, ivar, mcmc = False, 
														fullspec = False, polyorder = 0,
														lines = ['beta', 'gamma', 'delta', 'eps', 'h8'],
														make_plot = False)

			print(labels)
			print(e_labels)
			print(redchi)

			assert labels[0] < teff + 500 and labels[0] > teff - 500
			assert labels[1] < logg + 0.25 and labels[1] > logg - 0.25

if __name__ == '__main__':
	test_gfp()