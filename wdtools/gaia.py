import numpy as np
from scipy import stats
import emcee
import matplotlib.pyplot as plt
import galpy
from astropy import units as u
from galpy.orbit import Orbit
from astropy.coordinates import SkyCoord
from galpy.potential import MWPotential2014
import matplotlib


def gaia_cov(parallax_error, pmra_error, pmdec_error, 
             parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr):
    
    parallax_pmra_cov = parallax_pmra_corr * parallax_error * pmra_error
    parallax_pmdec_cov = parallax_pmdec_corr * parallax_error * pmdec_error
    pmra_pmdec_cov = pmra_pmdec_corr * pmra_error * pmdec_error
    
    cov = np.zeros((3, 3))
    
    cov[0,0] = parallax_error**2
    cov[0, 1] = parallax_pmra_cov
    cov[0, 2] = parallax_pmdec_cov
    cov[1,0] = parallax_pmra_cov
    cov[1,1] = pmra_error**2
    cov[1,2] = pmra_pmdec_cov
    cov[2,0] = parallax_pmdec_cov
    cov[2,1] = pmra_pmdec_cov
    cov[2,2] = pmdec_error**2
    
    return cov

def log_exp_dec_prior(parallax, L = 1350):
    r = 1000 / parallax #pc
    prior = (1 / (2 * L**3)) * (r**2) * np.exp(-r / L)
    if prior <= 0:
        return -np.Inf
    return np.log(prior)

def log_mvnorm(x, mu, cov):
    return stats.multivariate_normal(mean = mu, cov = cov).logpdf(x)

def get_post_samples(obj, walkers, burn, steps, progress = False, L = 1350):

    cov = gaia_cov(obj['parallax_error'], obj['pmra_error'], obj['pmdec_error'],
            obj['parallax_pmra_corr'], obj['parallax_pmdec_corr'], obj['pmra_pmdec_corr'])

    def logprior(params):
        plx, pmra, pmdec = params
        if plx <= 0:
            return -np.Inf
        else:
            return log_exp_dec_prior(plx, L)

    mu = [obj['parallax'], obj['pmra'], obj['pmdec']]
    gaia_mvnorm = stats.multivariate_normal(mean = mu, cov = cov)

    def loglik(params):
        return gaia_mvnorm.logpdf(params)

    def logprob(params):
        prior = logprior(params)

        if not np.isfinite(prior):
            return -np.Inf
        else:
            return prior + loglik(params)

    ndim,nwalkers = 3, walkers
    p0 = np.zeros((nwalkers, ndim))
    for jj in range(ndim):
        p0[:, jj] = gaia_mvnorm.rvs(size = nwalkers).T[jj]
    p0[:, 0][p0[:, 0] <= 0] = 1e-5

    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)

    b0 = sampler.run_mcmc(p0, burn, progress = progress);
    sampler.reset()
    b1 = sampler.run_mcmc(b0, steps, progress = progress)
    
    return sampler.flatchain

def get_vt_samples(obj, nsample):

    flatchain = get_post_samples(obj, nsample)

    pm_samples = np.sqrt(flatchain[:, 1]**2 +  flatchain[:, 2]**2)
    vt_samples = 4.7 * pm_samples / flatchain[:, 0]
    
    return vt_samples

def get_distance_samples(obj, nburn = 1e2, nsample = 1e3, progress = False):
    
    flatchain = get_post_samples(obj, nsample)
    plx = flatchain[:, 0]
    dist = 1000 / plx
    
    return dist

def get_distance_mode(obj, L = 1350):
    plx = obj['parallax'] / 1000
    e_plx = obj['parallax_error'] / 1000
    
    a =  - 1 / e_plx**2
    b = plx / e_plx**2
    c = -2
    d = 1 / L
    
    polycoef = [d,c,b,a]
    
    roots = np.roots(polycoef)
    dists = [];
    for root in roots:
        if root.imag == 0:
            dists.append(root.real)
        else:
            pass
        
    
    return np.min(dists)

def plot_orbits(name, obj, rv, e_rv, nmc = 10000, norbit = 50):
    present = 0 * u.Gyr
    plxfac = 1
    
    flatchain = get_post_samples(obj, walkers = 100, burn = 50, steps = 100)
    
    ts = np.linspace(0, 500, 1000) * u.Myr

    f, axs = plt.subplots(1, 2, figsize = (15, 7))
    ax1 = axs[0]
    ax2 = axs[1]
    lkwargs = dict(lw = 1, alpha = 0.75, zorder = 0)

    # Draw MW Left

    ax1.scatter([0], [0], marker = '+', s = 75, color = 'k', label = 'Sgr A*')
    ax1.scatter([8], [0], marker = '*', color = 'orange', s = 100, label = 'Sun', edgecolor = 'k', lw = 0.75)

    mw = matplotlib.patches.Ellipse((0,0), 40, 0.25, facecolor = 'none', edgecolor = 'k', linestyle = '--')
    ax1.add_patch(mw)

    # Draw MW Right

    ax2.scatter([0], [0], marker = '+', s = 75, color = 'k', label = 'Sgr A*')
    ax2.scatter([8], [0], marker = '*', color = 'orange', s = 100, label = 'Sun', edgecolor = 'k', lw = 0.75)

    mw = matplotlib.patches.Circle((0,0), 20, facecolor = 'none', edgecolor = 'k', linestyle = '--')
    ax2.add_patch(mw)
            
    ra_s = obj['ra'] + obj['ra_error'] * np.random.normal(size = nmc)
    dec_s = obj['dec'] + obj['dec_error'] * np.random.normal(size = nmc)
    pmra_s = flatchain[-nmc:, 1]
    pmdec_s = flatchain[-nmc:, 2]
    plx_s = flatchain[-nmc:, 0]
    rv_s = rv + e_rv * np.random.normal(size = nmc)

    coord = SkyCoord(
            ra = (ra_s) * u.degree, 
            dec = (dec_s) * u.degree,
            pm_ra_cosdec = pmra_s * u.mas / u.year, 
            pm_dec = pmdec_s * u.mas / u.year,
            radial_velocity = (rv_s) * u.km / u.s, 
            distance = (1000 / (plxfac * plx_s)) * u.pc
    )
            
    gcoord = coord.galactocentric
        
    Us = gcoord.v_x.value
    Vs = gcoord.v_y.value
    Ws = gcoord.v_z.value
                
    for kk in range(norbit):
        idx = np.random.randint(0, nmc)
        op = Orbit(coord[idx])
        rev_op = op.flip()

        op.integrate(ts, MWPotential2014)
        rev_op.integrate(ts, MWPotential2014)
        times = op.time() * u.Gyr

        ax1.scatter(op(present).x(), op(present).z(), color = 'tab:red', s = 100, marker = '*', alpha = 0.25,
                label = name, edgecolor = 'k', lw = 0.5, zorder = 10)
        ax1.plot(rev_op(times).x(), rev_op(times).z(), color = 'C0', label = 'Past', **lkwargs)
        ax1.plot(op(times).x(), op(times).z(), color = 'C1', label = 'Future', **lkwargs)

        ax2.scatter(op(present).x(), op(present).y(), color = 'tab:red', s = 100, marker = '*', alpha = 0.25,
                label = name, edgecolor = 'k', lw = 0.5, zorder = 10)
        ax2.plot(rev_op(times).x(), rev_op(times).y(), color = 'C0', label = 'Past', **lkwargs)
        ax2.plot(op(times).x(), op(times).y(), color = 'C1', label = 'Future', **lkwargs)
            
    U = np.median(Us)
    eU = stats.median_abs_deviation(Us, scale = 'normal')
    V = np.median(Vs)
    eV = stats.median_abs_deviation(Vs, scale = 'normal')
    W = np.median(Ws)
    eW = stats.median_abs_deviation(Ws, scale = 'normal')
        
    Vtot = np.sqrt(U**2 + V**2 + W**2)
    eVtot = np.sqrt( (U*eU/Vtot)**2 + (V*eV/Vtot)**2 + (W*eW/Vtot)**2)
    
    twargs = dict(transform = ax1.transAxes, fontsize = 14)
    ax1.text(0.1, 0.25, r'$v_x\ =\ %i\ \pm\ %i km/s$' % (U, eU), **twargs)
    ax1.text(0.1, 0.2, r'$v_y\ =\ %i\ \pm\ %i km/s$' % (V, eV), **twargs)
    ax1.text(0.1, 0.15, r'$v_z\ =\ %i\ \pm\ %i km/s$' % (W, eW), **twargs)
    ax1.text(0.1, 0.1, r'$v_{tot}\ =\ %i\ \pm\ %i km/s$' % (Vtot, eVtot), **twargs)
    
    ax1.set_xlim(-30, 30)
    ax1.set_ylim(-10, 10)

    ax1.set_xlabel('X (kpc)')
    ax1.set_ylabel('Z (kpc)')

    ax2.set_xlim(-50, 50)
    ax2.set_ylim(-50, 50)

    ax2.set_xlabel('X (kpc)')
    ax2.set_ylabel('Y (kpc)')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = plt.legend(by_label.values(), by_label.keys())

    for lh in leg.legendHandles: 
        lh.set_alpha(1)
        
    return op