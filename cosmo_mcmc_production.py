from classy import Class
import numpy as np
import pandas as pd
import emcee
import tqdm
from multiprocessing import Pool

# Import our model functions (this file also loads the PLANCK data)
import cosmo_helpers as ch

# Set up you 'guess' for the MCMC (IE, the first theta)
theta_start = np.array([0.0224, 0.120, 0.678, np.pi*2]) 
# omega_b, omega_cdm, h, amp
# These are the PLANCK best fit values, and a guess for an amplitude conversion factor

# Set up the number of walkers and iterations for each walker
n_dim  = len(theta_start)
n_walkers = 200

# Each walker will start at a 1e-7 fluctuation from the guess, except for the amplitude parameter which has a much larger fluctuation
p0 = np.array([np.array(theta_start) + 1e-4 * np.random.randn(n_dim) for i in range(n_walkers)])
a_s = theta_start[3]
for i in range(n_walkers):
    p0[i,3] = 2.5*np.pi + 0.3*np.random.rand()

def run_sampler(p0, nwalkers, niter, ndim, lnprob):
    # Define your sampler, giving it the likelihood and data

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (data['l'], data['Dl'], data['+dDl'], data['-dDl']), pool=pool)
        
        # Following the argument here (http://users.stat.umn.edu/~geyer/mcmc/burn.html) which I find convincing, I'm neglecting to include a burn-in sample, and instead we're starting with points near the PLANCK best-fits as defined above.
    
        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

    return sampler, pos, prob, state

n_iterations = 50
sampler, pos, prob, state = run_sampler(p0, n_walkers,n_iterations, n_dim, ch.ln_prob)

# things we want to save: 
flatsamples = sampler.flatchain
flatlnprobs = sampler.flatlnprobability
# Save the samples

df = pd.DataFrame({'omega_b': flat_samples[:,0], 
                   'omega_cdm': flat_samples[:,1], 
                   'h': flat_samples[:,2], 
                   'amp': flat_samples[:,3],
                   'lnprob': flat_lnprobs
                   })
                   
df.to_csv('cosmo_mcmc.csv', index=False)


