""" MCMC fits for phytoplankton """

import numpy as np

import emcee
import corner

from IPython import embed

def ph_model(param:np.ndarray, profiles:np.ndarray):
    """ Calculate the absorption coefficient for phytoplankton

    Args:
        param (np.ndarray): Array of parameters.
        profiles (np.ndarray): Array of profiles.

    Returns:
        np.ndarray: Array of absorption coefficients [m^-1].
    """
    # Calculate
    return np.sum(param[:,None] * profiles, axis=0)


def log_prob(param:np.ndarray, a_ph:np.ndarray, 
             profiles:np.ndarray, sig=0.0005):
    pred = ph_model(param, profiles)
    #
    #sig = scl_sig * Rs
    #
    #embed(header='mcmc_ph.py:log_prob 30')
    return -1*0.5 * np.sum( (pred-a_ph)**2 / sig**2)


def run_emcee(wave:np.ndarray, a_ph:np.ndarray, 
              profiles:np.ndarray, nwalkers:int=32, 
              nsteps:int=20000, skip:bool=False,
              save_file:str=None, p0=None):

    # Number of parameters
    nparms = profiles.shape[0]

    if p0 is None:
        p0 = np.ones((nwalkers, nparms))
        skip = True

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    if save_file is not None:
        backend = emcee.backends.HDFBackend(save_file)
        backend.reset(nwalkers, nparms)
    else:
        backend = None

    sampler = emcee.EnsembleSampler(
        nwalkers, nparms, log_prob, 
        args=[a_ph, profiles],
        backend=backend)

    # Burn in
    print("Running burn-in")
    state = sampler.run_mcmc(p0, 1000,
        skip_initial_state_check=skip)
    sampler.reset()

    # Run
    print("Running full model")
    sampler.run_mcmc(state, nsteps,
                     skip_initial_state_check=skip)

    print(f"All done: Wrote {save_file}")

    # Return
    return sampler