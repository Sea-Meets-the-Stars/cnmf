""" Utility functions for oceanographic applications """

import numpy as np


def prep(spec:np.ndarray, sigma:float=0.05):
    """ Prep IOP data for NMF analysis

    Args:
        spec (np.ndarray): IOPs (nspec, nwave)
        sigma (float, optional): Error to use. Defaults to 0.05.

    Returns:
        tuple: 
            - **new_spec** (*np.ndarray*) -- IOPs
            - **mask** (*np.ndarray*) -- Mask
            - **err** (*np.ndarray*) -- Error
    """
    # Prep
    new_spec = spec.copy()
    nspec, nwave = spec.shape

    # Reshape
    new_spec = np.reshape(new_spec, (new_spec.shape[0],
                     new_spec.shape[1], 1))

    # Build mask and error
    mask = (new_spec >= 0.).astype(int)
    err = np.ones_like(mask)*sigma

    # Return
    return new_spec, mask, err