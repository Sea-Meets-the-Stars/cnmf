""" This module contains functions for statistical analysis of NMF results 
WIP -- ignore this for now
"""

import numpy as np

from IPython import embed


def evar_computation(X:np.ndarray, W:np.ndarray, H:np.ndarray):
    """ Compute the explained variance of a model

    Args:
        X (np.ndarray): Variable to be modeled
        W (np.ndarray): Basis functions
        H (np.ndarray): Coefficients

    Returns:
        float: Explained variance
    """
    # Best estiamte
    X_est = np.dot(W, H)

    # Total original variance
    V_true = np.sum(np.std(X, axis=0)**2)
    V_est = np.sum(np.std(X-X_est, axis=0)**2)

    evar = 1 - V_est/V_true
    return evar
