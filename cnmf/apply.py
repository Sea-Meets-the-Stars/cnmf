""" Apply NMF solutions """

import numpy as np

from cnmf import zhu_nmf as nmf

def calc_coeff(M:np.ndarray, spec:np.ndarray, V:np.ndarray):

    # Build it up one component at a time
    H_tmp = None
    for nn in range(M.shape[0]):
        print("Working on component: ", nn+1)
        W_ini = M[0:nn+1,:].T
        H_rand = np.random.rand(1, spec.shape[0])
        if H_tmp is not None:
            H_ini = np.vstack((H_tmp, H_rand))
        else:
            H_ini = H_rand
    
        NMF = nmf.NMF(spec.T,
                    V=V.T, W=W_ini, H=H_ini,
                    n_components=nn+1)
        # Do it
        NMF.SolveNMF(H_only=True, verbose=True)
        # Save H
        H_tmp = NMF.H.copy()

    # Repackage
    coeff = NMF.H

    # Return
    return coeff