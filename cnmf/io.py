""" Methods for I/O of CNMF """

import os
from importlib import resources

import numpy as np

def pcanmf_filename(nmf_fit:str, decomp:str, N_NMF:int, iop:str=None):
    """
    Generate the filename for the pcanmf output file.

    Parameters:
        nmf_fit (str): The name of the NMF fit.
        decomp (str): The type of decomposition.
        N_NMF (int): The number of NMF components.
        iop (str, optional): Additional identifier for the output file. Defaults to None.

    Returns:
        str: The filename for the pcanmf output file.
    """
    path = os.path.join(resources.files('cnmf'),
                        'data', nmf_fit)
    outroot = os.path.join(path, f'{nmf_fit}_{decomp}_{N_NMF:02d}')
    if iop is not None:
        outroot += f'_{iop}'

    # FInish
    nmf_file = outroot+'.npz'
    return nmf_file

def save_nmf(outfile:str, M:np.ndarray, coeff:np.ndarray, spec:np.ndarray,
    mask:np.ndarray, err:np.ndarray, wave:np.ndarray,
    Rs:np.ndarray, UID:np.ndarray=None):
    """
    Save the NMF results to a file in npz format.

    Args:
        outfile (str): The path of the output file.
        M (np.ndarray): The spatial components matrix.
        coeff (np.ndarray): The temporal coefficients matrix.
        spec (np.ndarray): The spectral components matrix.
        mask (np.ndarray): The binary mask matrix.
        err (np.ndarray): The error matrix.
        wave (np.ndarray): The wavelength array.
        Rs (np.ndarray): The remote sensing reflectance array.
        UID (np.ndarray, optional): The unique ID array. Defaults to None.
    """
    # Create output directory?
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    # Save
    np.savez(outfile, M=M, coeff=coeff,
             spec=spec, mask=mask,
             err=err, wave=wave, Rs=Rs,
             UID=UID)
    print(f'Wrote: {outfile}')
             

def load_nmf(nmf_fit:str, N_NMF:int, iop:str=None,
             filename:str=None):
    # File name
    if filename is None:
        filename = pcanmf_filename(nmf_fit, 'NMF', N_NMF, iop=iop)

    # Load + Return
    print(f'Loading: {filename}')
    return np.load(filename)
