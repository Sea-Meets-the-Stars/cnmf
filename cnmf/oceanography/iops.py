""" Code related to Inherent Optical Properties (IOPs) """

import numpy as np

from oceancolor.hydrolight import loisel23
from oceancolor.tara import io as tara_io
from oceancolor.tara import spectra
from oceancolor.utils import spectra as spec_utils
from oceancolor import water

from IPython import embed




def tara_matched_to_l23(low_cut:float=405., high_cut:float=705., 
                        X:int=4, Y:int=0, for_nmf_imaging:bool=False,
                        include_water:bool=False):
    """ Generate Tara spectra matched to L23

    Restricted on wavelength and time of cruise
    as per Patrick Gray recommendations

    Args:
        low_cut (float, optional): low cut wavelength. Defaults to 400..
            The Microbiome dataset has a lowest wavelength
            of 408.5nm
        high_cut (float, optional): high cut wavelength. Defaults to 705..
            The Microbiome dataset has a highest wavelength
            of 726.3nm
        X (int, optional): simulation scenario. Defaults to 4.
        Y (int, optional): solar zenith angle used in the simulation.
            Defaults to 0.

    Returns:
        tuple: wavelength values, Tara spectra, Tara UIDs, L23 spectra
    """

    # Load up the data
    l23_ds = loisel23.load_ds(X, Y)

    # Load up Tara
    print("Loading Tara..")
    tara_db = tara_io.load_pg_db(expedition='Microbiome')

    # Spectra
    wv_nm, all_a_p, all_a_p_sig = spectra.spectra_from_table(tara_db)

    # Wavelengths, restricted to > 400 nm
    cut = (l23_ds.Lambda > low_cut) & (l23_ds.Lambda < high_cut)
    l23_a = l23_ds.a.data[:,cut]

    # Rebin
    wv_grid = l23_ds.Lambda.data[cut]
    tara_wv = np.append(wv_grid, [high_cut+5.]) - 2.5 # Because the rebinning is not interpolation
    rwv_nm, r_ap, r_sig = spec_utils.rebin_to_grid(
        wv_nm, all_a_p, all_a_p_sig, tara_wv)
    #embed(header='iops 104')

    # Add in water
    if include_water:
        print("Adding in water..")
        df_water = water.absorption.load_rsr_gsfc()
        a_w, _ =  spec_utils.rebin(df_water.wavelength.values, 
                            df_water.aw.values, np.zeros_like(df_water.aw),
                            wv_grid)

        final_tara = r_ap+np.outer(np.ones(r_ap.shape[0]), a_w)
    else:
        final_tara = r_ap


    # Polish Tara for PCA
    bad = np.isnan(final_tara) | (final_tara <= 0.)
    ibad = np.where(bad)

    mask = np.ones(final_tara.shape[0], dtype=bool)
    mask[np.unique(ibad[0])] = False

    # Cut down: Aggressive but necessary
    final_tara = final_tara[mask,:]
    tara_uid = tara_db.uid.values[mask]

    #embed(header='iops 126')

    # ##########################################
    # Return
    # For NMF imaging?
    if for_nmf_imaging:
        # Build mask and error
        mask = (wv_grid >= 0.).astype(int)
        mask = np.outer(np.ones(final_tara.shape[0]), mask)
        mask = np.reshape(mask, (mask.shape[0], 
                     mask.shape[1], 1))
        err = np.ones_like(mask)*0.005

        final_tara = np.reshape(final_tara, (final_tara.shape[0], 
                     final_tara.shape[1], 1))

        return wv_grid, final_tara, mask, err, tara_uid

    else:
        return wv_grid, final_tara, tara_uid, l23_a