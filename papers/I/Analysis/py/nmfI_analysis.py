""" NMF Analysis """

import os
import numpy as np
from importlib import resources

import sklearn

from ihop.iops import pca as ihop_pca

from oceancolor.hydrolight import loisel23 
from oceancolor.water import absorption 

from cnmf.oceanography import iops
from cnmf import nmf_imaging
from cnmf import io as cnmf_io
from cnmf import zhu_nmf as nmf
from cnmf import stats as cnmf_stats

from IPython import embed

# Wavelength range
min_wv=406.
high_cut=704.
pca_path = os.path.join(resources.files('cnmf'),
                            'data', 'L23')

def loisel23_components(iop:str, N_NMF:int=10, 
    min_wv:float=min_wv, high_cut:float=high_cut,
    clobber:bool=False, normalize:bool=True,
    X:int=4, Y:int=0):
    """
    Perform NMF analysis on Loisel23 data.

    Args:
        iop (str): The IOP dataset to use for analysis, e.g. 'a', 'aph'
        N_NMF (int, optional): The number of NMF components to extract. Defaults to 10.
        clobber (bool, optional): If True, overwrite existing output file. Defaults to False.
    """

    # Output file
    outfile = cnmf_io.pcanmf_filename('L23', 'NMF', N_NMF=N_NMF, iop=iop)
    if (not clobber) and (os.path.isfile(outfile)):
        print(f'File exists: {outfile}')
        return
    # Root
    outroot = outfile.replace('.npz','')

    # Load up the data
    l23_ds = loisel23.load_ds(X, Y)

    # Wavelengths, restricted to > 400 nm
    cut = (l23_ds.Lambda > min_wv) & (l23_ds.Lambda < high_cut)
    wave = l23_ds.Lambda.data[cut]
    Rs = l23_ds.Rrs.data[:,cut]
    
    # Grab the IOP
    if iop == 'a':
        l23_iop = l23_ds.a.data[:,cut]
        nspec = l23_iop.shape[0]
        a_w = absorption.a_water(wave, data='IOCCG')
        l23_iop = l23_iop - np.outer(np.ones(nspec), a_w)
    elif iop == 'bb':
        l23_iop = l23_ds.bb.data[:,cut]
    elif iop == 'aph':
        l23_iop = l23_ds.aph.data[:,cut]
    else:
        raise ValueError(f"Unknown IOP: {iop}")

    # Prep
    spec_nw, mask, err = iops.prep(l23_iop)

    # Do it
    comps = nmf_imaging.NMFcomponents(
        ref=spec_nw, mask=mask, ref_err=err,
        n_components=N_NMF,
        path_save=outroot, oneByOne=True,
        normalize=normalize,
        seed=12345)

    # Load
    M = np.load(outroot+'_comp.npy').T
    coeff = np.load(outroot+'_coef.npy').T

    # Save
    cnmf_io.save_nmf(outfile, M, coeff, spec_nw[...,0],
                     mask[...,0], err[...,0], wave, Rs)

    print(f'Wrote: {outfile}')

def l23_on_tara(sig:float=0.0005,
                    cut:int=None,
                    decomp:str='NMF'):
    """ Perform NMF analysis on Tara data, using the L23 fit as a basis.

    Args:
        sig (float, optional): _description_. Defaults to 0.0005.
        cut (int, optional): Cut on the number of spectra. Defaults to None.
    """

    # Load L23 fit
    nmf_fit, N_NMF, iop = 'L23', 4, 'a'
    d = cnmf_io.load_nmf(nmf_fit, N_NMF, iop)
    M = d['M']
    coeff = d['coeff']
    wave = d['wave']
    L23_pca_N20 = ihop_pca.load('pca_L23_X4Y0_a_N4.npz',
                                pca_path=pca_path)

    # Calculate Tara
    wv_grid, final_tara, tara_UIDs, l23_a = iops.tara_matched_to_l23(
        low_cut=min_wv, high_cut=high_cut)
    i0 = np.argmin(np.abs(wv_grid[0]-wave))
    assert np.isclose(wv_grid[0], wave[i0])
    i1 = np.argmin(np.abs(wv_grid[-1]-wave))
    assert i0 == 0 # Should be the same now
    #embed(header='nmf analysis 82')

    # Cut?
    if cut is not None:
        final_tara = final_tara[:cut]
        tara_UIDs = tara_UIDs[:cut]
    V = np.ones_like(final_tara) / sig**2
    M_tara = M[:,i0:i1+1]

    if decomp == 'NMF':
        # Build it up one component at a time
        H_tmp = None
        for nn in range(M_tara.shape[0]):
            print("Working on component: ", nn+1)
            W_ini = M_tara[0:nn+1,:].T
            H_rand = np.random.rand(1, final_tara.shape[0])
            if H_tmp is not None:
                H_ini = np.vstack((H_tmp, H_rand))
            else:
                H_ini = H_rand
        
            tara_NMF = nmf.NMF(final_tara.T,
                        V=V.T, W=W_ini, H=H_ini,
                        n_components=nn+1)
            # Do it
            tara_NMF.SolveNMF(H_only=True, verbose=True)
            # Save H
            H_tmp = tara_NMF.H.copy()
            #embed(header='iops 84')
        # Repackage
        save_coeff = tara_NMF.H
        save_M = M_tara
        outfile = cnmf_io.pcanmf_filename('Tara_L23', 'NMF', N_NMF, iop=iop)
    elif decomp == 'PCA':
        # Init the PCA
        pca = sklearn.decomposition.PCA(n_components=L23_pca_N20['M'].shape[0])
        pca.components_ = L23_pca_N20['M']
        pca.mean_ = L23_pca_N20['mean']
        # Apply
        save_coeff = pca.transform(final_tara)
        save_M = L23_pca_N20['M']
        outfile = cnmf_io.pcanmf_filename('Tara_L23', 'PCA', N_NMF, iop=iop)
    else:
        raise ValueError(f"Unknown decomp: {decomp}")

    # Save
    cnmf_io.save_nmf(outfile, save_M, save_coeff, final_tara,
                     None, V, wv_grid, None,
                     UID=tara_UIDs)

def tara_components(iop:str='a', N_NMF:int=10, clobber:bool=False,
        seed:int=12345):
    """
    Perform NMF analysis on Loisel23 data.

    Args:
        iop (str): The IOP dataset to use for analysis, e.g. 'a'
        N_NMF (int, optional): The number of NMF components to extract. Defaults to 10.
        clobber (bool, optional): If True, overwrite existing output file. Defaults to False.
        seed (int, optional): The random seed to use. Defaults to 12345.
    """

    # Output file
    outfile = cnmf_io.pcanmf_filename(
        'Tara', 'NMF', N_NMF=N_NMF, iop=iop)
    if (not clobber) and (os.path.isfile(outfile)):
        print(f'File exists: {outfile}')
        return
    # Root
    outroot = outfile.replace('.npz','')

    # Load
    wv_grid, final_tara, mask, err, tara_uid = iops.tara_matched_to_l23(
        low_cut=min_wv, for_nmf_imaging=True,
        high_cut=high_cut)

    # Do it
    comps = nmf_imaging.NMFcomponents(
        ref=final_tara, mask=mask, 
        ref_err=err, n_components=N_NMF,
        path_save=outroot, oneByOne=True,
        seed=seed, normalize=True)

    # Load
    M = np.load(outroot+'_comp.npy').T
    coeff = np.load(outroot+'_coef.npy').T

    # Save
    cnmf_io.save_nmf(outfile, M, coeff, final_tara[...,0],
                     mask[...,0], err[...,0], wv_grid, None,
                     UID=tara_uid)

    print(f'Wrote: {outfile}')

if __name__ == '__main__':



    # NMF on L23
    #for n in [3]:
    for n in range(1,10):
        #loisel23_components('a', N_NMF=n+1, min_wv=min_wv, high_cut=high_cut)
        #loisel23_components('bb', N_NMF=n+1, min_wv=min_wv, high_cut=high_cut)
        loisel23_components('aph', N_NMF=n+1, min_wv=min_wv, high_cut=high_cut)

    '''
    # PCA on L23
    pca_path = os.path.join(resources.files('cnmf'),
                            'data', 'L23')
    outroot = 'pca_L23'
    #ihop_pca.generate_l23_pca(clobber=True, Ncomp=20, X=4, Y=0,
    #                          min_wv=min_wv, high_cut=high_cut,
    #                          pca_path=pca_path, outroot=outroot)
    #ihop_pca.generate_l23_pca(clobber=True, Ncomp=4, X=4, Y=0,
    #                          min_wv=min_wv, high_cut=high_cut,
    #                          pca_path=pca_path, outroot=outroot)
    ihop_pca.generate_l23_pca(clobber=True, Ncomp=3, X=4, Y=0,
                              min_wv=min_wv, high_cut=high_cut,
                              pca_path=pca_path, outroot=outroot)

    # L23 PCA on Tara
    l23_on_tara(decomp='PCA')

    # L23 NMF on Tara
    l23_on_tara()#cut=40000)

    # NMF on Tara alone
    for n in [3,4]:
        # Do it
        tara_components('a', N_NMF=n)
        # Variance
        d = cnmf_io.load_nmf('Tara', n, 'a')
        evar_i = cnmf_stats.evar_computation(
                d['spec'], d['coeff'], d['M'])
        print(f"Explained variance: {evar_i}")
        # 3: Explained variance: 0.9975242528085961
        # 4: Explained variance: 0.9990660778255239
    #tara_components('a', N_NMF=10)
    '''

    # CNMF on L23 test