""" NMF Analysis """

import os
import numpy as np
from importlib import resources

from functools import partial


import sklearn

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

#from ihop.iops import pca as ihop_pca
from ihop.iops import decompose
from ihop.iops import io as ihop_io

from ocpy.hydrolight import loisel23 
from ocpy.water import absorption 
from ocpy.ph import absorption as ph_absorption
from ocpy.tara import io as tara_io
from ocpy.utils import cat_utils


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
    prefix_outfile:str='',
    sigma=0.05,
    X:int=4, Y:int=0):
    """
    Perform NMF analysis on Loisel23 data.

    Args:
        iop (str): The IOP dataset to use for analysis, e.g. 'a', 'aph'
        N_NMF (int, optional): The number of NMF components to extract. Defaults to 10.
        clobber (bool, optional): If True, overwrite existing output file. Defaults to False.
    """

    # Output file
    outfile = cnmf_io.pcanmf_filename(f'{prefix_outfile}L23', 'NMF', N_NMF=N_NMF, iop=iop)
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
    spec_nw, mask, err = iops.prep(l23_iop, sigma=sigma)

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

def l23_pca(X:int=4, Y:int=0, Ncomp:int=3, clobber:bool=False):


    # Load up the data
    ds = loisel23.load_ds(X, Y)


    # Loop on IOPs
    for iop in ['a', 'b', 'bb']:
        # Outfile
        #outfile = ihop_io.loisel23_filename(
        #    'PCA', iop, Ncomp, X, Y, d_path=pca_path)
        outfile = cnmf_io.pcanmf_filename(f'L23', 'PCA', 
                                          N_NMF=Ncomp, 
                                          iop=iop)
        # Cut on wavelength?
        data = ds[iop].data
        gd_wv = np.ones_like(ds.Lambda.data, dtype=bool)
        if min_wv is not None:
            gd_wv = gd_wv & (ds.Lambda.data >= min_wv)
        if high_cut is not None:
            gd_wv = gd_wv & (ds.Lambda.data <= high_cut)

        # Do it
        #if not os.path.exists(outfile) or clobber:
        #    pca.fit_normal(data[:,gd_wv], Ncomp, save_outputs=outfile,
        #                  extra_arrays={'Rs':ds.Rrs.data[:,gd_wv],
        #                                 'wavelength':ds.Lambda.data[gd_wv]})

    
        decompose.generate_pca(
                data[:,gd_wv], outfile, Ncomp,
                clobber=True) 
                #extra_arrays={'Rs':ds.Rrs.data[:,gd_wv],
                #                         'wavelength':ds.Lambda.data[gd_wv]})
                #                         'wavelength':ds.Lambda.data[gd_wv]})

    
def l23_on_tara(sig:float=0.0005,
                    cut:int=None, skip_save:bool=False,
                    decomp:str='NMF'):
    """ Perform NMF analysis on Tara data, using the L23 fit as a basis.

    Args:
        sig (float, optional): _description_. Defaults to 0.0005.
        cut (int, optional): Cut on the number of spectra. Defaults to None.
        skip_save (bool, optional): Skip saving the output. Defaults to False.
    """

    # Load L23 fit
    nmf_fit, N_NMF, iop = 'L23', 4, 'a'
    d = cnmf_io.load_nmf(nmf_fit, N_NMF, iop)
    M = d['M']
    coeff = d['coeff']
    wave = d['wave']

    # Load PCA
    #L23_pca_N20 = ihop_io.load('pca_L23_X4Y0_a_N4.npz',
    #                            pca_path=pca_path)
    L23_pca = cnmf_io.load_nmf(nmf_fit, N_NMF, iop,
                                decomp='PCA')

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
        pca = sklearn.decomposition.PCA(n_components=L23_pca['M'].shape[0])
        pca.components_ = L23_pca['M']
        pca.mean_ = L23_pca['mean']
        pca.explained_variance_ = None
        # Apply
        save_coeff = pca.transform(final_tara)
        save_M = L23_pca['M']
        outfile = cnmf_io.pcanmf_filename('Tara_L23', 'PCA', N_NMF, iop=iop)
    else:
        raise ValueError(f"Unknown decomp: {decomp}")

    # Variance 
    if decomp == 'NMF':
        X = final_tara
        X_est = np.dot(save_M.T, save_coeff).T
        V_true = np.sum(np.std(X, axis=0)**2)
        V_est = np.sum(np.std(X-X_est, axis=0)**2)

        evar = 1 - V_est/V_true

    # Save
    if not skip_save:
        cnmf_io.save_nmf(outfile, save_M, save_coeff, final_tara,
                     None, V, wv_grid, None,
                     UID=tara_UIDs)

def tara_components(iop:str='a', N_NMF:int=10, clobber:bool=False,
        seed:int=12345):
    """
    Perform NMF analysis on Tara data.

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

def fit_rmse_aph():

    # Load NMF
    nmf_fit = 'L23'
    N_NMF, iop = 2, 'aph'
    d = cnmf_io.load_nmf(nmf_fit, N_NMF, iop)
    M = d['M']
    coeff = d['coeff']
    NMF_wave = d['wave']

    # Reconstruct
    NMF_aph = np.dot(coeff, M)


    # Load L23 and unpack
    ds = loisel23.load_ds(4,0)
    wave = ds.Lambda.data
    aph = ds.aph.data

    # Load Bricaud
    b1998 = ph_absorption.load_bricaud1998()
    keep = (wave >= 410.) & (wave <= b1998['lambda'].max())
    fit_wave = wave[keep]

    # Interpolate
    f_b1998_A = interp1d(b1998['lambda'], b1998.Aphi)
    f_b1998_E = interp1d(b1998['lambda'], b1998.Ephi)

    # Apply
    L23_A = f_b1998_A(fit_wave)
    L23_E = f_b1998_E(fit_wave)

    '''
    # func to fit
    def func(x, Chl):
        b_aph = L23_A * Chl**(L23_E)
        return b_aph
    '''

    # Fit
    sv_chla = []
    b_rmses = []
    nmf_rmses = []
    aph_440 = []
    sigma = np.ones(fit_wave.size)*0.005

    i_440 = np.argmin(np.abs(fit_wave-440.))

    for idx in range(aph.shape[0]):
        # Grab
        i_aph = aph[idx, keep]
        # Bricaud
        #ans, cov =  curve_fit(func, fit_wave, i_aph, p0=0.05, sigma=sigma)
        ans, cov = fit_aph_with_bricaud(fit_wave, i_aph, sigma, 
                                        L23_A=L23_A, L23_E=L23_E)
        b_rmse = np.sqrt(np.mean((i_aph - 
                                  bricaud_func(fit_wave, ans[0], L23_A, L23_E))**2))
        # NMF
        nmf_rmse = np.sqrt(np.mean((i_aph - NMF_aph[idx, :])**2))

        # Save
        sv_chla.append(ans[0])
        b_rmses.append(b_rmse)
        nmf_rmses.append(nmf_rmse)
        aph_440.append(i_aph[i_440])
    
    # Write to disk
    outfile = 'L23_aph_fits.npz'
    np.savez(outfile, sv_chla=sv_chla, b_rmses=b_rmses, nmf_rmses=nmf_rmses,
             aph_440=aph_440)
    print(f"Wrote: {outfile}")

def bricaud_func(x, Chl, L23_A, L23_E):
    b_aph = L23_A * Chl**(L23_E)
    return b_aph

def fit_aph_with_bricaud(fit_wave, aph, sigma, L23_A=None, L23_E=None):

    if L23_A is None:
        # Load Bricaud
        b1998 = ph_absorption.load_bricaud1998()

        # Interpolate
        f_b1998_A = interp1d(b1998['lambda'], b1998.Aphi)
        f_b1998_E = interp1d(b1998['lambda'], b1998.Ephi)

        # Apply
        L23_A = f_b1998_A(fit_wave)
        L23_E = f_b1998_E(fit_wave)

    partial_func = partial(bricaud_func, L23_A=L23_A, L23_E=L23_E)

    ans, cov =  curve_fit(partial_func, fit_wave, aph, p0=0.05, sigma=sigma)
    # Return
    return ans, cov

def fit_H2_vs_LH(N_NMF:int=4):

    print("Loading Tara..")
    tara_db = tara_io.load_pg_db(expedition='Microbiome')

    # Load Tara
    d_tara = cnmf_io.load_nmf('Tara', N_NMF, 'a')
    tara_coeff = d_tara['coeff']
    NMF_chl = tara_coeff[:,1] 

    # Grab by ID
    midx = cat_utils.match_ids(d_tara['UID'], tara_db.uid.values)
    Tara_chlA = tara_db.Chl_lineheight.values[midx]
    keep = (Tara_chlA > 0.01) & (NMF_chl > 0.01)

    # Fit in log
    p = np.polyfit(np.log10(NMF_chl[keep]), np.log10(Tara_chlA[keep]), 1)

    # Calculate MAE
    log_calc_LH = p[0]*np.log10(NMF_chl[keep]) + p[1]
    abs_error = np.abs(Tara_chlA[keep] - 10**log_calc_LH)
    rel_error = abs_error / Tara_chlA[keep]
    mae = np.mean(rel_error)
    
    embed(header='nmfI_analysis 348')


if __name__ == '__main__':



    '''
    # NMF on L23
    #for n in [3]:
    for n in range(1,10):
        loisel23_components('a', N_NMF=n+1, min_wv=min_wv, high_cut=high_cut, clobber=True)
        loisel23_components('bb', N_NMF=n+1, min_wv=min_wv, high_cut=high_cut)
        loisel23_components('aph', N_NMF=n+1, min_wv=min_wv, high_cut=high_cut)
        # Lower sigma
        loisel23_components('a', N_NMF=n+1, min_wv=min_wv, high_cut=high_cut, sigma=0.005,
                            prefix_outfile='LOW')


    # PCA on L23
    #ihop_pca.generate_l23_pca(clobber=True, Ncomp=20, X=4, Y=0,
    #                          min_wv=min_wv, high_cut=high_cut,
    #                          pca_path=pca_path, outroot=outroot)
    #ihop_pca.generate_l23_pca(clobber=True, Ncomp=4, X=4, Y=0,
    #                          min_wv=min_wv, high_cut=high_cut,
    #                          pca_path=pca_path, outroot=outroot)
    l23_pca(Ncomp=4, clobber=True)
    l23_pca(Ncomp=20, clobber=True)
    '''

    # L23 PCA on Tara
    l23_on_tara(decomp='PCA')

    '''
    # L23 NMF on Tara
    l23_on_tara(skip_save=True)#cut=40000)



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

    # Bricaud aph
    #fit_rmse_aph()

    # Fit H2 vs LH
    #fit_H2_vs_LH()