""" Fuss with Gloria """
import os
import numpy as np

from matplotlib import pyplot as plt

from sklearn import decomposition

from ocpy.hydrolight import loisel23
from ocpy.insitu import gloria
from ocpy.utils import spectra
from ocpy.utils import plotting

from cnmf.oceanography import utils as co_utils
from cnmf import nmf_imaging
from cnmf import stats as cnmf_stats

from IPython import embed

def prep_gloria_l23(min_wv:float=400, high_cut:float=700):
    # Load Gloria
    df_meta, df_Rrs, df_Rrs_std, df_qc_flags = gloria.load_gloria()

    gloria_wv, gloria_Rrs, _ = gloria.parse_table(df_Rrs, 'Rrs')
    _, gloria_Rrs_std, _ = gloria.parse_table(df_Rrs_std, 'Rrs_std')

    # Load Loisel23
    X,Y = 4,0
    l23_ds = loisel23.load_ds(X, Y)

    # Wavelengths, restricted to 700nm > wv > 400 nm
    l23_a = l23_ds.a.data[:]
    l23_cut = (l23_ds.Lambda > min_wv) & (l23_ds.Lambda < high_cut)

    l23_wave = l23_ds.Lambda.data[l23_cut]
    l23_Rrs = l23_ds.Rrs.data[:,l23_cut]

    # Rebin Gloria to Loisel23
    new_gloria_wv = np.append(l23_wave, [l23_wave.max()+5.]) - 2.5 # Because the rebinning is not interpolation
    rwv_nm, gloria_rebin, gloria_rebin_sig = spectra.rebin_to_grid(
        gloria_wv, gloria_Rrs, gloria_Rrs_std, new_gloria_wv)


    # Clean up
    bad_Rrs = np.where(np.isnan(gloria_rebin))
    keep_gloria = np.ones(gloria_rebin.shape[0], dtype=bool)
    bad_rows = np.unique(bad_Rrs[0])
    keep_gloria[bad_rows] = False

    print(f'After all of the cuts, there are {np.sum(keep_gloria)} Gloria spectra left.')

    return l23_Rrs, gloria_rebin[keep_gloria], l23_wave


def nmf_gloria_l23(min_wv:float=400, high_cut:float=700, 
                   N_NMF:int=4):

    l23_Rrs, gloria_rebin, l23_wave = prep_gloria_l23(
        min_wv, high_cut)

    # Scale for NMF
    scaled_l23 = l23_Rrs * 1e4
    scaled_gloria = gloria_rebin * 1e4
    comb_Rrs = np.vstack((scaled_l23, scaled_gloria))

    # Prep for NMF
    l23_prepped, l23_mask, l23_err = co_utils.prep(scaled_l23, sigma=0.1)
    gloria_prepped, gloria_mask, gloria_err = co_utils.prep(scaled_gloria, sigma=0.1)
    comb_prepped, comb_mask, comb_err = co_utils.prep(comb_Rrs, sigma=0.1)

    # NMF
    normalize = True

    for spec, mask, err, pref in zip([l23_prepped, gloria_prepped, comb_prepped],
                               [l23_mask, gloria_mask, comb_mask],
                               [l23_err, gloria_err, comb_err],
                               ['L23', 'GLORIA', 'Comb']):
        outroot = f'{pref}_scaledRrs'
        for suffix in ['_coef.npy', '_comp.npy']:
            if os.path.exists(outroot+suffix):
                os.remove(outroot+suffix)
        #embed(header='nmf_gloria_l23 79')
        # NMF
        comps = nmf_imaging.NMFcomponents(
            ref=spec, mask=mask, ref_err=err,
            n_components=N_NMF,
            path_save=outroot, oneByOne=True,
            normalize=normalize,
            seed=12345, verbose=True)

        # Write the data too
        np.savez(outroot+'_data.npz', spec=spec, 
                 mask=mask, err=err, wave=l23_wave)
#

def plot_nmf(comb_only:bool=False):

    outfile = 'nmf_modes.png'
    if comb_only:
        outfile = 'nmf_modes_comb.png'

    # Load the NMF components
    dNMF = {}
    for ss, pref in enumerate(['L23', 'GLORIA', 'Comb']):
        outroot = f'{pref}_scaledRrs'
        dNMF[f'{pref}_M'] = np.load(outroot+'_comp.npy').T
        dNMF[f'{pref}_coeff'] = np.load(outroot+'_coef.npy').T
        # 
        d = np.load(outroot+'_data.npz')
        dNMF[f'{pref}_spec'] = d['spec'][...,0]
        # Explained variance
        dNMF[f'{pref}_evar'] = cnmf_stats.evar_computation(
            dNMF[f'{pref}_spec'], dNMF[f'{pref}_coeff'], dNMF[f'{pref}_M'])
        # 
        print(f'{pref} explained variance: {100*dNMF[f"{pref}_evar"]:0.2f}')
        #
        if ss == 0:
            wave = d['wave']


    #embed(header='plot_nmf 117')
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    #embed(header='pca_gloria_l23 77')
    clrs = ['b', 'g', 'r', 'c', 'm', 'y']
    for ii in range(dNMF['L23_M'].shape[0]):
        clr = clrs[ii]
        if ii == 0:
            l23_lbl = 'Loisel23'
            gloria_lbl = 'Gloria'
        else:
            l23_lbl = None
            gloria_lbl = None
        comb_lbl = f'Comb W{ii+1}'
        if not comb_only:
            ax.plot(wave, dNMF['L23_M'][ii], color=clr, ls='--', label=l23_lbl)
            ax.plot(wave, dNMF['GLORIA_M'][ii], color=clr, ls=':', label=gloria_lbl)
        ax.plot(wave, dNMF['Comb_M'][ii], color=clr, ls='-', label=comb_lbl)

    # Axes
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('NMF mode')
    ax.legend(fontsize=10)
    ax.grid()
    ax.minorticks_on()

    plotting.set_fontsize(ax, 15)
    # Save
    
    plt.savefig(outfile, dpi=300)
    print(f'Saved {outfile}')

def pca_gloria_l23(min_wv:float=400, high_cut:float=700):

    l23_Rrs, gloria_rebin, l23_wave = prep_gloria_l23(
        min_wv, high_cut)

    # PCA independently
    l23_pca_fit = decomposition.PCA(n_components=20).fit(l23_Rrs)
    gloria_pca_fit = decomposition.PCA(n_components=20).fit(gloria_rebin)

    # Combine em too
    comb_Rrs = np.vstack((l23_Rrs, gloria_rebin))
    comb_pca_fit = decomposition.PCA(n_components=20).fit(comb_Rrs)

    # Explained variance plot
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()

    xvals = np.arange(0, 20) + 1

    ax.plot(xvals, l23_pca_fit.explained_variance_ratio_, 'o', label='Loisel23')
    ax.plot(xvals, gloria_pca_fit.explained_variance_ratio_, 'o', label='Gloria')
    ax.plot(xvals, comb_pca_fit.explained_variance_ratio_, 'ok', label='Combined')

    # y-log
    ax.set_yscale('log')

    # Step in 5 on the x-axis 
    ax.set_xticks(np.arange(0, 20, 5))
    ax.minorticks_on()

    # Grid, including minor ticks
    ax.grid(which='both')

    # Line at 0.01
    ax.axhline(0.01, color='k', linestyle='--')

    ax.set_xlabel('Principal component')
    ax.set_ylabel('Explained variance ratio')
    ax.legend(fontsize=16)

    plotting.set_fontsize(ax, 18)

    # Save
    plt.savefig('pca_gloria_l23_expvar.png', dpi=300)
    print('Saved pca_gloria_l23_expvar.png')

    # Now let's compare the first 6 PC modes

    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()

    #embed(header='pca_gloria_l23 77')
    clrs = ['b', 'g', 'r', 'c', 'm', 'y']
    for ii, clr in enumerate(clrs):
        if ii == 0:
            l23_lbl = 'Loisel23'
            gloria_lbl = 'Gloria'
        else:
            l23_lbl = None
            gloria_lbl = None
        comb_lbl = f'Comb PC{ii+1}'
        ax.plot(l23_wave, l23_pca_fit.components_[ii], color=clr, ls='--', label=l23_lbl)
        ax.plot(l23_wave, gloria_pca_fit.components_[ii], color=clr, ls=':', label=gloria_lbl)
        ax.plot(l23_wave, comb_pca_fit.components_[ii], color=clr, ls='-', label=comb_lbl)

    # Axes
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('PC mode')
    ax.legend(fontsize=10)
    ax.grid()
    ax.minorticks_on()

    plotting.set_fontsize(ax, 15)
    # Save
    plt.savefig('pca_gloria_l23_modes.png', dpi=300)
    print('Saved pca_gloria_l23_modes.png')

    #embed(header='pca_gloria_l23 77')
 

if __name__ == '__main__':
    #pca_gloria_l23()

    # NMF
    #nmf_gloria_l23()
    plot_nmf()
    plot_nmf(comb_only=True)