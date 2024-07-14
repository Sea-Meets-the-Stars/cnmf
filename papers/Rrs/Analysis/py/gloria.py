""" Fuss with Gloria """
import numpy as np

from matplotlib import pyplot as plt

from sklearn import decomposition

from ocpy.hydrolight import loisel23
from ocpy.insitu import gloria
from ocpy.utils import spectra
from ocpy.utils import plotting

from IPython import embed

def pca_gloria_l23(min_wv:float=400, high_cut:float=700):
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
    l23_Rs = l23_ds.Rrs.data[:,l23_cut]

    # Rebin Gloria to Loisel23
    new_gloria_wv = np.append(l23_wave, [l23_wave.max()+5.]) - 2.5 # Because the rebinning is not interpolation
    rwv_nm, gloria_rebin, gloria_rebin_sig = spectra.rebin_to_grid(
        gloria_wv, gloria_Rrs, gloria_Rrs_std, new_gloria_wv)


    # Clean up
    bad_Rs = np.where(np.isnan(gloria_rebin))
    keep_gloria = np.ones(gloria_rebin.shape[0], dtype=bool)
    bad_rows = np.unique(bad_Rs[0])
    keep_gloria[bad_rows] = False

    print(f'After all of the cuts, there are {np.sum(keep_gloria)} Gloria spectra left.')

    # PCA independently
    l23_pca_fit = decomposition.PCA(n_components=20).fit(l23_Rs)
    gloria_pca_fit = decomposition.PCA(n_components=20).fit(gloria_rebin)

    # Combine em too
    comb_Rrs = np.vstack((l23_Rs, gloria_rebin))
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
    plt.savefig('pca_gloria_l23.png', dpi=300)
    print(f'Saved pca_gloria_l23.png')

    #embed(header='pca_gloria_l23 77')
 

if __name__ == '__main__':
    pca_gloria_l23()