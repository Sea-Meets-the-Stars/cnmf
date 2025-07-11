""" Talk Figures for CNMF """

import os
from importlib import resources

import numpy as np
from scipy.interpolate import interp1d 
from scipy.optimize import curve_fit

import seaborn as sns
import pandas

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy
mpl.rcParams['font.family'] = 'stixgeneral'

import corner

from oceancolor.utils import plotting 
from oceancolor.utils import cat_utils
from oceancolor.iop import cdom
from oceancolor.ph import pigments
from oceancolor.hydrolight import loisel23
from oceancolor.tara import io as tara_io


from ihop.iops import pca as ihop_pca

from cnmf import io as cnmf_io
from cnmf import stats as cnmf_stats

from IPython import embed

pca_path = os.path.join(resources.files('cnmf'),
                            'data', 'L23')

tformM = ccrs.Mollweide()
tformP = ccrs.PlateCarree()                        

# #############################################
def fig_many_examples(outfile='fig_many_examples.png',
    nmf_fit:str='L23', N_NMF:int=4, iop:str='a',
    seed:int=54321,
    norm:bool=True, Nex:int=100):

    # Load
    d_l23 = cnmf_io.load_nmf(nmf_fit, N_NMF, 'a')
    d_tara = cnmf_io.load_nmf('Tara_L23', N_NMF, iop)

    #embed(header='fig_examples 42')

    # #########################################################
    # Figure
    figsize=(10,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # #########################################################
    # Tara Spectra 
    ax_tara = plt.subplot(gs[0])
    ax_tara.grid(True)
    
    np.random.seed(seed)
    nTara = d_tara['spec'].shape[0]
    tara_idx = np.random.choice(np.arange(nTara), Nex, 
                           replace=False)

    # Tara spectra
    tara_mx = 0.
    for ss in tara_idx:
        ax_tara.plot(d_tara['wave'], d_tara['spec'][ss], 
                 #label=r'Tara: $a_{\rm p, 440} = 10^{'+f'{np.log10(a440):0.1f}'+r'} \, {\rm m}^{-1}$', 
                 color='blue', ls='-')
        tara_mx = max(tara_mx, np.max(d_tara['spec'][ss]))
    ax_tara.set_ylim(0., tara_mx*1.05)
    ax_tara.text(0.5, 0.90, 'Tara Microbiome', color='k',
        transform=ax_tara.transAxes,
        fontsize=22, ha='center')

    # #########################################################
    # L23 Spectra 
    ax_l23 = plt.subplot(gs[1])
    ax_l23.grid(True)
    
    np.random.seed(seed)
    nl23 = d_l23['spec'].shape[0]
    l23_idx = np.random.choice(np.arange(nl23), Nex, 
                           replace=False)

    # l23 spectra
    l23_mx = 0.
    for ss in l23_idx:
        ax_l23.plot(d_l23['wave'], d_l23['spec'][ss], 
                 #label=r'l23: $a_{\rm p, 440} = 10^{'+f'{np.log10(a440):0.1f}'+r'} \, {\rm m}^{-1}$', 
                 color='green', ls='-')
        l23_mx = max(l23_mx, np.max(d_l23['spec'][ss]))
    ax_l23.set_ylim(0., l23_mx*1.05)
    ax_l23.text(0.5, 0.90, 'Loisel+2023', color='k',
        transform=ax_l23.transAxes,
        fontsize=22, ha='center')

    '''
    # L23 spectra
    for ss, a440 in enumerate([2e-2, 2e-1]):
        ls = '-' if ss == 0 else ':'
        il_0 = np.argmin(np.abs(d_l23['spec'][:,i440_l23] - a440))
        # Normalize?
        if norm:
            iwv = np.argmin(np.abs(d_l23['wave']-440.))
            nrm = d_l23['spec'][il_0, iwv]
        else:
            nrm = 1.
        # Plot
        ax_spec.plot(d_l23['wave'], d_l23['spec'][il_0]/nrm, 
                 label=r'L23: $a_{\rm nw,440} = 10^{'+f'{np.log10(a440):0.1f}'+r'} \, {\rm m}^{-1}$', 
                 color='blue', ls=ls)
    # Label
    if norm:
        ax_spec.set_ylabel(r'Normalized $a_{\rm nw}$ [L23] or $a_{\rm p}$ [Tara]')
    else:
        ax_spec.set_ylabel(r'Absorption Coefficient (m$^{-1}$)')
    #ax_spec.set_yscale('log')

    ax_spec.legend(fontsize=10.)
    '''

    # Axes
    for ax in [ax_tara, ax_l23]:
        plotting.set_fontsize(ax, 15.)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_xlim(400,700)
        ax.set_ylabel(r'Absorption Coefficient (m$^{-1}$)')

    # Finish
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_l23_pca(outfile:str='fig_l23_pca.png',
                 nmf_fit:str='L23', Ncomp:int=4,
                 norm:bool=False):

    # Seaborn
    sns.set(style="whitegrid",
            rc={"lines.linewidth": 2.5})
            # 'axes.edgecolor': 'black'
    sns.set_palette("pastel")
    sns.set_context("paper")
    #sns.set_context("poster", linewidth=3)
    #sns.set_palette("husl")

    # Load
    ab, Rs, d, d_bb = ihop_pca.load_loisel_2023_pca(
        N_PCA=Ncomp, l23_path=pca_path)
    pca_N20 = ihop_pca.load('pca_L23_X4Y0_a_N20.npz',
                                pca_path=pca_path)

    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(1,2)

    # Explained variance
    ax_var = plt.subplot(gs[0])
    ax_var.plot(np.arange(pca_N20['explained_variance'].size-1)+2,
            1-np.cumsum(pca_N20['explained_variance'])[1:], 'o-',
            color='blue', label='PCA')
    ax_var.set_xlabel(r'Number of Components ($m$)')
    ax_var.set_ylabel('PCA Cumulative Unexplained Variance')
    ax_var.axhline(1., color='k', ls=':')
    ax_var.set_xlim(1., 10)
    ax_var.set_ylim(1e-5, 0.01)
    ax_var.set_yscale('log')

    # Basis functions
    ax_basis = plt.subplot(gs[1])
    wave = d['wavelength']
    M = d['M']
    nrm = 1
    itype = 'PCA'
    for ii in range(Ncomp):
        sns.lineplot(x=wave, y=M[ii]/nrm, 
                 label=f'{itype}:'+r'  $W_'+f'{ii+1}'+'$',
                 ax=ax_basis, lw=2)#, drawstyle='steps-pre')
    ax_basis.set_xlabel('Wavelength (nm)')
    ax_basis.set_xlim(400., 720.)
    ax_basis.set_ylabel('PCA Basis Functions')
    # Thick line around the border of the axis
    ax_basis.spines['top'].set_linewidth(2)
        
        # Horizontal line at 0
        #ax.axhline(0., color='k', ls='--')

    for ax in [ax_basis, ax_var]:
        plotting.set_fontsize(ax, 16)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Example spectra
    if flg & (2**0):
        fig_many_examples()

    # L23 PCA
    if flg & (2**1):
        fig_l23_pca()


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Many examples
        #flg += 2 ** 1  # 2 -- PCA 
    else:
        flg = sys.argv[1]

    main(flg)