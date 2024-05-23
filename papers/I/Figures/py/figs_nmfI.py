""" Figuers for the NMF paper"""

import sys
import os
from importlib import resources

from collections import deque

import numpy as np
from scipy.interpolate import interp1d 
from scipy.optimize import curve_fit
from scipy import stats
from scipy import signal

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
from oceancolor.ph import absorption as ph_absorption
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

sys.path.append(os.path.abspath("../Analysis/py"))
import nmfI_analysis

# #############################################
def fig_examples(outfile='fig_examples.png',
    nmf_fit:str='L23', N_NMF:int=4, iop:str='a',
    norm:bool=True):

    # Load
    d_l23 = cnmf_io.load_nmf(nmf_fit, N_NMF, 'a')
    d_tara = cnmf_io.load_nmf('Tara_L23', N_NMF, iop)

    #embed(header='fig_examples 42')

    # #########################################################
    # Figure
    figsize=(6,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    # a440 histogram
    ax_440 = plt.subplot(gs[2])

    i440_l23 = np.argmin(np.abs(d_l23['wave']-440.))
    a_440_l23 = d_l23['spec'][:,i440_l23]

    i440_tara = np.argmin(np.abs(d_tara['wave']-440.))
    a_440_tara = d_tara['spec'][:,i440_tara]

    d_440 = pandas.DataFrame()
    d_440['a440'] = np.concatenate((a_440_l23, a_440_tara))
    d_440['Sample'] = ['L23']*len(a_440_l23) + ['Tara']*len(a_440_tara)

    sns.histplot(data=d_440, x='a440', hue='Sample', ax=ax_440, stat='density', 
                 common_norm=False, log_scale=True)#, bins=20)
    ax_440.set_xlabel(r'$a_{440}$ (m$^{-1}$)')

    # Add a a675 histogram
    ax_675 = plt.subplot(gs[3])

    i675_l23 = np.argmin(np.abs(d_l23['wave']-675.))
    a_675_l23 = d_l23['spec'][:,i675_l23]

    i675_tara = np.argmin(np.abs(d_tara['wave']-675.))
    a_675_tara = d_tara['spec'][:,i675_tara]

    d_675 = pandas.DataFrame()
    d_675['a675'] = np.concatenate((a_675_l23, a_675_tara))
    d_675['Sample'] = ['L23']*len(a_675_l23) + ['Tara']*len(a_675_tara)

    sns.histplot(data=d_675, x='a675', hue='Sample', ax=ax_675, stat='density',
                    common_norm=False, log_scale=True)#, bins=20)
    ax_675.set_xlabel(r'$a_{675}$ (m$^{-1}$)')

    # Tick marks on the top
    for ax in [ax_440, ax_675]:
        ax.tick_params(axis='x', which='both', bottom=True, 
                       top=True, labelbottom=True, 
                       labeltop=False)

    # #########################################################
    # Spectra time
    ax_spec = plt.subplot(gs[0:2])
    ax_spec.grid(True)

    # Tara spectra
    for ss, a440 in enumerate([6e-3, 2e-2]):
        ls = '-' if ss == 0 else ':'
        it_0 = np.argmin(np.abs(d_tara['spec'][:,i440_tara] - a440))
        # Normalize?
        if norm:
            iwv = np.argmin(np.abs(d_tara['wave']-440.))
            nrm = d_tara['spec'][it_0, iwv]
        else:
            nrm = 1.
        ax_spec.plot(d_tara['wave'], d_tara['spec'][it_0]/nrm, 
                 label=r'Tara: $a_{\rm p, 440} = 10^{'+f'{np.log10(a440):0.1f}'+r'} \, {\rm m}^{-1}$', 
                 color='orange', ls=ls)
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
    ax_spec.set_xlabel('Wavelength (nm)')
    if norm:
        ax_spec.set_ylabel(r'Normalized $a_{\rm nw}$ [L23] or $a_{\rm p}$ [Tara]')
    else:
        ax_spec.set_ylabel(r'Absorption Coefficient (m$^{-1}$)')
    #ax_spec.set_yscale('log')

    ax_spec.legend(fontsize=10.)

    # Axes
    for ax in [ax_440, ax_675, ax_spec]:
        plotting.set_fontsize(ax, 10.)

    # Finish
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


# #############################################
def fig_l23_pca_nmf_var(
    outfile='fig_l23_pca_nmf_var.png',
    show_spec:bool=False, show_RMSE:bool=False,
    nmf_fit:str='L23'):

    # Load up
    if nmf_fit == 'L23':
        pca_N20 = ihop_pca.load('pca_L23_X4Y0_a_N20.npz',
                                    pca_path=pca_path)
        #pca_N20 = ihop_pca.load('pca_L23_X4Y0_Tara_a_N20.npz')
    #L23_Tara_pca = ihop_pca.load(f'pca_L23_X4Y0_Tara_a_N{N}.npz')
    #wave = L23_pca_N20['wavelength']


    # Variance in NMF
    evar_list, index_list = [], []
    for i in range(2, 11):
        # Load
        d = cnmf_io.load_nmf(nmf_fit, i, 'a')
        # eval
        evar_i = cnmf_stats.evar_computation(
            d['spec'], d['coeff'], d['M'])
        evar_list.append(evar_i)
        index_list.append(i)

    # Figure
    clrs = ['b', 'g']
    figsize=(6,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # #####################################################
    # PCA
    ax= plt.subplot(gs[0])

    if nmf_fit == 'L23':
        ax.plot(
            np.arange(pca_N20['explained_variance'].size-1)+2,
            1-np.cumsum(pca_N20['explained_variance'])[1:], 'o-',
            color=clrs[0], label='PCA')
    ax.plot(index_list, 1-np.array(evar_list), 'o-', color=clrs[1],
            label='NMF')

    ax.set_xlabel(r'Number of Components ($m$)')
    ax.set_ylabel('Cumulative Unexplained Variance')
    # Horizontal line at 1
    ax.axhline(1., color='k', ls=':')

    plotting.set_fontsize(ax, 17)
    ax.set_xlim(1., 10)
    ax.set_ylim(1e-5, 0.01)
    ax.set_yscale('log')
    ax.legend(fontsize=15)

    # Grid
    ax.grid(True)

    lbl = 'L23' if nmf_fit == 'L23' else 'Tara'
    ax.text(0.05, 0.90, lbl, color='k',
        transform=ax.transAxes,
        fontsize=22, ha='left')

    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_nmf_pca_basis(outfile:str='fig_nmf_pca_basis.png',
                 nmf_fit:str='L23', Ncomp:int=4,
                 norm:bool=False, iop:str='a',
                 skip_pca:bool=False):

    # Seaborn
    sns.set(style="whitegrid",
            rc={"lines.linewidth": 2.5})
            # 'axes.edgecolor': 'black'
    sns.set_palette("pastel")
    sns.set_context("paper")
    #sns.set_context("poster", linewidth=3)
    #sns.set_palette("husl")

    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(1,2)

    # a, bb
    for ss, itype in zip([0,1], ['PCA', 'NMF']):
        if skip_pca and ss == 0:
            continue

        # load
        if ss == 0:
            ab, Rs, d, d_bb = ihop_pca.load_loisel_2023_pca(
                N_PCA=Ncomp, l23_path=pca_path)
            wave = d['wavelength']
        elif ss == 1:
            d = cnmf_io.load_nmf(nmf_fit, Ncomp, iop)
            wave = d['wave']
        M = d['M']
        #embed(header='fig_nmf_pca_basis 376')

        ax = plt.subplot(gs[ss])

        # Plot
        for ii in range(Ncomp):
            # Normalize
            if norm:
                iwv = np.argmin(np.abs(wave-440.))
                nrm = M[ii][iwv]
            else:
                nrm = 1.
            # Step plot
            sns.lineplot(x=wave, y=M[ii]/nrm, 
                         label=f'{itype}:'+r'  $W_'+f'{ii+1}'+'$',
                         ax=ax, lw=2)#, drawstyle='steps-pre')
            #ax.step(wave, M[ii]/nrm, label=f'{itype}:'+r'  $\xi_'+f'{ii+1}'+'$')

        # Thick line around the border of the axis
        ax.spines['top'].set_linewidth(2)
        
        # Horizontal line at 0
        ax.axhline(0., color='k', ls='--')

        # Labels
        ax.set_xlabel('Wavelength (nm)')
        ax.set_xlim(400., 720.)

        lbl = 'PCA' if ss == 0 else 'NMF'
        ax.set_ylabel(lbl+' Basis Functions')

        loc = 'upper right'# if ss == 1 else 'upper left'
        ax.legend(fontsize=15, loc=loc)


        if ss == 0:
            xlbl, ha, flbl = 0.05, 'left', '(a)'
        else:
            xlbl, ha, flbl = 0.05, 'left', '(b)'

        ax.text(xlbl, 0.05, flbl, color='k',
            transform=ax.transAxes,
              fontsize=18, ha=ha)

        plotting.set_fontsize(ax, 18)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_aph_nmf(outfile:str='fig_aph_nmf.png',
                 nmf_fit:str='L23', Ncomp:int=2,
                 norm:bool=False, iop:str='aph',
                 skip_pca:bool=False):

    # Seaborn
    sns.set(style="whitegrid",
            rc={"lines.linewidth": 2.5})
            # 'axes.edgecolor': 'black'
    sns.set_palette("pastel")
    sns.set_context("paper")

    fig = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(1,1)

    # a, bb
    d = cnmf_io.load_nmf(nmf_fit, Ncomp, iop)
    wave = d['wave']
    M = d['M']

    ax = plt.subplot(gs[0])

    # Plot
    for ii in range(Ncomp):
        # Normalize
        if norm:
            iwv = np.argmin(np.abs(wave-440.))
            nrm = M[ii][iwv]
        else:
            nrm = 1.
        # Step plot
        sns.lineplot(x=wave, y=M[ii]/nrm, 
                        label=r'$W_'+f'{ii+1}'+r'^{\rm ph}$',
                        ax=ax, lw=2)#, drawstyle='steps-pre')
        #ax.step(wave, M[ii]/nrm, label=f'{itype}:'+r'  $\xi_'+f'{ii+1}'+'$')

    # Thick line around the border of the axis
    ax.spines['top'].set_linewidth(2)
    
    # Horizontal line at 0
    ax.axhline(0., color='k', ls='--')

    # Labels
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim(400., 720.)

    lbl = 'NMF'
    ax.set_ylabel(lbl+' Basis Functions')

    loc = 'upper right'# if ss == 1 else 'upper left'
    ax.legend(fontsize=15, loc=loc)

    plotting.set_fontsize(ax, 18)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_aph_fits(outfile:str='fig_aph_fits.png',
                 nmf_fit:str='L23', Ncomp:int=2,
                 idxs=[100, 1000],
                 norm:bool=False, iop:str='aph'):

    # Seaborn
    sns.set(style="whitegrid",
            rc={"lines.linewidth": 2.5})
            # 'axes.edgecolor': 'black'
    sns.set_palette("pastel")
    sns.set_context("paper")

    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(1,2)

    # Load
    # a, bb
    d = cnmf_io.load_nmf(nmf_fit, Ncomp, iop)
    wave = d['wave']
    M = d['M']
    coeff = d['coeff']
    recon = np.dot(coeff, M)


    ds = loisel23.load_ds(4,0)
    aph = ds.aph.data
    ds_wave = ds.Lambda.data

    # Load Bricaud
    b1998 = ph_absorption.load_bricaud1998()
    keep = (ds_wave >= 410.) & (ds_wave <= b1998['lambda'].max())
    fit_wave = ds_wave[keep]
    sigma = np.ones(fit_wave.size)*0.005

    # Interpolate
    f_b1998_A = interp1d(b1998['lambda'], b1998.Aphi)
    f_b1998_E = interp1d(b1998['lambda'], b1998.Ephi)

    # Apply
    L23_A = f_b1998_A(fit_wave)
    L23_E = f_b1998_E(fit_wave)

    for ss, idx in enumerate(idxs):
        ax = plt.subplot(gs[ss])
        #embed(header='fig_aph_fits 413')
        i_aph = aph[idx, keep]


        # Thick line around the border of the axis
        ax.spines['top'].set_linewidth(2)

        # Plot aph
        ax.plot(fit_wave, i_aph, 'ko', label=f'L23: idx={idx}', lw=2, zorder=1)

        # Fit
        ans, cov = nmfI_analysis.fit_aph_with_bricaud(
            fit_wave, i_aph, sigma, L23_A=L23_A, L23_E=L23_E)
        b_model = nmfI_analysis.bricaud_func(fit_wave, ans[0],
                                             L23_A, L23_E)
        ax.plot(fit_wave, b_model, 'g-', label='Bricaud', lw=2)

        # NMF
        ax.plot(wave, recon[idx], 'b-', label='NMF', lw=2)

        # Labels
        ax.set_xlabel('Wavelength (nm)')
        ax.set_xlim(400., 720.)

        lbl = 'NMF'
        ax.set_ylabel(r'$a_{\rm ph}$ (m$^{-1})$')

        loc = 'upper right'# if ss == 1 else 'upper left'
        ax.legend(fontsize=15, loc=loc)

        plotting.set_fontsize(ax, 18)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_nmf_rmse(outfile:str='fig_nmf_rmse.png',
                 nmf_fit:str='L23'):

    # RMSE
    rmss = []
    for n in range(1,10):
        # load
        d = cnmf_io.load_nmf(nmf_fit, N_NMF=n+1, iop='a')
        N_NMF = d['M'].shape[0]
        recon = np.dot(d['coeff'],
                       d['M'])
        #
        dev = recon - d['spec']
        rms = np.std(dev, axis=1)
        # Average
        avg_rms = np.mean(rms)
        rmss.append(avg_rms)

    # Plot

    fig = plt.figure(figsize=(6,6))
    plt.clf()
    ax = plt.gca()

    ax.plot(2+np.arange(N_NMF-1), rmss, 'o')

    ax.set_xlabel('Number of Components')
    ax.set_ylabel(r'Average RMSE (m$^{-1}$)')

    ax.set_yscale('log')
    
    # axes
    plotting.set_fontsize(ax, 15)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_nmf_basis(outroot:str='fig_nmf_basis',
                 nmf_fit:str='l23', N_NMF:int=4):

    outfile = f'{outroot}_{N_NMF}.png'
    # RMSE
    rmss = []
    # load
    d = load_nmf(nmf_fit, N_NMF=N_NMF)
    M = d['M']
    wave = d['wave']

    fig = plt.figure(figsize=(12,6))
    plt.clf()
    ax = plt.gca()

    # Plot
    for ss in range(N_NMF):
        ax.step(wave, M[ss], label=r'$W_'+f'{ss}'+'$')


    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Basis vector')

    ax.legend()

    #ax.set_yscale('log')
    
    # axes
    plotting.set_fontsize(ax, 15)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_fit_nmf(nmf_fit:str='L23', N_NMF:int=4,
                 icdom:int=0, # 0-indexing
                 ichl:int=1, # 0-indexing
                 outfile:str=None,
                 chl_min:float=430.,
                 cdom_max:float=530.,
                 add_gaussians:bool=False):
    """
    Generate a figure showing the fits of CDOM and chlorophyll using NMF.

    Args:
        nmf_fit (str): The type of NMF fit to use. Default is 'L23'.
        N_NMF (int): The number of NMF components. Default is 4.
        icdom (int): The index of the CDOM component to plot. Default is 1.
        ichl (int): The index of the chlorophyll component to plot. Default is 0.
        outfile (str): The output file name for the figure. If not provided, a default name will be used based on the nmf_fit parameter. Default is None.
        chl_min (float, optional): The minimum wavelength for the chlorophyll fit. Default is 450.0.
        cdom_max (float, optional): The maximum wavelength for the CDOM fit. Default is 600.0.
        add_gaussians (bool, optional): Whether to add additional Gaussian fits to the chlorophyll fit. Default is False.
    """
    if outfile is None:
        if nmf_fit == 'L23':
            outfile='fig_l23_fit_nmf.png'
        elif nmf_fit == 'Tara':
            outfile='fig_tara_fit_nmf.png'
    # Load
    d = cnmf_io.load_nmf(nmf_fit, N_NMF, 'a')
    M = d['M']
    wave = d['wave']
    a_cdom = M[icdom]

    # #########################################################
    # CDOM
    wv_cut = wave < cdom_max
    cut_wv = wave[wv_cut]

    # Fit exponentials
    exp_tot_coeff, cov = cdom.fit_exp_tot(
        wave[wv_cut], a_cdom[wv_cut])
    a_cdom_totexp_fit = exp_tot_coeff[0] * cdom.a_exp(
        wave[wv_cut], S_CDOM=exp_tot_coeff[1])
        #wave0=exp_tot_coeff[2])
    print(f'Tot exp coeff: {exp_tot_coeff}')
    exp_norm_coeff, cov = cdom.fit_exp_norm(wave[wv_cut], 
                                            a_cdom[wv_cut])
    a_cdom_exp_fit = exp_norm_coeff[0] * cdom.a_exp(wave[wv_cut])

    # Fit power-law
    pow_coeff, pow_cov = cdom.fit_pow(cut_wv, a_cdom[wv_cut])
    a_cdom_pow_fit = pow_coeff[0] * cdom.a_pow(cut_wv, S=pow_coeff[1])


    fig = plt.figure(figsize=(11,5))
    gs = gridspec.GridSpec(1,2)

    # #########################################################
    # Plot CDOM fits
    ax_cdom = plt.subplot(gs[min(icdom,1)])

    # NMF
    ax_cdom.step(wave, M[icdom], 
                 label=f'{nmf_fit}: '+r'$W_'+f'{icdom+1}'+'$', color='k',
                 lw=2)

    #ax_cdom.plot(cut_wv, a_cdom_exp_fit, 
    #        color='b', label='CDOM exp', ls='-')
    ax_cdom.plot(cut_wv, a_cdom_totexp_fit, 
            color='cyan', 
            label=r'Exponential ($S='+f'{exp_tot_coeff[1]:0.3f}'+r'$)', 
            ls='--', lw=2)
    ax_cdom.plot(cut_wv, a_cdom_pow_fit, 
            color='r', label='Power Law '+r'($\alpha='+f'{pow_coeff[1]:0.1f}'+r'$)', 
            ls=':', lw=2)

    ax_cdom.axvline(cdom_max, ls='--', color='gray')



    # #########################
    # Fit the chlorophyll basis functions
    ax_chl = plt.subplot(gs[ichl])

    # Gererate the profile
    if nmf_fit == 'L23':
        #a_chl = np.sum(M[1:], axis=0)
        dlbl = r'L23: $W_2 + W_4$'
        a_chl = M[1] + M[3]
    else:
        a_chl = M[ichl]
        dlbl=f'{nmf_fit}: '+r'$W_'+f'{ichl+1}'+'$'

    chla = pigments.a_chl(wave, ctype='a')
    chlb = pigments.a_chl(wave, ctype='b')
    chlc = pigments.a_chl(wave, ctype='c12')
    peri = pigments.a_chl(wave, pigment='Peri')
    beta = pigments.a_chl(wave, pigment='beta-Car')
    G584 = pigments.a_chl(wave, source='chase', pigment='G584')

    gd1 = (wave > chl_min) & (wave < 550.)
    gd2 = (wave > 640.) & (wave < 700.)
    gd_wave2 = gd1 | gd2
    if add_gaussians:
        gd3 = (wave > 560.) & (wave < 620.)
        gd_wave2 |= gd3

    add_pigments=[peri[gd_wave2], beta[gd_wave2]]
    if add_gaussians:
        add_pigments += [G584[gd_wave2]]

    # Fit
    sigma = np.ones_like(wave[gd_wave2])*0.05
    ans, cov = pigments.fit_a_chl(
        wave[gd_wave2], a_chl[gd_wave2], 
        add_pigments=add_pigments,
        fit_type='positive', sigma=sigma)
    print(f'Chl fit: {ans}')

    all_pigments=[peri, beta]
    if add_gaussians:
        all_pigments += [G584]
    def mk_model(*pargs):
        # pargs[0] is not used
        # Chl
        a = pargs[0]*chla + pargs[1]*chlb + pargs[2]*chlc
        # Others?
        if all_pigments is not None:
            for i, pigment in enumerate(all_pigments):
                a += pargs[3+i]*pigment
        # Return
        return a
    #embed(header='fig_fit_nmf 457')
    new_model = mk_model(*ans)
    
    # #########################
    # Plot
    ax_chl.plot(wave, a_chl, color='k', 
                label=dlbl)
    ax_chl.plot(wave[gd_wave2], new_model[gd_wave2], 'ro', 
                label='model')
    #https://www.allmovie.com/artist/akira-kurosawa-vn6780882/filmography

    # Chl
    for ss, pig, wv, lbl in zip(range(3), [chla,chlb,chlc], [673.,440.,440.], ['a', 'b', 'c12']):
        #iwv = np.argmin(np.abs(wave-wv))
        #nrm = pig[iwv]/M[0,iwv]
        #print(f'nrm: {1/nrm}')
        ax_chl.plot(wave, pig*ans[ss], label=f'Chl-{lbl}', ls=':')


    # New ones
    ax_chl.plot(wave, peri*ans[3], color='purple', label='Peri', ls=':')
    ax_chl.plot(wave, beta*ans[4], color='orange', label=r'$\beta$-Car', ls=':')
    if add_gaussians:
        ax_chl.plot(wave, G584*ans[5], color='gray', label=r'Chl-c(585)', ls=':')

    # Finish
    for ax in [ax_cdom, ax_chl]:
        plotting.set_fontsize(ax, 16)
        # Label the axes
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('NMF Basis')
        ax.legend(fontsize=15.)
        # Grid
        ax.grid(True)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_fit_W2(basis:str='W2', nmf_fit:str='L23', 
                N_NMF:int=4,
                outfile:str=None,
                chl_min:float=430):

    if outfile is None:
        outfile=f'fig_fit_{basis}_{nmf_fit}.png'

    # Load
    d = cnmf_io.load_nmf(nmf_fit, N_NMF, 'a')
    M = d['M']
    wave = d['wave']

    # #########################
    # Fit the chlorophyll basis functions
    fig = plt.figure(figsize=(11,5))
    gs = gridspec.GridSpec(1,1)
    ax_chl = plt.subplot(gs[0])

    # Gererate the profile
    dlbl = f'{nmf_fit}: {basis}'
    a_chl = M[int(basis[-1])-1]

    chla = pigments.a_chl(wave, ctype='a')
    chlb = pigments.a_chl(wave, ctype='b')
    chlc = pigments.a_chl(wave, ctype='c12')
    zea = pigments.a_chl(wave, pigment='Zea')
    peri = pigments.a_chl(wave, pigment='Peri')
    beta = pigments.a_chl(wave, pigment='beta-Car')
    G584 = pigments.a_chl(wave, source='chase', pigment='G584')

    gd1 = (wave > chl_min) & (wave < 550.)
    gd2 = (wave > 640.) & (wave < 700.)
    gd_wave2 = gd1 | gd2

    #add_pigments=[peri[gd_wave2], beta[gd_wave2]]
    add_pigments=[zea[gd_wave2]]

    # Fit
    sigma = np.ones_like(wave[gd_wave2])*0.05
    ans, cov = pigments.fit_a_chl(
        wave[gd_wave2], a_chl[gd_wave2], 
        add_pigments=add_pigments,
        fit_type='positive', sigma=sigma)
    print(f'Chl fit: {ans}')

    #all_pigments=[peri, beta]
    all_pigments=[zea]
    def mk_model(*pargs):
        # pargs[0] is not used
        # Chl
        a = pargs[0]*chla + pargs[1]*chlb + pargs[2]*chlc
        # Others?
        if all_pigments is not None:
            for i, pigment in enumerate(all_pigments):
                a += pargs[3+i]*pigment
        # Return
        return a
    #embed(header='fig_fit_nmf 457')
    new_model = mk_model(*ans)
    
    # #########################
    # Plot
    ax_chl.plot(wave, a_chl, color='k', 
                label=dlbl)
    ax_chl.plot(wave[gd_wave2], new_model[gd_wave2], 'ro', 
                label='model')
    #https://www.allmovie.com/artist/akira-kurosawa-vn6780882/filmography

    # Chl
    for ss, pig, wv, lbl in zip(range(3), [chla,chlb,chlc], [673.,440.,440.], ['a', 'b', 'c12']):
        #iwv = np.argmin(np.abs(wave-wv))
        #nrm = pig[iwv]/M[0,iwv]
        #print(f'nrm: {1/nrm}')
        ax_chl.plot(wave, pig*ans[ss], label=f'Chl-{lbl}', ls=':')


    # New ones
    #ax_chl.plot(wave, peri*ans[3], color='purple', label='Peri', ls=':')
    #ax_chl.plot(wave, beta*ans[4], color='orange', label=r'$\beta$-Car', ls=':')
    ax_chl.plot(wave, zea*ans[3], color='orange', label=r'Zea')


    # Finish
    for ax in [ax_chl]:
        plotting.set_fontsize(ax, 16)
        # Label the axes
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('NMF Basis')
        ax.legend(fontsize=15.)
        # Grid
        ax.grid(True)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_fit_W4(basis:str='W4', nmf_fit:str='L23', 
                N_NMF:int=4, 
                use_pigments=['FUCO', 'Diad'],
                outfile:str=None,
                chl_min:float=430):

    if outfile is None:
        outfile=f'fig_fit_{basis}_{nmf_fit}.png'

    # Load
    d = cnmf_io.load_nmf(nmf_fit, N_NMF, 'a')
    M = d['M']
    wave = d['wave']

    # #########################
    # Fit the chlorophyll basis functions
    fig = plt.figure(figsize=(11,5))
    gs = gridspec.GridSpec(1,1)
    ax_chl = plt.subplot(gs[0])

    # Gererate the profile
    dlbl = f'{nmf_fit}: {basis}'
    a_chl = M[int(basis[-1])-1]

    #chla = pigments.a_chl(wave, ctype='a')
    #chlb = pigments.a_chl(wave, ctype='b')
    #chlc = pigments.a_chl(wave, ctype='c12')
    #zea = pigments.a_chl(wave, pigment='Zea')
    peri = pigments.a_chl(wave, pigment='Peri')
    fuco = pigments.a_chl(wave, pigment='FUCO')
    diad = pigments.a_chl(wave, pigment='Diad')
    pdict = {'FUCO': fuco, 'Diad': diad, 'Peri': peri}
    #beta = pigments.a_chl(wave, pigment='beta-Car')
    #G584 = pigments.a_chl(wave, source='chase', pigment='G584')

    gd1 = (wave > chl_min) & (wave < 550.)
    #gd2 = (wave > 640.) & (wave < 700.)
    gd_wave2 = gd1 #| gd2

    add_pigments = []
    for pig in use_pigments:
        add_pigments += [pdict[pig][gd_wave2]]

    # Fit

    def func(*pargs):
        a = np.zeros_like(add_pigments[0])
        # Others?
        for i, pigment in enumerate(add_pigments):
            a += pargs[i+1]*pigment
        # Return
        return a
    p0 = np.ones(len(add_pigments))
    sigma = np.ones_like(wave[gd_wave2])*0.05
    ans, cov =  curve_fit(func, wave[gd_wave2],
                          a_chl[gd_wave2], 
                          p0=p0, sigma=sigma)

    #all_pigments=[peri, beta]
    all_pigments = []
    for pig in use_pigments:
        all_pigments += [pdict[pig]]
    def mk_model(*pargs, all_pigments=None):
        # pargs[0] is not used
        # Chl
        #a = pargs[0]*chla + pargs[1]*chlb + pargs[2]*chlc
        # Others?
        a = np.zeros_like(all_pigments[0])
        if all_pigments is not None:
            for i, pigment in enumerate(all_pigments):
                a += pargs[i]*pigment
        # Return
        return a
    #embed(header='fig_fit_nmf 457')
    new_model = mk_model(*ans, all_pigments=all_pigments)
    
    # #########################
    # Plot
    ax_chl.plot(wave, a_chl, color='k', 
                label=dlbl)
    ax_chl.plot(wave[gd_wave2], new_model[gd_wave2], 'ro', 
                label='model')
    #https://www.allmovie.com/artist/akira-kurosawa-vn6780882/filmography


    # New ones
    for ii, pig in enumerate(use_pigments):
        ax_chl.plot(wave, pdict[pig]*ans[ii], label=pig, ls=':')
    #ax_chl.plot(wave, peri*ans[0], color='purple', label='Peri', ls=':')
    #ax_chl.plot(wave, fuco*ans[1], color='blue', label='Fuco', ls=':')
    #ax_chl.plot(wave, diad*ans[2], color='green', label='Diad', ls=':')
    #ax_chl.plot(wave, beta*ans[4], color='orange', label=r'$\beta$-Car', ls=':')
    #ax_chl.plot(wave, zea*ans[0], color='orange', label=r'Zea')


    # Finish
    for ax in [ax_chl]:
        plotting.set_fontsize(ax, 16)
        # Label the axes
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('NMF Basis')
        ax.legend(fontsize=15.)
        # Grid
        ax.grid(True)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

# #########################################################
# #########################################################
def fig_l23_tara_coeffs(
    outfile:str='fig_l23_tara_coeffs.png',
    N_NMF:int=4, iop:str='a'):

    # Load L23 fit
    d_l23 = cnmf_io.load_nmf('L23', N_NMF, iop)
    M_l23 = d_l23['M']
    coeff = d_l23['coeff']
    wave_l23 = d_l23['wave']

    # Load Tara
    d_tara = cnmf_io.load_nmf('Tara_L23', N_NMF, iop)
    M_tara = d_tara['M']
    tara_coeff = d_tara['coeff']

    df = pandas.DataFrame()
    df['H1'] = coeff[:,0].tolist() + tara_coeff[0,:].tolist()
    df['H2'] = coeff[:,1].tolist() + tara_coeff[1,:].tolist()
    df['H3'] = coeff[:,2].tolist() + tara_coeff[2,:].tolist()
    df['H4'] = coeff[:,3].tolist() + tara_coeff[3,:].tolist()
    df['sample'] = ['L23']*len(coeff[:,0]) + ['Tara']*len(tara_coeff[0,:])

    # #########################################################
    # Figure
    figsize=(6,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    xmins = [1e-2, 1e-3, 1e-5, 1e-3]
    for ss in range(4):
        ax= plt.subplot(gs[ss])
        xmin = xmins[ss]
        #embed(header='fig_l23_tara_coeffs 529')
        keep = df[f'H{ss+1}'] > xmin
        sns.histplot(df[keep], x=f'H{ss+1}',
                     hue='sample', 
                     ax=ax, bins=100,
                common_bins=True, stat='density', common_norm=False,
                log_scale=True)
        # Label
        ax.set_xlabel(r'$H_'+f'{ss+1}'+'$')
        # Minor ticks
        ax.tick_params(axis='x', which='both', bottom=True, 
                       top=True, labelbottom=True, 
                       labeltop=False)
        # Fontsize
        plotting.set_fontsize(ax, 12)
        # Range
        #xmin = max(1e-15, np.min(df[f'a{ss+1}']))
        #xmax = np.max(df[f'a{ss+1}'])
        #ax.set_xlim(xmin, xmax)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")



# #########################################################
# #########################################################
def fig_l23_vs_tara_M(
    outfile:str='fig_l23_vs_tara_M.png',
    N_NMF:int=4, iop:str='a', lowsig:bool=False):

    # Load L23 fit
    l23_ds = 'L23' if not lowsig else 'LOWL23'
    d_l23 = cnmf_io.load_nmf(l23_ds, N_NMF, iop)
    M_l23 = d_l23['M']
    coeff_l23 = d_l23['coeff']
    wave_l23 = d_l23['wave']

    # Load Tara
    d_tara = cnmf_io.load_nmf('Tara', N_NMF, iop)
    M_tara = d_tara['M']
    wave_tara = d_tara['wave']

    # #########################################################
    # Figure
    figsize=(8,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    for ss in range(N_NMF):
    # Plot the Ms
        ax = plt.subplot(gs[ss])
        ax.plot(wave_l23, M_l23[ss], label=r'L23: $W_'+f'{ss+1}'+'$')#, ls=':')
        ax.plot(wave_tara, M_tara[ss], label=r'Tara: $W_'+f'{ss+1}'+'$')

        # Axes
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Basis vector')
        ax.legend(fontsize=13.)

        plotting.set_fontsize(ax, 13)
        # Grid
        ax.grid(True)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


# #########################################################
# #########################################################
def fig_l23_tara_a_contours(
    outfile:str='fig_l23_tara_a_contours.png',
    nmf_fit:str='L23',
    N_NMF:int=4, iop:str='a'):

    sns.set(style="whitegrid")

    # Load L23 fit
    d = cnmf_io.load_nmf(nmf_fit, N_NMF, iop)
    M = d['M']
    coeff = d['coeff']
    wave = d['wave']

    # Load Tara
    d_tara = cnmf_io.load_nmf('Tara_L23', N_NMF, iop)
    tara_coeff = d_tara['coeff']

    # Scale?
    scale = True
    if scale:
        for ss in range(4):
            med_l23 = np.median(coeff[ss,:])
            med_tara = np.median(tara_coeff[ss,:])
            print(f"Scale: {med_l23} {med_tara} {med_l23/med_tara}")
            #
            tara_coeff[ss,:] *= med_l23/med_tara


    # #########################################################
    # Figure
    figsize=(6,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(1,1)
    ax= plt.subplot(gs[0])

    # #########################################################
    # L23 Contours plot
    sns.kdeplot(
        x=coeff[:,0], 
        y=coeff[:,1],
        ax=ax,
        kind='kde', label='L23')

    # #########################################################
    # Tara Contours plot
    sns.kdeplot(
        x=tara_coeff[:,0], 
        y=tara_coeff[:,1],
        color='r',
        ax=ax,
        kind='kde', label='Tara', ls=':')

    # Finish
    ax.set_xlabel(r'$a_1$')
    ax.set_ylabel(r'$a_2$')

    ax.set_xlim(0., 0.02)
    ax.set_ylim(0., 0.04)

    ax.legend(fontsize=15.)

    # Finish
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_a_corner(dataset:str='L23'):

    outfile=f'fig_{dataset}_a_corner.png'

    # Load
    d = cnmf_io.load_nmf(dataset, 4, 'a')
    M = d['M']
    coeff = d['coeff']
    wave = d['wave']

    # Set minimums for presentation
    coeff[:,0] = np.maximum(coeff[:,0], 1e-3)
    coeff[:,1] = np.maximum(coeff[:,1], 1e-3)
    coeff[:,2] = np.maximum(coeff[:,2], 1e-3)
    coeff[:,3] = np.maximum(coeff[:,3], 1e-3)

    # Labels
    lbls = []
    for ss in range(4):
        lbls.append(r'$H_'+f'{ss+1}'+r'^{'+f'{dataset}'+'}$')

    fig = corner.corner(
        coeff[:,:4], labels=lbls,
        label_kwargs={'fontsize':17},
        color='blue',
        axes_scale='log',
        show_titles=True,
        title_kwargs={"fontsize": 12},
        )

    # Reset some things
    for ax in fig.get_axes():
        # Title
        if len(ax.get_title()) > 0:
            tit = ax.get_title()
            # Find the second $ sign
            #ipos = tit[1:].find('$')
            #ax.set_title(tit[:ipos+2])
            if dataset == 'L23':
                ax.set_title('Median '+tit[:21]+'$')
            else:
                ax.set_title(tit[:22]+'$')
            #embed(header='745 ')
            # Scrub the title
            #ax.set_title('')
            # Add a grid
            ax.grid(True)
        else: # Add a 1:1 line
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            if xlim[0] != 0.:
                xvals = np.linspace(xlim[0], xlim[1], 1000)
                yvals = xvals
                ax.plot(xvals, yvals, 'k:')
                ax.grid(True)
            

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_explore_corner(param:str, dataset:str):
    #
    X,Y = 4,0
    ds = loisel23.load_ds(X, Y)

    outfile=f'fig_explore_{param}_{dataset}.png'

    # Load
    d = cnmf_io.load_nmf(dataset, 4, 'a')
    M = d['M']
    coeff = d['coeff']
    wave = d['wave']

    # New param
    if param == 'adag' and dataset == 'L23':
        # Calculate 
        L23_wave = ds.Lambda.data
        i400 = np.argmin(np.abs(L23_wave-405.))
        d_param =  ds.ad[:,i400].data / ds.ag[:,i400].data
    elif param == 'chl' and dataset == 'L23':
        d_param = loisel23.calc_Chl(ds)
    else:
        raise ValueError(f'Bad combination of {param} and {dataset}')

    # Set minimums for presentation
    coeff[:,0] = np.maximum(coeff[:,0], 1e-3)
    coeff[:,1] = np.maximum(coeff[:,1], 1e-3)
    coeff[:,2] = np.maximum(coeff[:,2], 1e-3)
    coeff[:,3] = np.maximum(coeff[:,3], 1e-3)
    # Kludge
    coeff = np.append(coeff, d_param[:,None], axis=1)

    # Labels
    lbls = []
    for ss in range(4):
        lbls.append(r'$H_'+f'{ss+1}'+r'^{'+f'{dataset}'+'}$')
    lbls.append(param)

    fig = corner.corner(
        coeff[:,:5], labels=lbls,
        label_kwargs={'fontsize':17},
        color='blue',
        axes_scale='log',
        show_titles=False,
        title_kwargs={"fontsize": 12},
        )

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_nmf_indiv(idxs:list, nmf_fit:str='Tara', N_NMF:int=4,
                  seed=1234):

    # Init seed
    np.random.seed(seed)
    
    # Load
    d = cnmf_io.load_nmf(nmf_fit, N_NMF, 'a')

    # Reconstruction
    recon = np.dot(d['coeff'], d['M'])


    fig = plt.figure(figsize=(12,6))
    plt.clf()
    if isinstance(idxs, int):
        outfile = f'fig_nmf_{nmf_fit}_{idxs}.png'
        idxs = np.random.choice(np.arange(d['spec'].shape[0]), idxs)
        gs = gridspec.GridSpec(3,4)
    elif isinstance(idxs, list):
        # Choose random ones
        gs = gridspec.GridSpec(1,2)
        # Outfile
        outfile = f'fig_nmf_{nmf_fit}_i{idxs[0]}_{idxs[1]}.png'
    else:
        raise ValueError("idxs must be an int or list")

    for tt, idx in enumerate(idxs):
        print(f'id: {idx}')
        ax= plt.subplot(gs[tt])

        ax.plot(d['wave'], d['spec'][idx], 'k', label=f'data: i={idx}')
        #ax.plot(d['wave'], d['spec'][idx2], 'k', label='data2', ls='--')
        lbl = 'model' if tt == 0 else None
        ax.plot(d['wave'], recon[idx], label=lbl)

        # Stats
        dev = recon[idx] - d['spec'][idx]
        rel_dev = np.abs(dev) / d['spec'][idx]
        max_dev = np.max(np.abs(dev))
        irel = np.argmax(rel_dev)

        print(f'max_dev: {max_dev}')
        print(f'max_reldev: {rel_dev.max()} at {d["wave"][irel]}')

        # Break it down
        for ss in range(d['M'].shape[0]):
            ax.plot(d['wave'], d['M'][ss]*d['coeff'][idx][ss], 
                    label=r'$H_'+f'{ss+1}: {d["coeff"][idx][ss]:0.2f}'+'$', ls=':')

        ax.set_xlabel('Wavelength (nm)')


        if nmf_fit == 'Tara':
            ax.set_ylabel(r'$a_{\rm p} \; ({\rm m}^{-1})$')
            ax.legend(fontsize=8)
            plotting.set_fontsize(ax, 12)
        else:
            ax.set_ylabel(r'$a_{\rm nw}$ (m$^{-1}$)')
            plotting.set_fontsize(ax, 20)
            ax.legend(fontsize=15.)

        # Grid
        ax.grid(True)
        if nmf_fit == 'Tara':
            if tt>0: 
                ax.tick_params(labelbottom=False)  # Hide x-axis labels
            else:
                ax.set_xlabel('Wavelength (nm)')
        else:
            ax.set_xlabel('Wavelength (nm)')

    # Finish
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_H1_vs_adg(outfile:str='fig_H1_vs_adg.png',
                 nmf_fit:str='L23', N_NMF:int=4):

    # RMSE
    rmss = []
    # load
    d = cnmf_io.load_nmf(nmf_fit, N_NMF, 'a')
    M = d['M']
    wave = d['wave']
    coeff = d['coeff']
    L23_NMF_CDOM = coeff[:,0]

    ds = loisel23.load_ds(4,0)
    L23_wave = ds.Lambda.data
    i400 = np.argmin(np.abs(L23_wave-405.))
    L23_gd =  ds.ag[:,i400].data + ds.ad[:,i400].data


    fig = plt.figure(figsize=(6,6))
    plt.clf()
    ax = plt.gca()

    ax = sns.histplot(x=L23_NMF_CDOM, y=L23_gd, log_scale=True)
    #
    ax.set_xlabel(r'$H_1^{\rm L23}$')
    ax.set_ylabel(r'$a_{\rm dg}^{\rm L23}(405\,{\rm nm})$')

    # Add grid
    ax.grid(True)

    ax.legend()

    #ax.set_yscale('log')
    
    # axes
    plotting.set_fontsize(ax, 15)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_H3_vs_adg(outfile:str='fig_H3_vs_adg.png',
                 nmf_fit:str='L23', N_NMF:int=4, 
                 ax=None, skip_save:bool=False):

    # RMSE
    rmss = []
    # load
    d = cnmf_io.load_nmf(nmf_fit, N_NMF, 'a')
    M = d['M']
    wave = d['wave']
    coeff = d['coeff']
    #L23_NMF_CDOM = coeff[:,0]
    L23_NMF_H3 = coeff[:,2]

    ds = loisel23.load_ds(4,0)
    L23_wave = ds.Lambda.data
    i400 = np.argmin(np.abs(L23_wave-405.))
    i500 = np.argmin(np.abs(L23_wave-500.))
    L23_gd =  ds.ag.data + ds.ad.data
    L23_gd_400 =  ds.ag[:,i400].data + ds.ad[:,i400].data
    L23_dtog_400 =  ds.ad[:,i400].data/ds.ag[:,i400].data
    L23_gd_500_400 = L23_gd[:,i500]/L23_gd[:,i400].data 
    L23_g_500_400 = ds.ag[:,i500].data/ds.ag[:,i400].data 

    # Minimum
    L23_NMF_H3 = np.maximum(L23_NMF_H3, 1e-3)



    if ax is None:
        fig = plt.figure(figsize=(6,6))
        plt.clf()
        ax = plt.gca()

    ax = sns.histplot(x=L23_NMF_H3, 
                      #y=L23_dtog_400,
                      y=L23_gd_500_400, 
                      #y=L23_g_500_400, 
                      log_scale=True)
    #hb = ax.hexbin(L23_NMF_H3, L23_gd_500_400,
    #               gridsize=100, #bins='log', 
    #               xscale='log', yscale='log',
    #                cmap='Blues')

    #
    ax.set_xlabel(r'$H_3^{\rm L23}$')
    ax.set_ylabel(r'$a_{\rm dg}^{\rm L23}(500\,{\rm nm})/a_{\rm dg}^{\rm L23}(405\,{\rm nm})$')

    # Add grid
    ax.grid(True)

    # axes
    plotting.set_fontsize(ax, 15)

    # Stats

    # rank correlation test
    tau, p_value = stats.kendalltau(L23_NMF_H3, L23_gd_500_400)
    embed(header='1218 of figs_nmfI.py')

    if skip_save:
        return
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_H24_vs_aph(outfile:str='fig_H24_vs_aph.png',
                 nmf_fit:str='L23', N_NMF:int=4,
                 ax=None, save_fig:bool=True):

    # load
    d = cnmf_io.load_nmf(nmf_fit, N_NMF, 'a')
    M = d['M']
    wave = d['wave']
    coeff = d['coeff']
    L23_H_ph = coeff[:,1] + coeff[:,3]

    ds = loisel23.load_ds(4,0)
    L23_wave = ds.Lambda.data
    i440 = np.argmin(np.abs(L23_wave-440.))
    L23_ph =  ds.aph[:,i440].data 


    if ax is None:
        fig = plt.figure(figsize=(6,6))
        plt.clf()
        ax = plt.gca()

    hb = ax.hexbin(L23_H_ph, L23_ph,
                   gridsize=100, bins='log', 
                   xscale='log', yscale='log',
                    cmap='Blues')
    #
    cb = plt.colorbar(hb, ax=ax, label='counts')
    ax.set_xlabel(r'$H_{2}^{\rm Tara}$')
    ax.set_ylabel('Tara Chl Lineheight')

    plotting.set_fontsize(ax, 14)
    ax.set_xlabel(r'$H_2^{\rm L23} + H_4^{\rm L23}$')
    ax.set_ylabel(r'$a_{\rm ph}^{\rm L23}(440\,{\rm nm})$')

    ax.set_xlim(1e-2, None)

    # Add grid
    ax.grid(True)

    #ax.set_yscale('log')
    
    # axes
    plotting.set_fontsize(ax, 15)

    if not save_fig:
        return

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_variance_mode(outfile:str='fig_variance_mode.png'): 

    # Load PCAs
    pca_N20 = ihop_pca.load('pca_L23_X4Y0_a_N20.npz',
                                    pca_path=pca_path)

    fig = plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])

    xs = np.arange(len(pca_N20['explained_variance'])) + 1
    exponent = -2.
    y0 = 2
    ys = pca_N20['explained_variance'][y0] * (xs/xs[y0])**(exponent) 

    # Plot
    ax.plot(np.arange(pca_N20['explained_variance'].size)+1, 
            pca_N20['explained_variance'], 'o')
    #ax.plot(xs, d['explained_variance'], 'o', label='Explained Variance')
    ax.plot(xs, ys, '--', color='g', label=f'Power law: {exponent}')
    # Label
    ax.set_ylabel('Variance explained per mode')
    ax.set_xlabel('Number of PCA components')
    #
    #ax.set_xlim(0,10.)
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Minor ticks
    ax.minorticks_on()
    # Horizontal line at 0
    #ax.axhline(0., color='k', ls='--')

    #loc = 'upper right' if ss == 1 else 'upper left'
    ax.legend(fontsize=15)#, loc=loc)

    # Turn on grid
    ax.grid(True, which='both', ls='--', lw=0.5)

    plotting.set_fontsize(ax, 18)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_geo_tara(param:str, N_NMF:int=4, cmap:str='jet',
                 minval:float=-3, maxval:float=None): 

    sparam = param.replace('/','_')
    outfile=f'fig_geo_tara_{sparam}.png'

    print("Loading Tara..")
    tara_db = tara_io.load_pg_db(expedition='Microbiome')

    # Load Tara
    d_tara = cnmf_io.load_nmf('Tara', N_NMF, 'a')
    tara_coeff = d_tara['coeff']

    # Grab lat, lon
    midx = cat_utils.match_ids(d_tara['UID'], tara_db.uid.values)
    lats = tara_db.lat.values[midx]
    lons = tara_db.lon.values[midx]

    # Metric
    if param in ['H1', 'H2', 'H3', 'H4']:
        metric = np.log10(tara_coeff[:,int(param[1])-1])
    elif param == 'H1/H2+H4':
        metric = np.log10(tara_coeff[:,0]/(tara_coeff[:,1]+tara_coeff[:,3]))
        minval, maxval = -1.,1.
    elif '+' in param: 
        metric = np.zeros(tara_coeff.shape[0])
        for pp in param.split('+'):
            metric += tara_coeff[:,int(pp[1])-1]
        metric = np.log10(metric)
    elif '/' in param: 
        metric = np.log10(tara_coeff[:,int(param[1])-1]/tara_coeff[:,int(param[4])-1])
    else:
        raise ValueError(f'Bad param: {param}')

    if minval is not None:
        metric = np.maximum(metric, minval)
    if maxval is not None:
        metric = np.minimum(metric, maxval)

    fig = plt.figure(figsize=(7,8))
    plt.clf()

    ax = plt.subplot(projection=tformM)

    img = plt.scatter(x=lons,
        y=lats, c=metric, cmap=cmap,
            #vmin=0.,
            #vmax=vmax, 
        s=1,
        transform=tformP, label=param)

    # Color bar
    cbaxes = plt.colorbar(img, pad=0., fraction=0.030, orientation='horizontal') #location='left')
    cbaxes.set_label(param, fontsize=17.)
    cbaxes.ax.tick_params(labelsize=15)

    # Coast lines
    ax.coastlines(zorder=10)
    ax.add_feature(cartopy.feature.LAND, 
        facecolor='lightgray', edgecolor='black')
    #ax.set_global()

    lon_min, lon_max, lat_min, lat_max = -100, 30, -70, 50
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])


    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_tara_chl_W(N_NMF:int=4, ax=None,
                   save_fig:bool=True): 

    outfile='fig_tara_chl_W.png'

    print("Loading Tara..")
    tara_db = tara_io.load_pg_db(expedition='Microbiome')

    # Load Tara
    d_tara = cnmf_io.load_nmf('Tara', N_NMF, 'a')
    tara_coeff = d_tara['coeff']

    #tara_chl = tara_coeff[:,1] + tara_coeff[:,3]
    NMF_chl = tara_coeff[:,1] 

    # Grab lat, lon
    midx = cat_utils.match_ids(d_tara['UID'], tara_db.uid.values)
    Tara_chlA = tara_db.Chl_lineheight.values[midx]

    if ax is None:
        figsize=(8,6)
        fig = plt.figure(figsize=figsize)
        plt.clf()
        gs = gridspec.GridSpec(1,1)
        ax= plt.subplot(gs[0])

    keep = (Tara_chlA > 0.01) & (NMF_chl > 0.01)

    hb = ax.hexbin(NMF_chl[keep], Tara_chlA[keep], 
                   gridsize=50, bins='log', 
                   xscale='log', yscale='log',
                    cmap='Greens')
    #ax.set(xlim=xlim, ylim=ylim)
    cb = plt.colorbar(hb, ax=ax, label='counts')
    ax.set_xlabel(r'$H_{2}^{\rm Tara}$')
    ax.set_ylabel('Tara Chl Lineheight')
    plotting.set_fontsize(ax, 14)

    ax.grid(True)

    # Stats
    tau, p_value = stats.kendalltau(NMF_chl[keep], Tara_chlA[keep])
    print(f'LH: Kendall tau: {tau} p-value: {p_value}')


    if not save_fig:
        return

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_H3_combined(outfile='fig_H3_combined.png'):

    # Load
    nmf_fit = 'Tara'
    N_NMF, iop = 4, 'a'
    d = cnmf_io.load_nmf(nmf_fit, N_NMF, iop)
    M = d['M']
    coeff = d['coeff']
    NMF_wave = d['wave']
    tara_db = tara_io.load_pg_db(expedition='Microbiome')

    NMF_chl = coeff[:,1] 

    # Chl
    midx = cat_utils.match_ids(d['UID'], tara_db.uid.values)
    Tara_chlA = tara_db.Chl_lineheight.values[midx]

    # High outliers
    high_out = (NMF_chl > 2.) & (Tara_chlA < 0.5)
    all_high = np.where(high_out)[0]
    high_idx = all_high[0]

    # FIGURE
    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(1,2)

    # #############################################
    ax_spec = plt.subplot(gs[0])

    ax_spec.plot(d['wave'], d['spec'][high_idx])
    model = np.zeros_like(d['wave'])
    # Break it down
    for ss in range(d['M'].shape[0]):
        ax_spec.plot(d['wave'], d['M'][ss]*d['coeff'][high_idx][ss], 
            label=r'$H_'+f'{ss+1}: {d["coeff"][high_idx][ss]:0.2f}'+'$', ls=':')
        #
        model += d['M'][ss]*d['coeff'][high_idx][ss]
    ax_spec.plot(d['wave'], model, 'k:', label='Total')
    ax_spec.legend(fontsize=14)
    #
    ax_spec.set_xlabel('Wavelength (nm)')
    ax_spec.set_ylabel(r'$a_{\rm p}(\lambda) \; [\rm m^{-1}]$')

    plotting.set_fontsize(ax_spec, 15)
    #
    #ax_spec.text(0.15, 0.90, r'(a) High $H_2$', color='k',
    #    transform=ax_spec.transAxes,
    #    fontsize=22, ha='left')

    # #############################################
    ax_dg = plt.subplot(gs[1])

    fig_H3_vs_adg(ax=ax_dg, skip_save=True)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_aph_vs_H(outfile='fig_aph_vs_H.png'):

    # FIGURE
    fig = plt.figure(figsize=(12,5))
    gs = gridspec.GridSpec(1,2)

    # Tara
    ax_675 = plt.subplot(gs[0])
    fig_tara_chl_W(ax=ax_675, save_fig=False)

    # L23
    ax_440 = plt.subplot(gs[1])
    fig_H24_vs_aph(ax=ax_440, save_fig=False)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_fit_W24(outfile='fig_fit_W24.png',
                 ichl:int=1, # 0-indexing
                 N_NMF:int=4,
                 chl_min:float=420.,
                 add_gaussians:bool=False,
                dataset='L23'):

   # Load
    d = cnmf_io.load_nmf(dataset, N_NMF, 'a')
    M = d['M']
    wave = d['wave']

    # Gererate the profile
    if dataset == 'L23':
        #a_chl = np.sum(M[1:], axis=0)
        dlbl = r'$W_2^{\rm L23} + W_4^{\rm L23}$'
        a_chl = M[1] + M[3]#*1.25
    else:
        a_chl = M[ichl]
        dlbl=f'{dataset}: '+r'$W_'+f'{ichl+1}'+'$'


    chla = pigments.a_chl(wave, ctype='a')
    chlb = pigments.a_chl(wave, ctype='b')
    chlc = pigments.a_chl(wave, ctype='c12')
    peri = pigments.a_chl(wave, pigment='Peri')
    beta = pigments.a_chl(wave, pigment='beta-Car')
    G584 = pigments.a_chl(wave, source='chase', pigment='G584')

    gd1 = (wave > chl_min) & (wave < 550.)
    gd2 = (wave > 640.) & (wave < 700.)
    gd_wave2 = gd1 | gd2
    if add_gaussians:
        gd3 = (wave > 560.) & (wave < 620.)
        gd_wave2 |= gd3

    add_pigments=[peri[gd_wave2], beta[gd_wave2]]
    if add_gaussians:
        add_pigments += [G584[gd_wave2]]

    # Fit
    sigma = np.ones_like(wave[gd_wave2])*0.05
    ans, cov = pigments.fit_a_chl(
        wave[gd_wave2], a_chl[gd_wave2], 
        add_pigments=add_pigments,
        fit_type='positive', sigma=sigma)
    print(f'Chl fit: {ans}')

    all_pigments=[peri, beta]
    if add_gaussians:
        all_pigments += [G584]
    def mk_model(*pargs):
        # pargs[0] is not used
        # Chl
        a = pargs[0]*chla + pargs[1]*chlb + pargs[2]*chlc
        # Others?
        if all_pigments is not None:
            for i, pigment in enumerate(all_pigments):
                a += pargs[3+i]*pigment
        # Return
        return a
    #embed(header='fig_fit_nmf 457')
    new_model = mk_model(*ans)

    fig = plt.figure(figsize=(9,5))
    ax_chl = plt.gca()

    ax_chl.plot(wave, a_chl, color='k', 
                label=dlbl)
    ax_chl.plot(wave[gd_wave2], new_model[gd_wave2], 'ro', 
                label='model')
    #https://www.allmovie.com/artist/akira-kurosawa-vn6780882/filmography

    # Chl
    for ss, pig, wv, lbl in zip(range(3), [chla,chlb,chlc], [673.,440.,440.], ['a', 'b', 'c12']):
        #iwv = np.argmin(np.abs(wave-wv))
        #nrm = pig[iwv]/M[0,iwv]
        #print(f'nrm: {1/nrm}')
        ax_chl.plot(wave, pig*ans[ss], label=f'Chl-{lbl}', ls=':')


    # New ones
    ax_chl.plot(wave, peri*ans[3], color='purple', label='Peri', ls=':')
    ax_chl.plot(wave, beta*ans[4], color='orange', label=r'$\beta$-Car', ls=':')
    if add_gaussians:
        ax_chl.plot(wave, G584*ans[5], color='gray', label=r'Chl-c(585)', ls=':')


    ax = ax_chl
    plotting.set_fontsize(ax, 16)
    # Label the axes
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('NMF Basis')
    ax.legend(fontsize=15.)
    # Grid

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_tara_outliers():

    outfile = 'fig_tara_outliers.png'

    # Load
    nmf_fit = 'Tara'
    N_NMF, iop = 4, 'a'
    d = cnmf_io.load_nmf(nmf_fit, N_NMF, iop)
    M = d['M']
    coeff = d['coeff']
    NMF_wave = d['wave']
    tara_db = tara_io.load_pg_db(expedition='Microbiome')

    NMF_chl = coeff[:,1] 

    # Chl
    midx = cat_utils.match_ids(d['UID'], tara_db.uid.values)
    Tara_chlA = tara_db.Chl_lineheight.values[midx]

    # High outliers
    high_out = (NMF_chl > 2.) & (Tara_chlA < 0.5)
    all_high = np.where(high_out)[0]
    high_idx = all_high[0]

    # Low outliers
    low_out = (NMF_chl < 0.1) & (Tara_chlA > 0.5)
    all_low = np.where(low_out)[0]
    low_idx = all_low[0]

    fig = plt.figure(figsize=(12,7))
    gs = gridspec.GridSpec(1,2)

    # High
    ax_high = plt.subplot(gs[0])
    ax_high.plot(d['wave'], d['spec'][high_idx])
    model = np.zeros_like(d['wave'])
    # Break it down
    for ss in range(d['M'].shape[0]):
        ax_high.plot(d['wave'], d['M'][ss]*d['coeff'][high_idx][ss], 
            label=r'$H_'+f'{ss+1}: {d["coeff"][high_idx][ss]:0.2f}'+'$', ls=':')
        #
        model += d['M'][ss]*d['coeff'][high_idx][ss]
    ax_high.plot(d['wave'], model, 'k:', label='Total')
    ax_high.legend(fontsize=14)
    #
    ax_high.set_xlabel('Wavelength (nm)')
    ax_high.set_ylabel(r'$a_{\rm p}(\lambda) \; [\rm m^{-1}]$')
    #
    ax_high.text(0.15, 0.90, r'(a) High $H_2$', color='k',
        transform=ax_high.transAxes,
        fontsize=22, ha='left')

    # Low
    ax_low = plt.subplot(gs[1])
    ax_low.plot(d['wave'], d['spec'][low_idx])
    model = np.zeros_like(d['wave'])
    # Break it down
    for ss in range(d['M'].shape[0]):
        ax_low.plot(d['wave'], d['M'][ss]*d['coeff'][low_idx][ss], 
            label=r'$H_'+f'{ss+1}: {d["coeff"][low_idx][ss]:0.2f}'+'$', ls=':')
        #
        model += d['M'][ss]*d['coeff'][low_idx][ss]
    ax_low.plot(d['wave'], model, 'k:', label='Total')
    ax_low.legend(fontsize=14)
    #
    ax_low.set_xlabel('Wavelength (nm)')
    ax_low.set_ylabel(r'$a_{\rm p}(\lambda) \; [\rm m^{-1}]$')
    #
    ax_low.text(0.15, 0.90, r'(a) Low $H_2$', color='k',
        transform=ax_low.transAxes,
        fontsize=22, ha='left')

    for ax in [ax_high, ax_low]:
        plotting.set_fontsize(ax, 16)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_fit_W1(N_NMF:int=4,
               icdom:int=0, # 0-indexing
                 outfile:str='fig_W1_fits.png',
                 cdom_max:float=530.):
    """
    Generate a figure showing the fits of CDOM and chlorophyll using NMF.

    Args:
        nmf_fit (str): The type of NMF fit to use. Default is 'L23'.
        N_NMF (int): The number of NMF components. Default is 4.
        icdom (int): The index of the CDOM component to plot. Default is 1.
        outfile (str): The output file name for the figure. If not provided, a default name will be used based on the nmf_fit parameter. Default is None.
        cdom_max (float, optional): The maximum wavelength for the CDOM fit. Default is 600.0.
    """

    fig = plt.figure(figsize=(8,8))
    gs = gridspec.GridSpec(2,1)

    axes = []
    clrs = ['b', 'orange']
    for ss, dataset in enumerate(['L23', 'Tara']):

        # Load
        d = cnmf_io.load_nmf(dataset, N_NMF, 'a')
        M = d['M']
        wave = d['wave']
        a_cdom = M[icdom]

        # #########################################################
        # CDOM
        wv_cut = wave <= cdom_max
        cut_wv = wave[wv_cut]

        # Fit exponentials
        exp_tot_coeff, cov = cdom.fit_exp_tot(
            wave[wv_cut], a_cdom[wv_cut])
        a_cdom_totexp_fit = exp_tot_coeff[0] * cdom.a_exp(
            wave[wv_cut], S_CDOM=exp_tot_coeff[1])
            #wave0=exp_tot_coeff[2])
        print(f'Tot exp coeff: {exp_tot_coeff}')
        exp_norm_coeff, cov = cdom.fit_exp_norm(wave[wv_cut], 
                                                a_cdom[wv_cut])
        a_cdom_exp_fit = exp_norm_coeff[0] * cdom.a_exp(wave[wv_cut])

        # Fit power-law
        pow_coeff, pow_cov = cdom.fit_pow(cut_wv, a_cdom[wv_cut])
        a_cdom_pow_fit = pow_coeff[0] * cdom.a_pow(cut_wv, S=pow_coeff[1])

        # Plot
        # Plot CDOM fits
        ax_cdom = plt.subplot(gs[ss])

        # NMF
        ax_cdom.step(wave, M[icdom], 
                    label=r'$W_'+f'{icdom+1}'+r'^{\rm '+f'{dataset}'+r'}$', 
                    color=clrs[ss],
                    lw=2)

        #ax_cdom.plot(cut_wv, a_cdom_exp_fit, 
        #        color='b', label='CDOM exp', ls='-')
        ax_cdom.plot(cut_wv, a_cdom_totexp_fit, 
                color='k', 
                label=r'Exponential ($S='+f'{exp_tot_coeff[1]:0.3f}'+r'$)', 
                ls='--', lw=2)
        ax_cdom.plot(cut_wv, a_cdom_pow_fit, 
                color='k', label='Power Law '+r'($\beta='+f'{-1*pow_coeff[1]:0.1f}'+r'$)', 
                ls=':', lw=2)

        ax_cdom.axvline(cdom_max, ls='--', color='gray')
        axes.append(ax_cdom)
        if ss == 0:
            ax_cdom.tick_params(labelbottom=False)  # Hide x-axis labels
        else:
            ax_cdom.set_xlabel('Wavelength (nm)')

    # Finish
    for ax in axes:
        plotting.set_fontsize(ax, 19)
        # Label the axes
        ax.set_ylabel(r'$W_1$ Basis Function')
        ax.legend(fontsize=15.)
        # Grid
        ax.grid(True)
        ax.set_xlim(400., 650.)
        # Minor ticks
        ax.minorticks_on()

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_outliers(items:list=[(2298, 'L23'),
                           (120863, 'Tara'),
                           (1245, 'L23'),
                           (105191, 'Tara'),
                           ], 
                 N_NMF:int=4,
                outfile=f'fig_outliers.png',
                 seed=1234):

    data = {}
    for key in ['L23', 'Tara']:
        # Load
        data[key] = cnmf_io.load_nmf(key, N_NMF, 'a')
        # Reconstruction
        data[f'recon_{key}'] = np.dot(data[key]['coeff'], 
                               data[key]['M'])

    fig = plt.figure(figsize=(9,6))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    for tt, item in enumerate(items):
        idx, dataset = item
        print(f'id: {idx}')

        ax= plt.subplot(gs[tt])
        d = data[dataset]
        recon = data[f'recon_{dataset}']

        ax.plot(d['wave'], d['spec'][idx], 'k', 
                label=f'{dataset}: i={idx}')
        #ax.plot(d['wave'], d['spec'][idx2], 'k', label='data2', ls='--')
        lbl = 'model' if tt == 0 else None
        ax.plot(d['wave'], recon[idx], label=lbl)

        # Stats
        dev = recon[idx] - d['spec'][idx]
        rel_dev = np.abs(dev) / d['spec'][idx]
        max_dev = np.max(np.abs(dev))
        irel = np.argmax(rel_dev)

        print(f'max_dev: {max_dev}')
        print(f'max_reldev: {rel_dev.max()} at {d["wave"][irel]}')

        # Break it down
        for ss in range(d['M'].shape[0]):
            ax.plot(d['wave'], d['M'][ss]*d['coeff'][idx][ss], 
                    label=r'$H_'+f'{ss+1}: {d["coeff"][idx][ss]:0.2f}'+'$', ls=':')

        if dataset == 'Tara':
            ax.set_ylabel(r'$a_{\rm p} \; ({\rm m}^{-1})$')
        else:
            ax.set_ylabel(r'$a_{\rm nw}$ (m$^{-1}$)')
        ax.legend(fontsize=9)
        plotting.set_fontsize(ax, 15)

        # Grid
        ax.grid(True)
        if tt<2: 
            ax.tick_params(labelbottom=False)  # Hide x-axis labels
        else:
            ax.set_xlabel('Wavelength (nm)')

    # Finish
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_bricaud_rmse(outfile:str='fig_bricaud_rmse.png'):

    # Load
    bricaud = np.load('../Analysis/L23_aph_fits.npz')

    #df = pandas.DataFrame(bricaud)

    fig = plt.figure(figsize=(7,7))
    gs = gridspec.GridSpec(1,1)


    ax = plt.subplot(gs[0])

    # Scatter me
    sz = 1.
    ax.scatter(bricaud['aph_440'], bricaud['b_rmses'], color='g', label='Bricaud', s=sz)
    ax.scatter(bricaud['aph_440'], bricaud['nmf_rmses'], color='b', label='NMF', s=sz)

    print(f'Median ratio = {np.median(bricaud["b_rmses"]/bricaud["nmf_rmses"]):0.2f}')

    # Plot the running median
    #srt_b = np.argsort(bricaud['aph_440'])
    #med_b = signal.medfilt(bricaud['b_rmses'][srt_b], kernel_size=31)
    #med_n = signal.medfilt(bricaud['nmf_rmses'], kernel_size=51)

    #ax.plot(bricaud['aph_440'][srt_b], med_b, 'g-')
    #ax.plot(bricaud['aph_440'], med_n, 'b--')

    #ax.scatter(bricaud['nmf_rmses'], bricaud['b_rmses'])#, label='NMF')
    #ax.plot([1e-5, 1e-1], [1e-5,1e-1], 'k--')
    #ax.plot([1e-5, 1e-1], [3e-5,3e-1], 'k:')

    ax.set_xlabel(r'$a_{\rm ph}(440\,{\rm nm})$')
    ax.set_ylabel('RMSE')

    # Log on x-axis
    ax.set_xscale('log')
    ax.set_yscale('log')

    plotting.set_fontsize(ax, 17)

    ax.legend(fontsize=15)

    # Finish
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
        fig_examples()

    # PCA vs NMF explained variance on L23
    if flg & (2**1):
        fig_l23_pca_nmf_var()

    # L23: PCA and NMF basis functions
    if flg & (2**2):
        fig_nmf_pca_basis()
        #fig_nmf_pca_basis(Ncomp=3,
        #                  outfile='fig_nmf_pca_basis_N3.png')
        #fig_nmf_pca_basis(outfile='fig_nmf_pca_basis_aph.png',
        #                  iop='aph', Ncomp=3,
        #                  skip_pca=True)

    # Individual
    if flg & (2**3):
        #fig_nmf_indiv([100, 20000])
        fig_nmf_indiv([100, 2000], nmf_fit='L23')

    # Compare the NMF bases
    if flg & (2**4): # 16
        fig_l23_vs_tara_M()
        #fig_l23_vs_tara_M(lowsig=True)
        #fig_l23_vs_tara_M(N_NMF=3)
        #fig_l23_vs_tara_M(outfile='fig_l23_vs_tara_M_N3.png',
        #    N_NMF=3)

    # H1 vs adg
    if flg & (2**5):
        fig_fit_W1()

    # H1 vs adg
    if flg & (2**6):
        fig_H1_vs_adg()

    # H3
    if flg & (2**7):
        fig_H3_combined()

    # 
    if flg & (2**8):
        fig_aph_vs_H()

    # 
    if flg & (2**9):
        fig_fit_W24()

    # Outliers
    if flg & (2**11):
        fig_outliers()

    # APPENDIX
    # Individual for Tara
    if flg & (2**20):
        fig_aph_nmf()

    # aph Fits
    if flg & (2**21):
        fig_aph_fits()

    # Bricaud RMSE
    if flg & (2**22): # 16
        fig_bricaud_rmse()



    # Individual for Tara
    if flg & (2**13):
        fig_nmf_indiv(12, nmf_fit='Tara')

    # Coeff as corner plot
    if flg & (2**14):
        #fig_a_corner()
        fig_a_corner(dataset='Tara')

    # NMF RMSE
    if flg & (2**40):
        fig_nmf_rmse()

    # L23: a1, z2 contours
    if flg & (2**27):
        fig_l23_tara_a_contours()


    # Variance per mode
    if flg & (2**16):
        fig_variance_mode()

    # Corner parameter + H
    if flg & (2**17):
        #fig_explore_corner('adag', 'L23')
        fig_explore_corner('chl', 'L23')

    # Corner parameter + H
    if flg & (2**18):
        fig_geo_tara('H1', minval=-2.)
        #fig_geo_tara('H3')
        fig_geo_tara('H2+H4', minval=-2.)
        #fig_geo_tara('H1/H2+H4')#, cmap='viridis')
        #fig_geo_tara('H2/H4', maxval=2., minval=-1.)

    # aph vs H2+H4
    if flg & (2**19):
        fig_H24_vs_aph()
    
    # Fit W2 or W4
    if flg & (2**70):
        #fig_fit_W2(nmf_fit='L23', chl_min=460.)
        fig_fit_W2(nmf_fit='Tara', chl_min=460.)

    # Fit W2 or W4
    if flg & (2**71):
        #fig_fit_W4(nmf_fit='L23', chl_min=440.)
        fig_fit_W4(nmf_fit='Tara', chl_min=440.)

    # Tara Chl
    if flg & (2**72):
        fig_tara_chl_W()

    # Tara Chl outliers
    if flg & (2**23):
        fig_tara_outliers()

    # H1 vs adg
    if flg & (2**24):
        fig_H3_vs_adg()
        #fig_a_corner(nmf_fit='Tara')
    


    # L23: Fit NMF 1, 2
    if flg & (2**30):  # 8
        fig_fit_nmf()
        #fig_fit_nmf(nmf_fit='Tara')
        #fig_fit_nmf(outfile='fig_W3_l23_fit.png',
        #            icdom=2, cdom_max=550.)
        #fig_fit_nmf(nmf_fit='Tara', cdom_max=530.)

    # Coefficient distributions for L23 NMF
    if flg & (2**31): # 16
        fig_l23_tara_coeffs()

    # Fit nmr
    if flg & (2**99): # 64
        fig_fit_nmf(icdom=0, ichl=1, cdom_max=530.)
        #fig_fit_nmf(icdom=0, ichl=1, cdom_max=530.,
        #            N_NMF=3, outfile='fig_l23_fit_nmf_N3.png')
        #fig_fit_nmf(nmf_fit='Tara', cdom_max=530.,
        #            icdom=0, ichl=1)

    # NMF basis
    if flg & (2**51):
        fig_nmf_basis()
        fig_nmf_basis(N_NMF=5)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Figure 1 Example spectra
        #flg += 2 ** 1  # 2 -- Figure 2 L23: PCA vs NMF Explained variance
        #flg += 2 ** 2  # 4 -- Figure 3 L23: PCA and NMF basis
        #flg += 2 ** 3  # 8 --  Figure 4: Example fits of L23
        #flg += 2 ** 4  # 16 -- Figure 5: L23,Tara compare NMF basis functions
        #flg += 2 ** 5  # 32 -- Figure 6a: Fit to W1
        #flg += 2 ** 6  # 64 -- Figure 6b: L23 a_g + a_d
        #flg += 2 ** 7  # 128 -- Figure 7: H3
        #flg += 2 ** 8  # 256 -- Figure 8: H2 and H24
        #flg += 2 ** 9  # 512 -- Figure 9: Fit W2+W4

        #flg += 2 ** 11  # 2048 -- Figure 11: Outliers

        # Appendix
        #flg += 2 ** 20  # aph NMF
        #flg += 2 ** 21  # aph fits
        flg += 2 ** 22  # aph RMSE

        #flg += 2 ** XX  # 64 -- Fit l23 basis functions

        
        #flg += 2 ** 12  # L23 Indiv
        #flg += 2 ** 13  # Tara Indiv
        #flg += 2 ** 14  # L23/Tara H coefficients in a Corner plot

        #flg += 2 ** 16  # Variance per mode (PCA)
        #flg += 2 ** 17  # L23 H coefficients + ad/ag in a Corner plot
        #flg += 2 ** 18  # Explore Tara geographic distribution
        #flg += 2 ** 19  # L23 aph vs H2+H4
        #flg += 2 ** 20  # Fit W2 
        #flg += 2 ** 21  # Fit W4 

        #flg += 2 ** 22  # Tara Chl-a
        #flg += 2 ** 23  # Tara Chl-a outliers
        #flg += 2 ** 24  # Tara Chl-a outliers

        #flg += 2 ** 30  # 8 -- L23: Fit NMF basis functions with CDOM, Chl
        #flg += 2 ** 31  # 16 -- L23+Tara; W1, W2, W3, W4 coefficients


    else:
        flg = sys.argv[1]

    main(flg)