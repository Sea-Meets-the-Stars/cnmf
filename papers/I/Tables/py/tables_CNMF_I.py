"""  Module for Tables for the CNMF paper I """
# Imports
import os, sys


# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#import ssl_paper_analy

from cnmf import io as cnmf_io

from IPython import embed

def mktab_coeffs(dataset:str, outroot='tab_coeffs', 
                 sub=False, local=True):

    outfile = outroot+'_'+dataset+'.tex'
    if sub:
        outfile=outfile.replace('.tex', '_sub.tex')

    # Load up 
    tbl_dict = {}

    # Load
    N_NMF, iop = 4, 'a'
    d = cnmf_io.load_nmf(dataset, N_NMF, iop)
    M = d['M']
    coeff = d['coeff']
    
    # Open
    tbfil = open(outfile, 'w')

    # Header
    #tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{table*}\n')
    tbfil.write('\\centering\n')
    tbfil.write('\\caption{'+f'Derived Non-Negative Matrix Factorization Coefficients for the absorption coefficient spectra of the {dataset}'\
        ' dataset.  ')
    if dataset == 'L23':
        tbfil.write('The index is the row number for the dataset. ')
    else:
        tbfil.write('The UID refers to the Unix time stamp (in nanoseconds) of the observation.')
    tbfil.write('\\label{tab:'+f'{dataset}'+'}}\n')
    tbfil.write('\\begin{tabular}{cccccccccc}\n')
    tbfil.write('\\hline \n')
    if dataset == 'L23':
        tbfil.write('index & $H_1$ & $H_2$ & $H_3$ & $H_4$ \\\\ \n')
    else:
        tbfil.write('UID & $H_1$ & $H_2$ & $H_3$ & $H_4$ \\\\ \n')
    tbfil.write('\\\\ \n')
    tbfil.write('\\hline \n')


    tbfil.write('\\hline \n')

    # Loop me 
    for count in range(coeff.shape[0]):
        if sub and count > 20:
            break

        # Index
        if dataset == 'L23':
            slin = f'{count}'
        else:
            slin = f"{d['UID'][count]}"

        # Coeffs
        for ss in range(coeff.shape[1]):
            # H1
            slin += f'& {coeff[count,ss]:0.4f}'


        tbfil.write(slin)
        tbfil.write('\\\\ \n')

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    #tbfil.write('\\end{minipage} \n')
    tbfil.write('\\\\ \n')
    #tbfil.write('Notes: The \\DT\\ value listed here is measured from the inner $40 \\times 40$\,pixel$^2$ region of the cutout. \\\\ \n')
    #tbfil.write('LL is the log-likelihood metric calculated from the \\ulmo\\ algorithm. \\\\ \n')
    #tbfil.write('$U_{0,\\rm all}, U_{1,\\rm all}$ are the UMAP values for the UMAP analysis on the full dataset. \\\\ \n')
    #tbfil.write('$U_0, U_1$ are the UMAP values for the UMAP analysis in the \\DT\\ bin for this cutout. \\\\ \n')
    #tbfil.write('{$^b$}Assumes $\\nu=1$GHz, $n_e = 4 \\times 10^{-3} \\cm{-3}$, $z_{\\rm DLA} = 1$, $z_{\\rm source} = 2$.\\\\ \n')
    tbfil.write('\\end{table*} \n')

    tbfil.close()

    print('Wrote {:s}'.format(outfile))



# Command line execution
if __name__ == '__main__':

    mktab_coeffs('L23', sub=True)
    mktab_coeffs('Tara', sub=True)