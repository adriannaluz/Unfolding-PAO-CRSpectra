import numpy as np
import pyunfold
import matplotlib as mpl
import matplotlib.pyplot as plt
from astrotools import container as ctn
from astrotools import statistics as stats
from pyunfold import iterative_unfold
from pyunfold.callbacks import Logger
from funcs import efficiencies

composition = ["proton", "helium", "oxygen", "iron"]

path_scratch = '/net/scratch/Adrianna/data_analysis/'
path_scratch_CIC = '/net/scratch/Adrianna/data_analysis_data/SIMULATIONS/'

for i, nuclei in enumerate(composition):
    #==== Naming npz arrays ====#
    SD_array  = path_scratch + 'arrays/SD_only/%s_SDonly_merge_p3.npz' % nuclei
    array  = path_scratch_CIC + "arrays/Ecalibrated_%s" % nuclei

    #==== Opening data containers ====#
    SD_data = ctn.DataContainer(SD_array)
    data    = ctn.DataContainer(array)

    bins = np.arange(18., 20.1,0.1)
    true_samples = Etrue = np.log10(SD_data["MC_energy"])
    true_samples = Etrue = Etrue[np.log10(SD_data["MC_energy"]) >= bins[0]]
    Etrue_err = np.sqrt(Etrue)  # Poissonian counting errors
    observed_samples = Erec  = np.log10(data["ESD_calib"])
    observed_samples = Erec = Erec[np.log10(SD_data["MC_energy"]) >= bins[0]]
    Erec_err = np.sqrt(Erec)    # Poissonian counting errors

    _, b = np.histogram(Etrue, bins=bins)

    Ntrue, Btrue = np.histogram(Etrue, bins=b, density=True)
    data_true = Ntrue
    Ntrue_err = np.sqrt(Ntrue)
    Nrec , Brec  = np.histogram(Erec, bins=b, density=True)
    data_observed = Nrec
    Nrec_err = np.sqrt(Nrec)

    if False:
        fig, ax = plt.subplots()
        ax.hist(bins[:-1], bins, weights=data_true, histtype="step", lw=3,
                alpha=0.7, label='True distribution')
        ax.hist(bins[:-1], bins, weights=data_observed, histtype="step", lw=3,
                alpha=0.7, label='Observed distribution')

        ax.set(xlabel=r'log$_{10}$(E/eV)', ylabel='Density')
        plt.ylim([2*pow(10,-1),7*pow(10,-1)])
        plt.yscale("log")
        plt.legend()
        fig_name = "hist_true-obsdata_%s" % nuclei
        plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')
        plt.close()

    efficiencies = np.ones_like(data_observed, dtype=float)
    efficiencies_err = np.full_like(efficiencies, 0.1, dtype=float)

    response_hist = np.histogram2d(observed_samples,true_samples,bins=bins)[0]
    response_hist_err = np.sqrt(response_hist)

    Nevt_bin = response_hist.sum(axis=0)
    norm_factor = efficiencies / Nevt_bin

    response = response_hist * norm_factor   # Response or migration matrix
    response_err = response_hist_err * norm_factor

    if False:
        im = plt.imshow(response, origin='lower', cmap=plt.cm.RdPu)
        cbar = plt.colorbar(im, label='$P(E_i|C_{\mu})$')
        x = np.rint(np.linspace(0, len(bins)-2, 6)).astype(int)
        plt.plot(x,x, 'w-', linewidth=.75)
        plt.yticks(x, np.round(bins[x],1))
        plt.xticks(x, np.round(bins[x],1))
        fig_name = "test_2"
        plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
        plt.close()

    unfolded_results = iterative_unfold(data=data_observed,
                                    data_err=np.sqrt(data_observed),
                                    response=response,
                                    response_err=response_err,
                                    efficiencies=efficiencies,
                                    efficiencies_err=efficiencies_err,
                                    callbacks=[Logger()])

    if False:
        fig, ax = plt.subplots()
        ax.hist(bins[:-1], bins, weights=data_true, histtype="step", lw=3,
                alpha=0.7, label='True distribution')
        ax.hist(bins[:-1], bins, weights=data_observed, histtype="step", lw=3,
                alpha=0.7, label='Observed distribution')
        ax.errorbar(bins[:-1], unfolded_results['unfolded'],
                    yerr=unfolded_results['sys_err'],
                    alpha=0.7,
                    elinewidth=3,
                    capsize=4,
                    ls='None', marker='.', ms=10,
                    label='Unfolded distribution')

        ax.set(xlabel='X bins', ylabel='Counts')
        plt.ylim([2*pow(10,-1),7*pow(10,-1)])
        plt.yscale("log")
        ax.set(xlabel=r'log$_{10}$(E/eV)', ylabel='Density')
        plt.legend(loc='upper left')
        fig_name = "unfolding_%s" %nuclei
        plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
        plt.close()
