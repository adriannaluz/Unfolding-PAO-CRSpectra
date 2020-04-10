import numpy as np
import pyunfold
import matplotlib as mpl
import matplotlib.pyplot as plt
from astrotools import container as ctn
from astrotools import statistics as stats
from astrotools import auger
from pyunfold import iterative_unfold
from pyunfold.callbacks import Logger
from funcs import efficiencies

composition = ["proton", "helium", "oxygen", "iron"]
#composition = ["proton"]
evt_auger = [83143, 47500, 28657, 17843, 12435, 8715, 6050, 4111, 2620, 1691,
           991, 624, 372, 156, 83, 24, 9, 6]

#evt_bins = np.linspace(np.log10(2.5*pow(10,18)), np.log10(.3*pow(10,20)), len(evt_auger))
bin_width = 0.1
e_bins = np.arange(np.log10(2.5*pow(10,18)), 20.2, bin_width)

Junf_Jraw = [0.88 , 0.93, 0.935, 0.925, 0.945, 0.96, 0.975, 0.98,
             0.985, 0.98, 0.985, 0.99, 0.995, 0.995, 0.99,
             0.975, 0.965, 0.96, 0.965]

A = 3000. # km**2
t = 14.6 # years  # data from the Auger paper (unpublish)
Om = 2 * np.pi * (1 - np.cos(np.deg2rad(60.))) # solid angle in sr

path_scratch = '/net/scratch/Adrianna/data_analysis/'
path_scratch_CIC = '/net/scratch/Adrianna/data_analysis_data/SIMULATIONS/'
fig, ax = plt.subplots()

for i, nuclei in enumerate(composition):
    #==== Naming npz arrays ====#
    SD_array  = path_scratch + 'arrays/SD_only/%s_SDonly_merge_p3.npz' % nuclei
    array  = path_scratch_CIC + "arrays/Ecalibrated_%s" % nuclei

    #==== Opening data containers ====#
    SD_data = ctn.DataContainer(SD_array)
    data    = ctn.DataContainer(array)

    Erec = data["ESD_calib"]
    EMC  = SD_data["MC_energy"]
    Xmax = SD_data["MC_xmax"]

    bins = np.arange(18., 20.1,0.1)

    log10e_bins = np.arange(np.log10(2.5*pow(10,18)), 20.2, 0.1)  # np.arange(18., 20.25, 0.05)
    mask = np.log10(EMC) >= np.log10(2.5*pow(10,18))  # 18.
    EMC = EMC[mask]
    Erec = Erec[mask]

    H = np.histogram(np.log10(EMC), bins=log10e_bins)

    #  Sampling the simulated spectrum from auger analytic spectrum (ICRC2017)
    w = 1 / H[0][np.digitize(np.log10(EMC), bins=log10e_bins) - 2] # Correction for not having a flat spectrum
    p = EMC * auger.spectrum_analytic(np.log10(EMC), year=17) * w  # Differential spectrum
    rdm = np.random.choice(np.arange(len(Erec)), sum(evt_auger), replace=True, p=p/sum(p))
    Erec_new = Erec[rdm]
    EMC_new = EMC[rdm]


    #==== UNFOLDING ====#

    hist_MC  = np.histogram(np.log10(EMC_new) , bins=e_bins, weights=EMC_new**2)
    hist_rec = np.histogram(np.log10(Erec_new), bins=hist_MC[1], weights=Erec_new**2)

    true_samples = np.log10(EMC_new)
    Etrue_err = np.sqrt(true_samples)  # Poissonian counting errors
    observed_samples = np.log10(Erec_new)
    Erec_err = np.sqrt(observed_samples)    # Poissonian counting errors

    data_true = hist_MC[0]
    Ntrue_err = np.sqrt(data_true)
    data_observed = hist_rec[0]
    Nrec_err = np.sqrt(data_observed)

    efficiency = efficiencies(e_bins, Erec) #data_observed/data_true
    #efficiency[0] = 0.68
    #efficiency[1] = 0.91
    efficiency_err = 0.1 * efficiency

    response_hist = np.histogram2d(observed_samples,true_samples,bins=e_bins)[0]
    response_hist_err = np.sqrt(response_hist)

    Nevt_bin = response_hist.sum(axis=0)
    norm_factor = efficiency / Nevt_bin

    response = response_hist * norm_factor   # Response or migration matrix
    response_err = response_hist_err * norm_factor

    if False:  # Plotting migration matrix
        im = plt.imshow(response, origin='lower', cmap=plt.cm.RdPu)
        cbar = plt.colorbar(im, label='$P(E_i|C_{\mu})$')
        x = np.rint(np.linspace(0, len(e_bins)-2, 6)).astype(int)
        plt.plot(x,x, 'w-', linewidth=.75)
        plt.yticks(x, np.round(e_bins[x],1))
        plt.xticks(x, np.round(e_bins[x],1))
        fig_name = "resp_matrix_new_%s" % nuclei
        plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
        plt.close()

    #==== UNFOLDING ====#
    unfolded_results = iterative_unfold(data=data_observed,
                                        data_err=np.sqrt(data_observed),
                                        response=response,
                                        response_err=response_err,
                                        efficiencies=efficiency,
                                        efficiencies_err=efficiency_err,
                                        callbacks=[Logger()])


    ax.hist(stats.mid(e_bins), e_bins, weights=data_true/A/t/Om/bin_width, histtype="step", lw=1,
            alpha=0.7, label='Simulated %s' %nuclei, linestyle="--")#, color="C3")
    #ax.hist(stats.mid(e_bins), e_bins, weights=data_observed/A/t/Om, histtype="step", lw=3,
    #        alpha=0.7, label='Observed distribution', color="C2")
    #ax.hist(stats.mid(e_bins), e_bins, weights=evt_auger*(10**stats.mid(e_bins))**2/A/t/Om/bin_width, histtype="step", lw=3,
    #        alpha=0.7, label='Auger raw (2019)')#, color="C4")
    #ax.hist(stats.mid(e_bins), e_bins, weights=evt_auger*(10**stats.mid(e_bins))**2/A/t/Om/bin_width*Junf_Jraw[:-1], histtype="step", lw=3,
    #        alpha=0.7, label='Auger unfold. (2019)')#, color="C5")
    #ax.errorbar(stats.mid(e_bins), data_observed/A/t/Om/bin_width,
    #            yerr=Nrec_err*10**(stats.mid(e_bins))/A/t/Om/bin_width,
    #            alpha=0.7,
    #            elinewidth=3,
    #            capsize=4,
    #            ls='None', marker='.', ms=10,
    #            label='Reconstructed (sim)')#, color="C2")
    ax.errorbar(stats.mid(e_bins), unfolded_results['unfolded']/A/t/Om/bin_width,
                yerr=unfolded_results['sys_err']/A/t/Om/bin_width,
                alpha=0.7,
                elinewidth=3,
                capsize=4,
                ls='None', marker='.', ms=10,
                label='%s Unfolded (pyunfold)'%nuclei, color="C%s"%(i+3)) #"C3"

ax.hist(stats.mid(e_bins), e_bins, weights=evt_auger*(10**stats.mid(e_bins))**2/A/t/Om/bin_width*Junf_Jraw[:-1], histtype="step", lw=3,
        alpha=0.7, label='Auger unfold. (2019)')#, color="C5")

#plt.ylim([np.min(data_true/A/t/Om/bin_width)*0.6, 1.6*np.max(data_true/A/t/Om/bin_width)])
plt.ylim([1.*10**(36), 1.*10**(38)])
plt.yscale("log")
ax.set(xlabel=r'log$_{10}$(E/eV)', ylabel=r'J(E)$\times$E$^3$ [eV$^2$ km$^{-2}$ yr$^{-1}$ sr$^{-1}$]')
plt.legend()  #loc="lower left")
#fig_name = "unfolding_new_%s" % nuclei
fig_name = "unfolding_new_all"
plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
plt.close()
