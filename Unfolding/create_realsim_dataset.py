import numpy as np
import pyunfold
import matplotlib as mpl
import matplotlib.pyplot as plt
from astrotools import container as ctn
from astrotools import statistics as stats
from astrotools import auger
from scipy.interpolate import interp1d
from pyunfold import iterative_unfold
from pyunfold.callbacks import Logger
from funcs import efficiencies
from scipy.interpolate import interp1d

with_latex = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 20,
    "font.size": 20,
    "legend.fontsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fancybox": False}
mpl.rcParams.update(with_latex)

composition = ["proton", 'helium', 'oxygen', 'iron']

# spectral index for different energy in log scale from 18.3 to 19.9
sp_index = [3.4, 3.3, 3.2, 3.15, 3.05, 2.8, 2.6, 2.5, 2.55, 2.5, 2.6, 2.55,
            2.75, 2.8, 2.95, 2.9, 3.1, 3.1, 3.15, 3., 3.35, 3.75, 3.95,
            4.25, 4.7, 4.3, 4.5,4.75]
sp_index = np.array(sp_index)

sp_bins = np.linspace(np.log10(2.5*pow(10,18)), 20.2, len(sp_index))
int_sp = interp1d(sp_bins, sp_index, kind='cubic')
#plt.scatter(sp_bins, sp_index)
#bins = np.linspace(np.log10(2.5*pow(10,18)), 20.2, 50)
#plt.scatter(bins, int_sp(bins))
#plt.savefig("testo")
#plt.close()

evt_auger = [83143, 47500, 28657, 17843, 12435, 8715, 6050, 4111, 2620, 1691,
           991, 624, 372, 156, 83, 24, 9, 6]

evt_bins = np.linspace(np.log10(2.5*pow(10,18)), np.log10(.3*pow(10,20)), len(evt_auger))

#e =  np.linspace(np.log10(2.5*pow(10,18)), np.log10(.3*pow(10,20)), len(sp_index))
bin_width = 0.1
e_bins = np.arange(np.log10(2.5*pow(10,18)), 20.2, bin_width)

Junf_Jraw = [0.88 , 0.93, 0.935, 0.925, 0.945, 0.96, 0.975, 0.98,
             0.985, 0.98, 0.985, 0.99, 0.995, 0.995, 0.99,
             0.975, 0.965, 0.96, 0.965]

nuclei = "proton"
path_scratch = '/net/scratch/Adrianna/data_analysis/arrays/'
path_scratch_CIC = '/net/scratch/Adrianna/data_analysis_data/SIMULATIONS/arrays/'

#==== Naming npz arrays ====#
SD_array  = path_scratch + 'SD_only/%s_SDonly_merge_p3.npz' % nuclei
array  = path_scratch_CIC + "Ecalibrated_%s" % nuclei
all_mass = path_scratch_CIC + "E_all"

#==== Opening data containers ====#
SD_data  = ctn.DataContainer(SD_array)
data     = ctn.DataContainer(array)
all_data = ctn.DataContainer(all_mass)

Erec = all_data["ESD_rec"] #SD_data["SD_energy"]
EMC  = all_data["ESD_MC"] #SD_data["MC_energy"]
comp = all_data["composition"]
Xmax = all_data["Xmax"]


if False:
    #mass_num = [1, 4, 16, 56]  # mass number for mean_xmax in astrotools
    E = EMC[Xmax<=1.1*pow(10,3)]
    comp = comp[Xmax<=1.1*pow(10,3)]
    Xmax = Xmax[Xmax<=1.1*pow(10,3)]
    y = np.zeros(len(e_bins)-1)

    for j in range(len(composition)):
        mask_c = np.where(comp==j)[0]

        for i in range(len(e_bins)-1):
            mask = (np.log10(E[mask_c])>= e_bins[i]) & (np.log10(E[mask_c]) < e_bins[i+1])

            y[i] = np.mean(Xmax[mask_c][mask])
        plt.scatter(stats.mid(e_bins), y, label=composition[j])
        #plt.scatter(stats.mid(e_bins),auger.mean_xmax(stats.mid(e_bins), mass_num[j]), s=8., label="auger data %s" %composition[j])

    fig_name = "Xmax_energy"#_total"
    plt.ylabel(r"Xmax [gr/cm$^2$]")
    plt.xlabel(r"log$_{10}$(E/eV)")
    plt.legend()
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
    plt.close()

#==== Xmax distributions ====#
if False:
    comp = comp[Xmax<=1.1*pow(10,3)]
    Eref = Erec[Xmax<=1.1*pow(10,3)]
    Xmax = Xmax[Xmax<=1.1*pow(10,3)]

    bins = np.histogram(Xmax, bins=50)
    bins = plt.hist(Xmax, bins=bins[1], label="total", color="black", alpha=0.7)

    for i in range(4):
        plt.hist(Xmax[np.where(comp==i)[0]], bins=bins[1], histtype="step",
            label=composition[i], lw=2, color="C%s"%i)#, density=True)

    fig_name = "Xmax_dist"#_total"
    plt.xlim([580., 1120.])
    plt.xlabel(r"Xmax [gr/cm$^2$]")
    plt.ylabel("Number of events")
    plt.legend()
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
    plt.close()


#==== Unfolding ====#
mask_xmax = Xmax<=1.1*pow(10,3)
EMC  = EMC[mask_xmax]
Erec = Erec[mask_xmax]
comp = comp[mask_xmax]
Xmax = Xmax[mask_xmax]

A = 3000. # km**2
t = 14.6 # years  # data from the Auger paper (unpublish)
Om = 2 * np.pi * (1 - np.cos(np.deg2rad(60.))) # solid angle in sr

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

#==== Energy distribution for each bin ====#
if False:
    for i in range(len(e_bins)-1):
        mask = (np.log10(EMC_new) >= e_bins[i]) & (np.log10(EMC_new) < e_bins[i+1])
        plt.hist(np.log10(Erec_new[mask]), bins=np.arange(18., 20.2, 0.025), histtype="step")
        plt.vlines(18.5,0, 2.*10**(4), "k")

    plt.ylabel(r'N events')
    plt.xlabel(r'log$_{10}$(E/eV)')
    plt.xlim([18.,20.2])

    labels = [None for i in range(len(np.arange(18.,20.2,0.1)))]
    labels[0] = 18.0
    labels[5] = 18.5
    labels[10] = 19.0
    labels[15] = 19.5
    labels[20] = 20.

    plt.xticks(np.arange(18.,20.2,0.1),labels=labels)

    plt.ylim([0.9, 2.*pow(10,4)])
    plt.yscale("log")
    fig_name = "Edist_binning"
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
    plt.close()

if False:
    hist_MC  = np.histogram(np.log10(EMC_new) , bins=e_bins, weights=EMC_new**2)
    hist_rec = np.histogram(np.log10(Erec_new), bins=hist_MC[1], weights=Erec_new**2)
    comp = comp[mask][rdm]

    #for j in range(len(composition)):
    #    mask_c = np.where(comp==j)[0]
    #    Ep = Erec_new[mask_c]
    #    hist = np.histogram(np.log10(Ep) , bins=e_bins, weights=Ep**2)
    #    plt.errorbar(stats.mid(hist[1]), hist[0], yerr=np.sqrt(hist[0]), fmt="o", markersize=3.5, label=composition[j])#, s=9.0)

    plt.errorbar(stats.mid(hist_rec[1]), hist_rec[0]/A/Om/t/bin_width, yerr=np.sqrt(hist_rec[0]), fmt="ko", markersize=3.5, label=r"E$_{rec}$")#, s=9.0)
    plt.errorbar(stats.mid(hist_MC[1]),  hist_MC[0]/A/Om/t/bin_width, yerr=np.sqrt(hist_MC[0]), fmt="bo", markersize=3.5, label=r"E$_{MC}$")#, s=9.0)
    plt.yscale('log')
    plt.ylim([1.*10**(36), 1.*10**(38)])
    #plt.xscale('log')
    plt.xlabel(r"log$_{10}$(E/eV)")
    plt.ylabel(r'J(E)$\times$E$^3$ [eV$^2$ km$^{-2}$ yr$^{-1}$ sr$^{-1}$]')
    plt.legend()
    fig_name = "SP_energy_mass"
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')
    plt.close()


#==== UNFOLDING ====#
if False:
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

    if True:  # Plotting migration matrix
        im = plt.imshow(response, origin='lower', cmap=plt.cm.RdPu)
        cbar = plt.colorbar(im, label='$P(E_i|C_{\mu})$')
        x = np.rint(np.linspace(0, len(e_bins)-2, 6)).astype(int)
        plt.plot(x,x, 'w-', linewidth=.75)
        plt.yticks(x, np.round(e_bins[x],1))
        plt.xticks(x, np.round(e_bins[x],1))
        fig_name = "resp_matrix_new"
        plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
        plt.close()

    #==== UNFOLDING ====#
    if True:
        unfolded_results = iterative_unfold(data=data_observed,
                                            data_err=np.sqrt(data_observed),
                                            response=response,
                                            response_err=response_err,
                                            efficiencies=efficiency,
                                            efficiencies_err=efficiency_err,
                                            callbacks=[Logger()])

        fig, ax = plt.subplots()
        ax.hist(stats.mid(e_bins), e_bins, weights=data_true/A/t/Om/bin_width, histtype="step", lw=3,
                alpha=0.7, label='Simulated MC', color="C3")
        #ax.hist(stats.mid(e_bins), e_bins, weights=data_observed/A/t/Om, histtype="step", lw=3,
        #        alpha=0.7, label='Observed distribution', color="C2")
        ax.hist(stats.mid(e_bins), e_bins, weights=evt_auger*(10**stats.mid(e_bins))**2/A/t/Om/bin_width, histtype="step", lw=3,
        alpha=0.7, label='Auger raw (2019)', color="C4")
        ax.hist(stats.mid(e_bins), e_bins, weights=evt_auger*(10**stats.mid(e_bins))**2/A/t/Om/bin_width*Junf_Jraw[:-1], histtype="step", lw=3,
                alpha=0.7, label='Auger unfold. (2019)', color="C5")
        ax.errorbar(stats.mid(e_bins), data_observed/A/t/Om/bin_width,
                    yerr=Nrec_err*10**(stats.mid(e_bins))/A/t/Om/bin_width,
                    alpha=0.7,
                    elinewidth=3,
                    capsize=4,
                    ls='None', marker='.', ms=10,
                    label='Reconstructed (sim)', color="C2")
        ax.errorbar(stats.mid(e_bins), unfolded_results['unfolded']/A/t/Om/bin_width,
                    yerr=unfolded_results['sys_err']/A/t/Om/bin_width,
                    alpha=0.7,
                    elinewidth=3,
                    capsize=4,
                    ls='None', marker='.', ms=10,
                    label='Unfolded (pyunfold)', color="C3")

        #plt.ylim([np.min(data_true/A/t/Om/bin_width)*0.6, 1.6*np.max(data_true/A/t/Om/bin_width)])
        plt.ylim([1.*10**(36), 1.*10**(38)])
        plt.yscale("log")
        ax.set(xlabel=r'log$_{10}$(E/eV)', ylabel=r'J(E)$\times$E$^3$ [eV$^2$ km$^{-2}$ yr$^{-1}$ sr$^{-1}$]')
        plt.legend()  #loc="lower left")
        fig_name = "unfolding_new"
        plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
        plt.close()
