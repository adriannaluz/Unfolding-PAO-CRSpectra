import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astrotools import container as ctn
from astrotools import statistics as stats
from scipy.optimize import curve_fit
#from scipy.optimize import minimize
from scipy.stats import lognorm
from os.path import join

with_latex = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 20,
    "font.size": 20,
    "legend.fontsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fancybox": False}
mpl.rcParams.update(with_latex)


#==== Definition of the CIC fit ====#
def CIC_fit(x, a, b, c):
    #==== ax^3 + bx^2 + cx + d ====#
    return 1 + a * x + b * x**2 + c * x**3

def log_rel_diff(FD_Energy, MC_energy, bins):
    #==== Create the plot (FD_energy-MC_energy)/MC_energy vs. MC_energy ====#
    x = (FD_Energy-MC_energy)/MC_energy     #modify it with obs(array_name, observable)
    dig = np.digitize(np.log10(MC_energy), bins)
    n = len(bins) - 1
    mx, vx, n_in_bins = np.zeros(n), np.zeros(n), np.zeros(n)

    for i in range(n):
        idx = (dig == i+1)

        if not idx.any():  # check for empty bin
            mx[i] = np.nan
            vx[i] = np.nan
            continue

        mx[i], vx[i] = stats.mean_and_variance(x[idx])
        n_in_bins[i] = np.sum(idx)

    return mx, vx, n_in_bins

def w_bin(FD_Energy, MC_energy, bins):
    #==== Create the plot (FD_energy-MC_energy)/MC_energy vs. MC_energy ====#
    x = FD_Energy     #modify it with obs(array_name, observable)
    dig = np.digitize(np.log10(MC_energy), bins)
    n = len(bins) - 1
    mx, vx, n_in_bins = np.zeros(n), np.zeros(n), np.zeros(n)

    for i in range(n):
        idx = (dig == i+1)

        if not idx.any():  # check for empty bin
            mx[i] = np.nan
            vx[i] = np.nan
            continue

        mx[i], vx[i] = stats.mean_and_variance(x[idx])
        n_in_bins[i] = np.sum(idx)

    return mx, vx, n_in_bins

def fit(E,A,B):
    return B * np.log10(E) + A

def gaussian(x,mean,sigma):
    return np.exp(-0.5*(x/sigma)**2) / sigma / np.sqrt(2*np.pi)

def log_normal(x,s,loc,scale):
    return lognorm.pdf(x, s, loc, scale)
    #return np.exp(-0.5*(x/sigma)**2) / sigma / x / np.sqrt(2*np.pi)

composition = ["proton", "helium", "oxygen", "iron"]

cos_38 = np.cos(np.deg2rad(38.))**2
Xo = 879. # g/cm^2
bins = np.linspace(18.0, 20.2, 20)

path_scratch = '/net/scratch/Adrianna/data_analysis/arrays/'
path_scratch_CIC = '/net/scratch/Adrianna/data_analysis_data/SIMULATIONS/arrays/'
path_home = '/home/Adrianna/data_analysis/SIMULATIONS/arrays/'


SD_filename  = [(composition + '_SDonly_merge_p3.npz' ) for composition in composition]
GH_filename  = [(composition + '_FOVcuts_merge_p3.npz') for composition in composition]
CIC_filename = [('CIC_' + composition + '.npz' ) for composition in composition]

SD_array  = [join(path_scratch + 'SD_only/'   , filename) for filename in SD_filename]
GH_array  = [join(path_scratch + 'GH_FOVcuts/', filename) for filename in GH_filename]
CIC_array = [join(path_scratch_CIC, filename) for filename in CIC_filename]

SD_data = [ctn.DataContainer(SD_array) for SD_array in SD_array]
GH_data = [ctn.DataContainer(GH_array) for GH_array in GH_array]
CIC_data = [ctn.DataContainer(CIC_array) for CIC_array in CIC_array]

E_MC = []
s38_SD = []
E_SD = []
zen_SD =[]

E_MC_GH = []
s38_GH = []
E_GH = []
zen_GH = []

for i in range(0,3,2):
    E_MC     = np.append(E_MC    , np.concatenate((SD_data[i]['MC_energy'], SD_data[i+1]['MC_energy'])))
    E_MC_GH  = np.append(E_MC_GH , np.concatenate((GH_data[i]['MC_energy'], GH_data[i+1]['MC_energy'])))


for i in range(0,4,1):
    cos2_SD = np.cos(SD_data[i]["SD_zenith"])**2
    cos2_GH = np.cos(GH_data[i]["SD_zenith"])**2
    x_SD = cos2_SD - np.cos(np.deg2rad(38.))**2
    x_GH = cos2_GH - np.cos(np.deg2rad(38.))**2

    CIC_par = CIC_data[i]["CIC_par"]
    a, b = CIC_data[i]["a_b"]

    zen_SD = np.append(zen_SD, cos2_SD)
    s38_SD = np.append(s38_SD, SD_data[i]['s1000'] / CIC_fit(x_SD, *CIC_par))

    zen_GH = np.append(zen_GH, cos2_GH)
    s38_GH = np.append(s38_GH, GH_data[i]['s1000'] / CIC_fit(x_GH, *CIC_par))

    E_SD = np.append(E_SD, a * pow(10,18) * (SD_data[i]['s1000'] / CIC_fit(x_SD, *CIC_par))**b)
    E_GH = np.append(E_GH, a * pow(10,18) * (GH_data[i]['s1000'] / CIC_fit(x_GH, *CIC_par))**b)

####===== old plots (FILLED HISTOGRAMS SD - GH) ====####

#==== cos^2 distribution =====#
if True:
    hist_SD = np.histogram(cos2_SD, bins=50, density = True )#,normed=True)

    plt.hist(hist_SD[1][:-1], hist_SD[1], weights=hist_SD[0]/np.max(hist_SD[0]), histtype="stepfilled", color="C0", label="SD", alpha=0.5)

    #plt.hist(cos2_SD, histtype="stepfilled", alpha=0.5, bins=hist_SD[1], weights=np.ones(len(hist_SD[0]))/np.max(hist_SD[0]), color="C0", label="SD")
    hist_GH = np.histogram(cos2_GH, bins=hist_SD[1])
    plt.hist(hist_GH[1][:-1], hist_GH[1], weights=hist_GH[0]/np.max(hist_GH[0]), histtype="stepfilled", color="C0", label="GH")
    #plt.hist(cos2_GH, histtype="stepfilled", bins=hist_SD[1], weights=np.ones(len(hist_GH[0]))/np.max(hist_GH[0]), color="C0", label="GH")

    plt.xlabel(r"cos$^2(\theta)$")
    plt.ylabel("Counts")
    #plt.yscale("log")
    #plt.ylim([pow(10,-4),pow(10,1)])
    plt.xlim([0.2, 1.16])
    plt.legend(loc='upper right',prop={'size':10})
    fig_name = "Plots_IOANA/cos2_MC_all_norm"
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
    plt.close()
    print("Plot has been saved as", fig_name)

#==== (E_SD/E_SD -1) distribution
if False:
    mask    = (np.log10(E_MC)    >= 18.) & (np.log10(E_MC)    < 18.5)
    mask_GH = (np.log10(E_MC_GH) >= 18.) & (np.log10(E_MC_GH) < 18.5)

    # INCLINED EVENTS #
    #mask_c    = (cos2_SD[mask] <= 0.7)
    #mask_GH_c = (cos2_GH[mask_GH] <= 0.7)
    #mask_FD_c = (cos2_FD <= 0.7)

    # VERTICAL EVENTS #
    #mask_c = (cos2_SD[mask] > 0.7)
    #mask_FD_c = (cos2_FD > 0.7)
    #mask_GH_c = (cos2_GH[mask_GH] > 0.7)

    #plt.hist((E_SD-E_MC)/E_MC, histtype="stepfilled", alpha=0.5, bins=100, color="C%s"%i, density=True, label="%s SD" % nuclei)
    #bins = plt.hist((E_SD[mask][mask_c]-E_MC[mask][mask_c])/E_MC[mask][mask_c], histtype="stepfilled", alpha=0.5, bins=50, color="C%s"%i, density=True, label="SD")
    #plt.hist((E_GH[mask_GH][mask_GH_c]-E_MC_GH[mask_GH][mask_GH_c])/E_MC_GH[mask_GH][mask_GH_c], histtype="stepfilled", bins=bins[1], color="C%s"%i, density=True, label="GH")

    bins = plt.hist((E_SD[mask]-E_MC[mask])/E_MC[mask], histtype="stepfilled", alpha=0.5, bins=50, color="C0", density=True, label="SD")
    plt.hist((E_GH[mask_GH]-E_MC_GH[mask_GH])/E_MC_GH[mask_GH], histtype="stepfilled", bins=bins[1], color="C0", density=True, label="GH")

    plt.xlabel(r"(E - E$_{MC}$)/E$_{MC}$")
    plt.ylabel("Density")
    plt.yscale("log")
    #plt.ylim([pow(10,-4),pow(10,1)])
    plt.xlim([-1.5,1.5])
    plt.legend(loc='upper right',prop={'size':10})
    fig_name = "Plots_IOANA/dist_18_185_50_all"
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
    plt.close()
    print("Plot has been saved as", fig_name)
