import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from astrotools import container as ctn
from astrotools import statistics as stats
from scipy.optimize import curve_fit
#from scipy.optimize import minimize
from scipy.stats import lognorm
from scipy.interpolate import interp1d

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
    dig = np.digitize(MC_energy, bins)
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
    return np.exp(-0.5*(((x)-mean)/sigma)**2) / sigma / np.sqrt(2*np.pi)

def log_normal(x,s,loc,scale):
    return lognorm.pdf(x, s, loc, scale)
    #return np.exp(-0.5*(x/sigma)**2) / sigma / x / np.sqrt(2*np.pi)

composition = ["proton", "helium", "oxygen", "iron"]
# composition = ["proton"]
# composition = ["helium"]
# composition = ["oxygen"]
# composition = ["iron"]

cos_38 = np.cos(np.deg2rad(38.))**2
Xo = 879. # g/cm^2
bins = np.linspace(18.0, 20.2, 20)

path_scratch = '/net/scratch/Adrianna/data_analysis/'
path_scratch_CIC = '/net/scratch/Adrianna/data_analysis_data/SIMULATIONS/'
path_home = '/home/Adrianna/data_analysis/SIMULATIONS/'

for i, nuclei in enumerate(composition):
    #==== Naming npz arrays ====#
    SD_array  = path_scratch + 'arrays/SD_only/%s_SDonly_merge_p3.npz' % nuclei
    FD_array  = path_scratch + 'arrays/GH/%s_afterCuts_mod_merge_p3.npz' % nuclei
    GH_array  = path_scratch + 'arrays/GH_FOVcuts/%s_FOVcuts_merge_p3.npz' % nuclei  # After FOVcut
    CIC_array = path_scratch_CIC + "arrays/CIC_%s.npz" % nuclei

    #==== Opening data containers ====#
    SD_data  = ctn.DataContainer(SD_array)
    FD_data  = ctn.DataContainer(FD_array)
    GH_data  = ctn.DataContainer(GH_array)
    CIC_data = ctn.DataContainer(CIC_array)

    CIC_par = CIC_data["CIC_par"]
    A, B = CIC_data["A_B"]
    a, b = CIC_data["a_b"]
    sigma_a, sigma_b = CIC_data["Erra_Errb"]

    #==== SD observables ====#
    E_MC = SD_data["MC_energy"]
    zenith_SD  = SD_data["SD_zenith"]
    cos2_SD = np.cos(zenith_SD)**2
    s1000_SD = SD_data["s1000"]
    x_SD = cos2_SD - cos_38
    s38_SD = s1000_SD / CIC_fit(x_SD, *CIC_par)
    E_SD = a*pow(10,18)*(s38_SD)**b
    Xmax_SD = SD_data["MC_xmax"]
    DistXmax_SD = Xo/np.cos(zenith_SD) - Xmax_SD

    #==== HYBRID observables ====#
    E_MC_FD = FD_data["MC_energy"]
    zenith_FD  = FD_data["SD_zenith"]
    cos2_FD = np.cos(zenith_FD)**2
    s1000_FD = FD_data["s1000"]
    x_FD = cos2_FD - cos_38
    s38_FD = s1000_FD / CIC_fit(x_FD, *CIC_par)
    E_FD = a*pow(10,18)*(s38_FD)**b
    Xmax_FD = FD_data["MC_xmax"]
    DistXmax_FD = Xo/np.cos(zenith_FD) - Xmax_FD

    #==== GOLDEN-HYBRID observables ====#
    E_MC_GH = GH_data["MC_energy"]
    zenith_GH  = GH_data["SD_zenith"]
    cos2_GH = np.cos(zenith_GH)**2
    s1000_GH = GH_data["s1000"]
    x_GH = cos2_GH - cos_38
    s38_GH = s1000_GH / CIC_fit(x_GH, *CIC_par)
    E_GH = a*pow(10,18)*(s38_GH)**b
    Xmax_GH = GH_data["MC_xmax"]
    DistXmax_GH = Xo/np.cos(zenith_GH) - Xmax_GH
    GH_energy = GH_data["FD_energy"] # GH_data["true_FD_energy"]
