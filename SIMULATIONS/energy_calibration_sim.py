import numpy as np
import matplotlib.pyplot as plt
import argparse
#from funcs_data_analysis import *
from astrotools import container as ctn
from scipy.optimize import curve_fit
#from scipy.optimize import minimize

parser = argparse.ArgumentParser()
parser.add_argument('-nu', '--nucleus'   , type=str, help="composition in lower letters, ex: proton")
kw = parser.parse_args()

#==== SD keys ====#
#  ['SD_energy_error', 'SD_zenith_error', 'SD_zenith', 'event_highSAT',
# 'stat_ID', 'MC_energy', 'event_lowSAT', 's1000', 'stat_num', 'MC_e_energy',
# 'MC_zenith', 'station_lowSAT', 'SD_eventID', 'station_highSAT',
# 'Shw_size_error', 'SD_energy', 'Shw_size', 'MC_xmax', 'high_Sat', 'Total_ev',
# 'SD_idx', 'low_Sat']

#==== FD keys ====#
# ['Eye_ID', 'Xmax_to_core', 's1000', 'MC_e_energy', 'MC_zenith',
# 'hybrid_FDenergy', 'SD_eventID', 'FD_energy_error', 'FD_cal_energy_error',
# 'hybrid_SDenergy', 'Shw_size', 'FD_cal_energy', 'true_FD_energy',
# 'Shw_size_error', 'SD_energy_error', 'MC_energy', 'FD_Xmax', 'cos_zenith',
# 'cos_zenith_error', 'SD_zenith', 'SD_zenith_error', 'event_hybrid',
# 'fdEvent_ID', 'FD_zenith_error', 'FD_energy', 'DistXmax', 'SD_energy',
# 'MC_xmax', 'Total_ev', 'GH_events']

path_scratch = '/net/scratch/Adrianna/data_analysis/'
path_scratch_CIC = '/net/scratch/Adrianna/data_analysis_data/SIMULATIONS/'
#path_home = '/home/Adrianna/data_analysis/SIMULATIONS/'

#==== Naming npz arrays ====#
SD_array     = path_scratch + 'arrays/SD_only/%s_SDonly_merge_p3.npz' % kw.nucleus
GH_array     = path_scratch + 'arrays/GH/%s_afterCuts_mod_merge_p3.npz' % kw.nucleus
GH_array_FOV = path_scratch + 'arrays/GH_FOVcuts/%s_FOVcuts_merge_p3.npz' % kw.nucleus  # After FOVcut
CIC_array    = path_scratch_CIC + "arrays/CIC_%s.npz" % kw.nucleus

#==== Opening data containers ====#
SD_data = ctn.DataContainer(SD_array)
GH_data = ctn.DataContainer(GH_array)
GH_data_FOV = ctn.DataContainer(GH_array_FOV)
CIC_data = data = ctn.DataContainer(CIC_array)

CIC_par = CIC_data["CIC_par"]
cos_38 = np.cos(np.deg2rad(38.))**2
evt_cut = 6000

#==== Definition of the CIC fit ====#
def CIC_fit(x, a, b, c):
    #==== ax^3 + bx^2 + cx + d ====#
    return 1 + a * x + b * x**2 + c * x**3

#==== SD observables ====#
E_SD = SD_data["MC_energy"]    # SD Reconstructed energy
E_SD_e = SD_data["MC_e_energy"]    # SD Reconstructed energy error
theta_SD = SD_data["SD_zenith"]   # SD Reconstructed zenith
s1000 = SD_data["s1000"]
x = np.cos(theta_SD)**2 - cos_38
s_38 = s1000 / CIC_fit(x, *CIC_par)
id = SD_data['SD_eventID']
event_hybrid = GH_data_FOV['event_hybrid']

#==== HYBRID observables ====#
zenith_FD = GH_data["SD_zenith"]
x_FD = np.cos(zenith_FD)**2 - cos_38
E_FD = GH_data["MC_energy"]
E_FD_e = GH_data["MC_e_energy"]
logE_FD = np.log10(E_FD)
s1000_FD = GH_data["s1000"]
s1000_FD = np.array(s1000_FD)
id_fd = GH_data['SD_eventID']

s38_FD = s1000_FD / CIC_fit(x_FD, *CIC_par)
log_s38_FD = np.log10(s38_FD)

#==== GH observables ====#
zenith_GH = GH_data_FOV["SD_zenith"]
x_GH = np.cos(zenith_GH)**2 - cos_38
E_GH = GH_data_FOV["MC_energy"]
E_GH_e = GH_data_FOV["MC_e_energy"]
logE_GH = np.log10(E_GH)
s1000_GH = GH_data_FOV["s1000"]
s1000_GH = np.array(s1000_GH)
id_gh = GH_data_FOV['SD_eventID']

s38_GH = s1000_GH / CIC_fit(x_GH, *CIC_par)
log_s38_GH = np.log10(s38_GH)

#cond = (logE_GH>18.5)
#E_GH = E_GH[cond]
#logE_GH = logE_GH[cond]
#log_s38_GH = np.log10(s38_GH[cond])

#==== other observables ====#
#E_FD_NO_SD = E_FD[np.where(GH_data['hybrid_FDenergy']==0)[0]] # energy of the events recorded by FD and not by SD
#E_GH_NO_SD = E_GH[np.where(GH_data_FOV['hybrid_FDenergy']==0)[0]] # energy of the events recorded by FD and not by SD
#print(len(E_SD),len(E_GH),len(GH_data["FD_energy"]), len(E_FD_NO_SD) ,len(E_GH_NO_SD))

def fit(E,A,B):
    return B * np.log10(E) + A

mask = logE_GH>18.5#(logE_GH>18.5) & (logE_GH<19.5)
E_GH = E_GH[mask]
logE_GH = logE_GH[mask]
log_s38_GH = log_s38_GH[mask]

par = np.polyfit(logE_GH, log_s38_GH, 1)
A = par[1]
B = par[0]
#print(A,B)
opt_par, cov_m = curve_fit(fit, E_GH, log_s38_GH, p0=[A,B])
A = opt_par[0]
B = opt_par[1]
#print(A,B)
b = 1 / B
a = (1/10**(A))**(b) #/ pow(10,18)

sigma_a = a * np.log(10) / B * np.sqrt(cov_m[0,0] + (A/B)**2 * cov_m[1,1])
sigma_a = sigma_a / pow(10,18)
sigma_b = np.sqrt(cov_m[1,1])/B**2
print(sigma_a , sigma_b)
#print(A,B)
print(a/ pow(10,18),b)
#A = A *10**(18)

save_path = "arrays/CIC_%s" % kw.nucleus
data = ctn.DataContainer(save_path)
print(data["a_b"])
print(data["Erra_Errb"])
#data["A_B"] = (A,B)
#data["a_b"] = (a,b)
#data["Erra_Errb"] = (sigma_a, sigma_b)
#data["ErrA_ErrB"]) = (np.sqrt(cov_m[0,0]), np.sqrt(cov_m[1,1]))

#data.save(save_path)

"""
sigma_a = np.sqrt(cov_m[0,0] + cov_m[1,1]*(A)**2) * a * np.log10(10) / B**2 /(10**(18))
b = 1 / B
sigma_b = cov_m[1,1]**(0.5) / B**2

#savedata = '/net/scratch/Adrianna/data_analysis_data/SIMULATIONS/CIC_%s.npz' % kw.nucleus
#data = ctn.DataContainer(savedata)
#
#data["encal_a_b_195"] = [a,b]
#data["encal_A_B_195"] = [A,B]
#data.save(savedata)

def fit_par():
    A = opt_par[0]
    B = opt_par[1]
    a = (1/10**(A))**(1/B)/(10**(18))

    A = A *10**(18)
    sigma_a = np.sqrt(cov_m[0,0] + cov_m[1,1]*(A)**2) * a * np.log10(10) / B**2 /(10**(18))
    b = 1 / B
    sigma_b = cov_m[1,1]**(0.5) / B**2
    return a, b

#=== plot E_FD vs S_38 ====#
if False:
    #labels = ['fit', r'$\theta<38$', r'$\theta>=38$']
    plt.scatter(logE_GH, log_s38_GH, color='r', s=0.3, label='Data')
    plt.plot(logE_GH, fit(E_GH,*opt_par), 'k-', label='fit')
    #B = 1 / 1.031 # Value given in the paper in revision. Uncertainty +- 0.004
    #A = np.log10(1/(1.86*10**(17))**B) # Value given in the paper in revision. Uncertainty +- 0.03*10**(17)
    #plt.plot(logE_GH, fit(E_GH, A, B), 'b--', label='Auger fit')
    # plt.plot(logE_GH_s, fit(E_GH_s,*opt_par_), 'b-.', label='fit 2')
    # plt.scatter(logE_GH_s, log_s38_GH_s, color='b', s=0.2, label='Data')

    plt.legend(loc='upper left',prop={'size':10})
    plt.xlabel(r"log$_{10}$(E/eV)")
    plt.ylabel(r"log$_{10}$(S$_{38}$/VEM)")

    txt = r"E = (%.3f $\pm %.3f$)x$10^{18}$ eV (S$_{38}$/VEM)$^{(%.3f \pm %.3f)}$" % (round(a,3), round(sigma_a,3), round(b,3), round(sigma_b,3))
    #txt_auger = r"E$_{auger}$ = (%.3f $\pm %.3f$)x$10^{18}$ eV (S$_{38}$/VEM)$^{(%.3f \pm %.3f)}$" % (0.186, 0.003, 1.031, 0.004)
    # txt_s = r"E$_{2}$ = (%.3f x$10^{18} \pm %.3f$) eV (S$_{38}$/VEM)$^{(%.4f \pm %.4f)}$" % (round(a_,3), round(cov_m_[0,0]**0.5,3), b_, cov_m_[1,1]**0.5)
    plt.text(0.98, 0.09, txt, horizontalalignment="right", verticalalignment="bottom", transform=plt.gca().transAxes, fontsize=12)
    # plt.text(0.97, 0.03, txt_s, horizontalalignment="right", verticalalignment="bottom", transform=plt.gca().transAxes, fontsize=12)
    #plt.text(0.97, 0.03, txt_auger, horizontalalignment="right", verticalalignment="bottom", transform=plt.gca().transAxes, fontsize=12)
    fig_name = "s38_logE_GH_scatter_fits_%s" % kw.nucleus
    title = "%s" % kw.nucleus
    plt.title(title)
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
    plt.close()
    print("Plot has been saved as", fig_name)

#E_SD_cal = a * 10**(18)* (s38_GH[cond]) ** b
#logE_SD_cal = np.log10(E_SD_cal)

# plt.scatter(np.linspace(18.,20.2,len(E_SD_corr)),np.log10(E_SD_corr))
# plt.scatter(np.linspace(18.,20.2,len(E_corrected)),np.log10(E_corrected))
# #plt.yscale('log')
# plt.savefig('test')
# plt.close()

#def log_rel_diff(FD_Energy, MC_energy, bins):
#    #==== Create the plot (FD_energy-MC_energy)/MC_energy vs. MC_energy ====#
#    x = (np.log10(FD_Energy)-np.log10(MC_energy))/np.log10(MC_energy)     #modify it with obs(array_name, observable)
#    dig = np.digitize(np.log10(MC_energy), bins)
#    n = len(bins) - 1
#    mx, vx, n_in_bins = np.zeros(n), np.zeros(n), np.zeros(n)
#
#    for i in range(n):
#        idx = (dig == i+1)
#
#        if not idx.any():  # check for empty bin
#            mx[i] = np.nan
#            vx[i] = np.nan
#            continue
#
#        mx[i], vx[i] = stats.mean_and_variance(x[idx])
#        n_in_bins[i] = np.sum(idx)
#
#    return mx, vx, n_in_bins

#==== PLOT relative  difference ===#
bins = np.arange(18.,20.2,0.2)
if False: # look for a way to plot all nucleus at the same time
    zero_line = np.zeros(len(bins))
    plt.scatter(stats.mid(bins), log_rel_diff(E_corrected,E_GH,bins)[0], label='data', color='black')
    plt.plot(bins, zero_line, 'r--', alpha=0.4)
    plt.legend(shadow=True, loc='upper left', handlelength=0.8, fontsize=16)
    plt.xlabel(r'log(E$_{FD}$/eV)', {'fontsize': 16})
    plt.ylabel(r' [log(E$_{SD}$)-log(E$_{FD}$)] / log(E$_{FD}$)', {'fontsize': 16})
    fig_name = 'logRelDiff_Efd_Esd_data'
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')
    plt.close()
    print("Plot has been saved as", fig_name)

if False:
    plt.scatter(np.log10(E_GH), np.log10(E_corrected), color='black', s=0.3, label='Data')
    plt.plot(np.log10(E_GH), np.log10(E_GH), 'r-')
    plt.xlabel(r"log$_{10}$(E$_{FD}$/eV)")
    plt.ylabel(r"log$_{10}$(E$_{SD}$/eV)")
    plt.legend(loc='upper left',prop={'size':10})
    fig_name = "E_SD_E_FD_cal_data"
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
    plt.close()
    print("Plot has been saved as", fig_name)

if False:
    zero_line = np.zeros(len(bins))
    plt.scatter(logE_GH, (logE_SD_cal-logE_GH)/logE_GH, color='black', s=0.3, label='Data')
    plt.plot(bins, zero_line, 'r--')
    #plt.scatter(logE_GH, logE_SD_cal, color='black', s=0.3, label='Data')
    #plt.plot((E_SD_cal-E_GH)/E_GH, logE_GH, 'r-')
    plt.xlabel(r"log$_{10}$(E$_{FD}$/eV)")
    plt.ylabel(r' [log(E$_{SD}$)-log(E$_{SD}$)] / log(E$_{FD}$)', {'fontsize': 16})
    #plt.xscale('log')
    plt.legend(loc='upper left',prop={'size':10})
    fig_name = "logdiffE_E_FD_cal_data"
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
    plt.close()
    print("Plot has been saved as", fig_name)

#==== SLM ====#
"""
"""
E_FD_e_NO_SD = E_FD_e[np.where(GH_data['hybrid_FDenergy']==0)[0]]
E_GH_e_NO_SD = E_GH_e[np.where(GH_data_FOV['hybrid_FDenergy']==0)[0]]

E_GH = E_GH[E_GH>(3*10**18)] # Golden hybrid energies above 3*10**(18)


total_sum = []
def L_SLM(par):
    print('test combination (A, B)=', par)
    sum_k = []
    print(sum_k)
    print(total_sum)
    for k in range(len(E_GH)):
        sigma_S_k = ((0.2*((E_GH_NO_SD/par[0]/(10**(18)))**(1/par[1])))**2 + (E_GH_e_NO_SD)**2)**(0.5)

        sigma_E_k = E_GH_e_NO_SD

        c = 1 / sigma_E_k / sigma_S_k

        exp_1 = np.exp(-0.5 * (E_GH[k] - E_GH_NO_SD)**2 / sigma_E_k**2)

        exp_2 = np.exp(-0.5 * (s38_GH[k] - (E_GH_NO_SD/par[0]/10**(18))**(1/par[1]))**2 / sigma_S_k**2)

        sum_k.append(-np.log(np.sum(c * exp_1 * exp_2)))

    total_sum.append(np.sum(sum_k))

    return total_sum[0]

a_ = np.linspace(0.9*(a), 1.1*(a), 15)
b_ = np.linspace(0.9*b, 1.09*b, 15)
print(a_,b_)

likelihood = np.zeros((len(a_), len(b_)))
for i, ai in enumerate(a_):
    for j, bj in enumerate(b_):
        likelihood[i, j] = L_SLM([ai, bj])

#print(likelihood)
np.save('/net/scratch/Adrianna/data_analysis_data/SIMULATIONS/likelihood_scan_%s.npy' % kw.nucleus, likelihood)
"""
"""
def l_SLM(par):
    sigma_S_k = ((0.2*(par[0]*(E_GH_NO_SD[np.newaxis])**par[1]))**2 + (E_GH_e_NO_SD[np.newaxis])**2)**(0.5)
    sigma_E_k = E_GH_e_NO_SD[np.newaxis]
    c = 1 / sigma_E_k / sigma_S_k

    exp_1 = np.exp(-0.5 * (E_GH[:, np.newaxis] - E_GH_NO_SD[np.newaxis])**2 / (sigma_E_k)**2)
    exp_2 = np.exp(-0.5 * (s38_GH[:, np.newaxis] - a*(E_GH_NO_SD[np.newaxis])**b)**2 / sigma_S_k**2)
    total_exp = c * exp_1 * exp_2
    print(np.shape(total_exp))
    return np.sum(np.log(np.sum(total_exp,axis=-1)))

#print(minimize(l_SLM, [a,b], method = 'Nelder-Mead'))

a_ = np.arange(a-0.1,a+0.1,0.01)
a_ = a_[None,None,:, None]
b_ = np.arange(b-0.1,b+0.1,0.01)
b_ = b_[None,None,None,:]

E_GH = E_GH[:,None,None,None]
E_GH_NO_SD = E_GH_NO_SD[None,:,None,None]

sigma_S_k = ((0.2*(a*(E_GH_NO_SD)**b))**2 + (E_GH_e_NO_SD)**2)**(0.5)
sigma_E_k = E_GH_e_NO_SD
c = 1 / sigma_E_k / sigma_S_k
a = np.exp(-0.5 * (E_GH- E_GH_NO_SD)**2 / (sigma_E_k)**2)
b = np.exp(-0.5 * (s1000_GH - a*(E_GH_NO_SD)**b)**2 / sigma_S_k**2)
print(np.sum(a*b, axis=0))

def l_SLM(a,b):
    sum_k = 0
    for k in range(len(E_GH)):
        sum_i = 0
        for i in range(len(E_GH_NO_SD)): # Notation used as in the paper arXiv:1503.09027
            sigma_S_k = ((0.2*(a*(E_GH_NO_SD[i])**b))**2 + (E_GH_e_NO_SD[i])**2)**(0.5)
            sigma_E_k = E_GH_e_NO_SD[i]
            c = 1 / sigma_E_k / sigma_S_k
            a = np.exp(-0.5 * (E_GH[k] - E_GH_NO_SD[i])**2 / (sigma_E_k)**2)
            b = np.exp(-0.5 * (s1000_GH[k] - a*(E_GH_NO_SD[i])**b)**2 / sigma_S_k**2)
            sum_i.append(np.sum(a*b, axis=0))
            print(sum_i)
        sum_k.append(np.log(np.sum(sum_i, axis=0)))
    SUM_k = np.sum(sum_k,axis-=)
    print(SUM_k)

print(l_SLM(a,b))
"""
"""
#==== Drawing the vectors ====#
if False:
    B_v = np.sin(B_azimuth) * np.sin(B_zenith), np.cos(B_azimuth) * np.sin(B_zenith), np.cos(B_zenith)  # unit vector in the direction of the magnetic field
    #B_v = B_mag * np.array(B_v)
    B_v_inv = np.cos(B_azimuth) * np.sin(B_zenith), np.sin(B_azimuth) * np.sin(B_zenith), np.cos(B_zenith)  # unit vector in the direction of the magnetic field
    #B_v_inv = B_mag * np.array(B_v)
    theta = [0,30,60]

    plt.scatter(B_v[0],B_v[1])
    plt.scatter(B_v_inv[0],B_v_inv[1])
    plt.savefig('coord_sys')
    plt.close()

par = np.polyfit(np.log10(s38_GH), logE_GH, 1)
A = par[1]
B = par[0]
def fit(x,A,B):
    return B * np.log10(x) + A

opt_par, cov_m = curve_fit(fit, s38_GH, logE_GH, p0=[A,B])

cond = (logE_GH>18.5) & (logE_GH<20.)
logE_GH_s = logE_GH[cond]
s38_GH_s = s38_GH[cond]

par_s = np.polyfit(np.log10(s38_GH_s), logE_GH_s, 1)
A_s = par_s[0]
B_s = par_s[1]
opt_par_s, cov_m_s = curve_fit(fit, s38_GH_s, logE_GH_s, p0=[A,B])

#=== plot E_FD vs S_38 ====#
if False:
    #labels = ['fit', r'$\theta<38$', r'$\theta>=38$']
    plt.plot(np.log10(s38_GH), fit(s38_GH,*opt_par), 'k-', label='fit 1')
    plt.plot(np.log10(s38_GH), fit(s38_GH,*opt_par_s), 'b:', label='fit 2')
    plt.scatter(np.log10(s38_GH), logE_GH, color='r', s=0.3, label='Simulated data')

    plt.legend(loc='upper left',prop={'size':10})
    plt.ylabel(r"log$_{10}$(E/eV)")
    plt.xlabel(r"log$_{10}$(S$_{38}$/VEM)")
    #plt.yscale("log")
    #plt.xscale("log")
    #title = "%s" % kw.nucleus
    #plt.title(title)
    txt = r"E$_{1}$ = (%.3f x$10^{18} \pm %.3f$) eV (S$_{38}$/VEM)$^{(%.4f \pm %.4f)}$" % (10**(round(opt_par[0],3)-18), round(cov_m[0,0]**0.5,3), opt_par[1], cov_m[1,1]**0.5)
    txt_s = r"E$_{2}$ = (%.3f x$10^{18} \pm %.3f$) eV (S$_{38}$/VEM)$^{(%.3f \pm %.3f)}$" % (10**(round(opt_par_s[0],3)-18), round(cov_m_s[0,0]**0.5,3), round(opt_par_s[1],3), round(cov_m_s[1,1]**0.5,3))
    plt.text(0.97, 0.03, txt_s, horizontalalignment="right", verticalalignment="bottom", transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.97, 0.09, txt, horizontalalignment="right", verticalalignment="bottom", transform=plt.gca().transAxes, fontsize=12)
    fig_name = "test_s38_logE_GH_scatter_fits_data"
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
    plt.close()
    print("Plot has been saved as", fig_name)

#==== PLOT relative  difference E_SD ===#
bins = np.arange(18.5,20.2,0.2)
if False: # look for a way to plot all nucleus at the same time
    zero_line = np.zeros(len(bins))
    plt.scatter(stats.mid(bins), log_rel_diff(E_corrected,E_SD_cal,bins)[0], label='data', color='black')
    plt.plot(bins, zero_line, 'r--', alpha=0.4)
    plt.legend(shadow=True, loc='upper left', handlelength=0.8, fontsize=16)
    plt.xlabel(r'log(E$_{SD}$/eV)', {'fontsize': 16})
    plt.ylabel(r' [log(E$_{SD_c}$)-log(E$_{SD}$)] / log(E$_{SD}$)', {'fontsize': 16})
    fig_name = 'logRelDiff_Esd_Esd_data'
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')
    plt.close()
    print("Plot has been saved as", fig_name)
"""
