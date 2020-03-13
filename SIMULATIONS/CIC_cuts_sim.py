import numpy as np
import matplotlib.pyplot as plt
import argparse
from astrotools import container as ctn
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser()
parser.add_argument('-nu', '--nucleus'   , type=str, help="composition in lower letters, ex: proton")
kw = parser.parse_args()

#['Eye_ID', 'Xmax_to_core', 's1000', 'MC_e_energy', 'MC_zenith',
#'hybrid_FDenergy', 'SD_eventID', 'event_hybrid', 'FD_energy_error',
#'FD_cal_energy_error', 'hybrid_SDenergy', 'Total_ev', 'eye_array', 'eye_num',
#'FDEye_idx', 'Shw_size', 'SD_idx', 'true_FD_energy', 'Shw_size_error',
#'SD_energy_error', 'event_highSAT', 'stat_ID', 'MC_energy', 'hybrid_energy',
#'GH_events', 'FD_Xmax', 'event_lowSAT', 'cos_zenith', 'SD_zenith',
#'FD_cal_energy', 'SD_zenith_error', 'stat_num', 'fdEvent_ID', 'station_lowSAT',
#'FD_zenith_error', 'station_highSAT', 'FD_energy', 'DistXmax', 'SD_energy',
#'hybrid_true_energy']

#['Eye_ID', 'Xmax_to_core', 's1000', 'MC_e_energy', 'MC_zenith',
#'hybrid_FDenergy', 'SD_eventID', 'FD_energy_error', 'FD_cal_energy_error',
#'hybrid_SDenergy', 'Shw_size', 'FD_cal_energy', 'true_FD_energy',
#'Shw_size_error', 'SD_energy_error', 'MC_energy', 'FD_Xmax', 'cos_zenith',
#'cos_zenith_error', 'SD_zenith', 'SD_zenith_error', 'event_hybrid',
#'fdEvent_ID', 'FD_zenith_error', 'FD_energy', 'DistXmax', 'SD_energy',
#'MC_xmax', 'Total_ev', 'GH_events']

path_scratch = '/net/scratch/Adrianna/data_analysis/'
#path_home = '/home/Adrianna/data_analysis/SIMULATIONS/'

#==== Naming npz arrays ====#
SD_array = path_scratch + 'arrays/SD_only/%s_SDonly_merge_p3.npz' % kw.nucleus

#==== Opening data containers ====#
SD_data = ctn.DataContainer(SD_array)

#==== Variable Assigment ====#
SD_energy = SD_data["MC_energy"]    # MC energy for the SD dataset
s1000 = SD_data['s1000']      # S1000
zenith = SD_data["SD_zenith"]   # SD Reconstructed zenith
cos2_zenith = np.cos(zenith)**2

theta = []
binning = []

#==== Selection of the equal zenith binning size for the CIC method ====#
theta = np.arcsin((np.arange(11)/10.)**(0.5) * np.sin(np.deg2rad(60)))
binning = np.rad2deg(theta)

#===== Label of the angles in the zenith binning ====#
label = ["0 - 15.89", "15.89 - 22.79", "22.79 - 28.32", "28.32 - 33.21", "33.21 - 37.76",
         "37.76 - 42.13", "42.13 - 46.43", "46.43 - 50.77", "50.77 - 55.24",  "55.24 - 60"]

evt_bin = []
zenith_inbin = []
mean_cos2 = []
S_38_cut = []
s_1000 = []
s_2000 = []
idx_resamp = []
var = []

cos_2  = np.cos(np.deg2rad(binning))**2
cos_38 = np.cos(np.deg2rad(38.))**2

evt_cut = 6000  # Scut approx 40 VEM for evt_cut = 6000
Nshuffle = 500   # times of the shuffleling in the boostrap method for the variance

#==== Separation of the number of events in each zenith bin ====#
for i in range(len(theta)-1):

    cond = (zenith >= theta[i]) & (zenith <theta[i+1])  # Condition that selects the events in each zenith bin
    evt_bin.append(len(np.rad2deg(zenith[cond])))   # Number of events in each zenith bin
    zenith_inbin.append(np.rad2deg(zenith[cond]))
    mean_cos2.append(np.mean(np.cos(zenith[cond])**2))
    S_38_cut.append(np.sort(s1000[cond])[::-1][evt_cut])
    #s_1000.append(np.sort(s1000[cond])[::-1][:evt_cut + 1])#np.sort(s1000[cond])[::-1][:1000])
    s_2000.append(np.sort(s1000[cond])[::-1][:9000])#np.sort(s1000[cond])[::-1][:1000])
    idx_resamp.append(np.random.randint(0, high=len(s1000[cond]), size=(Nshuffle, len(s1000[cond]))))

    y_i = np.sort(s1000[cond][idx_resamp[i]],axis=1)[:,::-1][:,evt_cut]
    y = S_38_cut[i]
    var.append(np.sum((y_i-y)**2)/(Nshuffle - 1))
    #print(var[i], np.std(y_i))

var = np.array(var)
std_boot = np.sqrt(var)
s_1000 = np.array(s_2000)
mean_cos2 = np.array(mean_cos2)

#==== Definition of the CIC fit ====#
def CIC_fit(x, a, b, c):
    #==== ax^3 + bx^2 + cx + d ====#
    return 1 + a * x + b * x**2 + c * x**3

#==== Definition of the CIC fit parameters ====#
c, b, a, _ = np.polyfit(cos_2[:-1] - cos_38, s_1000[:,evt_cut]/s_1000[5,evt_cut], 3)

opt_param = (a,b,c)
a = np.linspace(opt_param[0]-1,opt_param[0]+1,50)
a = a[None,:, None, None]
b = np.linspace(opt_param[1]-1,opt_param[1]+1,50)
b = b[None,None, :, None]
c = np.linspace(opt_param[2]-1,opt_param[2]+1,50)
c = c[None,None, None, :]

s38_cut = S_38_cut[5] # s_1000[:,-1]

def Chi_2(s1000, a, b, c, evt_cut):
    x = mean_cos2 - cos_38
    x = x[:,None,None,None]
    var_ = var[:,None,None,None]
    s38_cut = s1000[5,evt_cut]
    s1000_cut = s1000[:,evt_cut]
    s1000 = s1000_cut[:,None,None,None]

    sum = np.sum((s1000 - s38_cut * CIC_fit(x, a, b, c))**2 / var_, axis=0)
    idx_a = np.unravel_index(np.argmin(sum,axis=None),sum.shape)[0]
    idx_b = np.unravel_index(np.argmin(sum,axis=None),sum.shape)[1]
    idx_c = np.unravel_index(np.argmin(sum,axis=None),sum.shape)[2]
    return a[:,idx_a,:,:][0,0,0], b[:,:,idx_b,:][0,0,0], c[:,:,:,idx_c][0,0,0]
    #return np.min(np.argmin(np.sum((s1000 - s38_cut * CIC)**2 / var, axis=0),axis=2))
    #return np.sum(s_1000[:,-1] - s38_cut * CIC)**2 / var**2
#print(Chi_2(s_1000, a, b, c, evt_cut))

#if __name__ == "__main__":

#n = len(s_1000)
#data = ctn.DataContainer(n)
#save_path = "arrays/CIC_%s" % kw.nucleus
#data = ctn.DataContainer(save_path)
#print(data['CIC_par'])
#data["s1000"] = s_1000
#data["s1000_error"] = std_boot
#data["s1000_variance"] = var
#data["evt_cut"] = evt_cut
#data["CIC_par"] = Chi_2(s_1000, a, b, c, evt_cut)
#data["zenith_bin"] = binning
#
#print(data.keys())
#data.save(save_path)


#==== Histogram number of events in equal binning (zenith) ====#
if False:
    w = evt_bin
    for i, lab in enumerate(label):
        r = np.cos(np.deg2rad(zenith_inbin[i]))**2
        plt.hist(r, range=(r.min(), r.max()), bins=1, histtype="step", color='C%s'%i)
        #plt.scatter(mean_cos2[i], w[i], color='C%s'%i, marker='o')
        plt.errorbar(mean_cos2[i], w[i], yerr=np.sqrt(w[i]), markersize=0.5, marker='o', color='C%s'%i) # yerr is the poissonian error of every bin

    N_avg = int(round(np.average(w)))
    label_hline = r"$\bar{N}$ = %s" % str(N_avg)

    plt.axhline(y=N_avg, color='r', linestyle='-.', alpha=0.5, label= label_hline)
    plt.ylim([0.,13000.])
    plt.legend(loc='lower right',prop={'size':14})
    plt.ylabel("Number of events", {'fontsize': 16})
    plt.xlabel(r'cos$^2$($\theta$)', {'fontsize': 16})

    title = '%s' % kw.nucleus
    plt.title(title)
    fig_name = 'events_equal_cos2bins_%s.png' % kw.nucleus
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')
    plt.close()
    print('Plot saved as ', fig_name)

#==== s1000 Attenuation ====#
if False:
    fig_name = 'attenuation_S1000_evtcut_%s' %kw.nucleus

    plt.scatter(mean_cos2, s_1000[:,1000], label=r'N$_{cut}$=1000')
    plt.scatter(mean_cos2, s_1000[:,4000], label=r'N$_{cut}$=4000' )
    plt.scatter(mean_cos2, s_1000[:,evt_cut], label=r'N$_{cut}$=%s'%evt_cut)

    plt.xlabel(r'cos$^2(\theta$)')
    plt.ylabel('S(1000) attenuation [VEM]')
    plt.legend(loc='upper left',prop={'size':10})
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')
    plt.close()

#==== s1000 attenuation + fit ====#
if False:
    from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
    fig_name = 'attenuationfit_S1000_%s' %kw.nucleus

    #plt.scatter(mean_cos2, s_1000[:,1000], label=r'N$_{cut}$=1000')
    #plt.plot(mean_cos2, s_1000[5,1000] * CIC_fit(mean_cos2-cos_38,*Chi_2(s_1000, a, b, c, 1000)), label='fit(N$_{cut}$=1000)')
    #plt.scatter(mean_cos2, s_1000[:,4000], label=r'N$_{cut}$=4000' )
    #plt.plot(mean_cos2, s_1000[5,4000] * CIC_fit(mean_cos2-cos_38,*Chi_2(s_1000, a, b, c, 4000)), label='fit(N$_{cut}$=4000)')
    #plt.scatter(mean_cos2, s_1000[:,evt_cut], label=r'N$_{cut}$=%s'%evt_cut)
    plt.plot(mean_cos2, s_1000[5,evt_cut] * CIC_fit(mean_cos2-cos_38,*Chi_2(s_1000, a, b, c, evt_cut)), color = 'green', label='CIC fit')
    plt.errorbar(mean_cos2, s_1000[:,evt_cut], yerr=std_boot, fmt='go', label=r'N$_{cut}$=%s'%evt_cut)

    plt.plot(mean_cos2, data["s1000"][5,evt_cut] * CIC_fit(mean_cos2-cos_38,*data["CIC_par"]), color = 'red', label='CIC fit')
    plt.errorbar(mean_cos2, data["s1000"][:,evt_cut], yerr=data["s1000_error"], fmt='ro')
    plt.xlabel(r'cos$^2(\theta$)')
    plt.ylabel('S(1000) attenuation [VEM]')
    plt.legend(loc='upper left',prop={'size':10})
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')
    plt.close()

#==== Attenuation factor curve (s_1000/S_38_cut) ====#
if False:
    fig_name = 'attenuation_factor_data'
    plt.scatter(1/mean_cos, s_1000[:,350]/s_1000[5,350], label=r'N$_{cut}$=350')
    plt.scatter(1/mean_cos, s_1000[:,evt_cut]/s_1000[5,evt_cut], label=r'N$_{cut}$=%s' %evt_cut)
    plt.scatter(1/mean_cos, s_2000[:,1000]/s_2000[5,1000], label='N$_{cut}$=1000')
    plt.scatter(1/mean_cos, s_3000[:,4000]/s_3000[5,4000], label='N$_{cut}$=4000')
    #plt.scatter(1/mean_cos, s_3000[:,8000]/s_3000[5,8000], label='N$_{cut}$=8000')
    #plt.xlim([0.,60.])
    label_hline = r'sec(38$^{\circ}$)'
    plt.axvline(1/cos_38, color='k', linestyle='--', alpha=0.5, label= label_hline)
    plt.xlabel(r'sec($\theta$)')
    plt.ylabel(r'S(1000)/S$_{38}$')
    plt.legend(loc='upper right',prop={'size':10})
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')
    plt.close()


#==== Uncertainties (bootstrap method), (errors in s1000) ====#
if False:
    plt.scatter(np.cos(theta[:-1])**2,  std_boot/s_1000[:,evt_cut]*100)
    plt.scatter(np.cos(theta[:-1])**2, -1*std_boot/s_1000[:,evt_cut]*100)
    #plt.ylim([-0.04,0.04])

    plt.ylabel(r'$\sigma(S_{1000})$/S$_{1000}$ [\%]')
    plt.xlabel(r'cos$^2(\theta$)')
    title = "%s" % kw.nucleus
    plt.title(title)
    fig_name = 'S1000_uncertainties_%s' % kw.nucleus
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')
    plt.close()

#==== Number of events (s1000>thr) vs s1000 ====#
if False:
    #theta = np.deg2rad([0., 26., 38., 49., 60.]) #np.arange(0.,70.,10)
    s1000_thresholds = np.logspace(0.01, 3., 1500, endpoint=True) #np.linspace(1., 10000.,1000) #[5., 10., 13., 20., 32., 50., 79., 100., 200., 300.]
    #label = [r'0 $\leq \theta <$ 26', r'26 $\leq \theta <$ 38', r'38 $\leq \theta <$ 49', r'49 $\leq \theta <$ 60']
    #labels = ['0 - 10', '10 - 20', '20 - 30', '20 - 30', '40 - 50', '50 - 60']
    label = [r" 0.0$^{\circ}$ $\leq \theta <$ 15.9$^{\circ}$", r"15.9$^{\circ}$ $\leq \theta <$ 22.8$^{\circ}$",
             r"22.8$^{\circ}$ $\leq \theta <$ 28.3$^{\circ}$", r"28.3$^{\circ}$ $\leq \theta <$ 33.2$^{\circ}$",
             r"33.2$^{\circ}$ $\leq \theta <$ 37.8$^{\circ}$", r"37.8$^{\circ}$ $\leq \theta <$ 42.1$^{\circ}$",
             r"42.1$^{\circ}$ $\leq \theta <$ 46.4$^{\circ}$", r"46.4$^{\circ}$ $\leq \theta <$ 50.8$^{\circ}$",
             r"50.8$^{\circ}$ $\leq \theta <$ 55.2$^{\circ}$", r"55.2$^{\circ}$ $\leq \theta <$ 60.0$^{\circ}$"]
    for k in range(len(theta)):
        if k < len(theta)-1:
            #cond = (zenith >= np.deg2rad(zen[k])) & (zenith < np.deg2rad(zen[k+1]))
            cond = (zenith >= theta[k]) & (zenith <theta[k+1])

            plot = []
            for thr in s1000_thresholds:
                plot.append(len(np.where(s1000[cond]>thr)[0])-1e-1)
            #plt.vlines(s1000_thresholds[np.argmin(np.abs(np.array(plot)-evt_cut))], 1, evt_cut, color="C%i" % k)
            plt.step(s1000_thresholds, plot, color="C%i" % k, label=label[k])
            #print(s1000_thresholds[np.argmin(np.abs(np.array(plot)-evt_cut))])
        #if k == 5:
        #    txt = r'S$_{38}^{cut}$=%0.2f' % s1000_thresholds[np.argmin(np.abs(np.array(plot)-evt_cut))]
    #N_cut = r"N$_{cut}$=%i" % evt_cut
    #plt.text(0.22, 0.92, N_cut, horizontalalignment="right", verticalalignment="bottom", transform=plt.gca().transAxes, fontsize=12)
    #plt.text(0.24, 0.85, txt, horizontalalignment="right", verticalalignment="bottom", transform=plt.gca().transAxes, fontsize=12)
    title = '%s' % kw.nucleus
    plt.title(title)
    #plt.axhline(evt_cut, color="black", linestyle="dashed")
    plt.ylim([1.,20000.])
    plt.xlim([10.,1000.])
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r"Number of events (S$_{1000}>$S$^{\textrm{thr}}_{1000}$)", {'fontsize': 16})
    plt.xlabel(r'S$^{\textrm{thr}}_{1000}$ [VEM]', {'fontsize': 16})

    plt.legend(label,loc='lower left',prop={'size':12})
    fig_name = 'Nevt_S1000_%s' % kw.nucleus
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
    plt.close()

#==== Number of events (s38>thr) vs s38 ====#
if False:
    s1000_thresholds = np.logspace(0.01, 3., 1500, endpoint=True)
    x = mean_cos2 - cos_38
    a, b, c = Chi_2(s_1000, a, b, c, evt_cut)

    label = [r" 0.0$^{\circ}$ $\leq \theta <$ 15.9$^{\circ}$", r"15.9$^{\circ}$ $\leq \theta <$ 22.8$^{\circ}$",
             r"22.8$^{\circ}$ $\leq \theta <$ 28.3$^{\circ}$", r"28.3$^{\circ}$ $\leq \theta <$ 33.2$^{\circ}$",
             r"33.2$^{\circ}$ $\leq \theta <$ 37.8$^{\circ}$", r"37.8$^{\circ}$ $\leq \theta <$ 42.1$^{\circ}$",
             r"42.1$^{\circ}$ $\leq \theta <$ 46.4$^{\circ}$", r"46.4$^{\circ}$ $\leq \theta <$ 50.8$^{\circ}$",
             r"50.8$^{\circ}$ $\leq \theta <$ 55.2$^{\circ}$", r"55.2$^{\circ}$ $\leq \theta <$ 60.0$^{\circ}$"]

    for k in range(len(theta)):
        if k < len(theta)-1:
            cond = (zenith >= theta[k]) & (zenith <theta[k+1])
            plot = []
            for thr in s1000_thresholds:
                s_38 = s1000 / CIC_fit(x, a, b, c)[k]
                plot.append(len(np.where(s_38[cond]>thr)[0])-1e-1)
            plt.step(s1000_thresholds, plot, color="C%i" % k, label=label[k]+r'$^\circ$')
        #if k==5:
        #    plt.vlines(s1000_thresholds[np.argmin(np.abs(np.array(plot)-evt_cut))], 1, evt_cut, color="C%i" % k)

    #N_cut = r"N$_{cut}$=%i" % evt_cut
    #plt.text(0.22, 0.92, N_cut, horizontalalignment="right", verticalalignment="bottom", transform=plt.gca().transAxes, fontsize=12)
    title = '%s' % kw.nucleus
    plt.title(title)
    #plt.axhline(evt_cut, color="black", linestyle="dashed")
    plt.ylim([1.,20000.])
    plt.xlim([10.,1000.])
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r"Number of events (S$_{38}>$S$^{\textrm{thr}}_{38}$)", {'fontsize': 16})
    plt.xlabel(r'S$^{\textrm{thr}}_{38}$ [VEM]', {'fontsize': 16})

    plt.legend(label,loc='lower left',prop={'size':12})
    fig_name = 'Nevt_S38_%s' % kw.nucleus
    plt.savefig(fig_name,figsize=(10,7), dpi = 300, bbox_inches = 'tight')#, color=colors)
    plt.close()
