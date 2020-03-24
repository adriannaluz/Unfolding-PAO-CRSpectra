import numpy as np
from astrotools import container as ctn

composition = ["proton", "helium", "oxygen", "iron"]

#==== Definition of the CIC fit ====#
def CIC_fit(x, a, b, c):
    #==== ax^3 + bx^2 + cx + d ====#
    return 1 + a * x + b * x**2 + c * x**3

cos_38 = np.cos(np.deg2rad(38.))**2
Xo = 879. # g/cm^2
bins = np.linspace(18.0, 20.2, 20)

path_scratch = '/net/scratch/Adrianna/data_analysis/'
path_scratch_CIC = '/net/scratch/Adrianna/data_analysis_data/SIMULATIONS/'

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

    ID_SD, idx_sd = np.unique(SD_data['SD_eventID'], return_index=True)
    ID_GH, idx_gh = np.unique(GH_data['SD_eventID'], return_index=True)
    mask = np.isin(SD_data['SD_eventID'][idx_sd], GH_data['SD_eventID'][idx_gh])

    n = len(E_SD)
    data = ctn.DataContainer(n)

    data["ESD_calib"] = E_SD
    data["EFD_reco"] = FD_data["FD_energy"]
    data["EFD_biascorr"] = FD_data["true_FD_energy"]
    data["EGH_reco"] = GH_data["FD_energy"]
    data["EGH_biascorr"] = GH_data["true_FD_energy"]
    data["SDGH_ID"] = GH_data['SD_eventID'][idx_gh]

    #print(data.keys())
    #save_path = path_scratch_CIC + "arrays/Ecalibrated_%s" % nuclei
    #data.save(save_path)
    #print("File saved as: ",save_path)
