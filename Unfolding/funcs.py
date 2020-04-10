import numpy as np

def efficiencies(bins, E_obs): # E_obs = np.array in eV
    logE = [18.0, 18.5, 19., 19.5, 20., 20.2]
    cor_p  = np.array([4830, 5862, 4845, 4849, 1989])
    cor_he = np.array([4958, 4895, 4941, 4825, 1755])
    cor_o  = np.array([4915, 4881, 4858, 4807, 1655])
    cor_fe = np.array([5071, 4879, 4848, 4735, 1770])
    corsika_evt = 6 * (cor_p + cor_he + cor_o + cor_fe) # 6 times the number of corsika (offline) files
    eff = [0.76957902, 0.91001192, 0.91507704, 0.90907764, 0.60265707]
    rec_evt = []
    a = np.zeros(len(bins))

    for i in range(len(bins)-1):
        for j in range(len(logE)-1):
            if logE[j] <= bins[i] < logE[j+1]:
                mask = (np.log10(E_obs) >= bins[i]) & (np.log10(E_obs) < bins[i+1])
                mask_c = (np.log10(E_obs) >= logE[j]) & (np.log10(E_obs) < logE[j+1])
                #print(logE[j], logE[j+1],len(E_obs[mask_c])/corsika_evt[j],len(E_obs[mask]))
                #rec_evt.append(len(E_obs[mask_c])/corsika_evt[j])
                rec_evt.append(eff[j])
    return np.array(rec_evt)

#def comp_efficiencies(bins, E_obs, composition): # E_obs = np.array in eV


#==== Definition of the energy fit parameters (a,b) ====#
def energy_fitpar(composition):

    if composition == "proton":
        a = 0.217937 # EeV
        b = 1.079747
        return a, b

    if composition == "helium":
        a = 0.204001 # EeV
        b = 1.084485
        return a, b

    if composition == "oxygen":
        a = 0.194436 # EeV
        b = 1.080255
        return a, b

    if composition == "iron":
        a = 0.186687 # EeV
        b = 1.080285
        return a, b


def comp_efficiencies(composition, Etrue): # Etrue = np.array in log scale
    logE = [18.0, 18.5, 19., 19.5, 20., 20.2]
    rec_evt = []
    efficiencies = np.zeros(len(Etrue))

    if composition == "proton":

        corsika_evt = 6 * np.array([4830, 5862, 4845, 4849, 1989]) # 6 times the number of corsika (offline) files
        for i in range(len(logE)-1):
            mask = (Etrue >= logE[i]) & (Etrue < logE[i+1])
            rec_evt.append(len(Etrue[mask]) / corsika_evt[i])

        return np.array(rec_evt)

    if composition == "helium":

        corsika_evt =  6 * np.array([4958, 4895, 4941, 4825, 1755])
        for i in range(len(logE)-1):
            mask = (Etrue >= logE[i]) & (Etrue < logE[i+1])
            rec_evt.append(len(Etrue[mask]) / corsika_evt[i])

        return np.array(rec_evt)

    if composition == "oxygen":
        corsika_evt = 6 * np.array([4915, 4881, 4858, 4807, 1655])
        for i in range(len(logE)-1):
            mask = (Etrue >= logE[i]) & (Etrue < logE[i+1])
            rec_evt.append(len(Etrue[mask]) / corsika_evt[i])

        return np.array(rec_evt)

    if composition == "iron":
        corsika_evt = 6 * np.array([5071, 4879, 4848, 4735, 1770])
        for i in range(len(logE)-1):
            mask = (Etrue >= logE[i]) & (Etrue < logE[i+1])
            rec_evt.append(len(Etrue[mask]) / corsika_evt[i])

        return np.array(rec_evt)

#[0.70465839 0.91407938 0.91623667 0.91149378 0.68183342]
#[0.7611268  0.9072523  0.91334413 0.91236615 0.57654321]
#[0.79579518 0.90896674 0.91337313 0.90805076 0.5897281 ]
#[0.81673569 0.90974927 0.91735424 0.90439986 0.56252354]
