import numpy as np

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

"""
def efficiencies(composition, Etrue): # Etrue = np.array in log scale
    logE = [18.0, 18.5, 19., 19.5, 20., 20.2]
    rec_evt = []
    efficiencies = np.zeros(len(Etrue))
    if composition == "proton":
        corsika_evt = 6 * np.array([4830, 5862, 4845, 4849, 1989]) # 6 times the number of corsika (offline) files
        for i in range(len(logE)-1):
            mask = (Etrue >= logE[i]) & (Etrue < logE[i+1])
            rec_evt.append(len(Etrue[mask]) / corsika_evt[i])
        #    efficiencies[mask] = len(Etrue[mask]) / corsika_evt[i]
        #return efficiencies
        return np.array(rec_evt)
    if composition == "helium":
        corsika_evt =  6 * np.array([4958, 4895, 4941, 4825, 1755])
        for i in range(len(logE)-1):
            mask = (Etrue >= logE[i]) & (Etrue < logE[i+1])
            efficiencies[mask] = len(Etrue[mask]) / corsika_evt[i]
        return efficiencies

    if composition == "oxygen":
        corsika_evt = 6 * np.array([4915, 4881, 4858, 4807, 1655])
        for i in range(len(logE)-1):
            mask = (Etrue >= logE[i]) & (Etrue < logE[i+1])
            efficiencies[mask] = len(Etrue[mask]) / corsika_evt[i]
        return efficiencies

    if composition == "iron":
        corsika_evt = 6 * np.array([5071, 4879, 4848, 4735, 1770])
        for i in range(len(logE)-1):
            mask = (Etrue >= logE[i]) & (Etrue < logE[i+1])
            efficiencies[mask] = len(Etrue[mask]) / corsika_evt[i]
        return efficiencies
"""
