import numpy as np

def efficiencies(composition, Etrue): # Etrue = np.array in log scale
    logE = [18.0, 18.5, 19., 19.5, 20., 20.2]
    rec_evt = []
    efficiencies = np.zeros(len(Etrue))
    if composition == "proton":
        corsika_evt = 6 * np.array([4830, 5862, 4845, 4849, 1989]) # 6 times the number of corsika (offline) files
        for i in range(len(logE)-1):
            mask = (Etrue >= logE[i]) & (Etrue < logE[i+1])
            efficiencies[mask] = len(Etrue[mask]) / corsika_evt[i]
        return efficiencies

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
