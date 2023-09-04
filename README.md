Unfolding the Cosmic Ray Spectra measured by the Pierre Auger Observatory

Calibration of the simulations consist in several steps. Below a step by step list of the them.


1) Get the data. I used the CORSIKA files already processed in Offline. That correspond to ADST.root files in the Lyon clsuter (/pauger/Simulations/libraries/rcola/Offline_v2r9p5/ADST/EPOS_LHC/). Files for proton, helium, oxygen and iron with log(E/eV) from 18.0 to 20.2 and zenith angles between 0 and 60 degrees are included.


2) Apply the corresponding cuts in order to obtain the three data set that are going to be used, "SD only", "FD" for FD Hybrid events and "GH" for Golden hybrid events with the additional Fidutial FOV cut. The last data set is contined in the two first ones. The second one is also included in the first one.


3) By using the SD only data set, the CIC method is applied ("CIC_cut_sim.py" file). This generate the attenuation curve for the different mass composition.


4) With the attenutation curve obtained in the previous step and by using the GH data set we can obtain the energy estimatior "S38". Afterwards, the energy estimator is converted into energy by using the FD energy (in the case of real data, this energy have to be corrected for the invisible energy). A linear fit is applied to get the parameters A and B in the equation E_FD = A (S38)^B.


Requirements

Astrotools (https://astro.pages.rwth-aachen.de/astrotools/)
