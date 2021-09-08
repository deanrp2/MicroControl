import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def get_jminus(typ = "wtd"): #wtd, refl or abs
    """Retrieve j- function from model data files."""
    #pull and load data
    configA_path = Path("model_data/configA_flux_%s_b.csv"%typ)
    configB_path = Path("model_data/configB_flux_%s_b.csv"%typ)
    jm_configA = pd.read_csv(configA_path, index_col = 0)
    jm_configB = pd.read_csv(configB_path, index_col = 0)

    #adjust 0 angle to be inward to the reactor
    jm_configA["centers"] -= 3.6820
    jm_configB["centers"] -= 4.0677

    #center periodic functions on 0
    jm_configA["centers"][jm_configA["centers"] < -np.pi] += 2*np.pi
    jm_configB["centers"][jm_configB["centers"] < -np.pi] += 2*np.pi

    #add buffer on each side for periodic integral evaluations
    lowcopy = jm_configA.copy()
    lowcopy["centers"] -= 2*np.pi
    highcopy = jm_configA.copy()
    highcopy["centers"] += 2*np.pi
    jm_configA = pd.concat([jm_configA, lowcopy, highcopy])

    lowcopy = jm_configB.copy()
    lowcopy["centers"] -= 2*np.pi
    highcopy = jm_configB.copy()
    highcopy["centers"] += 2*np.pi
    jm_configB = pd.concat([jm_configB, lowcopy, highcopy])

    #reorder from -pi to pi
    jm_configA = jm_configA.sort_values(by="centers")
    jm_configB = jm_configB.sort_values(by="centers")

    return jm_configA, jm_configB

def integrate(x, y, lbnd, ubnd): #TODO, add microlimit conditional
    """Integrate j- function given x & y across the bounds."""
    #make sure dealing with numpy arrays
    x, y = np.asarray(x), np.asarray(y)

    #integrate portions of functions where x-blocks are completely enclosed
    compl_ind = np.where(np.logical_and(x > lbnd, x < ubnd))[0]
    full_blocks_integral = 0#np.trapz(y[compl_ind], x[compl_ind])

    #integrate lower hanging partial block
    lidx = compl_ind.min()
    y_lbnd_approx = (lbnd - x[lidx-1])/(x[lidx] - x[lidx-1])*(y[lidx] - y[lidx-1]) + y[lidx-1]
    lower_block_integral = (x[lidx] - lbnd)*(y[lidx] + y_lbnd_approx)/2

    #integrate lower hanging partial block
    uidx = compl_ind.max()+1
    y_ubnd_approx = (ubnd - x[uidx-1])/(x[uidx] - x[uidx-1])*(y[uidx] - y[uidx-1]) + y[uidx-1]
    upper_block_integral = (ubnd - x[uidx - 1])*(y[uidx - 1] + y_ubnd_approx)/2

    return full_blocks_integral + lower_block_integral + upper_block_integral

def int_bounds(theta, cangle):
    """get bounds on j-^2 integrals given rotation angle """
    


#def drum_reactivity(pert, nom = None)
# ...
#    if nom:
#        return drum_reactivity(pert) - drum_reactivity(nom)

if __name__ == "__main__":
    from scipy.interpolate import interp1d
    jmA, jmB = get_jminus("wtd")














