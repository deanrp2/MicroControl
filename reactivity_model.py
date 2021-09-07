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

    #reorder from -pi to pi
    jm_configA = jm_configA.sort_values(by="centers")
    jm_configB = jm_configB.sort_values(by="centers")

    return jm_configA, jm_configB

def integratePP(x, y, lbnd, ubnd):
    """
    Integrate j- function given x & y across the 
    bounds.
    x and y must be ordered from lowest x to highest x.
    ubnd - lbnd must be less than 2pi
    """
    #determine if x sufficently covers bounds
    




#def drum_reactivity(pert, nom = None)
# ...
#    if nom:
#        return drum_reactivity(pert) - drum_reactivity(nom)

if __name__ == "__main__":
    jmA, jmB = get_jminus("wtd")

    plt.plot(jmA["centers"], jmA["hist"])
    plt.plot(jmB["centers"], jmB["hist"])
    plt.show()
