import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def get_jminus(typ = "wtd"): #wtd, refl or abs
    """Retrieve j- function from model data files."""
    configA_path = Path("model_data/configA_flux_%s_b.csv"%typ)
    configB_path = Path("model_data/configA_flux_%s_b.csv"%typ)
    jm_configA = pd.read_csv(configA_path, index_col = 0)
    jm_configB = pd.read_csv(configB_path, index_col = 0)
    return jm_configA, jm_configB



#def drum_reactivity(pert, nom = None)
# ...
#    if nom:
#        return drum_reactivity(pert) - drum_reactivity(nom)

if __name__ == "__main__":
    get_jminus()
