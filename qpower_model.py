import numpy as np
from pathlib import Path
import pandas as pd
from tensorflow.keras.models import load_model

def transform_features(x, f="cos"):
    if f == "cos":
        return np.cos(x)
    elif f == "sin":
        return np.sin(x)
    elif f == "tanh":
        return np.tanh(x)

class QPowerModel:
    """
    Use to evaluate quadrant power splits from control drum configurations.
    Set up as init, then separately use method call to minimize reading times.
    """
    def __init__(self):
        #Find and load file
        model_file = Path("pmdata/for_dean/power_model.h5")
        self.raw_model = load_model(model_file)

    def eval(self, pert):
        pertn = np.array([transform_features(pert), ])
        unorm = self.raw_model.predict(pertn).flatten()
        return unorm/unorm.sum()

def qPowerModel(pert):
    """Wrapper for QPowerModel that initializes and runs"""
    a = QPowerModel()
    return a.eval(pert)
