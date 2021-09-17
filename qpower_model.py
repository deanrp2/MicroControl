import numpy as np
from pathlib import Path
import pandas as pd

class QPowerModel:
    """
    Use to evaluate quadrant power splits from control drum configurations.
    Set up as init, then separately use method call to minimize reading times.
    """

    def __init(self):
        # MAJDI HERE
        model_file = Path("path/to/model file name") #nonsense
        self.eval = load_ML_model(model_file) #nonsense

def qPowerModel(config):
    """Wrapper for QPowerModel that initializes and runs"""
    a = QPowerModel()
    return a.eval(config)
