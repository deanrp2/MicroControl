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

def get_alphas(typ = "wtd"): #wtd, refl, abs
    """read alphas from file"""
    fpath = Path("model_data/alpha_%s_gr.csv"%typ)
    return pd.read_csv(fpath, index_col = 0)

def integrate(x, y, lbnd, ubnd):
    """Integrate j- function given x & y across the bounds."""
    #make sure dealing with numpy arrays
    x, y = np.asarray(x), np.asarray(y)

    #integrate portions of functions where x-blocks are completely enclosed
    compl_ind = np.where(np.logical_and(x > lbnd, x < ubnd))[0]

    if compl_ind.size == 0: #if both bounds fall within an interval
        idx = np.searchsorted(x, lbnd)
        y_lbnd_approx = (lbnd - x[idx-1])/(x[idx] - x[idx-1])*(y[idx] - y[idx-1]) + y[idx-1]
        y_ubnd_approx = (ubnd - x[idx-1])/(x[idx] - x[idx-1])*(y[idx] - y[idx-1]) + y[idx-1]
        return (ubnd - lbnd)*(y_ubnd_approx + y_lbnd_approx)/2

    full_blocks_integral = np.trapz(y[compl_ind], x[compl_ind])

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
    """get bounds on j-^2 integrals given rotation angle, theta and coating angle cangle
    returns (thetaA&thetaA0'), (thetaA'&thetaA0)"""
    assert (theta < np.pi + 1e-5) and (theta > -np.pi - 1e-5)

    if 0 < theta and theta < cangle:
        return ([cangle/2, theta + cangle/2], [-cangle/2, theta - cangle/2])
    if -cangle < theta and theta < 0:
        return ([theta - cangle/2, -cangle/2], [theta + alpha/2, cangle/2])
    else:
        return ([theta - cangle/2, theta + cangle/2], [-cangle/2, cangle/2])

def calc_zetatildes(theta, cangles, alphas, jminusA, jminusB):
    """
    calculate zetatilde functions for all drums
    thetas is numpy array of size 8 of rotation angles
    cangles is numpy array of size 8 of coating angles
    alphas is numpy array of shape (8,9) of model coefficients
    jminusA is pandas array with "centers" and "hist" columns
    jminusB is pandas array with "centers" and "hist" columns
    """
    configAids = [1, 4, 5, 8] #drum positions with configuration A

    #calculate gammastar by integrting j- over abs bounds for each drum
    gammastar = np.ones(9)
    for i in range(1, 9):
        if i in configAids:
            gammastar[i] = integrate(x = jminusA["centers"],
                                     y = jminusA["hist"],
                                     lbnd = theta[i-1] - cangles[i-1]/2,
                                     ubnd = theta[i-1] + cangles[i-1]/2)
        else:
            gammastar[i] = integrate(x = jminusB["centers"],
                                     y = jminusB["hist"],
                                     lbnd = theta[i-1] - cangles[i-1]/2,
                                     ubnd = theta[i-1] + cangles[i-1]/2)

    #calculate zetatilde for each drum
    return (gammastar@alphas.T).values

class ReactivityModel:
    """
    Used to evaluate reactivity insertion from control drum perturbation.
    Set up as init->method call to minimize file reading times
    """
    def __init__(self, typ = "abs"): #abs, wtd or refl
        """initialize to perform all file I/O"""
        self.jmA, self.jmB = get_jminus(typ)
        self.alphas = get_alphas(typ)
        self.cangles = np.array([130, 145, 145, 130,
                                 130, 145, 145, 130])/180*np.pi

    def eval(self, pert, nom = None, qpower = False):
        """
        Evaluate reactivity worth of drum perturbation.
        Pert is numpy array of 8 drum angles in radians with 
        coordinate systems described in the README.md.
        Nom is an optional starting state given same as pert
        qpower is whether or not to return 4-element fractional power array
        """

        zetatildes = calc_zetatildes(theta = pert,
                                     cangles = self.cangles,
                                     alphas = self.alphas,
                                     jminusA = self.jmA,
                                     jminusB = self.jmB)

        #loop through drums and calculate each contribution
        reactivities = np.zeros(8)
        for i in range(8):
            b1, b2 = int_bounds(pert[i], self.cangles[i])
            if i in [0, 3, 4, 7]: #if config A
                int1 = integrate(self.jmA["centers"], self.jmA["hist"], *b1)
                int2 = integrate(self.jmA["centers"], self.jmA["hist"], *b2)
            else: #if config A
                int1 = integrate(self.jmB["centers"], self.jmB["hist"], *b1)
                int2 = integrate(self.jmB["centers"], self.jmB["hist"], *b2)
            reactivities[i] = zetatildes[i]*(int1 - int2)

        if nom: #little trick
            reactivity = self.eval(pert) - self.eval(nom) #assume
                                                          #reactivites additive
        else:
            reactivity = reactivities.sum()

        if qpower:
            qpower = (zetatildes[::2] + zetatildes[1::2])/2
            return reactivity, qpower
        else:
            return reactivity

if __name__ == "__main__":
    a = ReactivityModel()
    print(a.eval(np.zeros(8)+np.pi/2))

