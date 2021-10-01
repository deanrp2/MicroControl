import numpy as np
from dataclasses import dataclass
import typing
import string
import random
import logging
import matplotlib.pyplot as plt
from pathlib import Path

def setup_logger(logger_name, log_file, level=logging.INFO):
    """
    Helper class to initialize logger
    """
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s, %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

@dataclass
class Objective:
    """
    In Multiobjective optimization, holds information
    about a single objective as well as the minimum
    and maximum of that objective

    indim : dimensionality of list/vector to be passed into calc
    calc : function with actually calculates the objective
    min : known minimum of objective
    max : known maximum of objective
    """
    name : str
    typ : str
    indim : int
    calc : typing.Callable
    min : float
    max : float

    def __post_init__(self):
        if self.typ != "max" and self.typ != "min":
            raise Exception("typ must be max or min")

class FitnessHelper:
    """
    Class to handle combining multiple objectives into 
    a single objective function
    also handles all objective logging
    always run minimization on these fitnesses
    """
    def __init__(self, objs, wts, fname):
        """
        objs : list of Objective objects must have same number of indims
        wts : weights corresponding to each objective (all pos)
        fname : Path for filename to write logging to
        """
        assert len(objs) == len(wts)

        self.objs = objs

        #adjust weights
        self.wts = np.asarray(wts)
        self.wts = self.wts/self.wts.sum()
        for i in range(self.wts.size):
            if self.objs[i].typ == "max":
                self.wts[i] *= -1

        self.fname = fname
        self.log_init()

    def log_init(self):
        logid = ''.join(random.choice(string.ascii_uppercase 
                + string.digits) for _ in range(25)) #get unique logger name
        setup_logger(logid, self.fname) #initialize logger
        self.log = logging.getLogger(logid)

        #print csv column headers
        headr = "\n"
        headr += "sys_time, "
        for i in range(1, self.objs[0].indim + 1):
            headr += "x%i, "%i
        for o in self.objs:
            headr += o.name + "_obj, "
            headr += o.name + "_scaled_obj, "
            headr += o.name + "_wtd_scaled_obj, "
        headr += "fitness"
        self.log.info(headr)

    def fitness(self, x):
        """
        function to actually pass into the optimizer
        """
        #calculating
        objectives = []
        objectives_scaled = []
        objectives_wtd_scaled = []
        for i in range(self.wts.size):
            o = self.objs[i].calc(x)
            o_scaled = (o - self.objs[i].min) \
                    / (self.objs[i].max - self.objs[i].min)
            o_wtd_scaled = self.wts[i]*o_scaled
            objectives.append(o)
            objectives_scaled.append(o_scaled)
            objectives_wtd_scaled.append(o_wtd_scaled)
        fitness = sum(objectives_scaled)

        #logging
        logline = ""
        for i in range(0, self.objs[0].indim):
            logline += ("%.6E,"%x[i]).rjust(15)
        for i in range(len(self.objs)):
            logline += ("%.6E,"%objectives[i]).rjust(15)
            logline += ("%.6E,"%objectives_scaled[i]).rjust(15)
            logline += ("%.6E,"%objectives_wtd_scaled[i]).rjust(15)
        logline += ("%.6E"%fitness).rjust(14)
        self.log.info(logline)

        return fitness



if __name__ == "__main__":
    from reactivity_model import ReactivityModel
    from qpower_model import QPowerModel

    a = ReactivityModel()
    #reactivity objective
    t_react = .04000
    def rtgt(x):
        react = a.eval(x)
        return np.abs(react - t_react)
    minn = 0
    maxx = max([rtgt(np.zeros(8)), rtgt(np.zeros(8)+np.pi)])
    tgt_react = Objective("react_err", "min", 8, rtgt, minn,  maxx)

    #differential worth objective
    def diffworth(x):
        return np.abs(a.evalg(x)).sum()
    diff_worth = Objective("diff_worth", "max", 8, diffworth, 0, 3800)

    #power split objective
    t_splits = np.zeros(4) + 0.25

    b = QPowerModel()

    xmax = np.zeros(8)
    xmax[[0, 1]] += np.pi
    off_splits = b.eval(xmax)

    max_off = np.abs(off_splits - t_splits).sum()
    def qpower(x):
        predicted = b.eval(x)
        return np.abs(predicted - t_splits).sum()

    tgt_splits = Objective("psplit_err", "min", 8, qpower, 0, max_off)

    #min max travel distance objective
    def tdist(x):
        return np.max(np.abs(x))
    minmax_dist = Objective("tdist", "min", 8, tdist, 0, np.pi)

    objs = [tgt_react, diff_worth, tgt_splits]
    wts = [.33, .33, .34]
    td = FitnessHelper(objs, wts, "text.txt")

    for i in range(400):
        angle = np.random.uniform(-np.pi, np.pi, 8)
        td.fitness(angle)





