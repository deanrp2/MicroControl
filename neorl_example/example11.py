import numpy as np
import matplotlib.pyplot as plt
import sys
from neorl import DE, ES, MFO

sys.path.append("..")

from reactivity_model import ReactivityModel
from qpower_model import QPowerModel

#import models from other files in repo
rm = ReactivityModel()
pm = QPowerModel()

#define unscaled objectives
def hatfc(x):
    thetas = np.zeros(8)
    thetas[0] = x[0]
    thetas[2:] = x[1:]
    react = rm.eval(thetas)
    return np.abs(react - 0.03308)

def hatfp(x):
    thetas = np.zeros(8)
    thetas[0] = x[0]
    thetas[2:] = x[1:]
    powers = pm.eval(thetas)
    targets = np.zeros(4)+0.25
    return np.abs(powers - targets).sum()

def hatfd(x):
    return np.max(np.abs(x))

#define objective scaling parameters
fc_max = 0.03308
fc_min = 0

fp_max = 0.0345
fp_min = 0

fd_max = np.pi
fd_min = 0

#define scaling objectives
fc = lambda x : (hatfc(x) - fc_min)/(fc_max - fc_min)
fp = lambda x : (hatfp(x) - fp_min)/(fp_max - fp_min)
fd = lambda x : (hatfd(x) - fd_min)/(fd_max - fd_min)

#define function weights
wc = 0.5
wp = 0.4
wd = 0.1

#define single objective function
F = lambda x : wc*fc(x) + wp*fp(x) + wd*fd(x)

#define drum rotation bounds
BOUNDS = {"x%i"%i : ["float", -1.*np.pi, 1.*np.pi] for i in range(1, 8)}

#run de optimization
npop = 20
F_de = 0.4
CR = 0.3
de = DE(mode = "min", bounds = BOUNDS, fit = F, npop = npop, F = F_de, CR = CR, seed = 1)
de_x, de_y, de_hist = de.evolute(100, verbose = True)

#run es optimization
mu = 25
cxpb = 0.6
mutpb = 0.3
es = ES(mode = "min", bounds = BOUNDS, fit = F, lambda_ = 50, mu = mu, cxpb = 0.6,
        mutpb = 0.3, seed = 1)
es_x, es_y, es_hist = es.evolute(100, verbose = True)

#run mfo optimization
nmoths = 55
mfo = MFO(mode = "min", bounds = BOUNDS, fit = F, nmoths = nmoths, b = 1, seed = 1)
mfo_x, mfo_y, mfo_hist = mfo.evolute(100, verbose = True)

plt.plot(de_hist["global_fitness"], label = "DE")
plt.plot(es_hist["global_fitness"], label = "ES")
plt.plot(mfo_hist["global_fitness"], label = "MFO")

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()

print("MFO fc hat")
print(hatfc(mfo_x))
print("MFO fp hat")
print(hatfp(mfo_x))
print("MFO fd hat")
print(hatfd(mfo_x))
