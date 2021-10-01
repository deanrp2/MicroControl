#--------------------------------------------------------
#Problem 6
#  Maximum differential worth, even power splits and critical
#--------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from neorl import DE, ES, PSO, PESA2
import sys
sys.path.append("..")

from reactivity_model import ReactivityModel
from qpower_model import QPowerModel

#accept target
target_reactivity = 3072 #pcm

#set up fitness function
rmodel = ReactivityModel()
pmodel = QPowerModel()

#maximum reactivity for normalization
thetasmax = np.zeros(8)
thetasmax += np.pi
reactivitymax = rmodel.eval(thetasmax)

#maximum quadpower for normalization
thetasmax = np.zeros(8)
thetasmax[[0, 1]] += np.pi
qpower_max = pmodel.eval(thetasmax)[0]
qpower_max_err = qpower_max - 0.25

thetasmin = np.zeros(8) + np.pi
thetasmin[[0,1]] -= np.pi
qpower_min = pmodel.eval(thetasmin)[0]
qpower_min_err = qpower_min - 0.25

#some estimate for maximum differential worth
mdiff_At = np.zeros(8)
mdiff_At[0] += np.pi/2
mdiff_A = rmodel.evald(mdiff_At, 1)
mdiff_Bt = np.zeros(8)
mdiff_Bt[1] += np.pi/2
mdiff_B = rmodel.evald(mdiff_Bt, 2)
mdiff = np.zeros(8)
mdiff[[0, 3, 4, 7]] = mdiff_A
mdiff[[1, 2, 5, 6]] = mdiff_B

def fitness(x):
    x = np.asarray(x)

    #calculate reactivity error
    rerr = (target_reactivity*1e-5 - rmodel.eval(x))**2/reactivitymax**2

    #calculate quadpower errors
    qpower = pmodel.eval(x)
    qpower_err = np.abs(qpower - 0.25)
    qpower_err_norm = sum(qpower_err/(qpower_max_err - qpower_min_err))

    #calculate total differential worth
    gradworth = rmodel.evalg(x)
    diff_worth_norm = np.abs(gradworth/mdiff).sum()/8

    return -.2*diff_worth_norm + .4*rerr + .4*qpower_err_norm


#set up bounds
BOUNDS = {"x%i"%i : ["float", -np.pi, np.pi] for i in range(1, 9)}

#optimize
#  Differential evolution
de = DE(mode = "min", bounds = BOUNDS, fit = fitness, npop=50,
        CR = 0.5, F = 0.7, ncores = 1, verbose = True)
de_x, de_y, de_hist = de.evolute(ngen = 4)

#  Evolution strategies
es = ES(mode = "min", bounds = BOUNDS, fit = fitness, lambda_ = 40,
        mu = 30, ncores = 1)
es_x, es_y, es_hist = de.evolute(ngen = 4)

# Particle Swarm
#pso = PSO(mode = "min", bounds = BOUNDS, fit = fitness, ncores = 1)
#print(pso.evolute(ngen = 100, verbose = False))

# Modern PESA
#mpesa = PESA2(mode = "min", bounds = BOUNDS, fit = fitness)
#print(type(mpesa.evolute(ngen = 4)))
#exit()

ans = {"Differential Evolution" : [de_x, de_hist],
        "Evolution Strategies" : [es_x, es_hist]}
#        "Particle Swarm" : pso_x}

plt.figure()
for key, value in ans.items():
    plt.plot(value[1], label = key)
    v = np.asarray(value[0])
    print("\n-----------------------\n", key, "\n-----------------------")
    s =["%.1f"%(a*180/np.pi) for a in v]
    print("Angles", " ".join(s))
    print("Reactivity Err", np.abs(target_reactivity*1e-5 - rmodel.eval(v))*1e5, "pcm")
    print("Qpowers", np.array(pmodel.eval(v))*100, "\%")
    print("Diff worth", np.abs(rmodel.evalg(v)).sum()*1e5*(np.pi/180), "pcm")

plt.legend()
plt.xlabel("Generations")
plt.ylabel("Obj")
plt.show()
