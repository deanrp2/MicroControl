#--------------------------------------------------------
#Problem 5
#  From full insertion, minimum maximum distance to
#  full insertion even power split
#--------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from neorl import DE, ES, PSO, PESA2

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

def fitness(xt):

    x = np.zeros(8)
    x[0] = xt[0]
    x[2:] = xt[1:] #leave element 1 at 0

    #calculate some travel distance metric
    traveldist = x.sum()
    max_traveldist = np.pi
    norm_traveldist = np.abs(x).max()/(np.pi)

    #calculate reactivity error
    rerr = (target_reactivity*1e-5 - rmodel.eval(x))**2/reactivitymax**2

    #calculate quadpower errors
    qpower = pmodel.eval(x)
    qpower_err = np.abs(qpower - 0.25)
    qpower_err_norm = sum(qpower_err/(qpower_max_err - qpower_min_err))

    return .1*norm_traveldist + .7*rerr + .2*qpower_err_norm


#set up bounds
BOUNDS = {"x%i"%i : ["float", -np.pi, np.pi] for i in range(1, 8)}

#optimize
#  Differential evolution
de = DE(mode = "min", bounds = BOUNDS, fit = fitness, npop=50,
        CR = 0.5, F = 0.7, ncores = 1, verbose = True)
de_x, de_y, de_hist = de.evolute(ngen = 200)

#  Evolution strategies
es = ES(mode = "min", bounds = BOUNDS, fit = fitness, lambda_ = 40,
        mu = 30, ncores = 1)
es_x, es_y, es_hist = de.evolute(ngen = 200)

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

for key, value in ans.items():
    plt.plot(value[1], label = key)
    vt = np.asarray(value[0])
    v = np.zeros(8)
    v[0] = vt[0]
    v[2:] = vt[1:]
    print("\n-----------------------\n", key, "\n-----------------------")
    print("Reactivity Err", np.abs(target_reactivity*1e-5 - rmodel.eval(v))*1e5, "pcm")
    print("Qpowers", np.array(pmodel.eval(v))*100, "\%")
    print("Traveldist", np.sum(np.abs(v))*180/np.pi)

plt.legend()
plt.xlabel("Generations")
plt.ylabel("Obj")
plt.show()
