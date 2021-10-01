#--------------------------------------------------------
#Problem 2
#  Rotate all 8 drums independently to check min and max
#  reactivity
#--------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from neorl import DE, ES, FNEAT

from reactivity_model import ReactivityModel

#select either "max" or "min"
m = "max"

#set up fitness function
model = ReactivityModel()

def fitness(x):
    return model.eval(np.array(x))

#set up bounds
BOUNDS = {"x%i"%i : ["float", -1.1*np.pi, 1.1*np.pi] for i in range(1, 9)}

#optimize MIN/MAX
#  Differential evolution
de = DE(mode = m, bounds = BOUNDS, fit = fitness, npop = 70,
        CR = 0.5, F = 0.7, ncores = 1, verbose = True)
de_x, de_y, de_hist = de.evolute(ngen = 60)

#  Evolution strategies
es = ES(mode = m, bounds = BOUNDS, fit = fitness, lambda_ = 70,
        mu = 30, ncores = 1)
es_x, es_y, es_hist = de.evolute(ngen = 60)

#  FNEAT
config = {"pop_size" : 75,
          "num_hidden" : 2,
          "activation_mutate_rate" : 0.1,
          "survival_threshold" : 0.3}
fneat = FNEAT(mode = m, bounds = BOUNDS, fit = fitness, config = config,
        ncores = 1)
x0 = np.random.uniform(-np.pi, np.pi, 8)
fneat_x, fneat_y, fneat_hist = fneat.evolute(ngen = 1, x0 = x0, startpoint = None,
        checkpoint_itv=None)
plt.plot(np.array(es_hist)*1e5, "--k", label = "ES")
plt.plot(np.array(de_hist)*1e5, "--r", label = "DE")
plt.plot(np.array(fneat_hist["global_fitness"])*1e5, "--b", label = "FNEAT")
plt.xlabel("Iterations")
plt.ylabel("reactivity [pcm]")
plt.legend()
plt.show()

