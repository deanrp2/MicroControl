import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import sys
import scipy.interpolate as si
import scipy.ndimage as nd

sys.path.append("..")

from reactivity_model import ReactivityModel

rmodel = ReactivityModel()

plot = True

#use maximum diffferential worth config from demo problem 8
#set of configs to run scram with, name and config as dict
configs = {"maximum_diff" : np.array([-142.9, -97.5, 46.6, -28.9, 51.6, 46.9, -94.6, 136.6])/180*np.pi}

#run optimization to find even config that matches the integral worth
target_reactivity = rmodel.eval(configs["maximum_diff"])

def fit(x):
    thetas = np.zeros(8) + x
    return np.abs(target_reactivity - rmodel.eval(thetas))

res = so.minimize(fit, x0 = np.array([0]), bounds = [(-np.pi, np.pi)])
configs["even"] = np.ones(8)*res.x


#print difference in configs[n]
for n in configs:
    print(n, "diff worth", np.abs(rmodel.evalg(configs[n])).sum())

# generate curves of reactivity vs time for a shutdown
max_drum_speed = 2 #deg/s
max_drum_speed = max_drum_speed/180*np.pi #rad/s

#run simulation
dt = 1

reactivity_log = {} #reactivity log
config_log = {} #drum config logs
for n, c in configs.items():
    reactivity_log[n] = []
    config_log[n] = []

ts = [] #time axis

t = 0.
tf = 80.

#time loop
while t < tf:
    print(t)
    for n, c in configs.items():
        config_log[n].append(configs[n].copy())
        reactivity_log[n].append(rmodel.eval(c))
    ts.append(t)

    #move drums
    for n in configs:
        grad = rmodel.evalg(configs[n])
        for i in range(8):
            if grad[i] < 0:
                configs[n][i] += max_drum_speed*dt
            else:
                configs[n][i] -= max_drum_speed*dt

    #update time
    t += dt

for n in configs:
    reactivity_log[n] = np.array(reactivity_log[n])
    config_log[n] = np.array(config_log[n])


if plot:
    fig, ax = plt.subplots(1,len(list(configs.keys())), sharey = True)
    for i, n in enumerate(list(configs.keys())):
        ax[i].set_title(n)
        for j in range(8):
            ax[i].plot(ts, config_log[n][:,j]*180/np.pi, "k", alpha = .1*j+.3)
    fig.tight_layout()
    ax[0].set_ylabel("drum angles")
    fig, ax = plt.subplots(1, 1)
    for n in configs:
        ax.plot(ts, reactivity_log[n], ".-", label = n)
    plt.legend()
    plt.show()





