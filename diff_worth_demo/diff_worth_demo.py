#demonstration of reactivity worth as fxn of time after accident
# for all even drums and maximum differential worth for drums

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so


from reactivity_model import ReactivityModel

rmodel = ReactivityModel()

#use maximum diffferential worth config from demo problem 8
mdiff_config = np.array([-142.9, -97.5, 46.6, -28.9, 51.6, 46.9, -94.6, 136.6])/180*np.pi
#mdiff_config = np.array([120, 100, 120, 120, 100, 100, 120, 100])/180*np.pi


#run optimization to find even config that matches the integral worth
target_reactivity = rmodel.eval(mdiff_config)

def fit(x):
    thetas = np.zeros(8) + x
    return np.abs(target_reactivity - rmodel.eval(thetas))

res = so.minimize(fit, x0 = np.array([0]), bounds = [(-np.pi, np.pi)])
even_config = np.ones(8)*res.x


#print difference in diffy worth
print("Even diff worth", np.abs(rmodel.evalg(even_config)).sum())
print("Max diff worth", np.abs(rmodel.evalg(mdiff_config)).sum())

# generate curves of reactivity vs time for a shutdown
max_drum_speed = 2 #deg/s
max_drum_speed = max_drum_speed/180*np.pi #rad/s

dt = 0.1

mdiff_r = []
even_r = []
ts = []

#time loop 
t = 0.

md = []
ev = []
while rmodel.eval(mdiff_config) > 3e-5:
    md.append(mdiff_config.copy())
    ev.append(even_config.copy())
    #print(rmodel.eval(mdiff_config))
    #calculate integral worth
    mdiff_r.append(rmodel.eval(mdiff_config))
    even_r.append(rmodel.eval(even_config))
    ts.append(t)

    #move drums
    mdiff_grad = rmodel.evalg(mdiff_config)
    even_grad = rmodel.evalg(even_config)
    for i in range(8):
        if mdiff_grad[i] < 0:
            mdiff_config[i] += max_drum_speed*dt
        else:
            mdiff_config[i] -= max_drum_speed*dt
        if even_grad[i] < 0:
            even_config[i] += max_drum_speed*dt
        else:
            even_config[i] -= max_drum_speed*dt
    #update time
    t += dt
md = np.array(md)
ev = np.array(ev)
fig, ax = plt.subplots(1,2, sharey = True)
for i in range(8):
    ax[0].plot(ts, md[:,i])
    ax[1].plot(ts, ev[:,i])
ax[0].set_title("max worth")
ax[1].set_title("even")
plt.figure()
plt.plot(ts, mdiff_r, "k.-", label = "max diff worth")
plt.plot(ts, even_r, "b.-", label = "even drums")
plt.legend()
plt.show()
