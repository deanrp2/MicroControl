import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4)

from mfo_expl import mfo_expl

import sys

sys.path.append("..")

x, y, hist, res, _ = mfo_expl(10000, seed = 5)

opt = res["fitness"].argmin()
b = res.iloc[opt]
for ol in range(4, 8):
    b["x" + str(ol)] *= -1

print("Radians")
for i in range(1, 8):
    print(b["x" + str(i)])
print("Degrees")
for i in range(1, 8):
    print(b["x" + str(i)]*180/np.pi)

print("Reactivity Error:", b["react_err_obj"])
print("Psplit Error:", b["psplit_err_obj"])
print("Travel Dist:", b["tdist_obj"])

from reactivity_model import ReactivityModel
from qpower_model import QPowerModel

print("Basic checks:")
thetas = np.array([b["x1"], 0] + [b["x"+str(d)] for d in range(2, 8)])
a = ReactivityModel()
b = QPowerModel()
print("Injected Reactivity:", a.eval(thetas))
print("Target Reactivity:", 0.03308)
print("Reactivity Error", a.eval(thetas) - 0.03308)
print("Qpower:", b.eval(thetas))
print("Th. Psplit Error", np.sum(np.abs(b.eval(thetas)-0.25)))
print("Travel dist:", np.abs(thetas).max())
