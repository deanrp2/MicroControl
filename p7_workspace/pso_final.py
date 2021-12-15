import numpy as np
import matplotlib.pyplot as plt

from pso_expl import pso_expl

import sys

sys.path.append("..")

x, y, hist, res, _ = pso_expl(100, seed = 5)

opt = res["fitness"].argmin()
b = res.iloc[opt]

for i in range(1, 9):
    print(b["x" + str(i)])

print("Reactivity Error:", b["react_err_obj"])
print("Psplit Error:", b["psplit_err_obj"])
print("Diff Worth:", b["diff_worth_obj"])

from reactivity_model import ReactivityModel
from qpower_model import QPowerModel

print("Basic checks:")
thetas = np.array([b["x"+str(d)] for d in range(1, 9)])
a = ReactivityModel()
b = QPowerModel()
print("Injected Reactivity:", a.eval(thetas))
print("Target Reactivity:", 0.03072)
print("Reactivity Error", a.eval(thetas) - 0.03072)
print("Qpower:", b.eval(thetas))
print("Diff worth:", np.abs(a.evalg(thetas)).sum())
