import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

#welch's t test

algos = []
means = []
stds = []

fname = "results_table.txt"

with open(fname, "r") as f:
    c = f.readlines()

N = int(c[1].split()[-1])
for i in range(6):
    algos.append(c[2 + 11*i].strip())
    means.append(float(c[5 + 11*i].split()[2]))
    stds.append(float(c[6 + 11*i].split()[2]))


pvals = np.zeros((6,6))

for i in range(6):
    for j in range(6):
        sX1 = stds[i]/np.sqrt(N)
        sX2 = stds[j]/np.sqrt(N)
        sDX = np.sqrt(sX1**2 + sX2**2)
        t_stat = (means[i] - means[j])/sDX
        print(t_stat)

        dof = sDX**4/(sX1**4/(N-1) + sX2**4/(N-1))
        pvals[i,j] = t.sf(np.abs(t_stat), int(dof))

plt.imshow(pvals < 0.1)
plt.show()
