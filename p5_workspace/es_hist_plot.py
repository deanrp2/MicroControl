import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fs = ["log/hist0p8.dat", "log/hist0p9.dat"]

titles = [".86 react wt", ".90 react wt"]

colmns = ["rerr", "psplit", "tdist", "obj"] + ["x%i"%i for i in range(1, 8)]
for i, f in enumerate(fs):
    fig, ax = plt.subplots(1, 1)
    a = np.genfromtxt(f, delimiter = ",")
    ara = pd.DataFrame(a, columns = colmns)

    ax.hist(ara["rerr"], bins = "auto", color = "#0504aa", rwidth=0.85,
            density = False)
    ax.set_xlabel("Reactivity Error")
    ax.set_ylabel("Freq")
    ax.set_xlim([0, 800])

    fig, ax = plt.subplots(1, 1)
    ax.plot(ara["rerr"], ara["psplit"], "k.")
    ax.set_xlabel("reactivity error")
    ax.set_ylabel("power split error")



plt.show()
