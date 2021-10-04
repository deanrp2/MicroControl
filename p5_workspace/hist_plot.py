import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def group(ara):
    temp = (ara - ara.mean(0))/ara.std(0)
    km = KMeans(n_clusters = 2).fit(temp)
    return km.labels_


fs = ["log/hist_150_even.dat", "log/hist_500_0p98.dat"]

titles = ["150 generations", "500 generations"]

colmns = ["rerr", "psplit", "tdist", "obj"] + ["x%i"%i for i in range(1, 8)]
for i, f in enumerate(fs):
    a = np.genfromtxt(f, delimiter = ",")
    ara = pd.DataFrame(a, columns = colmns)
    #gp = group(ara[["x%i"%ii for ii in range(1, 8)]]) #for only drum positions used in grouping
    gp = group(ara) #for all used in grouping

    fig, ax = plt.subplots(1, 1)
    ax.hist(ara["rerr"], bins = "auto", color = "#0504aa", rwidth=0.85,
            density = True)
    ax.set_xlabel("Reactivity Error")
    ax.set_ylabel("Freq")
    ax.set_xlim([0, 800])
    ax.set_title(titles[i])

    fig, ax = plt.subplots(1, 1)
    ax.plot(ara["rerr"].iloc[(gp == 0)], ara["psplit"].iloc[(gp == 0)], "k.")
    ax.plot(ara["rerr"].iloc[(gp == 1)], ara["psplit"].iloc[(gp == 1)], "kx")
    ax.set_xlabel("reactivity error")
    ax.set_ylabel("power split error")
    ax.set_title(titles[i])

    fig, ax = plt.subplots(1,1)
    ax.plot(ara["rerr"].iloc[(gp == 0)], ara["obj"].iloc[(gp == 0)], "k.")
    ax.plot(ara["rerr"].iloc[(gp == 1)], ara["obj"].iloc[(gp == 1)], "kx")
    ax.set_xlabel("reactivity error")
    ax.set_ylabel("fitness")
    ax.set_title(titles[i])



plt.show()
