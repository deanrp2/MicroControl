import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def group(ara2):
    temp = (ara - ara.mean(0))/ara.std(0)
    km = KMeans(n_clusters = 2).fit(temp)
    return km.labels_

grouping = True
sets = [0, 1, 2, 3] #which of the figures to plot

fs = ["log/hist_even.dat"]


titles = ["even", "react .5", "react .6", "react .7", "react .8" ]

colmns = ["rerr", "psplit", "diffworth", "obj"] + ["x%i"%i for i in range(1, 8)]
for i, f in enumerate(fs):
    a = np.genfromtxt(f, delimiter = ",")
    ara = pd.DataFrame(a, columns = colmns)
    if grouping:
        #gp = group(ara[["x%i"%ii for ii in range(1, 8)]]) #for only drum positions used in grouping
        gp = group(ara) #for all used in grouping
    else:
        gp = np.zeros(ara.shape[0])

    if 0 in sets:
        fig, ax = plt.subplots(1, 1)
        ax.hist(ara["rerr"], bins = "auto", color = "#0504aa", rwidth=0.85,
                density = True)
        ax.set_xlabel("Reactivity Error")
        ax.set_ylabel("Freq")
        ax.set_xlim([0, 800])
        ax.set_title(titles[i])

    if 1 in sets:
        fig, ax = plt.subplots(1, 1)
        ax.hist(ara["psplit"], bins = "auto", color = "purple", rwidth=0.85,
                density = True)
        ax.set_xlabel("Power Split Error")
        ax.set_ylabel("Freq")
        ax.set_title(titles[i])

    if 2 in sets:
        fig, ax = plt.subplots(1, 1)
        ax.plot(ara["rerr"].iloc[(gp == 0)], ara["psplit"].iloc[(gp == 0)], "k.")
        ax.plot(ara["rerr"].iloc[(gp == 1)], ara["psplit"].iloc[(gp == 1)], "kx")
        ax.set_xlabel("reactivity error")
        ax.set_ylabel("power split error")
        ax.set_title(titles[i])

    if 3 in sets:
        fig, ax = plt.subplots(1,1)
        ax.plot(ara["rerr"].iloc[(gp == 0)], ara["obj"].iloc[(gp == 0)], "k.")
        ax.plot(ara["rerr"].iloc[(gp == 1)], ara["obj"].iloc[(gp == 1)], "kx")
        ax.set_xlabel("reactivity error")
        ax.set_ylabel("fitness")
        ax.set_title(titles[i])

    if 4 in sets:
        fig, ax = plt.subplots(2, 4, figsize = (12, 7), sharex = True, sharey = True)
        fig.suptitle(titles[i])
        ax = ax.flatten()
        colors = ["#0504aa", "red"]
        labels = [".","x"]
        for k in [0, 1]:
            for j in range(7):
                ax[j].hist(ara["x%i"%(j + 1)].iloc[(gp==k)], bins = 50, color = colors[k], rwidth=0.85,
                        density = True, alpha = 0.4, label = labels[k])
                ax[j].set_xlim([-180, 180])
                ax[j].legend()
            print(np.sum(gp == k), "in group", k)
        plt.subplots_adjust(wspace=0, hspace=0)



plt.show()
