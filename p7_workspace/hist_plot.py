import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

fnames = ["hist_p33p33p33.dat",
          "hist_p5p3p2.dat",
          "hist_p5p4p1.dat",
          "hist_p6p3p1.dat",
          "hist_p55p4p05.dat"]
wts = [[.33,.33,.33],[.5,.3,.2],[.5,.4,.1], [.6,.3,.1],[.55,.4,.05]]
gh = [0,3,4]
fnames = [fnames[l] for l in gh]
wts = [wts[l] for l in gh]
ylims = [[0,25],[0,0.001],[0, 50]]
nbins = 22

fig, ax = plt.subplots(3,3, figsize = (8,7), sharey = "row", sharex = "row")

colmns = ["rerr", "psplit", "diffworth", "obj"] + ["x%i"%i for i in range(1, 8)]
nicenames = [r"$\hat{f}_c$ [pcm]", r"$\hat{f}_p$",r"$\hat{f}_w$ [pcm/$^\circ$]"]
colors = ["k", "r", "g"]
for i, wt in enumerate(wts):
    data = np.genfromtxt("log/" + fnames[i], delimiter = ",")
    ara = pd.DataFrame(data, columns = colmns)
    for j, n in enumerate(colmns[:3]):
        bbs = np.linspace(*ylims[j], nbins)
        ax[j,i].hist(ara[n], bins = bbs, color = colors[j], rwidth = 1, edgecolor = "k",
                density = True, orientation="horizontal", linewidth = 1, alpha = .5)
        plt.setp(ax[j,i].get_xticklabels(), fontsize="small")
        plt.setp(ax[j,i].get_yticklabels(), fontsize="small")
    ax[i, 0].set_ylabel(nicenames[i], rotation = 0, labelpad = 12, ha = "right")
    ax[0, i].set_title(r"w$_c$=%.2f, w$_ p$=%.2f, w$_ w$=%.2f"%(wt[0], wt[1], wt[2]), fontsize = 10)
    ax[2, i].set_xlabel("density")

    #ax[i,0].set_ylim(ylims[i])

plt.tight_layout()

fig.subplots_adjust(wspace = .15)
plt.show()
