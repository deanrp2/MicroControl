import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
print("no bimodal behavior observed")

fnames = ["hist_p25p45p3_g100.dat", "hist_p2p5p3_g100.dat", "hist_p3p4p3_g100.dat"]
wts = [[.25,.45,.3],[.2,.5,.3],[.3,.4,.3]]
ylims = [[0,1750],[0,0.0075],[50, 200]]
nbins = 40

fig, ax = plt.subplots(3,3, figsize = (8,7), sharey = "row")

colmns = ["rerr", "psplit", "tdist", "obj"] + ["x%i"%i for i in range(1, 8)]
nicenames = [r"$\hat{f}_c$ [pcm]", r"$\hat{f}_p$",r"$\hat{f}_d$ [$^\circ$]"]
colors = ["k", "r", "b"]
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
    ax[0, i].set_title(r"w$_c$=%.2f, w$_ p$=%.2f, w$_ d$=%.2f"%(wt[0], wt[1], wt[2]), fontsize = 10)
    ax[2, i].set_xlabel("density")

    ax[i,0].set_ylim(ylims[i])

plt.tight_layout()

fig.subplots_adjust(wspace = .15)
plt.show()

