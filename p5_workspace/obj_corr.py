import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fnames = ["hist_p5p4p1.dat", "hist_p6p2p1.dat"]
wts = [[.5,.4,.1],[.6,.2,.1]]
ylims = [[140,180],[0,0.008]]
labels = [r"w$_c$=%.2f, w$_ p$=%.2f, w$_ d$=%.2f"%(wt[0], wt[1], wt[2]) for wt in wts]

columns = ["rerr", "psplit", "tdist", "obj"] + ["x%i"%i for i in range(1, 8)]
nicenames = [r"$\hat{f}_c$ [pcm]", r"$\hat{f}_p$",r"$\hat{f}_d$ [$^\circ$]"]
grid = plt.GridSpec(1, 3)
fig = plt.gcf()
fig.set_size_inches(9.5, 3.3)

data1 = np.genfromtxt("log/" + fnames[0], delimiter = ",")
ara1 = pd.DataFrame(data1, columns  = columns)
data2 = np.genfromtxt("log/" + fnames[1], delimiter = ",")
ara2 = pd.DataFrame(data2, columns  = columns)

plt.subplot(grid[0])
plt.ylim(ylims[1])
plt.xlim(ylims[0])
#plt.plot(ara1["rerr"], ara1["psplit"], "k.", markersize = 1.2, alpha = .8, label = labels[0])
#plt.plot(ara2["rerr"], ara2["psplit"], "r.", markersize = 1.2, alpha = .8, label = labels[1])
plt.plot(ara1["tdist"], ara1["psplit"], "k.", markersize = 1.8, alpha = .9, label = labels[0])
plt.plot(ara2["tdist"], ara2["psplit"], "r.", markersize = 1.8, alpha = .9, label = labels[1])
plt.xlabel(nicenames[2])
plt.ylabel(nicenames[1])



plt.subplot(grid[1:])
x_pos = np.arange(7)
x1_means = np.abs(ara1[columns[-7:]]).mean(0)
x1_2sd = np.abs(ara1[columns[-7:]]).std(0)
x2_means = np.abs(ara2[columns[-7:]]).mean(0)
x2_2sd = np.abs(ara2[columns[-7:]]).std(0)
width = 0.35
plt.bar(x_pos, x1_means, width, label = labels[0], yerr = x1_2sd, capsize = 8, color = "k", alpha = .7)
plt.bar(x_pos+width, x2_means, width, label = labels[1], yerr = x2_2sd, capsize = 8, color = "r", alpha = .7)
plt.legend()
#plt.xticks(x_pos + width/2, [r"|$\theta_1$|"] + [r"|$\theta_%i$|"%t for t in range(3, 9)])
plt.xticks(x_pos + width/2, [r"Drum 1"] + [r"Drum %i"%t for t in range(3, 9)])
plt.ylim([0, 230])
plt.ylabel(r"|$\theta$| [$^\circ$]")

plt.tight_layout()


fig, ax = plt.subplots(2, 4, figsize = (10, 6))
ax = ax.flatten()

for i in range(1, 8):
    ax[i].hist(ara1["x" + str(i)], color = "k", alpha = .5)
    ax[i].hist(ara2["x" + str(i)], color = "r", alpha = .5)

plt.show()

exit()
data1 = np.genfromtxt("log/" + fnames[0], delimiter = ",")
ara1 = pd.DataFrame(data1, columns  = columns)
data2 = np.genfromtxt("log/" + fnames[1], delimiter = ",")
ara2 = pd.DataFrame(data1, columns  = columns)
exit()

fig, ax = plt.subplots(3,3, figsize = (8,7), sharey = "row", sharex = "row")

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

