import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fnames = ["hist_p6p3p1.dat","hist_p55p4p05.dat"]
wts = [[.6,.3,.1],[.55,.4,.05]]
ylims = [[140,180],[0,0.008]]
labels = [r"w$_c$=%.2f, w$_ p$=%.2f, w$_ w$=%.2f"%(wt[0], wt[1], wt[2]) for wt in wts]

columns = ["rerr", "psplit", "diffworth", "obj"] + ["x%i"%i for i in range(1, 8)]
nicenames = [r"$\hat{f}_c$ [pcm]", r"$\hat{f}_p$",r"$\hat{f}_w$ [$^\circ$]"]
grid = plt.GridSpec(1, 3)
fig = plt.gcf()
fig.set_size_inches(9.5, 3.3)

data1 = np.genfromtxt("log/" + fnames[0], delimiter = ",")
ara1 = pd.DataFrame(data1, columns  = columns)
data2 = np.genfromtxt("log/" + fnames[1], delimiter = ",")
ara2 = pd.DataFrame(data2, columns  = columns)

plt.subplot(grid[0])
#plt.ylim(ylims[1])
#plt.xlim(ylims[0])
plt.plot(ara1["diffworth"], ara1["psplit"], "k.", markersize = 1.8, alpha = .9, label = labels[0])
plt.plot(ara2["diffworth"], ara2["psplit"], "r.", markersize = 1.8, alpha = .9, label = labels[1])
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

