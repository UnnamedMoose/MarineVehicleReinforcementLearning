from scipy.interpolate import CubicSpline
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas

import dynamicsModel_BlueROV2_Heavy_6DoF

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

DOFs = ["X", "Y", "Z", "PHI", "THETA", "PSI"]
ds = 1.
dsFine = ds/100.
nPts = 5
range_pos = 0.2
range_angle = 30./180.*np.pi

s = np.arange(0, nPts*ds+ds/2., ds)
snew = np.arange(s[0], s[-1]+dsFine/2., dsFine)

ys = []
ynews = []
for i, dof in enumerate(DOFs):
    if dof in ["X", "Y", "Z"]:
        y = (np.random.rand(len(s))-0.5)/0.5 * range_pos
    else:
        y = (np.random.rand(len(s))-0.5)/0.5 * range_angle
    spl = CubicSpline(s, y)
    ynew = spl(snew)
    ys.append(y)
    ynews.append(ynew)

df = pandas.DataFrame(np.vstack([s]+ys).T, columns=["s"]+DOFs)
dfFine = pandas.DataFrame(np.vstack([snew]+ynews).T, columns=["s"]+DOFs)

# Plot individual DOFs.
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
plt.subplots_adjust(top=0.937, bottom=0.091, left=0.108, right=0.98, hspace=0.405, wspace=0.304)
axes[-1, 0].set_xlabel("s")
axes[-1, 1].set_xlabel("s")
order = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
for i, dof in enumerate(DOFs):
    ax = axes[order[i][0], order[i][1]]
    ax.set_ylabel(dof)
    lns = ax.plot(s, ys[i], "o", color="r", ms=6, label="Generated points")
    lns += ax.plot(snew, ynews[i], "k-", lw=2, zorder=-10, label="Smooth trajectory")
fig.legend(lns, [l.get_label() for l in lns], ncol=2, loc="upper center")

# Plot trajectory in 3D.
fig  = plt.figure(figsize=(8, 8))
plt.subplots_adjust(top=0.973, bottom=0.034, left=0.03, right=0.97)
ax = fig.add_subplot(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_aspect("equal")
lim = range_pos*1.1
ax.set_xlim((-lim, lim))
ax.set_ylim((-lim, lim))
ax.set_zlim((-lim, lim))
ax.invert_yaxis()  # NED and y +ve to stbd
ax.invert_zaxis()
ax.plot(ys[0], ys[1], ys[2], "o", color="r", ms=6, label="Generated points")
ax.plot(ynews[0], ynews[1], ynews[2], "k-", lw=2, zorder=-10, label="Smooth trajectory")
ax.legend(ncol=2, loc="upper center")

plt.show()
