from scipy.interpolate import CubicSpline
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas
import scipy.integrate

from dynamicsModel_BlueROV2_Heavy_6DoF import BlueROV2Heavy6DoF
import resources

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

DOFs = ["X", "Y", "Z", "PHI", "THETA", "PSI"]
ds = 1.
dsFine = ds/10.
nPts = 5

dofRange = dict(zip(DOFs, [0.2, 0.2, 0.2, 5./180.*np.pi, 5./180.*np.pi, np.pi]))

s = np.arange(0, nPts*ds+ds/2., ds)
snew = np.arange(s[0], s[-1]+dsFine/2., dsFine)

ys = []
ynews = []
for i, dof in enumerate(DOFs):
    y = (np.random.rand(len(s))-0.5)/0.5 * dofRange[dof]
    spl = CubicSpline(s, y)
    ynew = spl(snew)
    ys.append(y)
    ynews.append(ynew)

dfWps = pandas.DataFrame(np.vstack([s]+ys).T, columns=["s"]+DOFs)
dfWpsFine = pandas.DataFrame(np.vstack([snew]+ynews).T, columns=["s"]+DOFs)

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

# Create an rov object.
tMaxPerWp = 10.
dtEval = 0.05
state = np.append(dfWps.loc[0, DOFs].values, np.zeros(6))
rov = BlueROV2Heavy6DoF(dfWps.loc[1, DOFs].values)

def wpReached(t, y):
    xPrev = dfWps.loc[iWp-1, ["X", "Y", "Z"]].values
    xCur = dfWps.loc[iWp, ["X", "Y", "Z"]].values
    return np.linalg.norm(xCur - y[:3]) - min(0.02, np.linalg.norm(xCur-xPrev)/2.)

wpReached.terminal = True
wpReached.direction = 0

history = []
iWp = 0
while iWp < dfWps.shape[0]-1:
    iWp += 1
    print(iWp, dfWps.shape[0])
    rov.setPoint = dfWps.loc[iWp, DOFs].values

    result_solve_ivp = scipy.integrate.solve_ivp(
        rov.derivs, (0, tMaxPerWp), state, method='RK45', events=wpReached,
        t_eval=np.arange(0, tMaxPerWp+dtEval/2., dtEval), rtol=1e-3, atol=1e-3)

    state = result_solve_ivp.y[:, -1]
    history.append(result_solve_ivp)

# Plot individual DoFs
tTot = 0
fig, ax = plt.subplots()
ax.set_xlim((0, tMax))
for result_solve_ivp in history:
    lns = []
    colours = plt.cm.Set2(np.linspace(0.1, 0.9, 6))
    for i, v in enumerate(["x", "y", "z", "phi", "theta", "psi"]):
        lns += ax.plot(result_solve_ivp.t+tTot, result_solve_ivp.y[i, :], c=colours[i], label=v)
        ax.hlines(rov.setPoint[i], result_solve_ivp.t[0]+tTot, result_solve_ivp.t[-1]+tTot,
                  color=lns[-1].get_color(), linestyle="dashed")
    tTot += result_solve_ivp.t[-1]

ax.legend(lns, [l.get_label() for l in lns], loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)

# Plot trajectory in 3D.
fig  = plt.figure(figsize=(8, 8))
plt.subplots_adjust(top=0.973, bottom=0.034, left=0.03, right=0.97)
ax = fig.add_subplot(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_aspect("equal")
lim = dofRange["X"]*1.1
ax.set_xlim((-lim, lim))
ax.set_ylim((-lim, lim))
ax.set_zlim((-lim, lim))
ax.invert_yaxis()  # NED and y +ve to stbd
ax.invert_zaxis()
ax.plot(ys[0], ys[1], ys[2], "o", color="r", ms=6, label="Generated points")
ax.plot(ynews[0], ynews[1], ynews[2], "k-", lw=2, zorder=-10, label="Smooth trajectory")
ax.legend(ncol=2, loc="upper center")
for i in dfWps.index:
    rov.updateMovingCoordSystem(dfWps.loc[i, ["PHI", "THETA", "PSI"]].values)
    resources.plotCoordSystem(ax, rov.iHat, rov.jHat, rov.kHat, x0=dfWps.loc[i, ["X", "Y", "Z"]].values,
        ds=dofRange["X"]/4., ls="--")
for result_solve_ivp in history:
    ax.plot(result_solve_ivp.y[0, :], result_solve_ivp.y[1, :], result_solve_ivp.y[2, :],
        "-", lw=2, label="ROV path")

plt.show()
