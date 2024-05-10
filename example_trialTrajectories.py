from scipy.interpolate import CubicSpline
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas
import scipy.integrate

from dynamicsModel_BlueROV2_Heavy_6DoF import BlueROV2Heavy6DoF, BlueROV2Heavy6DoF_PID_controller
import resources

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

# DOFs = ["X", "Y", "Z", "PHI", "THETA", "PSI"]
DOFs = ["x", "y", "z", "phi", "theta", "psi"]
velocityNames = ["u", "v", "w", "p", "q", "r"]
forceNames = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
massNames = ["mx", "my", "mz", "Ix", "Iy", "Iz"]

ds = 1.
dsFine = ds/15.
nPts = 5

dofRange = dict(zip(DOFs, [0.2, 0.2, 0.2, 15./180.*np.pi, 15./180.*np.pi, np.pi]))

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

dfWpsCoarse = pandas.DataFrame(np.vstack([s]+ys).T, columns=["s"]+DOFs)
dfWpsFine = pandas.DataFrame(np.vstack([snew]+ynews).T, columns=["s"]+DOFs)

dfWps = dfWpsCoarse

# Create an rov object.
tMaxPerWp = 10.
dtEval = 0.05
state = np.append(dfWps.loc[0, DOFs].values, np.zeros(6))
rov = BlueROV2Heavy6DoF(BlueROV2Heavy6DoF_PID_controller(np.zeros(6)))

def wpReached(t, y):
    xPrev = dfWps.loc[iWp-1, ["x", "y", "z"]].values
    xCur = dfWps.loc[iWp, ["x", "y", "z"]].values
    return np.linalg.norm(xCur - y[:3]) - min(0.02, np.linalg.norm(xCur-xPrev)/2.)

wpReached.terminal = True
wpReached.direction = 0

history = None
iWp = 0
while iWp < dfWps.shape[0]-1:
    # Set the new waypoint.
    iWp += 1
    print(iWp, dfWps.shape[0])
    rov.controller.setPoint = dfWps.loc[iWp, DOFs].values

    # Solve until WP reached or time elapsed.
    result_solve_ivp = scipy.integrate.solve_ivp(
        rov.derivs, (0, tMaxPerWp), state, method='RK45', events=wpReached,
        t_eval=np.arange(0, tMaxPerWp+dtEval/2., dtEval), rtol=1e-3, atol=1e-3)

    # Store the data.
    df = pandas.DataFrame(
        data=np.hstack([result_solve_ivp.t[:, np.newaxis], result_solve_ivp.y.T]),
        columns=["t"] + DOFs + velocityNames)
    df["iWP"] = iWp

    # Store the set point.
    for i, t in enumerate(DOFs):
        df[t+"_d"] = rov.controller.setPoint[i]

    # Prepare fields to be computed at each time step.
    for i, t in enumerate(DOFs):
        df["GCF_"+t] = 0.
    for i in range(8):
        df["rpm_{:d}".format(i)] = 0.
    df[forceNames] = 0.
    df[massNames] = 0.

    # Clear past data.
    rov.controller.reset()

    # Loop over all stored time steps.
    for iTime in df.index:
        pos = df.loc[iTime, ["x", "y", "z"]].values
        angles = df.loc[iTime, ["phi", "theta", "psi"]].values
        vel = df.loc[iTime, velocityNames].values

        # Set the vehicle axes.
        rov.updateMovingCoordSystem(angles)

        # Generalised forces and moments from the controller in the global coordinates.
        rov.generalisedControlForces = rov.controller.computeControlForces(**df.loc[iTime, DOFs + ["t"]])
        for i, t in enumerate(DOFs):
            df.loc[i, "GCF_"+t] = rov.generalisedControlForces[i]

        # Go from forces to rpm.
        cv = rov.allocateThrust()
        for i in range(8):
            df.loc[iTime, "rpm_{:d}".format(i)] = cv[i]

        # Call the force model for the current state.
        M, RHS = rov.forceModel(pos, angles, vel, cv)
        df.loc[iTime, forceNames] = RHS
        df.loc[iTime, massNames] = M[range(6), range(6)]

    # Store.
    if history is None:
        history = df.copy()
    else:
        df["t"] += history["t"].max()
        history = pandas.concat([history, df], ignore_index=True)

    # Grab the initial state for the next WP.
    state = result_solve_ivp.y[:, -1]

# Plot the generated trajectory.
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
plt.subplots_adjust(top=0.937, bottom=0.091, left=0.108, right=0.98, hspace=0.405, wspace=0.304)
axes[-1, 0].set_xlabel("s")
axes[-1, 1].set_xlabel("s")
order = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
scales = [1., 1., 1., 180./np.pi, 180./np.pi, 180./np.pi]
for i, dof in enumerate(DOFs):
    ax = axes[order[i][0], order[i][1]]
    ax.set_ylabel(dof)
    lns = ax.plot(s, ys[i]*scales[i], "o", color="r", ms=6, label="Generated points")
    lns += ax.plot(snew, ynews[i]*scales[i], "k-", lw=2, zorder=-10, label="Smooth trajectory")
fig.legend(lns, [l.get_label() for l in lns], ncol=2, loc="upper center")

# Plot individual DoFs
fig, ax = plt.subplots()
colours = plt.cm.Set2(np.linspace(0.1, 0.9, 6))
lns = []
for i, t in enumerate(DOFs):
    lns += ax.plot(history["t"], history[t], "-", lw=2, c=colours[i], label=t)
    ax.plot(history["t"], history[t+"_d"], "--", lw=1, c=colours[i])

ax.legend(lns, [l.get_label() for l in lns], loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)
plt.show()

# Plot trajectory in 3D.
fig  = plt.figure(figsize=(8, 8))
plt.subplots_adjust(top=0.973, bottom=0.034, left=0.03, right=0.97)
ax = fig.add_subplot(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_aspect("equal")
lim = dofRange["x"]*1.1
ax.set_xlim((-lim, lim))
ax.set_ylim((-lim, lim))
ax.set_zlim((-lim, lim))
ax.invert_yaxis()  # NED and y +ve to stbd
ax.invert_zaxis()
ax.plot(ys[0], ys[1], ys[2], "o", color="r", ms=6, label="Waypoints")
ax.plot(ynews[0], ynews[1], ynews[2], "k-", lw=2, alpha=0.2, zorder=-10, label="Smooth trajectory")
ax.plot(history["x"], history["y"], history["z"], "-", lw=2, c="orange", label="ROV path")
ax.legend(ncol=3, loc="upper center")
for i in dfWpsCoarse.index:
    rov.updateMovingCoordSystem(dfWpsCoarse.loc[i, ["phi", "theta", "psi"]].values)
    resources.plotCoordSystem(ax, rov.iHat, rov.jHat, rov.kHat, x0=dfWpsCoarse.loc[i, ["x", "y", "z"]].values,
        ds=dofRange["x"]/4., ls="--")

plt.show()
