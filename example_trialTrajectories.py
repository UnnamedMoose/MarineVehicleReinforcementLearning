from scipy.interpolate import CubicSpline
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas
import scipy.integrate

from dynamicsModel_BlueROV2_Heavy_6DoF import BlueROV2Heavy6DoF, BlueROV2Heavy6DoF_PID_controller
import resources

# TODO
# Try Pseudo-Random Binary Sequences (PRBS)
# Try Orthogonal Multisine Signals
# Try Adjoint-Based Optimal Design for Multibody Systems - Targeting rich dynamics (e.g., 3D multibody), one maximizes the Fisher information
# Try https://arxiv.org/abs/2409.13088?utm_source=chatgpt.com
# Try https://arxiv.org/abs/2501.16625?utm_source=chatgpt.com
# Orthogonal multisine signals	Multiple sinusoids structured so that each multisine input is uncorrelated with others—enabling simultaneous multi-input excitation with minimal cross-correlation.
# Harmonic square waves	Square-wave versions of harmonic sinusoids, preserving orthogonality under specific frequency relationships—great for discrete actuators.
# Walsh functions / Rademacher functions	Sets of ±1 orthogonal functions used to represent signals in a digital-friendly, non-sinusoidal basis.
# Polynomial orthogonal bases	Inputs or outputs expanded into orthogonal polynomial series (Chebyshev, Legendre, etc.), aiding theoretical tractability in model fitting.
# Wavelet-based excitation design Crafting maneuvers using wavelet transforms to control time–frequency characteristics in a compact, flexible way.
# TODO

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

saveFigs = False

saveData = False
dataLabel = "N_51_v0"

# DOFs = ["X", "Y", "Z", "PHI", "THETA", "PSI"]
DOFs = ["x", "y", "z", "phi", "theta", "psi"]
velocityNames = ["u", "v", "w", "p", "q", "r"]
forceNames = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
massNames = ["mx", "my", "mz", "Ix", "Iy", "Iz"]

ds = 1.
dsFine = ds/15.
nPts = 51

dofRange = dict(zip(DOFs, [0.2, 0.2, 0.2, 25./180.*np.pi, 25./180.*np.pi, np.pi]))

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

# Common for plotting individual DOFs.
order = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
scales = [1., 1., 1., 180./np.pi, 180./np.pi, 180./np.pi]
units = ["m", "m", "m", "deg", "deg", "deg"]

# Plot the generated trajectory.
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
plt.subplots_adjust(top=0.937, bottom=0.091, left=0.108, right=0.98, hspace=0.405, wspace=0.304)
axes[-1, 0].set_xlabel("s")
axes[-1, 1].set_xlabel("s")
for i, dof in enumerate(DOFs):
    ax = axes[order[i][0], order[i][1]]
    ax.set_ylabel("{} [{}]".format(dof, units[i]))
    lns = ax.plot(s, ys[i]*scales[i], "o", color="r", ms=6, label="Generated points")
    lns += ax.plot(snew, ynews[i]*scales[i], "k-", lw=2, zorder=-10, label="Smooth trajectory")
fig.legend(lns, [l.get_label() for l in lns], ncol=2, loc="upper center")
if saveFigs:
    plt.savefig("./Figures/randomTrajectory_input.png", dpi=200, bbox_inches="tight")

# Plot individual DoFs
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
plt.subplots_adjust(top=0.937, bottom=0.091, left=0.108, right=0.98, hspace=0.405, wspace=0.304)
axes[-1, 0].set_xlabel("Time [s]")
axes[-1, 1].set_xlabel("Time [s]")
for i, dof in enumerate(DOFs):
    ax = axes[order[i][0], order[i][1]]
    ax.set_ylabel("{} [{}]".format(dof, units[i]))
    lns = ax.plot(history["t"], history[dof]*scales[i], "-", lw=2, c="r", label="Result")
    lns += ax.plot(history["t"], history[dof+"_d"]*scales[i], "--", lw=1, c="k", label="Demand")
fig.legend(lns, [l.get_label() for l in lns], ncol=2, loc="upper center")
if saveFigs:
    plt.savefig("./Figures/randomTrajectory_result_detailed.png", dpi=200, bbox_inches="tight")

# Plot trajectory in 3D.
fig = plt.figure(figsize=(8, 8))
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
if saveFigs:
    plt.savefig("./Figures/randomTrajectory_result.png", dpi=200, bbox_inches="tight")

if saveData:
    history.to_csv("tempData/randomWaypoints_{}.csv".format(dataLabel), index=False)

plt.show()


"""
# Python script to generate 6-DOF sinusoidal trajectories that respect per-DOF displacement & velocity limits.
# - 6 DOF order: [surge, sway, heave, roll, pitch, yaw]
# - translations in meters, rotations in radians (option to use degrees when inputting limits)
# - analytical derivatives (velocity and acceleration) are provided
# - supports two modes:
#     * 'single': excite one DOF at a time (useful for isolated identification)
#     * 'combined': excite all DOFs simultaneously using a grid or random combinations
# - saves trajectories and metadata to CSV (download link provided at the end of execution)
#
# Usage examples are at the bottom of this cell.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

def _ensure_array(x, n=6, name="value"):
    x = np.asarray(x, dtype=float)
    if x.size == 1:
        return np.full(n, float(x))
    if x.size != n:
        raise ValueError(f"{name} must be scalar or length-{n}. Got length {x.size}.")
    return x

def generate_sinusoid_6dof(duration=60.0, dt=0.01,
                           max_disp=None, max_vel=None,
                           amp_fractions=None, freq_specs=None,
                           mode='single', # 'single' or 'combined'
                           n_combined=20,
                           phases=None,
                           rotations_in_degrees=False):
    # """
    # Generate sinusoidal 6-DOF trajectories.
    #
    # Parameters
    # ----------
    # duration : float
    #     Total time (s).
    # dt : float
    #     Time step (s).
    # max_disp : scalar or array-like (6,)
    #     Maximum allowed displacement for each DOF. For rotations, interpreted as radians unless rotations_in_degrees=True.
    #     Order: [surge, sway, heave, roll, pitch, yaw]
    # max_vel : scalar or array-like (6,)
    #     Maximum allowed velocity magnitude for each DOF. For rotations, rad/s unless rotations_in_degrees=True.
    # amp_fractions : scalar or array-like, or list of fractions
    #     If scalar or length-6, interpreted as fraction of max_disp to use as amplitude for **combined** mode (or per DOF in single).
    #     If a list, used as the amplitude fractions to sweep in 'single' mode (e.g. [0.25, 0.5, 1.0]).
    # freq_specs : dict or None
    #     If provided, should contain either:
    #       - 'freqs': explicit list/array of frequencies (Hz) to use (will be clipped to feasible values based on max_vel)
    #       - OR 'f_max': scalar or length-6 giving maximum frequency to consider for each DOF (Hz)
    #     If None, frequencies are chosen based on max_vel and amplitude to ensure A*2*pi*f <= max_vel.
    # mode : str
    #     'single' => generate trajectories that excite one DOF at a time (useful for isolated identification)
    #     'combined' => generate trajectories that excite all DOFs simultaneously (grid or random)
    # n_combined : int
    #     Number of combined trajectories (used for 'combined' mode when freq/amp lists are not full grid)
    # phases : None or array-like
    #     If provided, phases in radians for the 6 DOFs. If None, randomized per trial.
    # rotations_in_degrees : bool
    #     If True, interprets rotational entries of max_disp and max_vel in degrees; converts to radians internally.
    #
    # Returns
    # -------
    # trials : list of dict
    #     Each dict contains 't', 'pos' (6xN), 'vel' (6xN), 'acc' (6xN), 'meta' (dictionary with amplitude, freq, phase, mode, dof_index)
    # """
    # default limits if not provided (safe small values)
    if max_disp is None:
        max_disp = np.array([0.5, 0.5, 0.5, np.deg2rad(5), np.deg2rad(5), np.deg2rad(10)])
    if max_vel is None:
        max_vel = np.array([0.5, 0.5, 0.5, np.deg2rad(5), np.deg2rad(5), np.deg2rad(10)])

    max_disp = _ensure_array(max_disp, 6, "max_disp")
    max_vel  = _ensure_array(max_vel, 6, "max_vel")

    if rotations_in_degrees:
        # convert rotational entries (indices 3,4,5) from degrees to radians
        max_disp[3:] = np.deg2rad(max_disp[3:])
        max_vel[3:]  = np.deg2rad(max_vel[3:])

    t = np.arange(0.0, duration + dt/2, dt)
    omega = 2.0 * np.pi

    # prepare amplitude fractions
    if amp_fractions is None:
        amp_fractions = [0.5, 1.0]  # default sweep fractions for 'single' mode
    if np.asarray(amp_fractions).ndim == 0 or np.asarray(amp_fractions).size == 6:
        # if scalar or per-DOF array provided, expand into list for combined mode
        amp_fracs_arr = _ensure_array(amp_fractions, 6, "amp_fractions") if np.asarray(amp_fractions).size == 6 else None
    else:
        amp_fracs_arr = None  # we'll use list of fractions in sweep

    # Helper to compute safe frequencies for a given amplitude A and max_vel per DOF
    def feasible_f_max(A, maxv):
        # want A * 2*pi*f <= maxv  => f <= maxv / (2*pi*A)
        # if A==0 => infinite f is allowed but meaningless; set f_max to 0
        f_max = np.zeros_like(A)
        for i, (Ai, mv) in enumerate(zip(A, maxv)):
            if Ai <= 0:
                f_max[i] = 0.0
            else:
                f_max[i] = float(mv) / (omega * float(Ai))
        return f_max

    trials = []

    if mode == 'single':
        # For each DOF, sweep through amplitude fractions and frequencies
        amp_frac_list = amp_fractions if (isinstance(amp_fractions, (list,tuple,np.ndarray)) and np.asarray(amp_fractions).ndim==1) else [0.25,0.5,1.0]
        for dof in range(6):
            for frac in amp_frac_list:
                A = np.zeros(6)
                A[dof] = frac * max_disp[dof]
                f_max = feasible_f_max(A, max_vel)
                # select some frequencies to try (0.1 to f_max for that DOF, clipped)
                if f_max[dof] <= 0:
                    freqs = np.array([0.0])
                else:
                    # choose up to 3 frequencies spaced in log space (including a low and near-maximum)
                    freqs = np.unique(np.clip(np.array([0.1, 0.5*f_max[dof], 0.9*f_max[dof]]), 0.0, f_max[dof]))
                for f in freqs:
                    phi = np.random.uniform(0, 2*np.pi) if phases is None else phases[dof] if np.asarray(phases).size==6 else phases
                    # build trajectory
                    pos = np.zeros((6, t.size))
                    vel = np.zeros_like(pos)
                    acc = np.zeros_like(pos)
                    if f > 0:
                        pos[dof] = A[dof] * np.sin(omega * f * t + phi)
                        vel[dof] = A[dof] * omega * f * np.cos(omega * f * t + phi)
                        acc[dof] = -A[dof] * (omega * f)**2 * np.sin(omega * f * t + phi)
                    else:
                        # zero-motion case (safe fallback)
                        pos[dof] = 0.0
                        vel[dof] = 0.0
                        acc[dof] = 0.0
                    meta = {'mode':'single', 'dof':dof, 'amp':A.copy(), 'freq':np.array([f]), 'phase':phi}
                    trials.append({'t':t, 'pos':pos, 'vel':vel, 'acc':acc, 'meta':meta})

    elif mode == 'combined':
        # For combined excitation, either use explicit amplitude fractions per DOF or sample random combinations
        if amp_fracs_arr is not None:
            A = amp_fracs_arr * max_disp
            # compute feasible frequencies per DOF
            f_max = feasible_f_max(A, max_vel)
            if freq_specs and 'freqs' in freq_specs:
                # clip given freqs by feasible f_max (per DOF)
                freqs = np.asarray(freq_specs['freqs'], dtype=float)
                # create per-DOF freq choices (clip)
                freq_choices = [np.clip(freqs, 0, f_max[i]) for i in range(6)]
                # form grid across DOFs might explode; we'll sample combinations uniformly at random
                combos = []
                import itertools
                all_choices = [list(fc) for fc in freq_choices]
                max_combos = np.prod([len(c) if len(c)>0 else 1 for c in all_choices])
                if max_combos <= n_combined:
                    # take full grid
                    for combo in itertools.product(*all_choices):
                        combos.append(combo)
                else:
                    # sample random combinations
                    for _ in range(n_combined):
                        combo = [np.random.choice(c) if len(c)>0 else 0.0 for c in all_choices]
                        combos.append(tuple(combo))
                for combo in combos:
                    phases_local = np.random.uniform(0, 2*np.pi, size=6) if phases is None else (np.asarray(phases) if np.asarray(phases).size==6 else phases)
                    pos = np.zeros((6, t.size))
                    vel = np.zeros_like(pos)
                    acc = np.zeros_like(pos)
                    for i in range(6):
                        fi = float(combo[i])
                        Ai = A[i]
                        if fi > 0 and Ai != 0:
                            pos[i] = Ai * np.sin(omega * fi * t + phases_local[i])
                            vel[i] = Ai * omega * fi * np.cos(omega * fi * t + phases_local[i])
                            acc[i] = -Ai * (omega * fi)**2 * np.sin(omega * fi * t + phases_local[i])
                    meta = {'mode':'combined', 'amp':A.copy(), 'freq':np.array(combo), 'phase':phases_local.copy()}
                    trials.append({'t':t, 'pos':pos, 'vel':vel, 'acc':acc, 'meta':meta})
        else:
            # amp_fractions provided as list (sweep); generate n_combined random combos choosing amplitude fraction per DOF and frequency per DOF based on feasibility
            amp_frac_list = np.asarray(amp_fractions)
            for _ in range(n_combined):
                frac_choice = np.random.choice(amp_frac_list, size=6)
                A = frac_choice * max_disp
                f_max = feasible_f_max(A, max_vel)
                # choose fi uniformly in [0, f_max] for each DOF
                freqs = np.array([np.random.uniform(0, f_max[i]) if f_max[i]>0 else 0.0 for i in range(6)])
                phases_local = np.random.uniform(0, 2*np.pi, size=6) if phases is None else (np.asarray(phases) if np.asarray(phases).size==6 else phases)
                pos = np.zeros((6, t.size))
                vel = np.zeros_like(pos)
                acc = np.zeros_like(pos)
                for i in range(6):
                    fi = float(freqs[i])
                    Ai = A[i]
                    if fi > 0 and Ai != 0:
                        pos[i] = Ai * np.sin(omega * fi * t + phases_local[i])
                        vel[i] = Ai * omega * fi * np.cos(omega * fi * t + phases_local[i])
                        acc[i] = -Ai * (omega * fi)**2 * np.sin(omega * fi * t + phases_local[i])
                meta = {'mode':'combined', 'amp':A.copy(), 'freq':freqs.copy(), 'phase':phases_local.copy()}
                trials.append({'t':t, 'pos':pos, 'vel':vel, 'acc':acc, 'meta':meta})
    else:
        raise ValueError("mode must be 'single' or 'combined'")

    return trials

def save_trial_to_csv(trial, filename):
    # """
    # Save a single trial to CSV with columns:
    #   time, pos_surge, pos_sway, pos_heave, pos_roll, pos_pitch, pos_yaw,
    #         vel_surge, vel_sway, ..., acc_yaw
    # """
    t = trial['t']
    pos = trial['pos']
    vel = trial['vel']
    acc = trial['acc']
    cols = {
        'time': t
    }
    names = ['surge','sway','heave','roll','pitch','yaw']
    for i,name in enumerate(names):
        cols[f'pos_{name}'] = pos[i]
    for i,name in enumerate(names):
        cols[f'vel_{name}'] = vel[i]
    for i,name in enumerate(names):
        cols[f'acc_{name}'] = acc[i]
    df = pd.DataFrame(cols)
    df.to_csv(filename, index=False)
    return filename

# ----------------------- Example run & plotting -----------------------
# Define limits (translations in m, rotations in degrees for convenience)
max_disp = [0.6, 0.4, 0.3, 10.0, 8.0, 20.0]  # last three in degrees (roll,pitch,yaw)
max_vel  = [0.6, 0.4, 0.3, 4.0, 3.0, 6.0]    # last three in deg/s

trials = generate_sinusoid_6dof(duration=30.0, dt=0.02,
                                max_disp=max_disp, max_vel=max_vel,
                                amp_fractions=[0.25, 0.5, 1.0],
                                mode='single',
                                rotations_in_degrees=True)

print(f"Generated {len(trials)} trials (single-DOF sweeps). Showing first trial metadata:")
print(trials[0]['meta'])

# Plot position, velocity, acceleration for first trial as demonstration (6 lines per plot)
names = ['surge','sway','heave','roll','pitch','yaw']

def plot_multi(trial, quantity, title):
    plt.figure(figsize=(10,4))
    t = trial['t']
    data = trial[quantity]
    for i in range(6):
        plt.plot(t, data[i], label=names[i])
    plt.xlabel('time [s]')
    plt.ylabel({'pos':'position','vel':'velocity','acc':'acceleration'}[quantity])
    plt.title(title)
    plt.legend(loc='upper right', ncol=3)
    plt.grid(True)
    plt.tight_layout()
    # Do not set explicit colors (left to matplotlib's defaults)
    # Show will be handled by the notebook environment

# Show plots for the first trial
plot_multi(trials[0], 'pos', 'Positions (trial 0)')
plot_multi(trials[0], 'vel', 'Velocities (trial 0)')
plot_multi(trials[0], 'acc', 'Accelerations (trial 0)')

# Save first trial to CSV
outdir = '/mnt/data/6dof_trials'
os.makedirs(outdir, exist_ok=True)
csv_path = os.path.join(outdir, 'trial_0.csv')
save_trial_to_csv(trials[0], csv_path)
print(f"Saved trial 0 to: {csv_path}")

# Also save metadata summary for all trials
meta_records = []
for idx, tr in enumerate(trials):
    m = tr['meta']
    rec = {'trial_index': idx, 'mode': m.get('mode',''), 'dof': m.get('dof',None),
           'amp': np.array(m.get('amp')).tolist(), 'freq': np.array(m.get('freq')).tolist(),
           'phase': np.array(m.get('phase')).tolist() if m.get('phase') is not None else None}
    meta_records.append(rec)
meta_df = pd.DataFrame(meta_records)
meta_csv = os.path.join(outdir, 'meta_summary.csv')
meta_df.to_csv(meta_csv, index=False)
print(f"Saved metadata summary to: {meta_csv}")

# Display the dataframe for the metadata (helpful to inspect)
import ace_tools as tools; tools.display_dataframe_to_user("6DOF Trials Metadata", meta_df)
"""
