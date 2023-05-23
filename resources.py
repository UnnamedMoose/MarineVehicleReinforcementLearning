# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:46:39 2023

@author: ALidtke
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import matplotlib.animation as animation
import datetime
import shutil
import pandas
import yaml

# Configure the coordinate system. Either use directions like they would appear
# on a map (consistent with manoeuvring theory) or like they appear in traditional
# Cartesian coordinate systems (e.g. inside a CFD code).
# orientation = "north_east_clockwise"
orientation = "right_up_anticlockwise"


def evaluate_agent(agent, env, num_episodes=1, num_steps=None, deterministic=True,
                   num_last_for_reward=None, render=False, init=None, saveDir=None):
    """
    Evaluate a RL agent
    :param agent: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    frames = []

    keepHistory = False
    if saveDir is not None:
        os.makedirs(saveDir, exist_ok=True)
        keepHistory = True

    # This function will only work for a single Environment
    all_episode_rewards = []
    for iEp in range(num_episodes):
        episode_rewards = []
        done = False

        obs = env.reset(fixedInitialValues=init, keepTimeHistory=keepHistory)
        if num_steps is None:
            num_steps = 1000000
        for i in range(num_steps):
            # _states are only useful when using LSTM policies
            action, _states = agent.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            if render:
                frames.append(env.render(mode="rgb_array"))
            if done:
                if saveDir is not None:
                    env.timeHistory.to_csv(os.path.join(
                        saveDir, "ep_{:d}.csv".format(iEp)), index=False)
                break

        if num_last_for_reward is None:
            all_episode_rewards.append(sum(episode_rewards))
        else:
            all_episode_rewards.append(np.mean(episode_rewards[-num_last_for_reward:], 1))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("  Mean reward:", mean_episode_reward)
    print("  Num episodes:", num_episodes)

    if render:
        return frames, mean_episode_reward, all_episode_rewards
    else:
        return mean_episode_reward, all_episode_rewards


def plot_horizontal(ax, x, y, psi, scale=1, markerSize=1, arrowSize=1, vehicleColour="y", alpha=0.5):
    """ Plot a representation of the AUV on the given axes. """

    x0 = np.array([x, y])

    Length = 0.457 # m (overall dimensions of the AUV)
    Width = 0.338
    # Only used for plotting now.
    D_thruster = 0.1  # Diameter of thrusters

    xyHull = np.array([
        [Length/2., -Width/2.+D_thruster],
        [Length/2., Width/2.-D_thruster],
        [Length/2.-D_thruster, Width/2.],
        [-Length/2.+D_thruster, Width/2.],
        [-Length/2., Width/2.-D_thruster],
        [-Length/2., -Width/2.+D_thruster],
        [-Length/2.+D_thruster, -Width/2.],
        [Length/2.-D_thruster, -Width/2.],
        [Length/2., -Width/2.+D_thruster],
    ])

    xyCentreline = np.array([
        [np.min(xyHull[:,0]), 0.],
        [np.max(xyHull[:,0]), 0.],
    ])

    xyDir = np.array([
        [Length/2.-Width/4., -Width/4.],
        [Length/2., 0.],
        [Length/2.-Width/4., Width/4.],
    ])

    def rotate(xy, psi):
        xyn = np.zeros(xy.shape)
        xyn[:,0] = np.cos(psi)*xy[:,0] - np.sin(psi)*xy[:,1]
        xyn[:,1] = np.sin(psi)*xy[:,0] + np.cos(psi)*xy[:,1]
        return xyn

    xyHull = rotate(xyHull*scale, psi) + x0
    xyCentreline = rotate(xyCentreline*scale, psi) + x0
    xyDir = rotate(xyDir*scale, psi) + x0

    if orientation == "north_east_clockwise":
        i0 = 1
        i1 = 0
    elif orientation == "right_up_anticlockwise":
        i0 = 0
        i1 = 1
    else:
        raise ValueError("Wrong orientation")

    objects = []
    objects += ax.fill(xyHull[:,i0], xyHull[:,i1], vehicleColour, alpha=alpha)
    objects += ax.plot(xyCentreline[:,i0], xyCentreline[:,i1], "k--", lw=2*markerSize)
    objects += ax.plot(xyDir[:,i0], xyDir[:,i1], "k-", lw=2*markerSize)
    objects += ax.plot(x0[i0], x0[i1], "ko", mew=3, mfc="None", ms=14*markerSize)

    return objects


def plotEpisode(env_to_plot, title=""):

    # TODO this shares a lot of code with animateEpisode. Could just use the latter

    fig, ax = plt.subplots(figsize=(10, 10))

    if orientation == "north_east_clockwise":
        ax.set_xlabel("y [m, +ve east]")
        ax.set_ylabel("x [m, +ve north]")
        i0 = 1
        i1 = 0
    elif orientation == "right_up_anticlockwise":
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        i0 = 0
        i1 = 1
    else:
        raise ValueError("Wrong orientation")

    ax.set_xlim(np.array(env_to_plot.xMinMax)+[-0.2, 0.2])
    ax.set_ylim(np.array(env_to_plot.yMinMax)+[-0.2, 0.2])
    ax.set_aspect("equal", "box")

    lns = []

    lns += ax.plot([env_to_plot.positionStart[i0], env_to_plot.positionTarget[i0]],
                    [env_to_plot.positionStart[i1], env_to_plot.positionTarget[i1]],
                    "go--", lw=2, ms=8, mew=2, mec="g", mfc="None", label="$Waypoints$")

    xyHeading = 0.5*np.cos(env_to_plot.headingTarget), 0.5*np.sin(env_to_plot.headingTarget)
    ax.plot([0, xyHeading[i0]], [0, xyHeading[i1]], "g-", lw=4, alpha=0.5)

    lns += ax.plot(env_to_plot.timeHistory[["x", "y"]].values[0, i0], env_to_plot.timeHistory[["x", "y"]].values[0, i1],
            "bs", ms=11, mew=2, mec="b", mfc="None", label="$Start$")

    lns += ax.plot(env_to_plot.timeHistory[["x", "y"]].values[-1, i0], env_to_plot.timeHistory[["x", "y"]].values[-1, i1],
            "bd", ms=11, mew=2, mec="b", mfc="None", label="$End$")

    lns += ax.plot(env_to_plot.timeHistory[["x", "y"]].values[:, i0], env_to_plot.timeHistory[["x", "y"]].values[:, i1], "k-",
        mec="k", lw=2, mew=2, ms=9, mfc="None", label="$Trajectory$")

    ax.legend(lns, [l.get_label() for l in lns], loc="lower center",
                    bbox_to_anchor=(0.5, 1.01), prop={"size":18}, ncol=4)

    tPlot = np.linspace(0, env_to_plot.timeHistory["time"].max(), 5)
    iToPlot = [np.argmin(np.abs(env_to_plot.timeHistory["time"] - t)) for t in tPlot]

    for i in iToPlot:
        plot_horizontal(ax, env_to_plot.timeHistory["x"].values[i],
                        env_to_plot.timeHistory["y"].values[i],
                        env_to_plot.timeHistory["psi"].values[i],
                        markerSize=0.5, arrowSize=1, scale=1)

    ax.text(ax.get_xlim()[1], ax.get_ylim()[0], "  "+title, va="bottom", ha="right", size=18)

    return fig, ax


def animateEpisode(env_plot, caseName, flipX=False, Uinf=1.):

    # Plot contours (and animate)
    if flipX:
        xm = -1.
    else:
        xm = 1.

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 9))

    if orientation == "north_east_clockwise":
        ax.set_xlabel("y [m, +ve east]")
        ax.set_ylabel("x [m, +ve north]")
        i0 = 1
        i1 = 0
    elif orientation == "right_up_anticlockwise":
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        i0 = 0
        i1 = 1
    else:
        raise ValueError("Wrong orientation")

    ax.set_xlim(np.array(env_plot.xMinMax)+[-0.2, 0.2])
    ax.set_ylim(np.array(env_plot.yMinMax)+[-0.2, 0.2])
    ax.set_aspect("equal", "box")

    # Plot constant elements
    lns = []

    lns += ax.plot([env_plot.positionTarget[i0]*xm],
                    [env_plot.positionTarget[i1]],
                    "go--", lw=2, ms=8, mew=2, mec="g", mfc="None", label="$Waypoint$")

    xyHeading = 0.5*np.cos(env_plot.headingTarget), 0.5*np.sin(env_plot.headingTarget)
    lns += ax.plot([0, xyHeading[i0]*xm], [0, xyHeading[i1]], "g-", lw=4, alpha=0.5, label="Target heading")

    lns += ax.plot(env_plot.timeHistory[["x", "y"]].values[0, i0]*xm, env_plot.timeHistory[["x", "y"]].values[0, i1],
            "bs", ms=11, mew=2, mec="b", mfc="None", label="$Start$")

    lns += ax.plot(env_plot.timeHistory[["x", "y"]].values[-1, i0]*xm, env_plot.timeHistory[["x", "y"]].values[-1, i1],
            "bd", ms=11, mew=2, mec="b", mfc="None", label="$End$")

    # Plot the flow field.
    iField = 0
    levels = np.sort(np.linspace(0.75*Uinf*xm, 1.25*Uinf*xm, 11))
    cs = ax.contourf(env_plot.flow.coords[:, :, 0]*xm, env_plot.flow.coords[:, :, 1],
                     env_plot.flow.interpField(env_plot.timeHistory.loc[0, "time"])[:, :, iField]*xm,
                     levels=levels, extend="both", cmap=plt.cm.PuOr_r, zorder=-100, alpha=0.5)

    # Plot the AUV outline.
    position = fig.add_axes([0.85, .25, 0.03, 0.5])
    cbar = fig.colorbar(cs, cax=position, orientation="vertical")
    cbar.set_label("u [m/s]")
    plt.subplots_adjust(right=0.8)
    auvObjects = plot_horizontal(ax, env_plot.timeHistory["x"].values[0]*xm,
                                 env_plot.timeHistory["y"].values[0],
                                 env_plot.timeHistory["psi"].values[0],
                                 markerSize=0.5, arrowSize=1, scale=1, alpha=0.8)

    # Plot vehicle and current speeds.
    arrLen = 0.5
    maxU = np.linalg.norm(env_plot.timeHistory[["u", "v"]].values, axis=1).max()
    maxUc = np.linalg.norm(env_plot.timeHistory[["u_current", "v_current"]].values, axis=1).max()

    arr_v = ax.arrow(
        env_plot.timeHistory[["x", "y"]].values[0, i0]*xm,
        env_plot.timeHistory[["x", "y"]].values[0, i1],
        env_plot.timeHistory[["u", "v"]].values[0, i0]*arrLen/maxU*xm,
        env_plot.timeHistory[["u", "v"]].values[0, i1]*arrLen/maxU,
        length_includes_head=True, zorder=100,
        width=0.01, edgecolor="None", facecolor="red")
    arr_v_dummy = ax.plot([0, 0], [0, 1], "r-", lw=2, label="V$_{AUV}$")[0]
    lns.append(arr_v_dummy)
    arr_v_dummy.remove()

    arr_c = ax.arrow(
        env_plot.timeHistory[["x", "y"]].values[0, i0]*xm,
        env_plot.timeHistory[["x", "y"]].values[0, i1],
        env_plot.timeHistory[["u_current", "v_current"]].values[0, i0]*arrLen/maxUc*xm,
        env_plot.timeHistory[["u_current", "v_current"]].values[0, i1]*arrLen/maxUc,
        length_includes_head=True, zorder=100,
        width=0.01, edgecolor="None", facecolor="magenta")
    arr_c_dummy = ax.plot([0, 0], [0, 1], "m-", lw=2, label="V$_{flow}$")[0]
    lns.append(arr_c_dummy)
    arr_c_dummy.remove()

    # Plot the entire trajectory
    lns += ax.plot(env_plot.timeHistory[["x", "y"]].values[:, i0]*xm, env_plot.timeHistory[["x", "y"]].values[:, i1], "k-",
        mec="k", lw=2, mew=2, ms=9, mfc="None", label="$Trajectory$")

    # Add the legend.
    ax.legend(lns, [l.get_label() for l in lns], loc="lower center",
                    bbox_to_anchor=(0.5, 1.01), prop={"size":18}, ncol=4)

    class AuvPlot(object):
        def __init__(self, cs, auvObjects, arr_c, arr_v):
            self.cs = cs
            self.auvObjects = auvObjects
            self.arr_c = arr_c
            self.arr_v = arr_v

        def animate(self, i):
            # global cs, auvObjects, arr_c, arr_v
            # removes only the contours, leaves the rest intact
            for c in self.cs.collections:
                c.remove()
            for c in self.auvObjects:
                c.remove()
            self.cs = ax.contourf(env_plot.flow.coords[:, :, 0]*xm, env_plot.flow.coords[:, :, 1],
                             env_plot.flow.interpField(env_plot.timeHistory.loc[i, "time"])[:, :, iField]*xm,
                             levels=levels, extend="both", cmap=plt.cm.PuOr_r, zorder=-100, alpha=0.5)

            self.auvObjects = plot_horizontal(ax, env_plot.timeHistory["x"].values[i]*xm,
                                         env_plot.timeHistory["y"].values[i],
                                         env_plot.timeHistory["psi"].values[i],
                                         markerSize=0.5, arrowSize=1, scale=1, alpha=0.8)

            self.arr_v.set_data(
                x=env_plot.timeHistory[["x", "y"]].values[i, i0]*xm,
                y=env_plot.timeHistory[["x", "y"]].values[i, i1],
                dx=env_plot.timeHistory[["u", "v"]].values[i, i0]*arrLen/maxU*xm,
                dy=env_plot.timeHistory[["u", "v"]].values[i, i1]*arrLen/maxU)
            self.arr_c.set_data(
                x=env_plot.timeHistory[["x", "y"]].values[i, i0]*xm,
                y=env_plot.timeHistory[["x", "y"]].values[i, i1],
                dx=env_plot.timeHistory[["u_current", "v_current"]].values[i, i0]*arrLen/maxUc*xm,
                dy=env_plot.timeHistory[["u_current", "v_current"]].values[i, i1]*arrLen/maxU)

            return self.cs, self.auvObjects

    auvPlot = AuvPlot(cs, auvObjects, arr_c, arr_v)

    anim = animation.FuncAnimation(fig, auvPlot.animate, repeat=False, frames=env_plot.timeHistory.shape[0])
    writer = animation.PillowWriter(fps=20)
    anim.save('./episodeAnim_{}.gif'.format(caseName), writer=writer, dpi=75)


def plotDetail(envs_to_plot, labels=None, title=""):
    fields = ["x", "y", "psi"]
    flabels = ["x [m]", "y [m]", "$\psi$ [deg]"]
    try:
        nEnvs = len(envs_to_plot)
    except TypeError:
        nEnvs = 1
        envs_to_plot = [envs_to_plot]
    if labels is None:
        labels = ["Env{:d}".format(i) for i in range(nEnvs)]
    caseColours = plt.cm.jet(np.linspace(0, 1, nEnvs))
    mults = [1., 1., 180./np.pi]
    for i, f in enumerate(fields):
        fig, ax = plt.subplots()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(flabels[i])
        if f == "psi":
            ax.set_ylim((0, 360.))
            ax.yaxis.set_ticks(np.arange(0, 361, 60))
        for iEnv in range(nEnvs):
            ax.plot(envs_to_plot[iEnv].timeHistory["time"].values,
                    envs_to_plot[iEnv].timeHistory[f].values*mults[i],
                    "-", lw=2, c=caseColours[iEnv], label=labels[iEnv])
            ax.plot(envs_to_plot[iEnv].timeHistory["time"].values,
                    envs_to_plot[iEnv].timeHistory[f+"_d"].values*mults[i],
                    "--", lw=1, c=caseColours[iEnv])
        if nEnvs > 1:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)

    flabs = ["Reward", "u$_{current}$ [m/s]", "v$_{current}$ [m/s]"]
    for i, f in enumerate(["reward", "u_current", "v_current"]):
        fig, ax = plt.subplots()
        ax.set_xlabel("Time step")
        ax.set_ylabel(flabs[i])
        for iEnv in range(nEnvs):
            ax.plot(envs_to_plot[iEnv].timeHistory[f], "-", lw=2,
                    c=caseColours[iEnv], label=labels[iEnv])

    for f in envs_to_plot[0].timeHistory.keys():
        if re.match("a[0-9]+", f) or re.match("s[0-9]+", f) or re.match("r[0-9]+", f):
            fig, ax = plt.subplots()
            ax.set_xlabel("Time step")
            ax.set_ylabel(f.replace("a", "Action ").replace("s", "State ").replace("r", "Reward "))
            for iEnv in range(nEnvs):
                ax.plot(envs_to_plot[iEnv].timeHistory[f], "-", lw=2,
                        c=caseColours[iEnv], label=labels[iEnv])
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)

# %% Functions for training.

def trainAgent(agent, nTrainingSteps, saveFile, log_interval=100000000, progress_bar=True):

    # os.mkdirs(logDir, exist_ok=True)

    # Train the agent for N steps
    starttime = datetime.datetime.now()
    print("\nTraining started at", str(starttime))
    agent.learn(total_timesteps=nTrainingSteps, log_interval=log_interval, progress_bar=progress_bar)
    endtime = datetime.datetime.now()
    trainingTime = (endtime-starttime).total_seconds()
    print("Training took {:.0f} seconds ({:.0f} minutes)".format(trainingTime, trainingTime/60.))

    # Save.
    agent.save(saveFile)

    # Retain convergence info and agent.
    convergenceData = pandas.read_csv(saveFile+".monitor.csv", skiprows=1)

    print("Final reward {:.2f}".format(convergenceData.rolling(200).mean()["r"].values[-1]))

    return convergenceData, trainingTime

def plotTraining(convHistories, saveAs=None):
    try:
        convHistories[0]
    except TypeError:
        convHistories = [convHistories]
    nAgents = len(convHistories)

    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(14, 7))
    plt.subplots_adjust(top=0.91, bottom=0.12, left=0.1, right=0.98, wspace=0.211)
    colours = plt.cm.plasma(np.linspace(0, 0.9, nAgents))
    lns = []
    iBest = 0
    rewardBest = -1e6
    for iAgent, convergence in enumerate(convHistories):
        rol = convergence.rolling(200).mean()
        if rol["r"].values[-1] > rewardBest:
            iBest = iAgent
            rewardBest = rol["r"].values[-1]
        for i, f in enumerate(["r", "l"]):
            ax[i].set_xlabel("Episode")
            ax[i].set_ylabel(f.replace("r", "Reward").replace("l", "Episode length"))
            ax[i].plot(convergence.index, convergence[f], ".", ms=4, alpha=0.5, c=colours[iAgent], zorder=-100)
            ln, = ax[i].plot(convergence.index, rol[f], "-", c=colours[iAgent], lw=2)
            if i == 0:
                lns.append(ln)
    ax[0].set_ylim((max(ax[0].get_ylim()[0], -1500), ax[0].get_ylim()[1]))
    fig.legend(lns, ["M{:d}".format(iAgent) for iAgent in range(nAgents)],
               loc="upper center", ncol=10)
    if saveAs is not None:
        plt.savefig(saveAs, dpi=200, bbox_inches="tight")

    return iBest, fig, ax

def saveHyperparameteres(agentName, agent_kwargs, policy_kwargs, env_kwargs, nTrainingSteps, trainingTimes, nProc):
    with open("./agentData/{}_hyperparameters.yaml".format(agentName), "w") as outf:
        try:
            trainingTimes[0]
        except TypeError:
            trainingTimes = [trainingTimes]

        data = {
            "agentName": agentName,
            "agent_kwargs": agent_kwargs.copy(),
            "policy_kwargs": policy_kwargs.copy(),
            "env_kwargs": env_kwargs.copy(),
            "nTrainingSteps": nTrainingSteps,
            "trainingTime": [float(t) for t in trainingTimes],
            "nProc": nProc,
        }
        # Change noise to human-readable format.
        try:
            data["agent_kwargs"]["action_noise"] = {
                "mu": [float(v) for v in data["agent_kwargs"]["action_noise"].noises[0]._mu],
                "sigma": [float(v) for v in data["agent_kwargs"]["action_noise"].noises[0]._sigma],
                }
        except KeyError:
            pass
        # Convert types because yaml is yaml.
        data["policy_kwargs"]["activation_fn"] = str(policy_kwargs["activation_fn"])
        data["agent_kwargs"]["train_freq"] = list(agent_kwargs["train_freq"])
        # Write.
        yaml.dump(data, outf, default_flow_style=False)
