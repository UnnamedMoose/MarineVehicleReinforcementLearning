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

def computeThrustAllocation(thrusterPositions, thrusterNormals, x0=None):
    if x0 is None:
        x0 = np.zeros(3)

    # Use pseudo-inverse of the control allocation matrix in order to go from
    # desired generalised forces to actuator demands in rpm.
    # NOTE the original 3 DoF notation is inconsistent with page 48 in Wu (2018),
    # what follows is (or should be) the same. See Figure 4.2 and Eq. 4.62 in their work.
    A = np.zeros((6, thrusterPositions.shape[0]))
    for i in range(thrusterPositions.shape[0]):
        A[:, i] = np.append(
            thrusterNormals[i, :],
            np.cross(thrusterPositions[i, :]-x0, thrusterNormals[i, :])
        )
    Ainv = np.linalg.pinv(A)

    return A, Ainv

def plotCoordSystem(ax, iHat, jHat, kHat, x0=np.zeros(3), ds=0.45, ls="-"):
    x1 = x0 + iHat*ds
    x2 = x0 + jHat*ds
    x3 = x0 + kHat*ds
    lns = ax.plot([x0[0], x1[0]], [x0[1], x1[1]], [x0[2], x1[2]], "r", ls=ls, lw=2)
    lns += ax.plot([x0[0], x2[0]], [x0[1], x2[1]], [x0[2], x2[2]], "g", ls=ls, lw=2)
    lns += ax.plot([x0[0], x3[0]], [x0[1], x3[1]], [x0[2], x3[2]], "b", ls=ls, lw=2)
    return lns

def plotIvpRes6dof(result, comp="disp"):
    fig, ax1 = plt.subplots()
    colours = plt.cm.gist_rainbow(np.linspace(0.1, 0.9, 6))
    ax1.set_xlim((0, result.t[-1]))
    ax2 = ax1.twinx()
    ax1.set_xlabel("Time [s]")
    if comp == "disp":
        ax1.set_ylabel("Linear displacement [m]")
        ax2.set_ylabel("Angular displacement [deg]")
        names = ["x", "y", "z", "$\phi$", "$\\theta$", "$\psi$"]
        di = 0
    elif comp == "vel":
        ax1.set_ylabel("Linear velocity [m]")
        ax2.set_ylabel("Angular velocity [deg/s]")
        names = ["u", "v", "w", "p", "q", "r"]
        di = 6
    else:
        raise ValueError("what are we plotting, exactly?")
    plt.subplots_adjust(top=0.85, bottom=0.137, left=0.144, right=0.896)
    lns = []
    for i in range(3):
        lns += ax1.plot(result.t, result.y[i+di, :], c=colours[i], label=names[i])
    for i in range(3):
        lns += ax2.plot(result.t, result.y[i+3+di, :]/np.pi*180., c=colours[i+3], label=names[i+3])
    fig.legend(lns, [l.get_label() for l in lns], loc="upper center", ncol=3)

    return fig, ax1, ax2


def angleError(psi_d, psi):
    """
    Function used for computing signed heading error that wraps around pi properly.

    Parameters
    ----------
    psi_d : float
        Target heading in radians <0, 2pi).
    psi : float
        Current heading in radians <0, 2pi).

    Returns
    -------
    diff : float
        Signed difference in radians <-pi, pi).

    """
    a = (psi_d - psi) % (2.*np.pi)
    b = (psi - psi_d) % (2.*np.pi)
    diff = a if a < b else -b
    return diff


def coordinateTransform(phi, theta, psi, dof=["x", "y", "psi"]):
    """ Return a coordinate transform matrix given the active DoF and the
    required roll, pitch and yaw angles. """

    if type(dof) is int:
        if dof == 3:
            dof = ["x", "y", "psi"]
        elif dof == 6:
            dof = ["x", "y", "z", "phi", "theta", "psi"]

    if set(dof) == set(["x", "y", "psi"]):
        coordTransform = np.array([
            [np.cos(psi), -np.sin(psi), 0.],
            [np.sin(psi), np.cos(psi), 0.],
            [0., 0., 1.],
        ])

    elif set(dof) == set(["x", "y", "z", "phi", "theta", "psi"]):
        cosThetaDenom = np.cos(theta)
        if np.abs(cosThetaDenom) < 1e-12:
            cosThetaDenom = 1e-6
        elif np.abs(cosThetaDenom) < 1e-6:
            cosThetaDenom = 1e-6 * np.sign(cosThetaDenom)

        J1 = np.array([
            [np.cos(psi)*np.cos(theta), -np.sin(psi)*np.cos(phi) + np.cos(psi)*np.sin(theta)*np.sin(phi), np.sin(psi)*np.sin(phi) + np.cos(psi)*np.sin(theta)*np.sin(phi)],
            [np.sin(psi)*np.cos(theta), np.cos(psi)*np.cos(phi) + np.sin(psi)*np.sin(theta)*np.sin(phi), -np.cos(psi)*np.sin(phi) + np.sin(psi)*np.sin(theta)*np.cos(phi)],
            [-np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi)],
        ])

        J2 = np.array([
            [1., np.sin(phi)*np.sin(theta)/cosThetaDenom, np.cos(phi)*np.sin(theta)/cosThetaDenom],
            [0., np.cos(phi), -np.sin(phi)],
            [0., np.sin(phi)/cosThetaDenom, np.cos(phi)/cosThetaDenom],
        ])

        coordTransform = np.array([
            [J1[0,0], J1[0,1], J1[0,2], 0., 0., 0.],
            [J1[1,0], J1[1,1], J1[1,2], 0., 0., 0.],
            [J1[2,0], J1[2,1], J1[2,2], 0., 0., 0.],
            [0., 0., 0., J2[0,0], J2[0,1], J2[0,2]],
            [0., 0., 0., J2[1,0], J2[1,1], J2[1,2]],
            [0., 0., 0., J2[2,0], J2[2,1], J2[2,2]],
        ])

    return coordTransform

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
    median_episode_reward = np.median(all_episode_rewards)
    print("  Mean reward:  ", mean_episode_reward)
    print("  Median reward:", median_episode_reward)
    print("  Num episodes: ", num_episodes)

    if render:
        return frames, mean_episode_reward, median_episode_reward, all_episode_rewards
    else:
        return mean_episode_reward, median_episode_reward, all_episode_rewards

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
        try:
            data["agent_kwargs"]["train_freq"] = list(agent_kwargs["train_freq"])
        except KeyError:
            pass
        # Write.
        yaml.dump(data, outf, default_flow_style=False)
