# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 07:48:17 2023

@author: alidtke
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import re
import platform
import stable_baselines3

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

import verySimpleAuv as auv
import resources

# %% Set up, load and evaluate
if __name__ == "__main__":
    # An ugly fix for OpenMP conflicts in my installation.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    nEpisodesEval = 100

    env_kwargs_evaluation = {
        "currentVelScale": 1.,
        "currentTurbScale": 2.,
        "noiseMagActuation": 0.,
        "noiseMagCoeffs": 0.,
    }

    agentSaves = {
        "SAC": "SAC_try9_0",
        "DDPG": "DDPG_try0_1",
        "TD3": "TD3_try0_0",
    }

    env_eval = auv.AuvEnv(**env_kwargs_evaluation)

    agents = {}
    for name in agentSaves:
        if name == "SAC":
            agents[name] = stable_baselines3.SAC.load("./agentData/{}".format(agentSaves[name]))
        elif name == "DDPG":
            agents[name] = stable_baselines3.DDPG.load("./agentData/{}".format(agentSaves[name]))
        elif name == "TD3":
            agents[name] = stable_baselines3.TD3.load("./agentData/{}".format(agentSaves[name]))
        else:
            raise ValueError("Agent {} not set up".format(name))

    meanRewards = {}
    allRewards = {}
    for name in agents:
        print("\n"+name)
        meanRewards[name], allRewards[name] = resources.evaluate_agent(
            agents[name], env_eval, num_episodes=nEpisodesEval)

# %% Plot

    colours = plt.cm.nipy_spectral(np.linspace(0, 0.95, len(agentSaves)))

    # Compare mean rewards.
    fig, ax = plt.subplots()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    x = np.array(range(nEpisodesEval))
    ds = 0.8/len(agentSaves)
    rewardMax = -1e6
    rewardMin = 1e6
    for i, name in enumerate(agentSaves):
        ax.bar(x+i*ds, allRewards[name], ds, align="edge", label=name, color=colours[i])
        if np.max(allRewards[name]) > rewardMax:
            rewardMax = np.max(allRewards[name])
        if np.min(allRewards[name]) < rewardMin:
            rewardMin = np.min(allRewards[name])
    xlim = ax.get_xlim()
    for i, name in enumerate(agentSaves):
        ax.plot(xlim, [meanRewards[name]]*2, "--", c=colours[i], lw=4, alpha=0.5)
    ax.plot(xlim, [0]*2, "k-", lw=1)
    ax.set_xlim(xlim)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4)

    # Plot reward distributions.
    fig, ax = plt.subplots()
    ax.set_xlabel("Evaluation reward distribution")
    ax.set_ylabel("Episode count")
    bins = np.linspace(rewardMin, rewardMax, 21)
    x = (bins[1:] + bins[:-1])/2
    ds = (x[1]-x[0])*0.8/len(agentSaves)
    for i, name in enumerate(agentSaves):
        h, _ = np.histogram(allRewards[name], bins=bins)
        plt.bar(x+i*ds, h, color=colours[i], alpha=1, label=name, width=ds)
    ax.legend()
