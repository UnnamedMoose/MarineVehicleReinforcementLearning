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
import sb3_contrib

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

    saveFigs = True

    comparisonLabel = "differentAgents"
    agentSaves = {
        "SAC": "SAC_try9",
        "DDPG": "DDPG_try0",
        "TD3": "TD3_try0",
        "LSTM PPO": "RecurrentPPO_try0",
        "TQC": "TQC_try0",
    }

    # comparisonLabel = "experienceTransformation"
    # agentSaves = {
    #     "SAC": "SAC_try9",
    #     "TQC": "TQC_try0",
    #     "TQC+experience transformations": "TQC_customBuffer_try0",
    # }

    env_eval = auv.AuvEnv(**env_kwargs_evaluation)

    agents = {}
    meanRewards = {}
    allRewards = {}
    bestVersions = {}
    for name in agentSaves:
        # Evaluate each saved version and pick the best one.
        files = [f for f in os.listdir("agentData") if re.match(agentSaves[name]+"_[0-9]+.zip", f)]

        print("\n{} - evaluating {:d} versions".format(name, len(files)))

        classDict = {
            "SAC": stable_baselines3.SAC,
            "DDPG": stable_baselines3.DDPG,
            "TD3": stable_baselines3.TD3,
            "LSTM PPO": sb3_contrib.RecurrentPPO,
            "TQC": sb3_contrib.TQC,
            "TQC+experience transformations": sb3_contrib.TQC,
        }

        for i, filename in enumerate(files):
            agents[name] = classDict[name].load("./agentData/{}".format(filename))

            meanReward, allReward = resources.evaluate_agent(
                agents[name], env_eval, num_episodes=nEpisodesEval)

            if name not in meanRewards:
                meanRewards[name] = [meanReward]
            else:
                meanRewards[name] = np.append(meanRewards[name], meanReward)

            if name not in allRewards:
                allRewards[name] = allReward
                bestVersions[name] = filename
            elif meanReward > np.max(meanRewards[name]):
                allRewards[name] = allReward
                bestVersions[name] = filename

# %% Plot

    colours = plt.cm.nipy_spectral(np.linspace(0, 0.95, len(agentSaves)))

    # Compare mean rewards for all variants of each agent.
    fig, ax = plt.subplots()
    ax.set_xlabel("Agent variant")
    ax.set_ylabel("Reward")
    for i, name in enumerate(agents):
        x = np.array(range(len(meanRewards[name]))) + 1
        ds = 0.8/len(agents)
        ax.bar(x+i*ds-0.8/2, meanRewards[name], ds, align="edge", label=name, color=colours[i])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=6)
    if saveFigs:
        plt.savefig("./Figures/comparativeEvaluation_meanRewards_{}.png".format(comparisonLabel), dpi=200, bbox_inches="tight")

    # Compare rewards from the variant with the highest mean.
    fig, ax = plt.subplots()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    x = np.array(range(nEpisodesEval)) + 1
    ds = 0.8/len(agentSaves)
    rewardMax = -1e6
    rewardMin = 1e6
    for i, name in enumerate(agentSaves):
        ax.bar(x+i*ds, allRewards[name], ds, align="edge", label=name, color=colours[i])
        if np.max(allRewards[name]) > rewardMax:
            rewardMax = np.max(allRewards[name])
        if np.min(allRewards[name]) < rewardMin and np.min(allRewards[name]) > 0:
            rewardMin = np.min(allRewards[name])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=6)

    # Plot reward distributions.
    fig, ax = plt.subplots()
    ax.set_xlabel("Evaluation reward distribution")
    ax.set_ylabel("Episode count")
    bins = np.linspace(rewardMin, rewardMax, 11)
    x = (bins[1:] + bins[:-1])/2
    ds = (x[1]-x[0])*0.8/len(agentSaves)
    for i, name in enumerate(agentSaves):
        h, _ = np.histogram(allRewards[name], bins=bins)
        ax.plot(x, h, c=colours[i], lw=3, label=name)
        ax.fill_between(x, np.zeros_like(h), h, color=colours[i], alpha=0.25)

        # ax.plot(np.append(x, x[-1]+(x[-1]-x[-2])), np.append(h, 0), c=colours[i], lw=3, label=name)
        # plt.bar(x+i*ds, h, color=colours[i], alpha=1, label=name, width=ds)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=6)
    if saveFigs:
        plt.savefig("./Figures/comparativeEvaluation_rewardDist_{}.png".format(comparisonLabel), dpi=200, bbox_inches="tight")
