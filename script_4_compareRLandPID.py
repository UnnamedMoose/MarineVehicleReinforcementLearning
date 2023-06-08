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

# %% Set up and load
if __name__ == "__main__":
    # An ugly fix for OpenMP conflicts in my installation.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    env_kwargs_evaluation = {
        "currentVelScale": 1.,
        "currentTurbScale": 2.,
        "noiseMagActuation": 0.,
        "noiseMagCoeffs": 0.,
    }

    agentName = "SAC_try9_0"

    env_eval = auv.AuvEnv(**env_kwargs_evaluation)

    agents = {
        "SAC": stable_baselines3.SAC.load("./agentData/{}".format(agentName)),
        "PID": auv.PDController(env_eval.dt),
    }

# %% Evaluate and plot.

    colours = plt.cm.nipy_spectral(np.linspace(0, 0.95, len(agents)))

    # nEpisodesEval = 100

    # meanRewards = {}
    # allRewards = {}
    # for name in agents:
    #     print("\n"+name)
    #     meanRewards[name], allRewards[name] = resources.evaluate_agent(
    #         agents[name], env_eval, num_episodes=nEpisodesEval)

    # # Compare mean rewards.
    # fig, ax = plt.subplots()
    # ax.set_xlabel("Episode")
    # ax.set_ylabel("Reward")
    # x = np.array(range(nEpisodesEval))
    # ds = 0.8/len(agents)
    # rewardMax = -1e6
    # rewardMin = 1e6
    # for i, name in enumerate(agents):
    #     ax.bar(x+i*ds, allRewards[name], ds, align="edge", label=name, color=colours[i])
    #     if np.max(allRewards[name]) > rewardMax:
    #         rewardMax = np.max(allRewards[name])
    #     if np.min(allRewards[name]) < rewardMin:
    #         rewardMin = np.min(allRewards[name])
    # xlim = ax.get_xlim()
    # for i, name in enumerate(agents):
    #     ax.plot(xlim, [meanRewards[name]]*2, "--", c=colours[i], lw=4, alpha=0.5)
    # ax.plot(xlim, [0]*2, "k-", lw=1)
    # ax.set_xlim(xlim)
    # ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4)

    # # Plot reward distributions.
    # fig, ax = plt.subplots()
    # ax.set_xlabel("Evaluation reward distribution")
    # ax.set_ylabel("Episode count")
    # bins = np.linspace(rewardMin, rewardMax, 21)
    # x = (bins[1:] + bins[:-1])/2
    # ds = (x[1]-x[0])*0.8/len(agents)
    # for i, name in enumerate(agents):
    #     h, _ = np.histogram(allRewards[name], bins=bins)
    #     plt.bar(x+i*ds, h, color=colours[i], alpha=1, label=name, width=ds)
    # ax.legend()

# %% Manufacture an episode by applying known displacements.

    actions = {}
    for name in agents:
        agent = agents[name]
        actions[name] = []

        state = env_eval.reset()

        env_eval.headingTarget = 0.
        env_eval.heading = 0.
        env_eval.position = np.array([-0.5, 0.])

        states = [state]

        x = np.linspace(-0.5, 0.5, 501)
        for i in range(len(x)):
            action, _states = agent.predict(states[-1], deterministic=True)
            actions[name].append(action)
            states.append(env_eval.dataToState(np.array([x[i], 0]), env_eval.heading, np.zeros(3)))
        actions[name] = np.array(actions[name])

    fig, ax = plt.subplots()
    ax.set_xlabel("x-position error [m]")
    ax.set_ylabel("x-direction actuation")
    for i, name in enumerate(agents):
        ax.plot(x[1:], actions[name][1:, 0], c=colours[i], lw=2, label=name)
    ax.legend()

# %%

    # actions = {}
    # for name in agents:
    #     agent = agents[name]
    #     actions[name] = []

    #     state = env_eval.reset()

    #     env_eval.headingTarget = 0.
    #     env_eval.heading = -1.
    #     env_eval.position = np.zeros(2)

    #     states = [state]

    #     x = np.linspace(-1, 1, 501)
    #     for i in range(len(x)):
    #         action, _states = agent.predict(states[-1], deterministic=True)
    #         actions[name].append(action)
    #         states.append(env_eval.dataToState(np.zeros(2), x[i], np.zeros(3)))
    #     actions[name] = np.array(actions[name])

    # fig, ax = plt.subplots()
    # ax.set_xlabel("Heading [deg]")
    # ax.set_ylabel("Moment actuation")
    # for i, name in enumerate(agents):
    #     ax.plot(x[1:]/np.pi*180, actions[name][1:, 2], c=colours[i], lw=2, label=name)
    # ax.legend()