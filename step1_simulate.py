# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 07:43:14 2022

@author: ALidtke
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import yaml
import stable_baselines3

import verySimpleAuv as auv
import resources

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

if __name__ == "__main__":
    # An ugly fix for OpenMP conflicts in my installation.
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    # ---
    # Controls
    makeAnimation = True

    modelName = "SAC_try7"

    env_kwargs = {
        "currentVelScale": 1.,
        "currentTurbScale": 2.,
    }
    # ---

    # Create the environment and load the best model to-date.
    env_eval = auv.AuvEnv(**env_kwargs)
    model = stable_baselines3.SAC.load("./bestModel/{}".format(modelName))

    # Load the hyperparamters as well for demonstration purposes.
    with open("./bestModel/{}_hyperparameters.yaml".format(modelName), "r") as outf:
        hyperparameters = yaml.safe_load(outf)

    # Load the convergence history of the model for demonstration purposes.
    convergence = pandas.read_csv("./bestModel/{}_monitor.csv".format(modelName), skiprows=1)

    # Plot model convergence history.
    fig, ax = plt.subplots()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.plot(convergence.index, convergence["r"], ".", ms=1, alpha=0.2, c="r", zorder=-100)
    ax.plot(convergence.index, convergence.rolling(200).mean()["r"], "-", c="r", lw=2)

    # Evaluate for a large number of episodes to test robustness.
    print("\nRL agent")
    mean_reward, allRewards = resources.evaluate_agent(
        model, env_eval, num_episodes=100)

    # Dumb agent.
    print("\nSimple control")
    env_eval_pd = auv.AuvEnv(**env_kwargs)
    pdController = auv.PDController(env_eval_pd.dt)
    mean_reward_pd, allRewards_pd = resources.evaluate_agent(
        pdController, env_eval_pd, num_episodes=100, saveDir="testEpisodes")

    # Evaluate once with fixed initial conditions.
    print("\nLike-for-like comparison")
    resources.evaluate_agent(model, env_eval, num_episodes=1,
                             init=[[-0.5, -0.5], 0.785, 1.57])
    resources.evaluate_agent(pdController, env_eval_pd, num_episodes=1,
                             init=[[-0.5, -0.5], 0.785, 1.57])
    resources.plotEpisode(env_eval, "RL control fixed init")
    fig, ax = resources.plotEpisode(env_eval_pd, "Simple control fixed init")

    # Compare stats.
    fig, ax = plt.subplots()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    x = np.array(range(len(allRewards)))
    ax.bar(x-0.4, allRewards, 0.4, align="edge", color="r", label="RL control")
    ax.bar(x, allRewards_pd, 0.4, align="edge", color="b", label="Simple control")
    xlim = ax.get_xlim()
    ax.plot(xlim, [mean_reward]*2, "r--", lw=4, alpha=0.5)
    ax.plot(xlim, [mean_reward_pd]*2, "b--", lw=4, alpha=0.5)
    ax.plot(xlim, [0]*2, "k-", lw=1)
    ax.set_xlim(xlim)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)

    # Compare detail
    resources.plotDetail([env_eval_pd, env_eval], labels=["Simple control", "RL control"])

    # Animate. Takes a long time.
    if makeAnimation:
        resources.animateEpisode(env_eval, "RL_control",
                                 flipX=True, Uinf=env_kwargs["currentVelScale"])
        resources.animateEpisode(env_eval_pd, "naive_control",
                                 flipX=True, Uinf=env_kwargs["currentVelScale"])
