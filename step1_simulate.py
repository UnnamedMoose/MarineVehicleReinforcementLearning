# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 07:43:14 2022

@author: ALidtke
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
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

    # ---
    # Controls
    makeAnimation = False
    modelName = "SAC_try5"
    # ---

    # Create the environment and load the best model to-date.
    env_eval = auv.AuvEnv()
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
    mean_reward, allRewards = resources.evaluate_agent(model, env_eval, num_episodes=100)
    fig, ax = plt.subplots()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.bar(range(len(allRewards)), allRewards, color="r")
    xlim = ax.get_xlim()
    ax.plot(xlim, [mean_reward]*2, "r--", lw=4, alpha=0.5)
    ax.plot(xlim, [0]*2, "k-", lw=1)
    ax.set_xlim(xlim)

    # Trained agent.
    print("\nSingle episode")
    mean_reward,_ = resources.evaluate_agent(model, env_eval)
    resources.plotEpisode(env_eval, "RL control")

    # Dumb agent.
    print("\nSimple control")
    env_eval_pd = auv.AuvEnv()
    pdController = auv.PDController(env_eval_pd.dt)
    mean_reward,_ = resources.evaluate_agent(pdController, env_eval_pd)
    fig, ax = resources.plotEpisode(env_eval_pd, "Simple control")

    # Compare detail
    resources.plotDetail([env_eval_pd, env_eval], labels=["Simple control", "RL control"])

    # Animate. Takes a long time.
    if makeAnimation:
        resources.animateEpisode(env_eval, "RL_control")
        resources.animateEpisode(env_eval_pd, "naive_control")
