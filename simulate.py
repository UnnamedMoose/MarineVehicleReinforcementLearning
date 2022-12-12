# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 07:43:14 2022

@author: ALidtke
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import datetime
import os
import gym
from gym.utils import seeding
import torch
import re
import stable_baselines3
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common import results_plotter
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise

import verySimpleAuv as auv

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

if __name__ == "__main__":

    saveFile = "./modelData/SAC_try1_0"

    env_eval = auv.AuvEnv()
    model = stable_baselines3.SAC.load(saveFile)

    # Evaluate for a large number of episodes to test robustness.
    mean_reward, allRewards = auv.evaluate_agent(model, env_eval, num_episodes=100)
    fig, ax = plt.subplots()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.bar(range(len(allRewards)), allRewards, color="r")
    xlim = ax.get_xlim()
    ax.plot(xlim, [mean_reward]*2, "r--", lw=4, alpha=0.5)
    ax.set_xlim(xlim)

    # Trained agent.
    print("\nAfter training")
    mean_reward,_ = auv.evaluate_agent(model, env_eval)
    auv.plotEpisode(env_eval, "RL control")

    # Dumb agent.
    print("\nSimple control")
    env_eval_pd = auv.AuvEnv()
    pdController = auv.PDController(env_eval_pd.dt)
    mean_reward,_ = auv.evaluate_agent(pdController, env_eval_pd)
    fig, ax = auv.plotEpisode(env_eval_pd, "Simple control")

    # Compare detail
    auv.plotDetail([env_eval_pd, env_eval], labels=["Simple control", "RL control"])

