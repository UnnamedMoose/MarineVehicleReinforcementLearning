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

    saveFile = "./modelData/SAC_try0_2"

    env_eval = auv.AuvEnv()
    env_eval.reset()
    model = stable_baselines3.SAC("MlpPolicy", env_eval)
    model.load(saveFile)

    # Trained agent.
    print("\nAfter training")
    mean_reward = auv.evaluate_agent(model, env_eval)
    auv.plotEpisode(env_eval, "RL control")

    # Dumb agent.
    print("\nSimple control")
    env_eval_pd = auv.AuvEnv()
    pdController = auv.PDController(env_eval_pd.dt)
    mean_reward = auv.evaluate_agent(pdController, env_eval_pd)
    fig, ax = auv.plotEpisode(env_eval_pd, "Simple control")

    # Compare detail
    auv.plotDetail([env_eval_pd, env_eval], labels=["Simple control", "RL control"])

