# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:24:37 2023

@author: alidtke
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import re
import platform
import yaml

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

# %% Load
trainings = {
    # Baseline agent with nothing special.
    # "Random init.": "SAC_try8",

    # 2 dummy state vars for CFD integration
    "Random init long state": "SAC_try9",

    # Restart tests.
    # "For restart": "SAC_try8_forRestart",
    # "Restart": "SAC_try8_restart",
    # "Restart no replay buffer": "SAC_try8_restart_noReplayBuffer",

    # Own pretraining - see separate branch.
    # "Pretrained actor":
    #     "SAC_customInit_try0_copy_LR_5e-4_targetEntropy_-4_actionNoise_0.05",
    # "Pretrained critic":
    #     "SAC_customInit_try1_copyCritic_LR_5e-4_targetEntropy_-4_actionNoise_0.05",
    # "Pretrained actor and critic":
    #     "SAC_customInit_try1_copyActorCritic_LR_5e-4_targetEntropy_-4_actionNoise_0.05",

    # SBL pretraining.
    "Pretrained from PID": "SAC_sblPretrain_try0_fromPID",
}

colours = plt.cm.nipy_spectral(np.linspace(0., 0.95, len(trainings)))

data = {}
for t in trainings:
    files = [f for f in os.listdir("agentData") if re.match(trainings[t]+"_[0-9]+.monitor.csv", f)
             or re.match(trainings[t]+"_[0-9]+.zip.monitor.csv", f)]
    for i, f in enumerate(files):
        df = pandas.read_csv(os.path.join("agentData", f), skiprows=1)
        data["{} {:d}".format(t, i)] = df[["r", "l"]].rolling(200).mean().dropna()

# %% Plot.
fig, ax = plt.subplots(1, 2, sharex=True, figsize=(14, 8))
plt.subplots_adjust(top=0.87, bottom=0.12, left=0.1, right=0.98, wspace=0.211)
# ax[0].set_xlim((0, 6000))
# ax[1].set_xlim((0, 6000))
ax[0].set_ylim((-100, 700))
ax[0].set_xlabel("Episode")
ax[1].set_xlabel("Episode")
ax[0].set_ylabel("Moving average of reward")
ax[1].set_ylabel("Moving everage of episode length")
ax[0].grid(axis="y", linestyle="dashed")
ax[1].grid(axis="y", linestyle="dashed")

lns = []
for i, t in enumerate(trainings):
    kBest = None
    for k in data:
        if t not in k:
            continue
        elif kBest is None:
            kBest = k
        ln = ax[0].plot(data[k]["r"], c=colours[i], label=t)
        if np.mean(data[k]["r"].values[-50:]) > np.mean(data[kBest]["r"].values[-50:]):
            kBest = k
        ax[1].plot(data[k]["l"], c=colours[i], label=t)
    # print(t, kBest)
    # ax[0].plot(data[kBest]["r"], "b-", lw=4, alpha=0.25)
    lns += ln
fig.legend(lns, [l.get_label() for l in lns], loc="upper center", ncol=3, framealpha=1)

