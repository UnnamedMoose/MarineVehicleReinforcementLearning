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

baseCaseName = "SAC_try7_performanceCheck"
files = [f for f in os.listdir("agentData/performanceCheck") if re.match(baseCaseName+"_[0-9]+_hyperparameters.yaml", f)]

times = []
for f in files:
    with open(os.path.join("./agentData/performanceCheck", f), "r") as inf:
        hyperparameters = yaml.safe_load(inf)
    times.append([hyperparameters["nProc"], hyperparameters["trainingTime"]])
times = np.array(times)
iSort = np.argsort(times[:, 0])
times = times[iSort, :]

fig, ax = plt.subplots()
ax.set_xlabel("No. processors")
ax.set_ylabel("Average training time [s]")
ax.plot(times[:, 0], times[:, 1], "ro--", lw=2, ms=7)
ax.xaxis.set_ticks(times[:, 0])
plt.savefig("Figures/scalingTest.png", dpi=200, bbox_inches="tight")
