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

    env_eval = auv.AuvEnv(**env_kwargs_evaluation)


# %% Evaluate and plot.

    state = env_eval.reset()

    # Recall that the state looks like so:
    # newState = np.concatenate([
    #     np.array([
    #         min(1., max(-1., perr[0]/0.2)),
    #         min(1., max(-1., perr[1]/0.2)),
    #         min(1., max(-1., herr/(45./180.*np.pi))),
    #         min(1., max(-1., (herr-self.herr_o)/(2./180*np.pi))),
    #         min(1., max(-1., (perr[0]-self.perr_o[0])/0.025)),
    #         min(1., max(-1., (perr[1]-self.perr_o[1])/0.025)),
    #     ]),
    #     np.clip(velocities/[0.2, 0.2, 30./180.*np.pi], -1., 1.),
    #     np.zeros(2),  # Placeholder for additional state variables used only in CFD
    # ])

    transformations = [
    # Mirror around the origin.
    # [-1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1.,  1.]
        {
            "t_pos": [-1, -1],
            "t_head": 1.,
            "t_vel": [-1, -1, 1],
        },

    # Mirror around the y axis
    # [-1.,  1.,  1.,  1., -1.,  1., -1.,  1.,  1.,  1.,  1.]
        {
            "t_pos": [-1, 1],
            "t_head": 1.,
            "t_vel": [-1, 1, 1],
        },
    # Mirror around the x axis
    # [ 1., -1.,  1.,  1.,  1., -1.,  1., -1.,  1.,  1.,  1.]
        {
            "t_pos": [1, -1],
            "t_head": 1.,
            "t_vel": [1, -1, 1],
        }
    ]

    pos = np.array([-0.5, 0.5])
    heading = 1.
    env_eval.headingTarget = np.pi/2
    velocities = np.array([1., 1., 0.])
    state = env_eval.dataToState(pos, heading, velocities)

    vScale = 0.3
    fig, ax = plt.subplots()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", "datalim")
    ax.plot([0], [0], "ro", ms=8, mew=2, mfc="None")
    ax.plot([pos[0], pos[0]+vScale*velocities[0]],
            [pos[1], pos[1]+vScale*velocities[1]], "m-", lw=3)
    resources.plot_horizontal(ax, pos[0], pos[1], heading,
                              vehicleColour="y", alpha=1)

    for t in transformations:
        t_pos = t["t_pos"]
        t_head = t["t_head"]
        t_vel = t["t_vel"]
        state_t = env_eval.dataToState(pos*t_pos, heading*t_head, velocities*t_vel)
        print(list(np.nan_to_num(state_t/state, nan=1)))

        ax.plot([pos[0]*t_pos[0], pos[0]*t_pos[0]+vScale*velocities[0]*t_vel[0]],
                [pos[1]*t_pos[1], pos[1]*t_pos[1]+vScale*velocities[1]*t_vel[1]], "m-", lw=3)
        resources.plot_horizontal(ax, pos[0]*t_pos[0], pos[1]*t_pos[1], heading*t_head,
                                  vehicleColour="orange", alpha=1)

    plt.savefig("./Figures/stateTransformations_xy.png", dpi=200, bbox_inches="tight")

# %% Heading transformation
    # Flip the heading error sign (mirror about target heading)

    # [1., 1., -1., 1., 1., 1., 1., 1., -1., 1., 1.]

    pos = [0, 0]

    fig, ax = plt.subplots()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", "datalim")
    ax.plot([0], [0], "ro", ms=8, mew=2, mfc="None")
    resources.plot_horizontal(ax, pos[0], pos[1], heading,
                              vehicleColour="y", alpha=1)

    xyHeading = 0.3*np.cos(env_eval.headingTarget), 0.3*np.sin(env_eval.headingTarget)
    ax.plot(np.array([0, xyHeading[0]])+pos[0],
            np.array([0, xyHeading[1]])+pos[1], "b-", lw=4, alpha=1)

    head = env_eval.headingTarget + (env_eval.headingTarget - heading)

    state_t = env_eval.dataToState(pos, head, velocities)
    print(list(np.nan_to_num(state_t/state, nan=1)))

    resources.plot_horizontal(ax, pos[0], pos[1], head,
                              vehicleColour="orange", alpha=1)

    plt.savefig("./Figures/stateTransformations_psi.png", dpi=200, bbox_inches="tight")