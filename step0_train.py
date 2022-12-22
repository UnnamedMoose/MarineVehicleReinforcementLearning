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
import yaml
import stable_baselines3
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise

import verySimpleAuv as auv

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

if __name__ == "__main__":

    modelName = "SAC_try3"

    # TODO review https://github.com/eleurent/highway-env/blob/master/highway_env/envs/parking_env.py
    #   and see if something could be used here, too.

    # TODO review "Deep Reinforcement Learning Algorithms for Ship Navigation in Restricted Waters"

    # TODO review "Path-following optimal control of autonomous underwater vehicle based on deep reinforcement learning"

    # No. parallel processes.
    nProc = 16
    # Do everything N times to rule out random successes and failures.
    nModels = 3

    # TODO adjust the hyperparameters here.
    nTrainingSteps = 3_000_000

    model_kwargs = {
        'learning_rate': 5e-4,
        'gamma': 0.95,
        'verbose': 1,
        'buffer_size': int(1e6),
        "use_sde_at_warmup": True,
        'batch_size': 2048,
        'learning_starts': 1024,
        'train_freq': (4, "step"),
        "action_noise": VectorizedActionNoise(NormalActionNoise(
            np.zeros(3), 0.1*np.ones(3)), nProc)
    }
    policy_kwargs = {
        "use_sde": True,
        # ReLU GELU Sigmoid SiLU SELU
        "activation_fn": torch.nn.GELU,
        "net_arch": dict(
            # Actor - determines action for a specific state
            pi=[64, 64],
            # Critic - estimates value of each state-action combination
            qf=[128, 128],
        )
    }

    # TODO compare weights somehow to see if some common features appear?
    # model.actor.latent_pi[0].weight.shape
    # model.critic.qf0[0].weight.shape

    # Train several times to make sure the agent doesn't just get lucky.
    convergenceData = []
    models = []
    for iModel in range(nModels):
        # Set up constants etc.
        saveFile = "./modelData/{}_{:d}".format(modelName, iModel)
        logDir = "./modelData/{}_{:d}_logs".format(modelName, iModel)
        os.makedirs(logDir, exist_ok=True)

        # Create the environments.
        env_eval = auv.AuvEnv()
        env = SubprocVecEnv([auv.make_env(i, envKwargs={}) for i in range(nProc)])
        env = VecMonitor(env, logDir)

        # Create the model using stable baselines.
        model = stable_baselines3.SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, **model_kwargs)

        # Train the agent for N steps
        starttime = datetime.datetime.now()
        print("\nTraining of model", iModel, "started at", str(starttime))
        model.learn(total_timesteps=nTrainingSteps, log_interval=100000000, progress_bar=True)
        endtime = datetime.datetime.now()
        trainingTime = (endtime-starttime).total_seconds()
        print("Training took {:.0f} seconds ({:.0f} minutes)".format(trainingTime, trainingTime/60.))

        # Save.
        model.save(saveFile)

        # Retain convergence info and model.
        convergenceData.append(pandas.read_csv(os.path.join(logDir, "monitor.csv"), skiprows=1))
        models.append(model)

        print("Final reward {:.2f}".format(convergenceData[-1].rolling(200).mean()["r"].values[-1]))

    # Save metadata in human-readable format.
    with open("./modelData/{}_hyperparameters.yaml".format(modelName), "w") as outf:
        data = {
            "modelName": modelName,
            "model_kwargs": model_kwargs.copy(),
            "policy_kwargs": policy_kwargs.copy(),
            "nTrainingSteps": nTrainingSteps,
        }
        # Change noise to human-readable format.
        data["model_kwargs"]["action_noise"] = {
            "mu": [float(v) for v in data["model_kwargs"]["action_noise"].noises[0]._mu],
            "sigma": [float(v) for v in data["model_kwargs"]["action_noise"].noises[0]._sigma],
            }
        # Convert types because yaml is yaml.
        data["policy_kwargs"]["activation_fn"] = str(policy_kwargs["activation_fn"])
        data["model_kwargs"]["train_freq"] = list(model_kwargs["train_freq"])
        # Write.
        yaml.dump(data, outf, default_flow_style=False)

    # Plot convergence of each model.
    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(14, 7))
    plt.subplots_adjust(top=0.91, bottom=0.12, left=0.1, right=0.98, wspace=0.211)
    colours = plt.cm.plasma(np.linspace(0, 1, nModels))
    lns = []
    iBest = 0
    rewardBest = -1e6
    for iModel, convergence in enumerate(convergenceData):
        rol = convergence.rolling(200).mean()
        if rol["r"].values[-1] > rewardBest:
            iBest = iModel
            rewardBest = rol["r"].values[-1]
        for i, f in enumerate(["r", "l"]):
            ax[i].set_xlabel("Episode")
            ax[i].set_ylabel(f.replace("r", "Reward").replace("l", "Episode length"))
            ax[i].plot(convergence.index, convergence[f], ".", ms=4, alpha=0.25, c=colours[iModel], zorder=-100)
            ln, = ax[i].plot(convergence.index, rol[f], "-", c=colours[iModel], lw=2)
            if i == 0:
                lns.append(ln)
    ax[0].set_ylim((max(ax[0].get_ylim()[0], -1500), ax[0].get_ylim()[1]))
    fig.legend(lns, ["M{:d}".format(iModel) for iModel in range(nModels)],
               loc="upper center", ncol=10)
    plt.savefig("./modelData/{}_convergence.png".format(modelName), dpi=200, bbox_inches="tight")

    # Pick the best model.
    model = models[iBest]

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

