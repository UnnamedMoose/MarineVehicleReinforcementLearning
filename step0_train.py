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
import shutil
from gym.utils import seeding
import torch
import re
import yaml
import stable_baselines3
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise

import verySimpleAuv as auv
import resources

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

if __name__ == "__main__":

    modelToRestart = None
    # modelToRestart = "SAC_try4"
    # modelToRestart = "SAC_try5_noCurrent_noTurb"
    # modelToRestart = "SAC_try5_noTurb"

    # modelName = "SAC_try5_noCurrent_noTurb"
    # modelName = "SAC_try5_noTurb"
    # modelName = "SAC_try6_LR_5e-3_gamma_0.95_batchSize_256"
    modelName = "SAC_try6"

    # No. parallel processes.
    nProc = 16
    # Do everything N times to rule out random successes and failures.
    nModels = 3

    nTrainingSteps = 3_000_000

    model_kwargs = {
        'learning_rate': 5e-3,
        'gamma': 0.95,
        'verbose': 1,
        'buffer_size': (128*2)*512,
        "use_sde_at_warmup": True,
        'batch_size': 256,
        'learning_starts': 256,
        'train_freq': (1, "step"),
        # "action_noise": VectorizedActionNoise(NormalActionNoise(
        #     np.zeros(3), 0.1*np.ones(3)), nProc)
    }
    policy_kwargs = {
        "use_sde": False,
        "activation_fn": torch.nn.GELU,
        "net_arch": dict(
            # Actor - determines action for a specific state
            pi=[128, 128],
            # Critic - estimates value of each state-action combination
            qf=[128, 128],
        )
    }
    env_kwargs = {
        # Set to zero to disable the flow - much faster training.
        "currentVelScale": 1.,
        "currentTurbScale": 2.,
        # Use noise in coefficients for training only.
        "noiseMagActuation": 0.1,
        "noiseMagCoeffs": 0.1,
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
        env = SubprocVecEnv([auv.make_env(i, env_kwargs=env_kwargs) for i in range(nProc)])
        env = VecMonitor(env, logDir)

        # Create the model using stable baselines.
        if modelToRestart is None:
            model = stable_baselines3.SAC(
                "MlpPolicy", env, policy_kwargs=policy_kwargs, **model_kwargs)
        else:
            model = stable_baselines3.SAC.load("./bestModel/{}".format(modelToRestart))
            model.set_env(env)

        # Train the agent for N steps
        starttime = datetime.datetime.now()
        print("\nTraining of model", iModel, "started at", str(starttime))
        model.learn(total_timesteps=nTrainingSteps, log_interval=100000000, progress_bar=True)
        endtime = datetime.datetime.now()
        trainingTime = (endtime-starttime).total_seconds()
        print("Training took {:.0f} seconds ({:.0f} minutes)".format(trainingTime, trainingTime/60.))

        # Save.
        model.save(saveFile)

        # Move the monitor to make it easier to find.
        shutil.copyfile(os.path.join(logDir, "monitor.csv"), saveFile+"_monitor.csv")

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
            "env_kwargs": env_kwargs.copy(),
            "nTrainingSteps": nTrainingSteps,
        }
        # Change noise to human-readable format.
        try:
            data["model_kwargs"]["action_noise"] = {
                "mu": [float(v) for v in data["model_kwargs"]["action_noise"].noises[0]._mu],
                "sigma": [float(v) for v in data["model_kwargs"]["action_noise"].noises[0]._sigma],
                }
        except KeyError:
            pass
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
