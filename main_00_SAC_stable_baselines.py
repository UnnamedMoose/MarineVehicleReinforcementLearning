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
import time
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
    # An ugly fix for OpenMP conflicts in my installation.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # For saving trained agents.
    agentName = "SAC_try9"

    # Set to None to pick the best agent from the trained set. Specify as string
    # to load a particular saved model.
    agentName_eval = None
    # agentName_eval = "SAC_try8_0"

    # Top-level switches
    do_training = True
    do_evaluation = False

    # --- Training parameters ---
    loadReplayBuffer = True  # For a "perfect" restart keep this on.
    agentToRestart = None
    # agentToRestart = "SAC_try8_forRestart_0"

    # No. parallel processes.
    nProc = 16

    # Do everything N times to rule out random successes and failures.
    nAgents = 5

    # Any found agent will be left alone unless this is set to true.
    overwrite = True

    nTrainingSteps = 1_500_000
    # nTrainingSteps = 500_000
#
    agent_kwargs = {
        'learning_rate': 5e-4,
        'gamma': 0.95,
        'verbose': 1,
        'buffer_size': (128*3)*512,
        'batch_size': 256,
        'learning_starts': 256,
        'train_freq': (1, "step"),
        "gradient_steps": 1,
        "action_noise": VectorizedActionNoise(NormalActionNoise(
            np.zeros(3), 0.05*np.ones(3)), nProc),
        "use_sde_at_warmup": False,
        # "target_entropy": -4.,
        "target_entropy": "auto",
        "ent_coef": "auto_0.1",
    }
    policy_kwargs = {
        "activation_fn": torch.nn.GELU,
        "net_arch": dict(
            pi=[128, 128, 128],
            qf=[128, 128, 128],
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

    # --- Evaluation parameters ---
    makeAnimation = False

    env_kwargs_evaluation = {
        "currentVelScale": 1.,
        "currentTurbScale": 2.,
        "noiseMagActuation": 0.,
        "noiseMagCoeffs": 0.,
    }

# %% Training.
    if do_training:

        # Train several times to make sure the agent doesn't just get lucky.
        convergenceData = []
        trainingTimes = []
        agents = []
        for iAgent in range(nAgents):
            # Set up constants etc.
            saveFile = "./agentData/{}_{:d}".format(agentName, iAgent)

            if not overwrite:
                if os.path.isfile(saveFile) or os.path.isfile(saveFile+".zip"):
                    print("Skipping training of existing agent", saveFile)
                    continue

            # Create the environments.
            env_eval = auv.AuvEnv()
            env = SubprocVecEnv([auv.make_env(i, env_kwargs=env_kwargs) for i in range(nProc)])
            env = VecMonitor(env, saveFile)

            # Create the agent using stable baselines.
            if agentToRestart is None:
                agent = stable_baselines3.SAC(
                    "MlpPolicy", env, policy_kwargs=policy_kwargs, **agent_kwargs)
            else:
                agent = stable_baselines3.SAC.load("./agentData/{}".format(agentToRestart),
                                                   env=env, force_reset=False)
                if loadReplayBuffer:
                    agent.load_replay_buffer("./agentData/{}_replayBuffer".format(agentToRestart))

            # Train the agent for N steps
            conv, trainingTime = resources.trainAgent(agent, nTrainingSteps, saveFile)
            convergenceData.append(conv)
            trainingTimes.append(trainingTime)
            agents.append(agent)

            # Save the model and replay buffer.
            agent.save(saveFile)
            agent.save_replay_buffer(saveFile+"_replayBuffer")

            # Evaluate
            env_eval = auv.AuvEnv()
            resources.evaluate_agent(agent, env_eval, num_episodes=100)

            # Plot convergence of each agent. Redo after each agent to provide
            # intermediate updates on how the training is going.
            iBest, fig, ax = resources.plotTraining(
                convergenceData, saveAs="./agentData/{}_convergence.png".format(agentName))

        # Save metadata in human-readable format.
        resources.saveHyperparameteres(
            agentName, agent_kwargs, policy_kwargs, env_kwargs, nTrainingSteps, trainingTimes, nProc)

        # Pick the best agent.
        agent = agents[iBest]

        # Override for evaluation
        if agentName_eval is None:
            agentName_eval = "{}_{:d}".format(agentName, iBest)

        # Trained agent.
        print("\nAfter training")
        mean_reward,_ = resources.evaluate_agent(agent, env_eval)
        resources.plotEpisode(env_eval, "RL control")

        # Dumb agent.
        print("\nSimple control")
        env_eval_pd = auv.AuvEnv()
        pdController = auv.PDController(env_eval_pd.dt)
        mean_reward,_ = resources.evaluate_agent(pdController, env_eval_pd)
        fig, ax = resources.plotEpisode(env_eval_pd, "Simple control")

        # Compare detail
        resources.plotDetail([env_eval_pd, env_eval], labels=["Simple control", "RL control"])

# %% Evaluation
    if do_evaluation:
        # Create the environment and load the best agent to-date.
        env_eval = auv.AuvEnv(**env_kwargs_evaluation)
        agent = stable_baselines3.SAC.load("./agentData/{}".format(agentName_eval))

        # Load the hyperparamters as well for demonstration purposes.
        # with open("./agentData/{}_hyperparameters.yaml".format(agentName_eval), "r") as outf:
        #     hyperparameters = yaml.safe_load(outf)

        # Load the convergence history of the agent for demonstration purposes.
        convergence = pandas.read_csv("./agentData/{}.monitor.csv".format(agentName_eval), skiprows=1)

        # Plot agent convergence history.
        fig, ax = plt.subplots()
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.plot(convergence.index, convergence["r"], ".", ms=1, alpha=0.2, c="r", zorder=-100)
        ax.plot(convergence.index, convergence.rolling(200).mean()["r"], "-", c="r", lw=2)

        # Evaluate for a large number of episodes to test robustness.
        print("\nRL agent")
        mean_reward, allRewards = resources.evaluate_agent(
            agent, env_eval, num_episodes=100)

        # Dumb agent.
        print("\nSimple control")
        env_eval_pd = auv.AuvEnv(**env_kwargs_evaluation)
        pdController = auv.PDController(env_eval_pd.dt)
        mean_reward_pd, allRewards_pd = resources.evaluate_agent(
            pdController, env_eval_pd, num_episodes=100, saveDir="testEpisodes")

        # Evaluate once with fixed initial conditions.
        print("\nLike-for-like comparison")
        print("RL agent")
        resources.evaluate_agent(agent, env_eval, num_episodes=1,
                                 init=[[-0.5, -0.5], 0.785, 1.57])
        print("PD controller")
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
                                     flipX=True, Uinf=env_kwargs_evaluation["currentVelScale"])
            resources.animateEpisode(env_eval_pd, "naive_control",
                                     flipX=True, Uinf=env_kwargs_evaluation["currentVelScale"])