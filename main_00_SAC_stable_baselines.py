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
    # An ugly fix for OpenMP conflicts in my installation.
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    agentName = "SAC_try7"

    # Top-level switches
    do_training = False
    do_evaluation = True

    # --- Training parameters ---

    agentToRestart = None
    # agentToRestart = "SAC_try6"

    # No. parallel processes.
    nProc = 16

    # Do everything N times to rule out random successes and failures.
    nAgents = 1

    nTrainingSteps = 3_000_000

    agent_kwargs = {
        'learning_rate': 2e-3,
        'gamma': 0.95,
        'verbose': 1,
        'buffer_size': (128*3)*512,
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
            pi=[128, 128, 128],
            # Critic - estimates value of each state-action combination
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
    }

# %% Training.
    if do_training:

        # Train several times to make sure the agent doesn't just get lucky.
        convergenceData = []
        agents = []
        for iAgent in range(nAgents):
            # Set up constants etc.
            saveFile = "./agentData/{}_{:d}".format(agentName, iAgent)
            logDir = "./agentData/{}_{:d}_logs".format(agentName, iAgent)
            os.makedirs(logDir, exist_ok=True)

            # Create the environments.
            env_eval = auv.AuvEnv()
            env = SubprocVecEnv([auv.make_env(i, env_kwargs=env_kwargs) for i in range(nProc)])
            env = VecMonitor(env, logDir)

            # Create the agent using stable baselines.
            if agentToRestart is None:
                agent = stable_baselines3.SAC(
                    "MlpPolicy", env, policy_kwargs=policy_kwargs, **agent_kwargs)
            else:
                agent = stable_baselines3.SAC.load("./bestAgent/{}".format(agentToRestart))
                agent.set_env(env)

            # Train the agent for N steps
            convergenceData.append(resources.trainAgent(agent, nTrainingSteps, saveFile, logDir))
            agents.append(agent)

            # Evaluate
            env_eval = auv.AuvEnv()
            resources.evaluate_agent(agent, env_eval, num_episodes=100)

        # Save metadata in human-readable format.
        resources.saveHyperparameteres(
            agentName, agent_kwargs, policy_kwargs, env_kwargs, nTrainingSteps)

        # Plot convergence of each agent.
        iBest, _ = resources.plotTraining(
            convergenceData, saveAs="./agentData/{}_convergence.png".format(agentName))

        # Pick the best agent.
        agent = agents[iBest]

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
        agent = stable_baselines3.SAC.load("./bestAgent/{}".format(agentName))

        # Load the hyperparamters as well for demonstration purposes.
        with open("./bestAgent/{}_hyperparameters.yaml".format(agentName), "r") as outf:
            hyperparameters = yaml.safe_load(outf)

        # Load the convergence history of the agent for demonstration purposes.
        convergence = pandas.read_csv("./bestAgent/{}_monitor.csv".format(agentName), skiprows=1)

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
        resources.evaluate_agent(agent, env_eval, num_episodes=1,
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
                                     flipX=True, Uinf=env_kwargs_evaluation["currentVelScale"])
            resources.animateEpisode(env_eval_pd, "naive_control",
                                     flipX=True, Uinf=env_kwargs_evaluation["currentVelScale"])