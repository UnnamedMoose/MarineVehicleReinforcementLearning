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
import sb3_contrib
import stable_baselines3
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise

from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union
try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

import verySimpleAuv as auv
import resources

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)


class CustomReplayBuffer(ReplayBuffer):
    """ Specialised buffer that applies geometric transformations to the observations
    and actions in order to fill up the space more quickly and (hopefully) provide
    higher sample efficiency.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs,
                         optimize_memory_usage=False, handle_timeout_termination=True)
        self.nRollovers = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

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

        transformations_obs = [
            # "Standard"
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            # Mirror everything around the origin.
            [-1., -1., 1., 1., -1., -1., -1., -1., 1., 1., 1.],
            # Mirror around x and y axes only.
            [-1.,  1.,  1.,  1., -1.,  1., -1.,  1.,  1.,  1.,  1.],
            [ 1., -1.,  1.,  1.,  1., -1.,  1., -1.,  1.,  1.,  1.],
            # Flip the heading.
            [1., 1., -1., 1., 1., 1., 1., 1., -1., 1., 1.],
        ]
        transformations_act = [
            [1., 1., 1.],
            [-1., -1., 1.],
            [-1., 1., 1.],
            [1., -1., 1.],
            [1., 1., -1.],
        ]

        # Once the buffer gets full-ish, reduce the number of artificially generated
        # states to zero and only rely on actual data.
        # if self.full:
        #     vals = [0]
        # else:
        #     nMax = len(transformations_act)-1
        #     x0 = int(self.buffer_size*0.5)
        #     dx = int(self.buffer_size*0.5)
        #     n = max(0, min(nMax, int((self.pos - x0) * -nMax // dx + nMax)))
        #     vals = np.append([0], 1+np.random.choice(range(nMax), size=n, replace=False))

        # Apply transformations and store the experience.
        for i in range(len(transformations_obs)):
            # if i not in vals:
            #     continue
            # Stop generating synthetic data once the buffer has rolled over a few times.
            if (self.nRollovers > 5) and (i != 0):
                continue
            # Copy to avoid modification by reference, apply transformation.
            self.observations[self.pos] = np.array(obs).copy() * transformations_obs[i]
            self.next_observations[self.pos] = np.array(next_obs).copy() * transformations_obs[i]
            self.actions[self.pos] = np.array(action).copy() * transformations_act[i]
            # Reward and done are unchanged.
            self.rewards[self.pos] = np.array(reward).copy()
            self.dones[self.pos] = np.array(done).copy()
            if self.handle_timeout_termination:
                self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
            # Advance in the buffer. Check if full.
            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0
                self.nRollovers += 1


if __name__ == "__main__":
    # An ugly fix for OpenMP conflicts in my installation.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # For saving trained agents.
    agentName = "TQC_customBuffer_try2"

    # Set to None to pick the best agent from the trained set. Specify as string
    # to load a particular saved model.
    agentName_eval = None
    # agentName_eval = "SAC_try8_0"

    # Top-level switches
    do_training = True
    do_evaluation = True

    # --- Training parameters ---
    loadReplayBuffer = True  # For a "perfect" restart keep this on.
    agentToRestart = None
    # agentToRestart = "TQC_customBuffer_try0_0"

    # No. parallel processes.
    nProc = 16

    # Do everything N times to rule out random successes and failures.
    nAgents = 5

    # Any found agent will be left alone unless this is set to true.
    overwrite = False

    nTrainingSteps = 1_500_000
    # nTrainingSteps = 500_000

    agent_kwargs = {
        'gamma': 0.95,
        'verbose': 1,
        'batch_size': 256,
        'learning_rate': 2e-3,
        'buffer_size': (128*3)*512,
        'learning_starts': 256,
        'train_freq': (1, "step"),
        "gradient_steps": 1,
        "action_noise": VectorizedActionNoise(NormalActionNoise(
            np.zeros(3), 0.05*np.ones(3)), nProc),
        # Set the custom buffer.
        "replay_buffer_class": CustomReplayBuffer
    }
    policy_kwargs = {
        "activation_fn": torch.nn.GELU,
        "net_arch": dict(
            pi=[128, 128, 128],
            qf=[128, 128, 128],
        ),
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
            env = SubprocVecEnv([auv.make_env(i, env_kwargs=env_kwargs) for i in range(nProc)])
            env = VecMonitor(env, saveFile)

            # Create the agent using stable baselines.
            if agentToRestart is None:
                agent = sb3_contrib.TQC("MlpPolicy", env, policy_kwargs=policy_kwargs, **agent_kwargs)

            else:
                agent = sb3_contrib.TQC.load("./agentData/{}".format(agentToRestart), env=env)
                agent.set_parameters("./agentData/{}".format(agentToRestart))
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
        env_eval = auv.AuvEnv()
        print("\nAfter training")
        mean_reward, _ = resources.evaluate_agent(agent, env_eval, num_episodes=100)
        resources.plotEpisode(env_eval, "RL control")

        # Dumb agent.
        print("\nSimple control")
        env_eval_pd = auv.AuvEnv()
        pdController = auv.PDController(env_eval_pd.dt)
        mean_reward, _ = resources.evaluate_agent(pdController, env_eval_pd, num_episodes=100)
        fig, ax = resources.plotEpisode(env_eval_pd, "Simple control")

        # Compare detail
        resources.plotDetail([env_eval_pd, env_eval], labels=["Simple control", "RL control"])

# %% Evaluation
    if do_evaluation:
        # Create the environment and load the best agent to-date.
        env_eval = auv.AuvEnv(**env_kwargs_evaluation)
        agent = sb3_contrib.TQC.load("./agentData/{}".format(agentName_eval))

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
