# -*- coding: utf-8 -*-
"""
Created on Wed May 24 09:56:15 2023

@author: ALidtke
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import re
import platform

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.data.types import TrajectoryWithRew
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env, make_seeds

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

if __name__ == "__main__":
    # An ugly fix for OpenMP conflicts in my installation.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Initialise random number generation.
    rng = np.random.default_rng(0)

    # Create source env and train the "expert"
    env = gym.make("CartPole-v1")
    expert = PPO(policy=MlpPolicy, env=env, n_steps=64)
    expert.learn(1000)

    # Generate rollouts from the expert
    # rollouts = rollout.rollout(
    #     expert,
    #     make_vec_env(
    #         "CartPole-v1",
    #         n_envs=5,
    #         post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    #         rng=rng,
    #     ),
    #     rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    #     rng=rng,
    # )

    # Create own rollout generation to circumvent the issues with the imitation wrappers.
    rollouts = []
    for iEp in range(60):
        observations = [env.reset()]
        actions = []
        rewards = []
        for i in range(30):
            action = np.random.randint(0, 2)
            obs, reward, done, info = env.step(action)
            observations = np.append(observations, obs[np.newaxis, :], axis=0)
            actions = np.append(actions, action)
            rewards = np.append(rewards, reward)

        out_dict_stacked = {"rews": rewards, "acts": actions, "obs": observations, "infos": None}
        traj = TrajectoryWithRew(**out_dict_stacked, terminal=done)
        assert traj.rews.shape[0] == traj.acts.shape[0] == traj.obs.shape[0] - 1

        rollouts.append(traj)

    # Create a vectorised environment for collecting further episodes.
    venv = make_vec_env("CartPole-v1", n_envs=8, rng=rng)

    # Create the target agent.
    learner = PPO(env=venv, policy=MlpPolicy)

    # Sanity check
    rewards_init, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)
    print("Initialised rewards:", np.mean(rewards_init))

    # Reward net and instance for the generative adversatial algorithm
    reward_net = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    # Train.
    gail_trainer.train(20_000)
    rewards_pre, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)
    print("Pretrained rewards:", np.mean(rewards_pre))

    # Train some more the "normal" way.
    learner.learn(50_000)

    # Final evaluation.
    rewards, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)
    print("Trained rewards:", np.mean(rewards))

    # Check.
    fig, ax = plt.subplots()
    h, x = np.histogram(rewards_init)
    x = (x[1:] + x[:-1])/2
    plt.bar(x, h, color="green", alpha=0.5, label="Initialised", width=20)
    h, x = np.histogram(rewards_pre)
    x = (x[1:] + x[:-1])/2
    plt.bar(x, h, color="blue", alpha=0.5, label="Pretrained", width=20)
    h, x = np.histogram(rewards)
    x = (x[1:] + x[:-1])/2
    plt.bar(x, h, color="red", alpha=0.5, label="Pretrained+Trained", width=20)
    ax.legend()