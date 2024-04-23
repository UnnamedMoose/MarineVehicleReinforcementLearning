# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:16:55 2023

@author: ALidtke
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import datetime
import os
import re

import stable_baselines3
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
import torch

from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data.types import TrajectoryWithRew
from imitation.rewards.reward_nets import BasicRewardNet, BasicShapedRewardNet
from imitation.util.networks import RunningNorm

import verySimpleAuv as auv
import resources

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

# %% Set up.
if __name__ == "__main__":
    # An ugly fix for OpenMP conflicts in my installation.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    agentName = "SAC_sblPretrain_try0_fromPID"

    # Top-level switches
    do_generateData = False

    # --- Pretraining parameters ---
    nPretrainEpisodes = 200
    nPretrainSteps = 1_000
    nProcPretrain = 1  # TODO the env doesn't work with pretraining in parallel yet

    # Note that the bounds check is disabled to ensure episodes are of equal length.
    env_kwargs_pretrain = {
        "currentVelScale": 1.,
        "currentTurbScale": 2.,
        "noiseMagActuation": 0.,
        "noiseMagCoeffs": 0.,
        "stopOnBoundsExceeded": False,
    }

    # --- Training parameters ---

    # No. parallel processes.
    nProc = 16

    # Do everything N times to rule out random successes and failures.
    nAgents = 10

    # Any found agent will be left alone unless this is set to true.
    overwrite = False

    nTrainingSteps = 1_500_000
    agent_kwargs = {
        'learning_rate': 5e-4,
        'gamma': 0.95,
        'verbose': 1,
        'buffer_size': (128*3)*512,
        'batch_size': 256,
        'learning_starts': 256,
        'train_freq': (1, "step"),
        "gradient_steps": 1,
# XXX Included explicitly due to different parallelisations at the pretraining and training stages.
# "action_noise": VectorizedActionNoise(NormalActionNoise(
#     np.zeros(3), 0.05*np.ones(3)), nProc),
        "use_sde_at_warmup": False,
        # "target_entropy": -4.0,
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

# %% Create test rollouts.

    # Create a random seed.
    seed = 3
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # Create an evaluation environment and Proportional-Derivative controller.
    env_eval = auv.AuvEnv(**env_kwargs_evaluation)
    pdController = auv.PDController(env_eval.dt)

    # Simply create the env without any wrappers.
    env_pretrain = auv.AuvEnv(**env_kwargs_pretrain)

    # Generate training data using a PD controller.
    if do_generateData:
        print("\nGenerating training data using a simple controller")
        env_eval_pre = auv.AuvEnv(**env_kwargs_pretrain)
        pdController = auv.PDController(env_eval_pre.dt)
        resources.evaluate_agent(
            pdController, env_eval_pre, num_episodes=nPretrainEpisodes*2, saveDir="testEpisodes")

    # Read the pre-computed data
    pretrainEpisodes = [pandas.read_csv(os.path.join("testEpisodes", f)) for f in os.listdir("testEpisodes")]
    print("Read {:d} pretraining episodes".format(len(pretrainEpisodes)))

    # Wrap the episodes into rollouts compatible with the imitate library.
    rollouts = []
    stateVars = [k for k in pretrainEpisodes[0].keys() if re.match("s[0-9]+", k)]
    actionVars = [k for k in pretrainEpisodes[0].keys() if re.match("a[0-9]+", k)]
    for ep in pretrainEpisodes:
        observations = ep[stateVars].values
        actions = ep[actionVars].values[:-1]
        rewards = ep["reward"].values[:-1]

        out_dict_stacked = {"rews": rewards, "acts": actions, "obs": observations, "infos": None}
        traj = TrajectoryWithRew(**out_dict_stacked, terminal=(rewards[-1] < -200))
        assert traj.rews.shape[0] == traj.acts.shape[0] == traj.obs.shape[0] - 1

        rollouts.append(traj)

    # Plot the different training episodes.
    fig, ax = plt.subplots()
    ax.set_xlabel("s0")
    ax.set_ylabel("s1")
    for e in pretrainEpisodes:
        cs = ax.scatter(e["s0"], e["s1"], c=e["r"], s=5)
    cbar = plt.colorbar(cs)
    cbar.set_label("Reward")

# %% Train the agents.

    # Create a parallel version of the environment.
    env_pretrain = SubprocVecEnv([auv.make_env(i, env_kwargs=env_kwargs_pretrain) for i in range(nProcPretrain)])

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

        # Create the agent using stable baselines.
        agent = stable_baselines3.SAC(
            "MlpPolicy", env_pretrain, policy_kwargs=policy_kwargs,
            action_noise=VectorizedActionNoise(NormalActionNoise(np.zeros(3), 0.05*np.ones(3)), nProcPretrain),
            **agent_kwargs)

        # Evaluate
        print("\nRandomly initialised agent")
        _, rewards_init = resources.evaluate_agent(agent, env_eval, num_episodes=100)

        # TODO Choose episodes for pretraining at random.
        iPretrain = np.random.default_rng().choice(
            len(pretrainEpisodes), size=(nPretrainEpisodes,), replace=False, shuffle=False)

        # Pretrain.
        reward_net = BasicRewardNet(
        # reward_net = BasicShapedRewardNet(
            env_pretrain.observation_space,
            env_pretrain.action_space,
            normalize_input_layer=RunningNorm,
        )
        pretrainer = GAIL(
        # pretrainer = AIRL(
            demonstrations=[rollouts[i] for i in iPretrain],
            demo_batch_size=1024,
            gen_replay_buffer_capacity=2048,
            n_disc_updates_per_round=4,
            venv=env_pretrain,
            gen_algo=agent,
            reward_net=reward_net,
            allow_variable_horizon=False,
            # log_dir="./tempData",
            # init_tensorboard=True,
        )
        pretrainer.train(nPretrainSteps)

        # Evaluate
        print("\nPretrained agent")
        _, rewards_pre = resources.evaluate_agent(agent, env_eval, num_episodes=100)

        # Save the pretrained agent.
        agent.save(saveFile+"_pretrained")
        del agent

        # Create and set a parallel enviornment for training
        env = SubprocVecEnv([auv.make_env(i, env_kwargs=env_kwargs) for i in range(nProc)])
        env = VecMonitor(env, saveFile)
        # agent.set_env(env)  # TODO if both envs would be parallelised the same way, just use this.

        # Load the model and change environments.
        # TODO this is only needed to run the rest of the pipeline in parallel.
        agent = stable_baselines3.SAC.load(
            saveFile+"_pretrained", env=env, force_reset=False,
            action_noise=VectorizedActionNoise(NormalActionNoise(np.zeros(3), 0.05*np.ones(3)), nProc))

        # Train the agent for N more steps
        conv, trainingTime = resources.trainAgent(agent, nTrainingSteps, saveFile)
        convergenceData.append(conv)
        trainingTimes.append(trainingTime)
        agents.append(agent)

        # Evaluate
        print("\nTrained agent")
        _, rewards_trained = resources.evaluate_agent(agent, env_eval, num_episodes=100)

        # Plot convergence of each agent. Redo after each agent to provide
        # intermediate updates on how the training is going.
        iBest, fig, ax = resources.plotTraining(
            convergenceData, saveAs="./agentData/{}_convergence.png".format(agentName))

        # Check the effects of pretraining.
        fig, ax = plt.subplots()
        ax.set_xlabel("Reward range")
        ax.set_ylabel("Episode count")
        bins = np.linspace(0, np.max(rewards_trained), 11)
        x = (bins[1:] + bins[:-1])/2
        h, _ = np.histogram(rewards_init, bins=bins)
        plt.bar(x, h, color="green", alpha=0.5, label="Initialised", width=20)
        h, _ = np.histogram(rewards_pre, bins=bins)
        plt.bar(x, h, color="blue", alpha=0.5, label="Pretrained", width=20)
        h, _ = np.histogram(rewards_trained, bins=bins)
        plt.bar(x, h, color="red", alpha=0.5, label="Pretrained+Trained", width=20)
        ax.legend()

    # Save metadata in human-readable format.
    resources.saveHyperparameteres(
        agentName, agent_kwargs, policy_kwargs, env_kwargs, nTrainingSteps, trainingTimes, nProc)
