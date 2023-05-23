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
import copy
import shutil

import stable_baselines3
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.torch_layers import create_mlp
from sklearn.model_selection import train_test_split
import torch as th
from torch import nn

import verySimpleAuv as auv
import resources

from script_2_trainActor import ActorNeuralNetwork
from script_3_trainCritic import CriticNeuralNetwork

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

# %% Set up.
if __name__ == "__main__":
    # An ugly fix for OpenMP conflicts in my installation.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Just copy the weights without doing anything. Keep the learning rate high
    # => converges very fast but final performance is not great
    # agentName = "SAC_customInit_try0_simpleCopy"
    # Noise in the weights, low learning rate
    # => need more thatn 1e6 time steps but it doesn't look like it will do much
    # agentName = "SAC_customInit_try1_noise_5e-3_LR_1e-4"
    # Noise in the weights, intermediate learning rate
    # => didn't do much
    # agentName = "SAC_customInit_try2_noise_1e-2_LR_5e-4_noWarmupSde"
    # => doesn't seem to converge at all.
    # agentName = "SAC_customInit_try3_noise_1e-2_LR_5e-4_noWarmupSde_constEntropyCoeff_0.5"
    # => doesn't seem to converge at all.
    # agentName = "SAC_customInit_try4_noise_1e-2_LR_5e-4_noWarmupSde_constEntropyCoeff_0.2"
    # => seems better but still does not improve relative to PD
    # agentName = "SAC_customInit_try5_noise_1e-2_LR_5e-4_noWarmupSde_targetEntropy_-15"
    # => same
    # agentName = "SAC_customInit_try6_noise_1e-2_LR_5e-4_noWarmupSde_targetEntropy_-30_actionNoise_0.1"
    # => multiplication introduces a lot of noise, too much even
    # agentName = "SAC_customInit_try7_noiseMult_LR_5e-4_targetEntropy_-30_actionNoise_0.05"
    # agentName = "SAC_customInit_try11_noiseAdd_5e-2_dropout_0.2_mag_0.5_LR_5e-4_targetEntropy_-9_actionNoise_0.05"
    # agentName = "SAC_customInit_try14_noiseAdd_1e-2_dropout_0.1_mag_0.5_LR_5e-4_targetEntropy_-4_actionNoise_0.05"
    # agentName = "SAC_customInit_try15_noiseAdd_1e-2_LR_5e-4_targetEntropy_-4_actionNoise_0.05"
    # agentName = "SAC_customInit_try18_blend_0.95_LR_5e-4_targetEntropy_-4_actionNoise_0.05"
    # agentName = "SAC_customInit_try19_blend_0.99_LR_5e-4_targetEntropy_-4_actionNoise_0.05"

    # agentName = "SAC_customInit_try0_copy_LR_5e-4_targetEntropy_-4_actionNoise_0.05"

    agentName = "SAC_customInit_try1_copyCritic_LR_5e-4_targetEntropy_-4_actionNoise_0.05"

    # nTrainingSteps = 1_500_000
    # nTrainingSteps = 1_000_000
    nTrainingSteps = 500_000
    # nTrainingSteps = 200_000
    nAgents = 3
    nProc = 16

    # Train and evaluate the agent.
    do_training = True
    do_evaluation = False

    # SAC agent settings.
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
        "target_entropy": -4.,
    }
    policy_kwargs = {
        "activation_fn": th.nn.GELU,
        "net_arch": dict(
            pi=[128, 128, 128],
            qf=[128, 128, 128],
        )
    }
    # Env settings for training and evaluation
    env_kwargs = {
        "currentVelScale": 1.,
        "currentTurbScale": 2.,
        "noiseMagActuation": 0.1,
        "noiseMagCoeffs": 0.1,
    }
    env_kwargs_evaluation = {
        "currentVelScale": 1.,
        "currentTurbScale": 2.,
        "noiseMagActuation": 0.,
        "noiseMagCoeffs": 0.,
    }

# %% Create a real SAC agent and copy the weights.
    if do_training:

        # Train several times to make sure the agent doesn't just get lucky.
        convergenceData = []
        trainingTimes = []
        agents = []
        for iAgent in range(nAgents):
            saveFile = "./agentData/{}_{:d}".format(agentName, iAgent)

            # Create the training environment
            env = SubprocVecEnv([auv.make_env(i, env_kwargs=env_kwargs) for i in range(nProc)])
            env = VecMonitor(env, saveFile)

            # Create the agent.
            sacAgent = stable_baselines3.SAC(
                "MlpPolicy", env, policy_kwargs=policy_kwargs, **agent_kwargs)

            # ===
            # !!! This is the critical bit. !!!
            # Load the pretrained nets.
            pretrained_actor = th.load("agentData/pretrained_actor.pt")
            pretrained_critic = th.load("agentData/pretrained_critic.pt")

            # Add some noise.
            # noise_scale = 0.01
            # for key in pretrained_actor.latent_pi.state_dict():
            #     pretrained_actor.latent_pi.state_dict()[key].add_(
            #         th.rand_like(pretrained_actor.latent_pi.state_dict()[key])*noise_scale)

            # for key in pretrained_actor.mu.state_dict():
            #     pretrained_actor.mu.state_dict()[key].add_(
            #         th.rand_like(pretrained_actor.mu.state_dict()[key])*noise_scale)

            # for key in pretrained_critic.qf.state_dict():
            #     pretrained_critic.qf.state_dict()[key].add_(
            #         th.rand_like(pretrained_critic.qf.state_dict()[key])*noise_scale)

            # Copy the weights from the pre-trained actor.
            # sacAgent.actor.latent_pi = copy.deepcopy(pretrained_actor.latent_pi)
            # sacAgent.actor.mu = copy.deepcopy(pretrained_actor.mu)
            sacAgent.critic.qf0 = copy.deepcopy(pretrained_critic.qf)
            sacAgent.critic.qf1 = copy.deepcopy(pretrained_critic.qf)

            # --- Approach 2 - copy element-wise and add modifications. ---
            # latent_pi_randomInit = copy.deepcopy(sacAgent.actor.latent_pi)
            # mu_randomInit = copy.deepcopy(sacAgent.actor.mu)
            # sacAgent.actor.latent_pi = copy.deepcopy(pretrained_actor.latent_pi)
            # sacAgent.actor.mu = copy.deepcopy(pretrained_actor.mu)
            """
            blend_trained = 1
            blend_random = 2
            scale_trained = 0.2
            with th.no_grad():
                # Actor
                for key in sacAgent.actor.latent_pi.state_dict().keys():
                    sacAgent.actor.latent_pi.state_dict()[key].add_(
                        -sacAgent.actor.latent_pi.state_dict()[key] + pretrained_actor.latent_pi.state_dict()[key])

                for key in sacAgent.actor.mu.state_dict().keys():
                    sacAgent.actor.mu.state_dict()[key].add_(
                        -sacAgent.actor.mu.state_dict()[key] + pretrained_actor.mu.state_dict()[key])

                    # sacAgent.actor.latent_pi.state_dict()[key] = \
                #         pretrained_actor.latent_pi.state_dict()[key]

                    # sacAgent.actor.latent_pi.state_dict()[key].multiply_(blend_trained)
                    # sacAgent.actor.latent_pi.state_dict()[key].multiply_(
                    #     th.rand_like(sacAgent.actor.latent_pi.state_dict()[key])*scale_trained+1.-scale_trained/2.)
                    # sacAgent.actor.latent_pi.state_dict()[key].add_(blend_random*latent_pi_randomInit.state_dict()[key])

                # for key in sacAgent.actor.mu.state_dict().keys():
                #     sacAgent.actor.mu.state_dict()[key] = \
                #         pretrained_actor.mu.state_dict()[key]

                    # sacAgent.actor.mu.state_dict()[key].multiply_(blend_trained)
                    # sacAgent.actor.mu.state_dict()[key].add_(blend_random*mu_randomInit.state_dict()[key])
                    # sacAgent.actor.mu.state_dict()[key].multiply_(
                    #     th.rand_like(sacAgent.actor.mu.state_dict()[key])*scale_trained+1.-scale_trained/2.)

                # Critic
                for key in sacAgent.critic.state_dict().keys():
                    keySrc = key.replace("qf0", "qf").replace("qf1", "qf")
                    sacAgent.critic.state_dict()[key].add_(
                        -sacAgent.critic.state_dict()[key]+pretrained_critic.state_dict()[keySrc])
                    sacAgent.critic_target.state_dict()[key].add_(
                        -sacAgent.critic_target.state_dict()[key] + pretrained_critic.state_dict()[keySrc])

                    # sacAgent.critic.state_dict()[key] = \
                    #     pretrained_critic.state_dict()[keySrc]
                    # sacAgent.critic_target.state_dict()[key] = \
                    #     copy.deepcopy(pretrained_critic.state_dict()[keySrc])

            # # Add noise to enhance exploration and avoid overfitting the simple controller.
            # with th.no_grad():
            #     spoilNet(sacAgent.actor.latent_pi, add_dropout=False)
            #     spoilNet(sacAgent.actor.mu)
            # New approach: copy the pre-trained weights by blending them with
            # an underrelaxation-like parameter

            # Blend copied and random init.
            # sacAgent.actor.latent_pi = copyWeightsRelaxed(
            #     sacAgent.actor.latent_pi, pretrained_actor.latent_pi)
            # sacAgent.actor.mu = copyWeightsRelaxed(
            #     sacAgent.actor.mu, pretrained_actor.mu)
            """
            # print(kupa)
            # ===

            # Train the agent for N steps
            conv, trainingTime = resources.trainAgent(sacAgent, nTrainingSteps, saveFile)
            convergenceData.append(conv)
            trainingTimes.append(trainingTime)
            agents.append(sacAgent)

            # Plot convergence of each agent. Redo after each agent to provide
            # intermediate updates on how the training is going.
            iBest, fig, ax = resources.plotTraining(
                convergenceData, saveAs="./agentData/{}_convergence.png".format(agentName))

        # Save metadata in human-readable format.
        resources.saveHyperparameteres(
            agentName, agent_kwargs, policy_kwargs, env_kwargs, nTrainingSteps, trainingTimes, nProc)

# %% Evaluation

    if do_evaluation:

        # Evaluate for a large number of episodes to test robustness.
        nEvalEpisodes = 100

        # Proper RL agent.
        print("\nRL agent")
        env_eval = auv.AuvEnv(**env_kwargs_evaluation)
        agent = agents[iBest]
        mean_reward, allRewards = resources.evaluate_agent(
            agent, env_eval, num_episodes=nEvalEpisodes)

        # Simple PD controller.
        print("\nSimple control")
        env_eval_pd = auv.AuvEnv(**env_kwargs_evaluation)
        pdController = auv.PDController(env_eval_pd.dt)
        mean_reward_pd, allRewards_pd = resources.evaluate_agent(
            pdController, env_eval_pd, num_episodes=nEvalEpisodes)

        # Plot all.
        fig, ax = plt.subplots()
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        x = np.array(range(len(allRewards)))
        w_bar = 0.3
        ax.bar(x-w_bar, allRewards, w_bar, align="edge", color="r", label="RL control")
        ax.bar(x+w_bar, allRewards_pd, w_bar, align="edge", color="b", label="Simple control")
        xlim = ax.get_xlim()
        ax.plot(xlim, [mean_reward]*2, "r--", lw=4, alpha=0.5)
        ax.plot(xlim, [mean_reward_pd]*2, "b--", lw=4, alpha=0.5)
        ax.plot(xlim, [0]*2, "k-", lw=1)
        ax.set_xlim(xlim)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)
