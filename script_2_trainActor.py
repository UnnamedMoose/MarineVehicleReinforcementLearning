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

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

# Construct a neural network for approximating the simple controller output.
# This mimics the actor part of stable baselines SAC agent exactly.
class ActorNeuralNetwork(nn.Module):
    def __init__(self):
        super(ActorNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        net_arch = policy_kwargs["net_arch"]["pi"]
        latent_pi_net = create_mlp(
            len(stateVars), -1, net_arch, policy_kwargs["activation_fn"])
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1]
        self.mu = nn.Linear(last_layer_dim, len(actionVars))

    def forward(self, x):
        features = self.flatten(x)
        latent_pi  = self.latent_pi(features)
        actions = self.mu(latent_pi)
        return actions

# %% Set up.
if __name__ == "__main__":
    # An ugly fix for OpenMP conflicts in my installation.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    n_epochs_supervised = 5

    # Top-level switches
    # Generate data using PD control for pre-training of the actor net.
    do_generateData = False
    # Use supervised learning to approximate the PD controller using the actor net.
    do_trainActor = False
    # evaluate the actor on its own
    do_evaluate = True

    # SAC agent settings.
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

# %% Generate training data using a PD controller.
    if do_generateData:
        print("\nGenerating training data using a simple controller")
        # NOTE using noise in the coeffs for better exploration and reducing
        # overfitting.
        env_training_pd = auv.AuvEnv(**env_kwargs)
        # Add normally-distributed noise to the actions as well.
        pdController = auv.PDController(env_training_pd.dt, noiseSigma=0.1)
        # Create training data.
        mean_reward_pd, allRewards_pd = resources.evaluate_agent(
            pdController, env_training_pd, num_episodes=5000, saveDir="testEpisodes")

# %% Train an actor network for approximating the control output of the simple controller.
    # Read the pre-computed data
    trainingEpisodes = [pandas.read_csv(os.path.join("testEpisodes", f)) for f in os.listdir("testEpisodes")]
    buffer = pandas.concat(trainingEpisodes).reset_index(drop=True)

    # Plot the different training episodes.
    fig, ax = plt.subplots()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cs = ax.scatter(buffer["x"], buffer["y"], c=buffer["reward"], s=5)
    cbar = plt.colorbar(cs)
    cbar.set_label("Reward")

    # Get the state and action variables from the buffer.
    stateVars = [k for k in buffer.keys() if re.match("s[0-9]+", k)]
    actionVars = [k for k in buffer.keys() if re.match("a[0-9]+", k)]

    # See what hardware is being used.
    device = "cuda" if th.cuda.is_available() else "cpu"

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        with th.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        print(f"Test error average loss: {test_loss:>8f}")

    def predict(model, state):
        return model.forward(th.from_numpy(state[np.newaxis, :]).float()).detach().numpy().flatten()

    if do_trainActor:
        # Extract data as arrays.
        X = buffer[stateVars].values
        Y = buffer[actionVars].values
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

        # Convert the data from Numpy to a TensorDataset from Pytorch
        X_torch = th.from_numpy(X).float()
        y_torch = th.from_numpy(Y).float()
        X_train_torch = th.from_numpy(X_train).float()
        y_train_torch = th.from_numpy(y_train).float()
        X_test_torch = th.from_numpy(X_test).float()
        y_test_torch = th.from_numpy(y_test).float()

        # These objects sample the data tensor with a given batch size
        batch_size = 128
        training_data = th.utils.data.TensorDataset(X_train_torch, y_train_torch)
        train_dataloader = th.utils.data.DataLoader(training_data, batch_size=batch_size)
        testing_data = th.utils.data.TensorDataset(X_test_torch, y_test_torch)
        test_dataloader = th.utils.data.DataLoader(testing_data, batch_size=batch_size)

        # Create the network
        pretrained_actor = ActorNeuralNetwork().to(device)

        # choose a loss function and an optimizer
        loss_fn = nn.MSELoss()
        optimizer = th.optim.Adam(pretrained_actor.parameters(), lr=1e-3)

        # Train.
        t_start = datetime.datetime.now()
        loss = np.zeros(n_epochs_supervised)
        val_loss = np.zeros(n_epochs_supervised)
        for t in range(n_epochs_supervised):
            print(f"\nEpoch {t+1}\n-------------------------------")
            train(train_dataloader, pretrained_actor, loss_fn, optimizer)
            test(test_dataloader, pretrained_actor, loss_fn)

            # keep track of the errors
            pred_train = pretrained_actor(X_train_torch)
            loss[t] = loss_fn(pred_train, y_train_torch)
            pred_test = pretrained_actor(X_test_torch)
            val_loss[t] = loss_fn(pred_test, y_test_torch)
        t_end = datetime.datetime.now()
        trainingTime = (t_end-t_start).total_seconds()
        print("\nTraining took {:.0f} seconds ({:.0f} minutes)".format(trainingTime, trainingTime/60.))

        # plot the evolution of the training error
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        ax.plot(loss, label='Training')
        ax.plot(val_loss, label='Validation')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Error')
        ax.legend(loc="upper center", ncol=2)

        # Save the mdodel.
        th.save(pretrained_actor, "agentData/pretrained_actor.pt")

    else:
        # load
        pretrained_actor = th.load("agentData/pretrained_actor.pt")


# %% Evaluation - only direct actor

    if do_evaluate:

        # Create a controller that uses the pretrined actor network.
        class NNController(object):
            def __init__(self, dt, net):
                self.dt = dt
                self.net = net

            def predict(self, obs, deterministic=True):
                # NOTE deterministic is a dummy kwarg needed to make this function look
                # like a stable baselines equivalent
                states = None
                actions = predict(self.net, obs)
                return actions, states

        # Evaluate for a large number of episodes to test robustness.
        nEvalEpisodes = 100

        # Pure NN actor.
        print ("\nDirect NN control")
        env_eval_nn = auv.AuvEnv(**env_kwargs_evaluation)
        nnController = NNController(env_eval_nn.dt, pretrained_actor)
        mean_reward_nn, allRewards_nn = resources.evaluate_agent(
            nnController, env_eval_nn, num_episodes=nEvalEpisodes)

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
        x = np.array(range(len(allRewards_nn)))
        w_bar = 0.3
        ax.bar(x, allRewards_nn, w_bar, align="edge", color="g", label="Direct NN control")
        ax.bar(x+w_bar, allRewards_pd, w_bar, align="edge", color="b", label="Simple control")
        xlim = ax.get_xlim()
        ax.plot(xlim, [mean_reward_nn]*2, "g--", lw=4, alpha=0.5)
        ax.plot(xlim, [mean_reward_pd]*2, "b--", lw=4, alpha=0.5)
        ax.plot(xlim, [0]*2, "k-", lw=1)
        ax.set_xlim(xlim)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)
