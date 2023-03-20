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

if __name__ == "__main__":
    # An ugly fix for OpenMP conflicts in my installation.
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    modelName = "test_try0"
    nNeurons = 32

    # Top-level switches
    do_generateData = True

    env_kwargs = {
        # Set to zero to disable the flow - much faster training.
        "currentVelScale": 1.,
        "currentTurbScale": 2.,
        # Use noise in coefficients for training only.
        "noiseMagActuation": 0.1,
        "noiseMagCoeffs": 0.1,
    }

# %% Generate training data using a PD controller.
    if do_generateData:
        print("\nGenerating training data using a simple controller")
        env_eval_pd = auv.AuvEnv(**env_kwargs)
        pdController = auv.PDController(env_eval_pd.dt)
        mean_reward_pd, allRewards_pd = resources.evaluate_agent(
            pdController, env_eval_pd, num_episodes=100, saveDir="testEpisodes")

# %% Train a network for approximating the control output of the simple controller.
    # Read the pre-computed data
    buffer = [pandas.read_csv(os.path.join("testEpisodes", f)) for f in os.listdir("testEpisodes")]
    buffer = pandas.concat(buffer).reset_index(drop=True)

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

    # Construct a neural network for approximating the simple controller output.
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.layer_stack = nn.Sequential(
                nn.Linear(len(stateVars), nNeurons),
                nn.GELU(),
                nn.Linear(nNeurons, nNeurons),
                nn.GELU(),
                nn.Linear(nNeurons, len(actionVars))
            )

        def forward(self, x):
            x = self.flatten(x)
            y = self.layer_stack(x)
            return y

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
    device = "cuda" if th.cuda.is_available() else "cpu"
    model = NeuralNetwork().to(device)

    # choose a loss function and an optimizer
    loss_fn = nn.MSELoss()
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)

    # Train.
    t_start = datetime.datetime.now()
    epochs = 25
    loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    for t in range(epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

        # keep track of the errors
        pred_train = model(X_train_torch)
        loss[t] = loss_fn(pred_train, y_train_torch)
        pred_test = model(X_test_torch)
        val_loss[t] = loss_fn(pred_test, y_test_torch)
    t_end = datetime.datetime.now()
    trainingTime = (t_end-t_start).total_seconds()
    print("Training took {:.0f} seconds ({:.0f} minutes)".format(trainingTime, trainingTime/60.))

    # plot the evolution of the training error
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(loss, label='Training')
    ax.plot(val_loss, label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error')
    ax.legend(loc="upper center", ncol=2)

# %% Evaluate the training and results.

    # Create a controller that uses the network to predict the correct action.
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

    # Create env and controller instance, then evaluate.
    env_eval_nn = auv.AuvEnv(**env_kwargs)
    nnController = NNController(env_eval_nn.dt, model)
    mean_reward_nn, allRewards_nn = resources.evaluate_agent(
        nnController, env_eval_nn, num_episodes=1)

    # Evaluate using the simple controller for comparison.
    env_eval_pd = auv.AuvEnv(**env_kwargs)
    pdController = auv.PDController(env_eval_pd.dt)
    mean_reward_pd, allRewards_pd = resources.evaluate_agent(
        pdController, env_eval_pd, num_episodes=1)

    # Plot.
    resources.plotEpisode(env_eval_nn, "NN control")
    resources.plotDetail([env_eval_pd, env_eval_nn], labels=["PD control", "NN control"])