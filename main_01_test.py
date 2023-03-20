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
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv
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

# %% Set up.
if __name__ == "__main__":
    # An ugly fix for OpenMP conflicts in my installation.
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    agentName = "test_try0"

    # Top-level switches
    # Generate data using PD control for pre-training of the actor net.
    do_generateData = False

    # SAC agent settings.
    agent_kwargs = {
        'learning_rate': 2e-3,
        'gamma': 0.95,
        'verbose': 1,
        'buffer_size': (128*3)*512,
        "use_sde_at_warmup": True,
        'batch_size': 256,
        'learning_starts': 256,
        'train_freq': (1, "step"),
    }
    policy_kwargs = {
        "use_sde": False,
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
        env_eval_pd = auv.AuvEnv(**env_kwargs)
        pdController = auv.PDController(env_eval_pd.dt)
        mean_reward_pd, allRewards_pd = resources.evaluate_agent(
            pdController, env_eval_pd, num_episodes=100, saveDir="testEpisodes")

# %% Train an actor network for approximating the control output of the simple controller.
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
    # This mimics the actor part of stable baselines SAC agent exactly.
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
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
    pretrained_actor = NeuralNetwork().to(device)

    # choose a loss function and an optimizer
    loss_fn = nn.MSELoss()
    optimizer = th.optim.Adam(pretrained_actor.parameters(), lr=1e-3)

    # Train.
    t_start = datetime.datetime.now()
    epochs = 100
    loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    for t in range(epochs):
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
    env_eval_nn = auv.AuvEnv(**env_kwargs_evaluation)
    nnController = NNController(env_eval_nn.dt, pretrained_actor)
    mean_reward_nn, allRewards_nn = resources.evaluate_agent(
        nnController, env_eval_nn, num_episodes=1)

    # Evaluate using the simple controller for comparison.
    env_eval_pd = auv.AuvEnv(**env_kwargs_evaluation)
    pdController = auv.PDController(env_eval_pd.dt)
    mean_reward_pd, allRewards_pd = resources.evaluate_agent(
        pdController, env_eval_pd, num_episodes=1)

    # Plot.
    resources.plotEpisode(env_eval_nn, "Direct NN control")
    # resources.plotDetail([env_eval_pd, env_eval_nn], labels=["PD control", "NN control"])

# %% Create a real SAC agent and copy the weights.

    nProc = 16
    nAgents = 3
    agentName = "SAC_customInit_try0"

    nTrainingSteps = 500_000

    # Train several times to make sure the agent doesn't just get lucky.
    convergenceData = []
    agents = []
    for iAgent in range(nAgents):
        saveFile = "./agentData/{}_{:d}".format(agentName, iAgent)
        logDir = "./agentData/{}_{:d}_logs".format(agentName, iAgent)

        # Create the training environment
        env = SubprocVecEnv([auv.make_env(i, env_kwargs=env_kwargs) for i in range(nProc)])
        env = VecMonitor(env, logDir)

        # Create the agent.
        sacAgent = stable_baselines3.SAC(
            "MlpPolicy", env, policy_kwargs=policy_kwargs, **agent_kwargs)

        # !!! This is the critical bit. !!!
        # Copy the weights from the pre-trained actor.
        sacAgent.actor.latent_pi = copy.deepcopy(pretrained_actor.latent_pi)
        sacAgent.actor.mu = copy.deepcopy(pretrained_actor.mu)

        # Train the agent for N steps
        convergenceData.append(resources.trainAgent(sacAgent, nTrainingSteps, saveFile, logDir))

    # Save metadata in human-readable format.
    resources.saveHyperparameteres(
        agentName, agent_kwargs, policy_kwargs, env_kwargs, nTrainingSteps)

    # Plot convergence of each agent.
    iBest, _ = resources.plotTraining(
        convergenceData, saveAs="./agentData/{}_convergence.png".format(agentName))
