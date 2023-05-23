# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 08:32:22 2023

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
from scipy.interpolate import griddata

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

# Define the network
class CriticNeuralNetwork(nn.Module):
    def __init__(self):
        super(CriticNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        net_arch = policy_kwargs["net_arch"]["qf"]
        qf_net = create_mlp(
            dimIn, -1, net_arch, policy_kwargs["activation_fn"])
        qf_net.append(nn.Linear(net_arch[-1], 1))
        self.qf = nn.Sequential(*qf_net)

    def forward(self, x):
        features = self.flatten(x)
        q  = self.qf(features)
        return q

# %% Settings.
if __name__ == "__main__":
    do_estimateQ = False
    do_trainCritic = True
    do_evaluate = False

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

    # Read the pre-computed data (created in the actor part)
    trainingEpisodes = [pandas.read_csv(os.path.join("testEpisodes", f)) for f in os.listdir("testEpisodes")]
    buffer = pandas.concat(trainingEpisodes).reset_index(drop=True)

    # Plot the different training episodes.
    fig, ax = plt.subplots()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cs = ax.scatter(buffer["x"], buffer["y"], c=buffer["reward"], s=5)
    cbar = plt.colorbar(cs)
    cbar.set_label("Reward")

# %% Estimate Q

    # Constants.
    learningRate = 0.1
    gamma = 0.95

    # Get the state and action variables from the buffer.
    stateVars = [k for k in buffer.keys() if re.match("s[0-9]+", k)]
    actionVars = [k for k in buffer.keys() if re.match("a[0-9]+", k)]

    # !!! for now, only keep first three states to enable basic control.
    stateVars = stateVars[:3]

    # Size of the discrete observation space. The same is used for actions
    discrete_var_size = 11
    discrete_var_delta = (1.-(-1.))/(discrete_var_size-1)

    # Continous to discrete state converter
    def get_discrete_var(state):
        discrete_var = np.round((np.clip(state, -1., 1.) - (-1.))/discrete_var_delta).astype(int)
        return discrete_var

    if do_estimateQ:
        # Initialization q table
        q_table = np.random.uniform(
            low=-2, high=0, size=([len(actionVars)]+[discrete_var_size]*(len(stateVars)+1)))

        # q_table2 = q_table.copy()

        # Run the estimation.
        time_start = datetime.datetime.now()

        for iEpisode in range(len(trainingEpisodes)):
            if iEpisode % 100 == 0:
                print("Estimating Q using episode {:d}".format(iEpisode))

            # Approach 1 - takes 77 s per 100 episodes

            for i in range(1, trainingEpisodes[iEpisode].shape[0]):
                # Retrieve the state, the action taken, and the state that resulted from the action.
                discrete_actions = tuple(get_discrete_var(trainingEpisodes[iEpisode].loc[i-1, actionVars]))
                current_discrete_state = tuple(get_discrete_var(trainingEpisodes[iEpisode].loc[i-1, stateVars]))
                new_discrete_state = tuple(get_discrete_var(trainingEpisodes[iEpisode].loc[i, stateVars]))
                reward = trainingEpisodes[iEpisode].loc[i, "r"]

                # Iterate in each action dimension to make matrix ops easier to read.
                for iAction, discrete_action in enumerate(discrete_actions):
                    # Maximum possible Q value in next step (for new state)
                    max_future_q = np.max(q_table[iAction][new_discrete_state])

                    # Current Q value (for current state and performed action)
                    current_q = q_table[iAction][current_discrete_state][discrete_action]

                    # Compute the new Q value for current state and action
                    new_q = (1 - learningRate) * current_q + learningRate * (reward + gamma * max_future_q)

                    # Update Q table.
                    q_table[iAction][current_discrete_state + (discrete_action,)] = new_q

            # Approach 2

            # TODO would need to recurse gamma

            # Iterate in each action dimension to make matrix ops easier to read.
            # for iAction, action in enumerate(actionVars):

            #     # Retrieve the state, the action taken, and the state that resulted from the action.
            #     discrete_action = get_discrete_var(trainingEpisodes[iEpisode][action].values[:-1])
            #     current_discrete_state = get_discrete_var(trainingEpisodes[iEpisode][stateVars].values[:-1])
            #     new_discrete_state = get_discrete_var(trainingEpisodes[iEpisode][stateVars].values[1:])
            #     reward = trainingEpisodes[iEpisode]["r"].values[1:]

            #     # Maximum possible Q value in next step (for new state)
            #     max_future_q = np.max([q_table2[iAction][tuple(s)] for s in new_discrete_state], axis=1)

            #     # Current Q value (for current state and performed action)
            #     current_q = np.array([q_table2[iAction][tuple(current_discrete_state[i]) + (discrete_action[i],)] for i in range(len(discrete_action))])

            #     # Compute the new Q value for current state and action
            #     new_q = (1 - learningRate) * current_q + learningRate * (reward + gamma * max_future_q)

            #     # Update Q table.
            #     for i in range(len(discrete_action)):
            #         q_table2[iAction][tuple(current_discrete_state[i]) + (discrete_action[i],)] = new_q[i]

        time_end = datetime.datetime.now()
        trainingTime = (time_end-time_start).total_seconds()
        print("Estimating Q took {:.0f} seconds ({:.0f} minutes)".format(trainingTime, trainingTime/60.))

        # Save
        np.savez("./agentData/Qmatrix.npz", q_table=q_table, stateVars=stateVars, actionVars=actionVars,
                 discrete_var_size=discrete_var_size, discrete_var_delta=discrete_var_delta)
    else:
        # Load pre-computed Q-matrix.
        qdat = np.load("./agentData/Qmatrix.npz")
        q_table = qdat["q_table"]
        stateVars = qdat["stateVars"]
        actionVars = qdat["actionVars"]
        discrete_var_size = int(qdat["discrete_var_size"])
        discrete_var_delta = float(qdat["discrete_var_delta"])

# %% Plot the Q value fro two state vars and one action dimension.
    iAction = 0
    fig, ax = plt.subplots(discrete_var_size, figsize=(8, 12))
    plt.subplots_adjust(top=0.96, bottom=0.105, left=0.125, right=0.9, hspace=0.27, wspace=0.2)
    for i in range(discrete_var_size):
        ax[i].set_title("Actuation level {:d}".format(i))
        ax[i].contourf(q_table[iAction, :, :, discrete_var_size//2, i])

# %% Simulate an episode using pure Q-learning.

    if do_evaluate:
        # Create a controller that uses the estimated Q matrix to take actions.
        class QmatrixController(object):
            def __init__(self, dt, Qmat):
                self.dt = dt
                self.Qmat = Qmat

            def predict(self, obs, deterministic=True):
                # NOTE deterministic is a dummy kwarg needed to make this function look
                # like a stable baselines equivalent
                states = None

                # Compute discrete state. Discard last N states that were not included
                # in the Q matrix (if any)
                nKeep = len(self.Qmat.shape) - 2
                discrete_state = tuple(get_discrete_var(obs[:nKeep]))

                # Compute the action.
                nActions = 3
                actions = []
                for iAction in range(nActions):
                    qVals = q_table[iAction][discrete_state]
                    actions = np.append(actions, np.linspace(-1, 1, len(qVals))[np.argmax(qVals)])

                return actions, states

        env_eval_q = auv.AuvEnv(**env_kwargs_evaluation)
        qController = QmatrixController(env_eval_q.dt, q_table)
        mean_reward_q, allRewards_q = resources.evaluate_agent(
            qController, env_eval_q, num_episodes=1)

        resources.plotEpisode(env_eval_q, "Q control")

# %% Generate samples for the critic training.

    # Some state vars are omitted in the q-value table estimates for robustness.
    # They need to be included in order to keep the network architecture correct.
    nStateVarsAll = len([k for k in buffer.keys() if re.match("s[0-9]+", k)])

    # Actor takes in actions and states to predict Q.
    dimIn = nStateVarsAll + len(actionVars)

    # Generate data by interpolating the approximated Q matrix. Assume no variation
    # along the dimensions that were not used in the estimation.
    nSamples = 10000
    X = np.random.rand(nSamples*dimIn).reshape((nSamples, dimIn))

    Y = np.zeros((nSamples, 1))
    for i in range(nSamples):
        # NOTE I wasn't sure so left this here for now.
        # ?? assume actions come first in the vector but this needs to be checked.
        # discrete_actions = tuple(get_discrete_var(X[i, :len(actionVars)]))
        # current_discrete_state = tuple(get_discrete_var(X[i, len(actionVars):len(actionVars)+len(stateVars)]))

        # Actions come last, this is how the critic is called inside SAC:
        # q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
        # min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        discrete_actions = tuple(get_discrete_var(X[i, nStateVarsAll:]))
        current_discrete_state = tuple(get_discrete_var(X[i, :len(stateVars)]))

        # !!! the way the Q matrix is constructed is kind of weird because there is
        # a separate level per action dimension. Maybe it'd be more appropriate to
        # describe the nAction-dimensional space in some flattened version and then
        # interpolate along that imaginary dimension?
        for iAction, discrete_action in enumerate(discrete_actions):
            current_q = q_table[iAction][current_discrete_state][discrete_action]
            # Use mean value.
            Y[i, 0] = current_q/len(actionVars)

# %% Use supervised learning to estimate Q using a NN.

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

    n_epochs_supervised = 25

    if do_trainCritic:
        # Split data.
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
        pretrained_critic = CriticNeuralNetwork().to(device)

        # choose a loss function and an optimizer
        loss_fn = nn.MSELoss()
        optimizer = th.optim.Adam(pretrained_critic.parameters(), lr=1e-3)

        # Train.
        t_start = datetime.datetime.now()
        loss = np.zeros(n_epochs_supervised)
        val_loss = np.zeros(n_epochs_supervised)
        for t in range(n_epochs_supervised):
            print(f"\nEpoch {t+1}\n-------------------------------")
            train(train_dataloader, pretrained_critic, loss_fn, optimizer)
            test(test_dataloader, pretrained_critic, loss_fn)

            # keep track of the errors
            pred_train = pretrained_critic(X_train_torch)
            loss[t] = loss_fn(pred_train, y_train_torch)
            pred_test = pretrained_critic(X_test_torch)
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
        th.save(pretrained_critic, "agentData/pretrained_critic.pt")

    else:
        # load
        pretrained_critic = th.load("agentData/pretrained_critic.pt")
