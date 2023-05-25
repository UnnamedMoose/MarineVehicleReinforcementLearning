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

    # --- Pretraining parameters ---
    nPretrainEpisodes = 100
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
    nAgents = 3

    # Any found agent will be left alone unless this is set to true.
    overwrite = True

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
        "target_entropy": -4.,
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

    # Create an evaluation environment and Proportional-Derivative controller.
    env_eval = auv.AuvEnv(**env_kwargs_evaluation)
    pdController = auv.PDController(env_eval.dt)

    # Create an equivalent to imitation::util::crete_vec_env that instantiates
    # the custom environment.
    rng = np.random.default_rng(0)

    # Simply create the env without any wrappers.
    env_pretrain = auv.AuvEnv(**env_kwargs_pretrain)

    # Create own rollout generation to circumvent the issues with the imitation wrappers.
    # NOTE: this is where data could also be read from files, for instance.
    rollouts = []
    for iEp in range(nPretrainEpisodes):
        obs = env_pretrain.reset()
        observations = [obs]
        actions = np.empty((0, env_pretrain.action_space.shape[0]))
        rewards = []
        for i in range(env_pretrain._max_episode_steps):
            action, _states = pdController.predict(obs, deterministic=True)
            obs, reward, done, info = env_pretrain.step(action)
            observations = np.append(observations, obs[np.newaxis, :], axis=0)
            actions = np.append(actions, action[np.newaxis, :], axis=0)
            rewards = np.append(rewards, reward)

        out_dict_stacked = {"rews": rewards, "acts": actions, "obs": observations, "infos": None}
        traj = TrajectoryWithRew(**out_dict_stacked, terminal=done)
        assert traj.rews.shape[0] == traj.acts.shape[0] == traj.obs.shape[0] - 1

        rollouts.append(traj)

    # Pack the generated rollouts into DataFrames for eventual saving and easier visualisation.
    pretrainedEpisodes = []
    for r in rollouts:
        pretrainedEpisodes.append(pandas.DataFrame(
            data=np.concatenate([r.obs[1:, :], r.acts, r.rews[:, np.newaxis]], axis=1),
            columns=["s{:d}".format(i) for i in range(r.obs.shape[1])] +
                ["a{:d}".format(i) for i in range(r.acts.shape[1])] + ["r"]))

    # Plot the different training episodes.
    fig, ax = plt.subplots()
    ax.set_xlabel("s0")
    ax.set_ylabel("s1")
    for e in pretrainedEpisodes:
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
        rewards_init, _ = resources.evaluate_agent(agent, env_eval, num_episodes=100)

        # Pretrain.
        reward_net = BasicRewardNet(
        # reward_net = BasicShapedRewardNet(
            env_pretrain.observation_space,
            env_pretrain.action_space,
            normalize_input_layer=RunningNorm,
        )
        pretrainer = GAIL(
        # pretrainer = AIRL(
            demonstrations=rollouts,
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
        rewards_pre, _ = resources.evaluate_agent(agent, env_eval, num_episodes=100)

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
            saveFile+"_pretrained", env=env,
            action_noise=VectorizedActionNoise(NormalActionNoise(np.zeros(3), 0.05*np.ones(3)), nProc))

        # Train the agent for N more steps
        conv, trainingTime = resources.trainAgent(agent, nTrainingSteps, saveFile)
        convergenceData.append(conv)
        trainingTimes.append(trainingTime)
        agents.append(agent)

        # Evaluate
        print("\nTrained agent")
        rewards_trained, _ = resources.evaluate_agent(agent, env_eval, num_episodes=100)

        # Plot convergence of each agent. Redo after each agent to provide
        # intermediate updates on how the training is going.
        iBest, fig, ax = resources.plotTraining(
            convergenceData, saveAs="./agentData/{}_convergence.png".format(agentName))

        # Check the effects of pretraining.
        fig, ax = plt.subplots()
        ax.set_xlabel("Reward range")
        ax.set_ylabel("Episode count")
        bins = np.linspace(0, rewards_trained.max(), 11)
        h, x = np.histogram(rewards_init, bins=bins)
        x = (x[1:] + x[:-1])/2
        plt.bar(x, h, color="green", alpha=0.5, label="Initialised", width=20)
        h, x = np.histogram(rewards_pre, bins=bins)
        x = (x[1:] + x[:-1])/2
        plt.bar(x, h, color="blue", alpha=0.5, label="Pretrained", width=20)
        h, x = np.histogram(rewards_trained, bins=bins)
        x = (x[1:] + x[:-1])/2
        plt.bar(x, h, color="red", alpha=0.5, label="Pretrained+Trained", width=20)
        ax.legend()

    # Save metadata in human-readable format.
    resources.saveHyperparameteres(
        agentName, agent_kwargs, policy_kwargs, env_kwargs, nTrainingSteps, trainingTimes, nProc)


# %% Generate training data using a PD controller.
    # if do_generateData:
    #     print("\nGenerating training data using a simple controller")
    #     pdController = auv.PDController(env_eval_pd.dt)
    #     mean_reward_pd, allRewards_pd = resources.evaluate_agent(
    #         pdController, env_eval_pd, num_episodes=100, saveDir="testEpisodes")

# %% Train an actor network for approximating the control output of the simple controller.
    # # Read the pre-computed data
    # pretrainEpisodes = [pandas.read_csv(os.path.join("testEpisodes", f)) for f in os.listdir("testEpisodes")]
    # print("Read {:d} pretraining episodes".format(len(pretrainEpisodes)))

    # # Choose episodes for pretraining at random.
    # iPretrain = np.random.default_rng().choice(
    #     len(pretrainEpisodes), size=(nPretrainEpisodes,), replace=False, shuffle=False)

# %%

    # # Flatten to a single buffer.
    # buffer = pandas.concat(pretrainEpisodes).reset_index(drop=True)

    # # Plot the different training episodes.
    # fig, ax = plt.subplots()
    # ax.set_xlabel("x [m]")
    # ax.set_ylabel("y [m]")
    # cs = ax.scatter(buffer["x"], buffer["y"], c=buffer["reward"], s=5)
    # cbar = plt.colorbar(cs)
    # cbar.set_label("Reward")

    # Get the state and action variables from the buffer.
    # stateVars = [k for k in buffer.keys() if re.match("s[0-9]+", k)]
    # actionVars = [k for k in buffer.keys() if re.match("a[0-9]+", k)]

#     # Construct a neural network for approximating the simple controller output.
#     # This mimics the actor part of stable baselines SAC agent exactly.
#     class NeuralNetwork(nn.Module):
#         def __init__(self):
#             super(NeuralNetwork, self).__init__()
#             self.flatten = nn.Flatten()
#             net_arch = policy_kwargs["net_arch"]["pi"]
#             latent_pi_net = create_mlp(
#                 len(stateVars), -1, net_arch, policy_kwargs["activation_fn"])
#             self.latent_pi = nn.Sequential(*latent_pi_net)
#             last_layer_dim = net_arch[-1]
#             self.mu = nn.Linear(last_layer_dim, len(actionVars))

#         def forward(self, x):
#             features = self.flatten(x)
#             latent_pi  = self.latent_pi(features)
#             actions = self.mu(latent_pi)
#             return actions

#     def train(dataloader, model, loss_fn, optimizer):
#         size = len(dataloader.dataset)
#         model.train()
#         for batch, (X, y) in enumerate(dataloader):
#             X, y = X.to(device), y.to(device)

#             # Compute prediction error
#             pred = model(X)
#             loss = loss_fn(pred, y)

#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if batch % 100 == 0:
#                 loss, current = loss.item(), batch * len(X)
#                 print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#     def test(dataloader, model, loss_fn):
#         num_batches = len(dataloader)
#         model.eval()
#         test_loss = 0
#         with th.no_grad():
#             for X, y in dataloader:
#                 X, y = X.to(device), y.to(device)
#                 pred = model(X)
#                 test_loss += loss_fn(pred, y).item()
#         test_loss /= num_batches
#         print(f"Test error average loss: {test_loss:>8f}")

#     def predict(model, state):
#         return model.forward(th.from_numpy(state[np.newaxis, :]).float()).detach().numpy().flatten()

#     # Extract data as arrays.
#     X = buffer[stateVars].values
#     Y = buffer[actionVars].values
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

#     # Convert the data from Numpy to a TensorDataset from Pytorch
#     X_torch = th.from_numpy(X).float()
#     y_torch = th.from_numpy(Y).float()
#     X_train_torch = th.from_numpy(X_train).float()
#     y_train_torch = th.from_numpy(y_train).float()
#     X_test_torch = th.from_numpy(X_test).float()
#     y_test_torch = th.from_numpy(y_test).float()

#     # These objects sample the data tensor with a given batch size
#     batch_size = 128
#     training_data = th.utils.data.TensorDataset(X_train_torch, y_train_torch)
#     train_dataloader = th.utils.data.DataLoader(training_data, batch_size=batch_size)
#     testing_data = th.utils.data.TensorDataset(X_test_torch, y_test_torch)
#     test_dataloader = th.utils.data.DataLoader(testing_data, batch_size=batch_size)

#     # Create the network
#     device = "cuda" if th.cuda.is_available() else "cpu"
#     pretrained_actor = NeuralNetwork().to(device)

#     # choose a loss function and an optimizer
#     loss_fn = nn.MSELoss()
#     optimizer = th.optim.Adam(pretrained_actor.parameters(), lr=1e-3)

#     # Train.
#     t_start = datetime.datetime.now()
#     epochs = 25
#     loss = np.zeros(epochs)
#     val_loss = np.zeros(epochs)
#     for t in range(epochs):
#         print(f"\nEpoch {t+1}\n-------------------------------")
#         train(train_dataloader, pretrained_actor, loss_fn, optimizer)
#         test(test_dataloader, pretrained_actor, loss_fn)

#         # keep track of the errors
#         pred_train = pretrained_actor(X_train_torch)
#         loss[t] = loss_fn(pred_train, y_train_torch)
#         pred_test = pretrained_actor(X_test_torch)
#         val_loss[t] = loss_fn(pred_test, y_test_torch)
#     t_end = datetime.datetime.now()
#     trainingTime = (t_end-t_start).total_seconds()
#     print("\nTraining took {:.0f} seconds ({:.0f} minutes)".format(trainingTime, trainingTime/60.))

#     # plot the evolution of the training error
#     fig, ax = plt.subplots()
#     ax.set_yscale("log")
#     ax.plot(loss, label='Training')
#     ax.plot(val_loss, label='Validation')
#     ax.set_xlabel('Epochs')
#     ax.set_ylabel('Error')
#     ax.legend(loc="upper center", ncol=2)

# # %% Evaluate the training and results.

#     # Create a controller that uses the network to predict the correct action.
#     class NNController(object):
#         def __init__(self, dt, net):
#             self.dt = dt
#             self.net = net

#         def predict(self, obs, deterministic=True):
#             # NOTE deterministic is a dummy kwarg needed to make this function look
#             # like a stable baselines equivalent
#             states = None
#             actions = predict(self.net, obs)
#             return actions, states

#     # Create env and controller instance, then evaluate.
#     env_eval_nn = auv.AuvEnv(**env_kwargs_evaluation)
#     nnController = NNController(env_eval_nn.dt, pretrained_actor)
#     print ("\nEvaluating direct NN control")
#     mean_reward_nn, allRewards_nn = resources.evaluate_agent(
#         nnController, env_eval_nn, num_episodes=1)

#     # Evaluate using the simple controller for comparison.
#     env_eval_pd = auv.AuvEnv(**env_kwargs_evaluation)
#     pdController = auv.PDController(env_eval_pd.dt)
#     print ("\nEvaluating simple PD control")
#     mean_reward_pd, allRewards_pd = resources.evaluate_agent(
#         pdController, env_eval_pd, num_episodes=1)

#     # Plot.
#     resources.plotEpisode(env_eval_nn, "Direct NN control")
#     # resources.plotDetail([env_eval_pd, env_eval_nn], labels=["PD control", "NN control"])

# # %% Create a real SAC agent and copy the weights.

#     # Train several times to make sure the agent doesn't just get lucky.
#     convergenceData = []
#     trainingTimeAvg = 0
#     agents = []
#     for iAgent in range(nAgents):
#         saveFile = "./agentData/{}_{:d}".format(agentName, iAgent)

#         # Create the training environment
#         env = SubprocVecEnv([auv.make_env(i, env_kwargs=env_kwargs) for i in range(nProc)])
#         env = VecMonitor(env, saveFile)

#         # Create the agent.
#         sacAgent = stable_baselines3.SAC(
#             "MlpPolicy", env, policy_kwargs=policy_kwargs, **agent_kwargs)

#         # !!! This is the critical bit. !!!
#         # Copy the weights from the pre-trained actor.
#         sacAgent.actor.latent_pi = copy.deepcopy(pretrained_actor.latent_pi)
#         sacAgent.actor.mu = copy.deepcopy(pretrained_actor.mu)
#         # Add noise to enhance exploration and avoid overfitting the simple controller.
#         # with th.no_grad():
#         #     for param in sacAgent.actor.latent_pi.parameters():
#         #         param.add_(th.randn(param.size())*5e-3)
#         #     for param in sacAgent.actor.mu.parameters():
#         #         param.add_(th.randn(param.size())*5e-3)

#         # Train the agent for N steps
#         conv, trainingTime = resources.trainAgent(sacAgent, nTrainingSteps, saveFile)
#         convergenceData.append(conv)
#         trainingTimeAvg += trainingTime/nAgents
#         agents.append(sacAgent)

#     # Save metadata in human-readable format.
#     resources.saveHyperparameteres(
#         agentName, agent_kwargs, policy_kwargs, env_kwargs, nTrainingSteps, trainingTimeAvg, nProc)

#     # Plot convergence of each agent.
#     iBest, fig, ax = resources.plotTraining(
#         convergenceData, saveAs="./agentData/{}_convergence.png".format(agentName))
