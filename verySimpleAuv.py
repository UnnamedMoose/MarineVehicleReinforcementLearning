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
import torch
import re
import stable_baselines3
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common import results_plotter

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)


# Configure the coordinate system. Either use directions like they would appear
# on a map (consistent with manoeuvring theory) or like they appear in traditional
# Cartesian coordinate systems (e.g. inside a CFD code).
# orientation = "north_east_clockwise"
orientation = "right_up_anticlockwise"


def headingError(psi_d, psi):
    a = (psi_d - psi) % (2.*np.pi)
    b = (psi - psi_d) % (2.*np.pi)
    diff = a if a < b else -b
    return diff


class AuvEnv(gym.Env):
    def __init__(self, dt=0.02):
        # Call base class constructor.
        super(AuvEnv, self).__init__()

        # Dry mass and inertia..
        self.m = 11.4
        self.Izz = 0.16

        # Basic forceand moment coefficients
        self.Xuu = -18.18 * 2.21 # kg/m
        self.Yvv = -21.66 * 4.87
        self.Nrr =  -1.55 # kg m^2 / rad^2
        self.Xu = -4.03 * 2.21
        self.Yv = -6.22 * 4.87
        self.Nr = -0.07

        # Max actuation
        self.maxForce = 150.  # N
        self.maxMoment = 20.  # Nm

        # Non-dimensional (x, y) force and yaw moment.
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # Observation space.
        # TODO change length here when adding new items to the state.
        lenState = 4
        self.observation_space = gym.spaces.Box(
            -1*np.ones(lenState, dtype=np.float32),
            np.ones(lenState, dtype=np.float32),
            shape=(lenState,))

        self._max_episode_steps = 1000

        # For deciding when the vehicle has moved too far away from the goal.
        self.xMinMax = [-1, 1]
        self.yMinMax = [-1, 1]

        # Updates at fixed intervals.
        self.iStep = 0
        self.dt = dt

        # Used in computation of the reward.
        self.state = None
        self.steps_beyond_done = None

        # time trace of all important quantities. Most retrieved from the vehicle model itself
        self.timeHistory = []

    def dataToState(self, pos, heading):
        # Non-dimensionalise the position error (unit vector towards the target).
        perr = self.positionTarget - pos
        dTarget = np.linalg.norm(perr)
        perr /= max(1e-6, dTarget)

        # Get heading error by comparing on both sides of zero.
        # Clip the heading error to <-1, 1>. This should be implicit in the headingError
        # function but a limiter never hurts.
        herr = min(1., max(-1., headingError(self.headingTarget, heading)/np.pi))

        # Actual state are values relative to the target and distance to target.
        newState = np.concatenate([perr, [herr, dTarget]])

        # TODO add more items here. Basic controller needs the first three elements
        #   to stay like this.

        return newState

    def reset(self, keepTimeHistory=False):
        self.position = np.random.rand(2) * 0.5 * [self.xMinMax[1]-self.xMinMax[0], self.yMinMax[1]-self.yMinMax[0]] \
            + [self.xMinMax[0], self.yMinMax[0]]
        self.positionStart = self.position.copy()
        self.positionTarget = np.zeros(2)
        self.heading = np.random.rand()*2.*np.pi
        self.headingStart = self.heading
        self.headingTarget = np.random.rand()*2.*np.pi
        self.velocities = np.zeros(3)
        self.accelerations = np.zeros(3)
        self.time = 0
        self.iStep = 0
        self.steps_beyond_done = 0
        self.state = self.dataToState(self.position, self.heading)
        self.timeHistory = []

        return self.state

    def step(self, action):
        # Set new time.
        self.iStep += 1
        self.time += self.dt

        # Scale the actions.
        Fset = action[:2]*self.maxForce
        Nset = action[2]*self.maxMoment

        # Compute total forces and moments in the global reference frame.
        # NOTE: this is a very simplified problem definition, ignoring rigid body
        #   accelerations and cross-coupling terms.
        X = (self.Xu + self.Xuu*np.abs(self.velocities[0]))*self.velocities[0] + Fset[0]
        Y = (self.Yv + self.Yvv*np.abs(self.velocities[1]))*self.velocities[1] + Fset[1]
        N = (self.Nr + self.Nrr*np.abs(self.velocities[2]))*self.velocities[2] + Nset

        # Advance using the Euler method.
        # NOTE: this ignores added mass and inertia due to fluid accelerations.
        self.accelerations = np.array([
            X/self.m,
            Y/self.m,
            N/self.Izz
        ])
        dydt = np.append(self.velocities, self.accelerations)
        y = np.concatenate([self.position, [self.heading], self.velocities])
        y = y + dydt*self.dt
        position = y[:2]
        heading = y[2] % (2.*np.pi)
        velocities = y[3:]

        # Compute state.
        self.state = self.dataToState(position, heading)

        # Compute the reward.
        # TODO add more components here.
        perr = self.positionTarget - position
        herr = headingError(self.headingTarget, heading)

        # --- Reward 1: sum of all absolute errors scaled to reasonable value. ---
        # angleScale = 180. / 180. * np.pi
        # distScale = 0.3
        # rewardTerms = -0.5*np.array([
        #     min(1., np.abs(perr[0])/distScale),
        #     min(1., np.abs(perr[1])/distScale),
        #     min(1., np.abs(herr)/angleScale)
        # ])

        # --- Reward 2: reduce error in both position and heading errors. ---
        # perr_o = self.positionTarget - self.position
        # herr_o = headingError(self.headingTarget, self.heading)
        # rewardTerms = np.array([
        #     0.1*max(-1., min(1., np.abs(perr_o[0]) / max(1e-6, np.abs(perr[0])) - 1.)),
        #     0.1*max(-1., min(1., np.abs(perr_o[1]) / max(1e-6, np.abs(perr[1])) - 1.)),
        #     0.5*max(-1., min(1., np.abs(herr_o) / max(1e-6, np.abs(herr)) - 1.)),
        #     -1.*sum(action**2.),
        # ])

        # --- Reward 3: reduce error in both position and heading errors. ---
        # Reward components equal to zero at value=scaleFactor
        # angleScale = 90. / 180. * np.pi
        distScale = 0.3
        rewardTerms = np.array([
            0.1*np.clip(-np.log(max(1e-12, np.abs(perr[0])/distScale)), -2., 2.)/2.,
            0.1*np.clip(-np.log(max(1e-12, np.abs(perr[1])/distScale)), -2., 2.)/2.,
            # 0.1*np.clip(-np.log(max(1e-12, (np.abs(herr)/angleScale)**0.5)), -2., 2.)/2.,
            -(np.abs(herr)/np.pi)**0.5,
            -0.1*sum(action**2.),
        ])

        # Get total reward.
        reward = np.sum(rewardTerms)

        # Update the position and heading at the new time value.
        self.position = position
        self.heading = heading
        self.velocities = velocities

        # Check if domain exceeded.
        done = False
        if (self.position[0] < self.xMinMax[0]) or (self.position[0] > self.xMinMax[1]):
            done = True
            reward += -1000.
        if (self.position[1] < self.yMinMax[0]) or (self.position[1] > self.yMinMax[1]):
            done = True
            reward += -1000.
        if self.iStep >= self._max_episode_steps:
            done = True

        # Store stats.
        self.timeHistory.append(dict(zip(
            ["step", "time", "reward", "x", "y", "psi", "x_d", "y_d", "psi_d"] \
                +["Fx", "Fy", "N", "Fx_set", "Fy_set", "N_set"] \
                +["r{:d}".format(i) for i in range(len(rewardTerms))] \
                +["a{:d}".format(i) for i in range(len(action))] \
                +["s{:d}".format(i) for i in range(len(self.state))],
            np.concatenate([[self.iStep, self.time, reward], self.position, [self.heading],
                            self.positionTarget, [self.headingTarget], [X, Y, N, Fset[0], Fset[1], Nset],
                            rewardTerms, action, self.state])
        )))
        if done:
            self.timeHistory = pandas.DataFrame(self.timeHistory)

        if done:
            self.steps_beyond_done += 1
        else:
            self.steps_beyond_done = 0

        return self.state, reward, done, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass


def make_env(rank, seed=0, envKwargs={}):
    """
    Utility function for multiprocessed env.

    :param filename: (str) path to file from which the env is created
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = AuvEnv(**envKwargs)
        env.seed(seed + rank)
        return env
    return _init
    pass


def evaluate_agent(model, env, num_episodes=1, num_steps=None,
                    num_last_for_reward=None, render=False):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    frames = []

    # This function will only work for a single Environment
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        if num_steps is None:
            num_steps = 1000000
        for i in range(num_steps):
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            if render:
                frames.append(env.render(mode="rgb_array"))
            if done:
                break

        if num_last_for_reward is None:
            all_episode_rewards.append(sum(episode_rewards))
        else:
            all_episode_rewards.append(np.mean(episode_rewards[-num_last_for_reward:], 1))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("  Mean reward:", mean_episode_reward)
    print("  Num episodes:", num_episodes)

    if render:
        return frames, mean_episode_reward
    else:
        return mean_episode_reward


def plot_horizontal(ax, x, y, psi, scale=1, markerSize=1, arrowSize=1, vehicleColour="y"):
    """ Plot a representation of the AUV on the given axes.
    Thuster forces should be non-dimensionalised with maximum value. """

    x0 = np.array([x, y])

    Length = 0.457 # m (overall dimensions of the AUV)
    Width = 0.338
    # Only used for plotting now.
    D_thruster = 0.1  # Diameter of thrusters

    xyHull = np.array([
        [Length/2., -Width/2.+D_thruster],
        [Length/2., Width/2.-D_thruster],
        [Length/2.-D_thruster, Width/2.],
        [-Length/2.+D_thruster, Width/2.],
        [-Length/2., Width/2.-D_thruster],
        [-Length/2., -Width/2.+D_thruster],
        [-Length/2.+D_thruster, -Width/2.],
        [Length/2.-D_thruster, -Width/2.],
        [Length/2., -Width/2.+D_thruster],
    ])

    xyCentreline = np.array([
        [np.min(xyHull[:,0]), 0.],
        [np.max(xyHull[:,0]), 0.],
    ])

    xyDir = np.array([
        [Length/2.-Width/4., -Width/4.],
        [Length/2., 0.],
        [Length/2.-Width/4., Width/4.],
    ])

    def rotate(xy, psi):
        xyn = np.zeros(xy.shape)
        xyn[:,0] = np.cos(psi)*xy[:,0] - np.sin(psi)*xy[:,1]
        xyn[:,1] = np.sin(psi)*xy[:,0] + np.cos(psi)*xy[:,1]
        return xyn

    xyHull = rotate(xyHull*scale, psi) + x0
    xyCentreline = rotate(xyCentreline*scale, psi) + x0
    xyDir = rotate(xyDir*scale, psi) + x0

    if orientation == "north_east_clockwise":
        i0 = 1
        i1 = 0
    elif orientation == "right_up_anticlockwise":
        i0 = 0
        i1 = 1
    else:
        raise ValueError("Wrong orientation")

    objects = []
    objects += ax.fill(xyHull[:,i0], xyHull[:,i1], vehicleColour, alpha=0.5)
    objects += ax.plot(xyCentreline[:,i0], xyCentreline[:,i1], "k--", lw=2*markerSize)
    objects += ax.plot(xyDir[:,i0], xyDir[:,i1], "k-", lw=2*markerSize)
    objects += ax.plot(x0[i0], x0[i1], "ko", mew=3, mfc="None", ms=14*markerSize)

    return objects


def plotEpisode(env_to_plot, title=""):

    fig, ax = plt.subplots(figsize=(10, 10))

    if orientation == "north_east_clockwise":
        ax.set_xlabel("y [m, +ve east]")
        ax.set_ylabel("x [m, +ve north]")
        i0 = 1
        i1 = 0
    elif orientation == "right_up_anticlockwise":
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        i0 = 0
        i1 = 1
    else:
        raise ValueError("Wrong orientation")

    ax.set_xlim(np.array(env_to_plot.xMinMax)+[-0.2, 0.2])
    ax.set_ylim(np.array(env_to_plot.yMinMax)+[-0.2, 0.2])
    ax.set_aspect("equal", "box")

    lns = []

    lns += ax.plot([env_to_plot.positionStart[i0], env_to_plot.positionTarget[i0]],
                    [env_to_plot.positionStart[i1], env_to_plot.positionTarget[i1]],
                    "go--", lw=2, ms=8, mew=2, mec="g", mfc="None", label="$Waypoints$")

    xyHeading = 0.5*np.cos(env_to_plot.headingTarget), 0.5*np.sin(env_to_plot.headingTarget)
    ax.plot([0, xyHeading[i0]], [0, xyHeading[i1]], "g-", lw=4, alpha=0.5)

    lns += ax.plot(env_to_plot.timeHistory[["x", "y"]].values[0, i0], env_to_plot.timeHistory[["x", "y"]].values[0, i1],
            "bs", ms=11, mew=2, mec="b", mfc="None", label="$Start$")

    lns += ax.plot(env_to_plot.timeHistory[["x", "y"]].values[-1, i0], env_to_plot.timeHistory[["x", "y"]].values[-1, i1],
            "bd", ms=11, mew=2, mec="b", mfc="None", label="$End$")

    lns += ax.plot(env_to_plot.timeHistory[["x", "y"]].values[:, i0], env_to_plot.timeHistory[["x", "y"]].values[:, i1], "k-",
        mec="k", lw=2, mew=2, ms=9, mfc="None", label="$Trajectory$")

    ax.legend(lns, [l.get_label() for l in lns], loc="lower center",
                    bbox_to_anchor=(0.5, 1.01), prop={"size":18}, ncol=4)

    tPlot = np.linspace(0, env_to_plot.timeHistory["time"].max(), 5)
    iToPlot = [np.argmin(np.abs(env_to_plot.timeHistory["time"] - t)) for t in tPlot]

    for i in iToPlot:
        plot_horizontal(ax, env_to_plot.timeHistory["x"].values[i],
                        env_to_plot.timeHistory["y"].values[i],
                        env_to_plot.timeHistory["psi"].values[i],
                        markerSize=0.5, arrowSize=1, scale=1)

    ax.text(ax.get_xlim()[1], ax.get_ylim()[0], "  "+title, va="bottom", ha="right", size=18)

    return fig, ax


def plotDetail(envs_to_plot, labels=None, title=""):
    fields = ["x", "y", "psi"]
    flabels = ["x [m]", "y [m]", "$\psi$ [deg]"]
    try:
        nEnvs = len(envs_to_plot)
    except TypeError:
        nEnvs = 1
        envs_to_plot = [envs_to_plot]
    if labels is None:
        labels = ["Env{:d}".format(i) for i in range(nEnvs)]
    caseColours = plt.cm.jet(np.linspace(0, 1, nEnvs))
    mults = [1., 1., 180./np.pi]
    for i, f in enumerate(fields):
        fig, ax = plt.subplots()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(flabels[i])
        if f == "psi":
            ax.set_ylim((0, 360.))
            ax.yaxis.set_ticks(np.arange(0, 361, 60))
        for iEnv in range(nEnvs):
            ax.plot(envs_to_plot[iEnv].timeHistory["time"].values,
                    envs_to_plot[iEnv].timeHistory[f].values*mults[i],
                    "-", lw=2, c=caseColours[iEnv], label=labels[iEnv])
            ax.plot(envs_to_plot[iEnv].timeHistory["time"].values,
                    envs_to_plot[iEnv].timeHistory[f+"_d"].values*mults[i],
                    "--", lw=1, c=caseColours[iEnv])
        if nEnvs > 1:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)

    fig, ax = plt.subplots()
    ax.set_xlabel("Time step")
    ax.set_ylabel("Reward")
    for iEnv in range(nEnvs):
        ax.plot(envs_to_plot[iEnv].timeHistory["reward"], "-", lw=2,
                c=caseColours[iEnv], label=labels[iEnv])

    for f in envs_to_plot[0].timeHistory.keys():
        if re.match("a[0-9]+", f) or re.match("s[0-9]+", f) or re.match("r[0-9]+", f):
            fig, ax = plt.subplots()
            ax.set_xlabel("Time step")
            ax.set_ylabel(f.replace("a", "Action ").replace("s", "State ").replace("r", "Reward "))
            for iEnv in range(nEnvs):
                ax.plot(envs_to_plot[iEnv].timeHistory[f], "-", lw=2,
                        c=caseColours[iEnv], label=labels[iEnv])
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)


class PDController(object):
    """ Simple controller that emulates the action of a model used in stable baselines.
    The returned forces are capped at +/- 1 to mimic actions of an RL agent.
    Observations are assumed to start with direction to target and scaled heading error. """
    # NOTE: this is prone to chatter due to the way the states are defined, i.e.
    #   when obs will flick constantly between -1/+1 the controller will respond
    #   too abruptly. For the heading this may also (rarely) lead to it getting stuck
    #   at the target angle + pi sometimes. Not important since it's only here
    #   to test the environment and not show optimal classical control.
    def __init__(self, dt, P=[1., 1., 1.], D=[0.05, 0.05, 0.01]):
        self.P = np.array(P)
        self.D = np.array(D)
        self.dt = dt
        self.oldObs = None

    def predict(self, obs):
        states = np.zeros(len(self.P))
        if self.oldObs is None:
            self.oldObs = obs
        actions = np.clip(obs[:3]*self.P + (obs[:3] - self.oldObs[:3])/self.dt*self.D, -1., 1.)
        self.oldObs = obs
        return actions, states


if __name__ == "__main__":

    modelName = "SAC_try0"

    nProc = 16

    # TODO adjust the hyperparameters here.
    nTrainingSteps = 1_000_000

    model_kwargs = {
        'learning_rate': 5e-4,
        'gamma':  0.99,
        'verbose': 1,
        'batch_size': 256,
        'buffer_size': int(5e6),
        'learning_starts': 100,
    }
    policy_kwargs = {
        "activation_fn": torch.nn.GELU,
        "net_arch": dict(
            # Actor - determines action for a specific state
            pi=[128],
            # Critic - estimates value of each state-action combination
            qf=[128, 128])
    }

    # Set up constants etc.
    saveFile = "./modelData/{}".format(modelName)
    logDir = "./modelData/{}_logs".format(modelName)
    os.makedirs(logDir, exist_ok=True)

    # Create the environments.
    env_eval = AuvEnv()
    env = SubprocVecEnv([make_env(i, envKwargs={}) for i in range(nProc)])
    env = VecMonitor(env, logDir)

    # Create the model using stable baselines.
    model = stable_baselines3.SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, **model_kwargs)

    # Train the agent for N steps
    starttime = datetime.datetime.now()
    print("\nTraining started at", str(starttime))
    model.learn(total_timesteps=nTrainingSteps, log_interval=200)
    endtime = datetime.datetime.now()
    trainingTime = (endtime-starttime).total_seconds()
    print("Training took {:.0f} seconds ({:.0f} minutes)".format(trainingTime, trainingTime/60.))

    # Plot training results.
    fig, ax = plt.subplots()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    try:
        results_plotter.plot_results(
            [logDir], num_timesteps=1e15, x_axis="episodes", task_name="", figsize=(10, 8))
    except IndexError:
        pass
    ax.set_ylim((max(ax.get_ylim()[0], -1500), ax.get_ylim()[1]))
    plt.tight_layout()

    # Trained agent.
    print("\nAfter training")
    mean_reward = evaluate_agent(model, env_eval)
    plotEpisode(env_eval, "RL control")

    # Dumb agent.
    print("\nSimple control")
    env_eval_pd = AuvEnv()
    pdController = PDController(env_eval_pd.dt)
    mean_reward = evaluate_agent(pdController, env_eval_pd)
    fig, ax = plotEpisode(env_eval_pd, "Simple control")

    # Compare detail
    plotDetail([env_eval_pd, env_eval], labels=["Simple control", "RL control"])
