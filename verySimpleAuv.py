# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 07:43:14 2022

@author: ALidtke
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import re
import warnings
import matplotlib.animation as animation
import collections
import gym
from gym.utils import seeding

import flowGenerator


class PDController(object):
    """ Simple controller that emulates the action of a model used in stable baselines.
    The returned forces are capped at +/- 1 to mimic actions of an RL agent.
    Observations are assumed to start with direction to target and scaled heading error. """
    def __init__(self, dt, P=[1., 1., 1.], D=[0.05, 0.05, 0.01], noiseSigma=None):
        self.P = np.array(P)
        self.D = np.array(D)
        self.dt = dt
        self.oldObs = None
        self.noiseSigma = noiseSigma

    def predict(self, obs, deterministic=True):
        # NOTE deterministic is a dummy kwarg needed to make this function look
        # like a stable baselines equivalent
        states = obs#np.zeros(len(self.P))

        x = obs[:3]

        if self.oldObs is None:
            self.oldObs = x

        actions = np.clip(x*self.P + (x - self.oldObs)/self.dt*self.D, -1., 1.)

        if self.noiseSigma is not None:
            actions += np.random.normal(loc=0., scale=self.noiseSigma, size=actions.shape)

        self.oldObs = x

        return np.clip(actions, -1., 1.), states


def headingError(psi_d, psi):
    """
    Function used for computing signed heading error that wraps around pi properly.

    Parameters
    ----------
    psi_d : float
        Target heading in radians <0, 2pi).
    psi : float
        Current heading in radians <0, 2pi).

    Returns
    -------
    diff : float
        Signed difference in radians <-pi, pi).

    """
    a = (psi_d - psi) % (2.*np.pi)
    b = (psi - psi_d) % (2.*np.pi)
    diff = a if a < b else -b
    return diff


class AuvEnv(gym.Env):
    def __init__(self, seed=None, dt=0.02, noiseMagCoeffs=0.0, noiseMagActuation=0.0,
                 currentVelScale=1.0, currentTurbScale=2.0, stopOnBoundsExceeded=True):
        # Call base class constructor.
        super(AuvEnv, self).__init__()
        self.seed = seed

        # Tied to the no. time values stored in the turbulence data set.
        # self._max_episode_steps = 1500
        self._max_episode_steps = 250

        # Whether or not to stop when bounds are exceeded. Used to disable this
        # check when generating data for adversarial pre-training of the agent
        # that requires episodes of equal length.
        self.stopOnBoundsExceeded = stopOnBoundsExceeded

        # Updates at fixed intervals.
        self.iStep = 0
        self.dt = dt

        self.state = None
        self.steps_beyond_done = None
        self.perr_0 = np.zeros(2)
        self.herr_o = 0.

        # Load the flow data and scale to reasonable values. Keep this fixed for now.
        dataDir = "./turbulenceData"
        self.flow = flowGenerator.ReconstructedFlow(dataDir)
        self.flow.scale(11., currentVelScale, currentTurbScale, translate=(-1.65, -1.1))

        # time trace of all important quantities. Most retrieved from the vehicle model itself
        self.timeHistory = []

        # For deciding when the vehicle has moved too far away from the goal.
        self.xMinMax = [-1, 1]
        self.yMinMax = [-1, 1]

        # Dry mass and inertia..
        self.m = 11.4
        self.Izz = 0.16

        # Basic forceand moment coefficients
        self.Xuu = -18.18 * 2.21 # kg/m
        self.Yvv = -21.66 * 4.87
        self.Nrr = -1.55 # kg m^2 / rad^2
        self.Xu = -4.03 * 2.21
        self.Yv = -6.22 * 4.87
        self.Nr = -0.07

        # Max actuation
        self.maxForce = 150.  # N
        self.maxMoment = 20.  # Nm

        # Noise amplitudes for coefficients and max actuation.
        # Used to encourage generalised training.
        self.noiseMagCoeffs = noiseMagCoeffs
        self.noiseMagActuation = noiseMagActuation

        # Non-dimensional (x, y) force and yaw moment.
        self.lenAction = 3
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0,
                                           shape=(self.lenAction,), dtype=np.float32)

        # Observation space.
        lenState = 9 + 2
        self.observation_space = gym.spaces.Box(
            -1*np.ones(lenState, dtype=np.float32),
            np.ones(lenState, dtype=np.float32),
            shape=(lenState,))

    def dataToState(self, pos, heading, velocities):
        # Non-dimensionalise the position error (unit vector towards the target).
        perr = self.positionTarget - pos
        # NOTE for arbitrary motion this would need to be scaled and clipped to <-1, 1/0>
        # dTarget = np.linalg.norm(perr)
        # perr /= max(1e-6, dTarget)

        # Get heading error by comparing on both sides of zero.
        herr = headingError(self.headingTarget, heading)

        # Initialise if called just after reset.
        if self.herr_o is None:
            self.herr_o = herr
            self.perr_o = perr

        # Basic controller needs the first three elements to stay as they are now.
        newState = np.concatenate([
            np.array([
                min(1., max(-1., perr[0]/0.2)),
                min(1., max(-1., perr[1]/0.2)),
                min(1., max(-1., herr/(45./180.*np.pi))),
                min(1., max(-1., (herr-self.herr_o)/(2./180*np.pi))),
                min(1., max(-1., (perr[0]-self.perr_o[0])/0.025)),
                min(1., max(-1., (perr[1]-self.perr_o[1])/0.025)),
            ]),
            np.clip(velocities/[0.2, 0.2, 30./180.*np.pi], -1., 1.),
            np.zeros(2),  # Placeholder for additional state variables used only in CFD
        ])

        return newState

    def reset(self, keepTimeHistory=False, applyNoise=True, fixedInitialValues=None):
        if self.seed is not None:
            self._np_random, self.seed = seeding.np_random(self.seed)

        # Multipliers to mass, inertia and force coefficients used to improve
        # exploration and test robustness.
        if applyNoise:
            self.mMult, self.IMult, self.XuuMult, self.YvvMult, self.NrrMult, self.XuMult, \
                self.YvMult, self.NrMult = 1. + self.noiseMagCoeffs/2. - np.random.rand(8)*self.noiseMagCoeffs
            self.XactMult, self.YactMult, self.NactMult = \
                1. + self.noiseMagActuation/2. - np.random.rand(3)*self.noiseMagActuation
        else:
            self.mMult, self.IMult, self.XuuMult, self.YvvMult, self.NrrMult, self.XuMult, \
                self.YvMult, self.NrMult, self.XactMult, self.YactMult, self.NactMult = np.ones(11)

        # Set initial and target parameters.
        if fixedInitialValues is None:
            self.position = (np.random.rand(2)-0.5) * 0.5 * [self.xMinMax[1]-self.xMinMax[0], self.yMinMax[1]-self.yMinMax[0]]
            self.heading = np.random.rand()*2.*np.pi
            self.headingTarget = np.random.rand()*2.*np.pi
        else:
            self.position = fixedInitialValues[0]
            self.heading = fixedInitialValues[1]
            self.headingTarget = fixedInitialValues[2]
        self.positionStart = self.position.copy()
        self.positionTarget = np.zeros(2)
        self.headingStart = self.heading

        # random initial time in the first 25% of flow data.
        self.flowDataTimeOffset = np.random.rand()*self.flow.time[self.flow.time.shape[0]//4]

        # Used for checking action history in the reward.
        self.recentActions = collections.deque(10*[None], 10)

        # Other stuff.
        self.velocities = np.zeros(3)
        self.time = 0
        self.iStep = 0
        self.steps_beyond_done = 0
        self.herr_o = None
        self.perr_o = None
        self.timeHistory = []

        # Get the initial state.
        self.state = self.dataToState(self.position, self.heading, self.velocities)

        return self.state

    def step(self, action):
        # Set new time.
        self.iStep += 1
        self.time += self.dt

        # Check if max episode length reached.
        done = False
        if self.iStep >= self._max_episode_steps:
            done = True

        # Store the actions.
        self.recentActions.appendleft(action)

        # Scale the actions.
        Fset = action[:2]*self.maxForce*[self.XactMult, self.YactMult]
        Nset = action[2]*self.maxMoment*self.NactMult

        # Coordinate transformation from vehicle to global reference frame.
        Jtransform = np.array([
            [np.cos(self.heading), -np.sin(self.heading), 0.],
            [np.sin(self.heading), np.cos(self.heading), 0.],
            [0., 0., 1.],
        ])
        # Inverse transform to go from global to vehicle axes.
        invJtransform = np.linalg.pinv(Jtransform)

        # Use pre-made turbulence data to generate a turbid current.
        velCurrent = self.flow.interp(self.time+self.flowDataTimeOffset, self.position)[:2]
        if self.time+self.flowDataTimeOffset > self.flow.time[-1]:
            warnings.warn("Time value outside of input turbulence data range!")

        # Relative fluid velocity in the vehicle reference frame. For added mass,
        # one could assume rate of change of fluid velocity is much smaller than
        # that of the vehicle, hence d/dt(v-vc) = dv/dt.
        velRel = np.dot(invJtransform[:-1, :-1], self.velocities[:2] - velCurrent)

        # Compute hydrodynamic forces and moments in the vehicle reference frame.
        # NOTE: this is a very simplified problem definition, ignoring rigid body
        #   accelerations and cross-coupling terms.
        Fhydro = np.array([
            (self.Xu*self.XuMult + self.Xuu*self.XuuMult*np.abs(velRel[0]))*velRel[0],
            (self.Yv*self.YvMult + self.Yvv*self.YvvMult*np.abs(velRel[1]))*velRel[1],
            (self.Nr*self.NrMult + self.Nrr*self.NrrMult*np.abs(self.velocities[2]))*self.velocities[2],
        ])

        # Transform the forces to the global coordinate system.
        Fhydro = np.dot(Jtransform, Fhydro)

        # Vector of accelerations in the global reference frame.
        # NOTE: this ignores added mass and inertia due to fluid accelerations.
        accelerations = np.array([
            (Fhydro[0]+Fset[0])/(self.m*self.mMult),
            (Fhydro[1]+Fset[1])/(self.m*self.mMult),
            (Fhydro[2]+Nset)/(self.Izz*self.IMult)
        ])

        # Advance using the Euler method.
        dydt = np.append(self.velocities, accelerations)
        y = np.concatenate([self.position, [self.heading], self.velocities])
        y = y + dydt*self.dt
        position = y[:2]
        heading = y[2] % (2.*np.pi)
        velocities = y[3:]

        # Compute state.
        self.state = self.dataToState(position, heading, velocities)

        # Compute the reward.
        bonus = 0.

        # Check if domain exceeded.
        if (position[0] < self.xMinMax[0]) or (position[0] > self.xMinMax[1]):
            if self.stopOnBoundsExceeded:
                done = True
            bonus += -100.
        if (position[1] < self.yMinMax[0]) or (position[1] > self.yMinMax[1]):
            if self.stopOnBoundsExceeded:
                done = True
            bonus += -100.

        # Compute errors.
        perr = self.positionTarget - position
        herr = headingError(self.headingTarget, heading)

        # Update for the next pass.
        self.herr_o = herr
        self.perr_o = perr

        # Compute rms of recent actions.
        rmsAc = np.array([x for x in self.recentActions if x is not None])
        rmsAc = np.sqrt(np.sum((rmsAc-np.mean(rmsAc, axis=0))**2., axis=0) / rmsAc.shape[0])
        rmsAc = np.mean(rmsAc)

        rewardTerms = np.array([
            # --- ver 0 ---
            # # Square error along all DoF
            # -np.sum(np.clip(np.abs([perr[0], perr[1], herr])/[0.3, 0.3, 0.5*np.pi], 0., 1.)**2.),
            # # Bonus for being close to the objective.
            # 0.333*np.sum(np.abs([perr[0], perr[1], herr]) < [0.02, 0.02, 25./180.*np.pi]),
            # # Penalty for actuation to encourage it to do nothing when possible.
            # -0.05*np.sum(action**2.),

            # --- inspider by Woo et al. (2019) ---
            np.exp(-5.*np.linalg.norm(perr)),
            np.exp(-0.1*np.abs(herr/np.pi*180.)) if np.abs(herr) < np.pi/2. else -np.exp(-0.1*(180. - np.abs(herr/np.pi*180.))),
            np.exp(-0.6*rmsAc),

            # Additional term which encourages as little actuation as possible.
            # np.exp(-5.*np.sum(np.abs(action))/len(action)),
            -0.1*np.sum(action**2.)/len(action),

            # ---
            # Additional bonuses or penalties.
            bonus,
        ])

        # Get total reward.
        reward = np.sum(rewardTerms)

        # Update the position and heading at the new time value.
        self.position = position
        self.heading = heading
        self.velocities = velocities

        # Store stats.
        self.timeHistory.append(dict(zip(
            ["step", "time", "reward", "x", "y", "psi", "x_d", "y_d", "psi_d"] \
                +["Fx", "Fy", "N", "Fx_set", "Fy_set", "N_set"] \
                +["u", "v", "r", "u_current", "v_current", "rmsAc"] \
                +["r{:d}".format(i) for i in range(len(rewardTerms))] \
                +["a{:d}".format(i) for i in range(len(action))] \
                +["s{:d}".format(i) for i in range(len(self.state))],
            np.concatenate([
                [self.iStep, self.time, reward], self.position, [self.heading], self.positionTarget, [self.headingTarget],
                Fhydro, Fset, [Nset],
                velocities, velCurrent, [rmsAc],
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


def make_env(rank, seed=0, env_kwargs={}):
    """
    Utility function for multiprocessed env.

    :param filename: (str) path to file from which the env is created
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = AuvEnv(seed=seed+rank, **env_kwargs)
        # env.seed(seed + rank)
        # env.reset(seed=seed+rank)
        return env
    return _init
    pass


# %% crude testing
if __name__ == "__main__":
    import resources

    font = {"family": "serif",
            "weight": "normal",
            "size": 16}
    matplotlib.rc("font", **font)

    # Run an example episode with classical control
    print("\nSimple control")
    env_eval_pd = AuvEnv()
    pdController = PDController(env_eval_pd.dt)
    mean_reward,_ = resources.evaluate_agent(pdController, env_eval_pd)

    # Plot time history on top of the flow field
    fig, ax = resources.plotEpisode(env_eval_pd, "Simple control")

    # Plot detailed time histories of state, action, etc.
    resources.plotDetail([env_eval_pd], labels=["Simple control"])

    # Plot an animated replay of the time history.
    resources.animateEpisode(env_eval_pd, "test")
