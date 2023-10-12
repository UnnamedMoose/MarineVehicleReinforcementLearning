# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:50:05 2023

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
from verySimpleAuv import PDController
from resources import headingError

class AuvEnvCyl(gym.Env):
    def __init__(self, seed=None, dt=0.02, noiseMagCoeffs=0.0, noiseMagActuation=0.0,
                 currentVelScale=1.0, currentTurbScale=2.0, stopOnBoundsExceeded=True):
        # Call base class constructor.
        super(AuvEnvCyl, self).__init__()
        self.seed = seed

        # Position and size of the cylinder
        self.Rcyl = 1.33
        self.xCyl = np.array([2.5, 0.])

        # Crete waypoints and target headings for a path along the cylinder.
        Rwp = self.Rcyl*1.3
        t = np.linspace(-30, 30, 21) * np.pi/180.
        x = -Rwp*np.cos(t) + self.xCyl[0]
        y = Rwp*np.sin(t) + self.xCyl[1]
        self.waypoints = np.vstack([x, y, -t]).T
        self.wpThreshold = self.Rcyl*0.05
        self.iWp = 0

        # Tied to the no. time values stored in the turbulence data set.
        self._max_episode_steps = 1200
        # self._max_episode_steps = 250

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
        self.xMinMax = [-2, 2]
        self.yMinMax = [-2, 2]

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
            # self.headingTarget = np.random.rand()*2.*np.pi
        else:
            self.position = fixedInitialValues[0]
            self.heading = fixedInitialValues[1]
            # self.headingTarget = fixedInitialValues[2]
        self.positionStart = self.position.copy()
        # self.positionTarget = np.zeros(2)
        self.positionTarget = self.waypoints[self.iWp, :2]
        self.headingTarget = self.waypoints[self.iWp, 2]
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

        # Check if waypoint reached.
        if np.linalg.norm(perr) < self.wpThreshold:
            self.iWp = min(self.waypoints.shape[0]-1, self.iWp+1)
            self.positionTarget = self.waypoints[self.iWp, :2]
            self.headingTarget = self.waypoints[self.iWp, 2]

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


# %% Run an episode.
if __name__ == "__main__":
    import resources

    font = {"family": "serif",
            "weight": "normal",
            "size": 16}
    matplotlib.rc("font", **font)

    # Run an example episode with classical control
    print("\nSimple control")
    env_eval_pd = AuvEnvCyl()
    pdController = PDController(env_eval_pd.dt)
    mean_reward,_ = resources.evaluate_agent(pdController, env_eval_pd)

# %% Postprocessing

    # Plot detailed time histories of state, action, etc.
    resources.plotDetail([env_eval_pd], labels=["Simple control"])

    # Plot time history on top of the flow field
    fig, ax = resources.plotEpisode(env_eval_pd, "", plotHeading=False, plotWaypoints=False)

    headingTarget = np.pi
    Rcyl = 1.33
    rCyl = 2.5
    xCyl = rCyl*np.array([-np.cos(headingTarget), -np.sin(headingTarget)])
    t = np.linspace(0, 2*np.pi, 101)
    ax.fill(xCyl[0]+Rcyl*np.cos(t), xCyl[1]+Rcyl*np.sin(t), "grey")
    ax.set_xlim((-2.5, 2.5))
    ax.set_ylim((-2.5, 2.5))

    # Define the waypoints.
    t = env_eval_pd.waypoints[:, 2]
    x = env_eval_pd.waypoints[:, 0]
    y = env_eval_pd.waypoints[:, 1]
    waypoints = np.vstack([x, y, t]).T
    ax.plot(x, y, "ro--")
    for i in range(len(t)):
        # ax.text(x[i], y[i], "{:d}".format(i))
        # Note: flip both to reverse the heading.
        vx = [x[i], x[i]+Rcyl*0.1*np.cos(t[i])]
        vy = [y[i], y[i]+Rcyl*0.1*np.sin(t[i])]
        ax.plot(vx, vy, "m-", lw=3, alpha=0.5)
