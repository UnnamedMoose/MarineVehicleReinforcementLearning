# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 11:33:45 2023

@author: alidtke
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import re
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate
import gym

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

from resources import headingError


class Ship3DoF:
    def __init__(self, setPoint):
        # Target set point (position and heading).
        # Should give reasonable behaviour - go to a point and
        # maintain a constant heading.
        self.setPoint = setPoint

        # Stuff for the PID controller.
        self.eOld = None
        self.eInt = np.zeros(3)
        self.tOld = 0.

        # ===
        # Physical properties
        self.rho_f = 1000.
        self.m = 79.4 # kg (50 kg dry mass + flooded areas)
        self.dispVol = self.m / self.rho_f # For neutral buoyancy.
        self.Length = 1.96 # m (overall length of the AUV)

        self.LhFwd =  0.60 # m distance from the CB to thruster axes
        self.LhAft = -0.65
        self.LvFwd =  0.52
        self.LvAft = -0.54

        self.D_thruster = 0.07 # Diameter of tunnel thrusters (the same for all of them)
        self.D_prop = 0.305 # Prop diameter.

        self.CB = np.zeros(3) # Coordinate system centred on CB
        self.CG = np.array([0., 0., 0.06]) # m (relative to the CB)
        self.CG[2] = 0.06 # True value measured by Kantapon.

        self.I = np.array([ # kg m2
            [10.,  0.,  0.],
            [ 0., 35.,  0.],
            [ 0.,  0., 35.],
        ])

        # ===
        # Manoeuvring coefficients.

        # ---
        # Added mass and intertia - from Table 3.6 in Kantapon's thesis
        self.Xudot =  -2.4 # kg
        self.Yvdot = -65.5
        self.Zwdot = -65.5
        self.Kpdot =   0.
        self.Mqdot = -14.17 * 2.5 # TODO !!!!!!!!!!!!!!!!!!!!! tweaked coeff
        self.Nrdot = -14.17 * 2.5 # kg m^2 / rad # TODO !!!!!!!!!!!!!!!!!!!!! tweaked coeff
        # Cross-coupling terms.
        self.Yrdot =   0.
        self.Nvdot =   0.

        # ---
        # Quadratic terms.
        self.Xuu =   -2.79 # kg/m (for a fully submerged vehicle)
        self.Yvv = -183.0
        self.Yrr =    0.
        self.Ypp =    0.
        self.Zww = -183.0
        self.Zqq =    0.
        self.Kvv =    0.
        self.Kpp =    0.
        self.Krr =    0.
        self.Mww =  -59.0
        self.Mqq =  -82.0 # kg m^2 / rad^2
        self.Nvv =   59.0 # kg
        self.Nrr =  -82.0 # kg m^2 / rad^2
        self.Npp =    0.

        # ---
        # Linear damping terms - same as Kantapon but divided by velocity and density
#        self.Xu = -16.2208 # TODO this doesn't appear in the tables but a value is given in the code; seems off though?
#        self.Xu = 0.5 * rho * self.Length**2. * state["u"] * -1e-3 # TODO made up to fit the data :/
        self.Xu = 0.
        self.Yv = 0.5 * self.Length**2. * -28.5e-3
        self.Yr = 0.5 * self.Length**3. *  12.6e-3
        self.Yp = 0.
        self.Nv = 0.5 * self.Length**3. *  4.539e-3
        self.Nr = 0.5 * self.Length**4. * -5.348e-3 * 1.2 # TODO !!!!!!!!!!!!!!!!!!!!! tweaked coeff
        self.Np = 0.
        self.Zw = 0.5 * self.Length**2. * -28.5e-3
        self.Zq = 0.5 * self.Length**3. * -12.6e-3 # TODO is the sign correct?
        self.Kv = 0.
        self.Kp = 0.
        self.Kr = 0.
        self.Mw = 0.5 * self.Length**3. * -4.539e-3
        self.Mq = 0.5 * self.Length**4. * -5.348e-3 * 1.2 # TODO !!!!!!!!!!!!!!!!!!!!! tweaked coeff

        # ===
        # Rudder.
        # TODO replace with a regular force model?
        self.XuuDrud2 = -0.0036 # kg / m / deg*2
        self.YuuDrud = -0.3241 # kg / m / deg
        self.NuuDrud = 0.3254

        # ===
        # Stern plane.
        self.XuuDsplane2 = -0.0036
        self.ZuuDsplane = 0.3241
        self.MuuDsplane = 0.3254

        # ===
        # Tunnel thruster.
        # In Kantapon's thesis he used rpm and NOT rps, contrary to convention.
        # This would have yielded an equivalent value of 0.4633.
        #Kt_thruster = 1.2870e-4
        # This comes from Palmer 2009 thesis, Figure 4.29 on page 100. Kt is
        # almost constant as a function of rps.
        self.Kt_thruster = 0.53 * 0.9 # TODO !!!!!!!!!!!!!!!!!!!!! tweaked coeff
        self.cThruster1 = 0.35
        self.cThruster2 = 1.50

        # ===
        # Main propeller.
        # Use regressed coefficients to compute delivered thrust.
        # Original:
        #self.cProp1 = 0.6999
        #self.cProp2 = 1.5205
        #self.Kt0_prop = 0.0946
        # Modified by recomputing a fit to Figure 3.6
        self.cProp1 = 0.47
        self.cProp2 = 1.52
        self.Kt0_prop = 0.065
        # Wake fraction and thrust deduction.
        self.wt = 0.36
        self.t = 0.11

        # Generalised control forces and moments - X Y N
        self.generalisedControlForces = np.zeros(3)
        # Rudder angle and propeller rpm
        self.controlVector = np.zeros(2)

    def derivs(self, t, state):

        # Unpack the satate
        x, y, psi, u, v, r = state
        vel = np.array([u, v, r])

        # Compute the coordinate transform to go from local to global coords.
        Jtransform = np.array([
            [np.cos(psi), -np.sin(psi), 0.],
            [np.sin(psi), np.cos(psi), 0.],
            [0., 0., 1.],
        ])

        # Generalised forces and moments from the controller
        windup = np.array([2., 2., 90./180.*np.pi])
        K_P = np.array([20., 20., 20.])
        K_I = np.array([0.1, 0.1, 0.1])
        K_D = np.array([5., 5., 0.5])
        e = np.append(self.setPoint[:2] - np.array([x, y]),
                      headingError(self.setPoint[2], psi))
        if self.eOld is None:
            self.eOld = e.copy()
        dedt = (e - self.eOld) / max(1e-9, t - self.tOld)
        self.eInt += 0.5*(self.eOld + e) * (t - self.tOld)
        self.eInt[np.where(np.abs(e) > windup)[0]] = 0.
        controlValues = K_P*e + K_D*dedt + K_I*self.eInt
        for i, m in enumerate([150., 150., 100.]):
            controlValues[i] = max(-m, min(m, controlValues[i]))
        self.eOld = e
        self.tOld = t

        # Resolve into the vehicle reference frame before force allocation.
        Xd = controlValues[0]*np.cos(psi) + controlValues[1]*np.sin(psi)
        Yd = -controlValues[0]*np.sin(psi) + controlValues[1]*np.cos(psi)
        Nd = controlValues[2]
        self.generalisedControlForces = np.array([Xd, Yd, Nd])

        # TODO add a current model
        invJtransform = np.linalg.pinv(Jtransform)
        velCurrent = np.zeros(3)

        # Resolve the current into the vehicle reference frame.
        if np.linalg.norm(velCurrent) > 0.:
            velCurrent = np.matmul(invJtransform, velCurrent)

        # Relative fluid velocity. For added mass, assume rate of change of fluid
        # velocity is much smaller than that of the vehicle, hence d/dt(v-vc) = dv/dt.
        velRel = vel - velCurrent
        uRel = vel[0] - velCurrent[0]
        vRel = vel[1] - velCurrent[1]

        # ===
        # Total Mass Matrix (or inertia matrix), including added mass terms.
        Mrb = np.array([
            [self.m,             0.,                -self.m*self.CG[1]],
            [0.,                 self.m,            self.m*self.CG[0]],
            [-self.m*self.CG[1], self.m*self.CG[0], self.I[2,2]],
        ])

        Ma = -1. * np.diag([self.Xudot, self.Yvdot, self.Nrdot])

        M = Mrb + Ma

        # ===
        # Rigid-body accelerations.
        Crb = np.array([
            [0., 0., -self.m*(self.CG[0]*r + v)],
            [0., 0., -self.m*(self.CG[1]*r - u)],
            [self.m*(self.CG[0]*r + v), self.m*(self.CG[1]*r - u), 0.],
        ])

        Ca = np.array([
            [0., 0., self.Yvdot*vRel],
            [0., 0., -self.Xudot*uRel],
            [-self.Yvdot*vRel, self.Xudot*uRel, 0.],
        ])

        # ===
        # Fluid damping.
        # Divided by rho to correct for environment models c.f. Kantapon's thesis
        # and by velocity to be made invariant of the current state.
        Dl = -1. * np.array([
            [self.Xu, 0., 0.],
            [0., self.Yv, self.Yr],
            [0., self.Nv, self.Nr],
        ]) * self.rho_f * np.abs(uRel)

        Dq = -1. * np.array([
            [self.Xuu, 0., 0.],
            [0., self.Yvv, self.Yrr],
            [0., self.Nvv, self.Nrr],
        ]) * np.array([
            [np.abs(uRel), 0., 0.],
            [0., np.abs(vRel), np.abs(r)],
            [0., np.abs(vRel), np.abs(r)],
        ])

        D = Dl + Dq

        # ===
        # Propeller forces
        propRpm = 2000.
        Fprop = self.rho_f * (propRpm/60.)**2. * np.sign(propRpm) \
            * self.D_thruster**4. * self.Kt_thruster

        # ===
        # Rudder forces.

        # Let op: hardcoded square wave for the rudder angle
        freq = 1./5.
        rudderAngle = 35.*np.sign(np.sin(2.*np.pi*t*freq))/180.*np.pi

        self.controlVector = np.array([propRpm, rudderAngle])

        # Applied density correction w.r.t. the original values in Kantapon's thesis.
        Xrud = self.XuuDrud2*self.rho_f/1000. * np.abs(uRel) * uRel * rudderAngle**2.
        Yrud = self.YuuDrud*self.rho_f/1000. * np.abs(uRel) * uRel * rudderAngle
        Nrud = self.NuuDrud*self.rho_f/1000. * np.abs(uRel) * uRel * rudderAngle

        # ===
        # Hydrodynamic forces excluding added mass terms.
        Xh = Fprop + Xrud
        Yh = Yrud
        Nh = Nrud
        H = np.array([Xh, Yh, Nh])

        # ===
        # Hydrostatic forces.
        G = np.zeros(3)

        # ===
        # Externally applied forces and moments.
        E = np.array([0., 0., 0.])

        # ===
        # Total forces and moments. Note the split between hydrodynamic
        # effects (using relative flow velocity) and rigid body terms.
        RHS = -np.dot(Crb, vel) - np.dot(Ca+D, velRel) - G + H + E

        # ===
        # Solve M*acc = F for accelerations
        acc = np.linalg.solve(M, RHS)

        # ===
        # Apply a coordinate transformation to get velocities in the global coordinates.
        # After the integration this will yield displacements in the global coordinates.
        vel = np.dot(Jtransform, vel)

        # ===
        # Return derivatives of the system along each degree of freedom. The first
        # part of the derivative vector are the rates of change of position in the
        # global reference frame; the second part are the accelerations,
        # i.e. force/moment divided by mass/inertia, including added mass effects,
        # in the body reference frame.
        return np.append(vel, acc)

    def plot_horizontal(self, ax, x, y, psi, psiD, forces, scale=1, markerSize=1, arrowSize=1, vehicleColour="y"):
        """ Plot a representation of the AUV on the given axes.
        Thuster forces should be non-dimensionalised with maximum value. """

        x0 = np.array([x, y])

        self.Width = 0.2
        xyHull = np.array([
            [self.Length/2.,                  -self.Width/2.+self.D_thruster],
            [self.Length/2.,                   self.Width/2.-self.D_thruster],
            [self.Length/2.-self.D_thruster,   self.Width/2.],
            [-self.Length/2.+self.D_thruster,  self.Width/2.],
            [-self.Length/2.,                  self.Width/2.-self.D_thruster],
            [-self.Length/2.,                 -self.Width/2.+self.D_thruster],
            [-self.Length/2.+self.D_thruster, -self.Width/2.],
            [self.Length/2.-self.D_thruster,  -self.Width/2.],
            [self.Length/2.,                  -self.Width/2.+self.D_thruster],
        ])

        xyCentreline = np.array([
            [np.min(xyHull[:,0]), 0.],
            [np.max(xyHull[:,0]), 0.],
        ])

        xyDir = np.array([
            [self.Length/2.-self.Width/4., -self.Width/4.],
            [self.Length/2., 0.],
            [self.Length/2.-self.Width/4., self.Width/4.],
        ])

        def rotate(xy, psi):
            xyn = np.zeros(xy.shape)
            xyn[:,0] = np.cos(psi)*xy[:,0] - np.sin(psi)*xy[:,1]
            xyn[:,1] = np.sin(psi)*xy[:,0] + np.cos(psi)*xy[:,1]
            return xyn

        xyHull = rotate(xyHull*scale, psi) + x0
        xyCentreline = rotate(xyCentreline*scale, psi) + x0
        xyDir = rotate(xyDir*scale, psi) + x0

        objects = []
        objects += ax.fill(xyHull[:,1], xyHull[:,0], vehicleColour, alpha=0.5)
        objects += ax.plot(xyCentreline[:,1], xyCentreline[:,0], "k--", lw=2*markerSize)
        objects += ax.plot(xyDir[:,1], xyDir[:,0], "k-", lw=2*markerSize)
        objects += ax.plot(x0[1], x0[0], "ko", mew=3, mfc="None", ms=14*markerSize)

        uD = np.array([np.sin(psiD), np.cos(psiD)])*self.Length/2.
        objects.append( ax.arrow(x0[1], x0[0], uD[0]*scale, uD[1]*scale, color="r",
                width=0.002*self.Length*scale/2.*arrowSize, lw=2*arrowSize) )

        return objects


class Ship3DoFEnv(gym.Env):
    def __init__(self, seed=None, dt=0.2, maxSteps=250):
        # Call base class constructor.
        super(Ship3DoFEnv, self).__init__()

        # Set common stuff.
        self.seed = seed
        self.dt = dt
        self._max_episode_steps = maxSteps

        # Define the action space.
        self.lenAction = 2
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.lenAction,), dtype=np.float32)

        # Define the observation space.
        self.lenObs = 5
        self.observation_space = gym.spaces.Box(
            -1*np.ones(self.lenObs, dtype=np.float32),
            np.ones(self.lenObs, dtype=np.float32),
            shape=(self.lenObs,))

    def dataToState(self, systemState):
        # Need to translate the system state (positions, velocities, etc.)
        # into the RL agent's state.
        return np.clip([
            # Dimensionless vector to target.
            (self.wp[0]-systemState[0]) / (self.vehicle.Length*3.),
            (self.wp[1]-systemState[1]) / (self.vehicle.Length*3.),
            # Heading error
            headingError(self.vehicle.setPoint[2], systemState[2]) / (45./180.*np.pi)
        ], -1., 1.)

    def reset(self, initialSetpoint=None):
        if self.seed is not None:
            self._np_random, self.seed = gym.utils.seeding.np_random(self.seed)

        self.iStep = 0
        self.time = 0.

        if initialSetpoint is None:
            # Generate a path consisting of two-point segments
            # with a random heading for both waypoints.
            self.wp = (np.random.rand(2)-[0.5, 0.5])*[10., 10.]
            self.targetHeading = np.random.rand()*2.*np.pi
        else:
            # Set up the vehicle with a fixed initial waypoint and desired heading.
            # Doing so should allow a dummy agent to be used such that simple A->B
            # behaviour could be simulated. This can be used for the tuning of the
            # controller and testing of system dynamics.
            self.wp = np.array(initialSetpoint[:2])
            self.targetHeading = initialSetpoint[2]

        # Create a vehicle instance with an initial set point.
        self.vehicle = Ship3DoF(np.append(self.wp, self.targetHeading))

        # Set the initial conditions.
        self.systemState = np.array([0., 0., 0./180.*np.pi, 0., 0., 0.])

        # Store the time history of system states and other data.
        self.timeHistory = [np.concatenate([
            [self.time], self.systemState, self.vehicle.generalisedControlForces,
            self.vehicle.controlVector, self.vehicle.setPoint])]

        # Create the initial RL state.
        self.state = self.dataToState(self.systemState)

        return self.state

    def step(self, action):
        # Set new time.
        self.iStep += 1
        self.time += self.dt

        # Advance in time
        result_solve_ivp = scipy.integrate.solve_ivp(
            self.vehicle.derivs, (self.time-self.dt, self.time), self.systemState,
            method='RK45', t_eval=np.array([self.time]), max_step=self.dt, rtol=1e-3, atol=1e-3)

        # Sort out the computed heading.
        result_solve_ivp.y[2, :] = result_solve_ivp.y[2, :] % (2.*np.pi)

        # Store the dynamical system state.
        self.systemState = result_solve_ivp.y[:, -1]

        # Extract RL state at the new time level.
        self.state = self.dataToState(self.systemState)

        # Check if max episode length reached.
        done = False
        if self.iStep >= self._max_episode_steps:
            done = True

        # TODO this could be the distance from the path, RMS deviation from the path
        # while it has been selected, some penalty for duration and excessive actuation.
        reward = 0.

        # Store and tidy up the data when done.
        self.timeHistory.append(np.concatenate([
            [self.time], self.systemState, self.vehicle.generalisedControlForces,
            self.vehicle.controlVector, self.vehicle.setPoint]))
        if done:
            self.timeHistory = pandas.DataFrame(
                data=np.array(self.timeHistory),
                columns=["t"]+[f"x{i:d}" for i in range(len(self.systemState))]
                    +[f"F{i:d}" for i in range(len(self.vehicle.generalisedControlForces))]
                    +[f"u{i:d}" for i in range(len(self.vehicle.controlVector))]
                    +["x_d", "y_d", "psi_d"])

        if done:
            self.steps_beyond_done += 1
        else:
            self.steps_beyond_done = 0

        return self.state, reward, done, {}


if __name__ == "__main__":

    env = Ship3DoFEnv(maxSteps=100)
    env.reset(initialSetpoint=[10., -10., 280./180.*np.pi])
    for i in range(100):
        env.step([0., 0.])

    # Plot trajectory
    fig, ax  = plt.subplots()
    ax.set_xlabel("y [m, +ve east]")
    ax.set_ylabel("x [m, +ve north]")
    ax.plot(env.timeHistory["x1"], env.timeHistory["x0"], "r", label="Trajectory")
    ax.plot(env.wp[1], env.wp[0], "m*--", ms=12, label="Waypoint")
    ax.set_aspect("equal")
    for i in [0, 5, 10, 15]:
        env.vehicle.plot_horizontal(ax, env.timeHistory["x0"].values[i], env.timeHistory["x1"].values[i],
            env.timeHistory["x2"].values[i], env.vehicle.setPoint[2],
            np.zeros(4),
            markerSize=0.5, arrowSize=1, scale=1)
    ax.legend(ncol=3, bbox_to_anchor=(0.5, 1.01), loc="lower center")

    # Plot individual DoFs
    fig, ax  = plt.subplots()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Displacement [m, rad]")
    ax.set_xlim((0, env.timeHistory["t"].max()))
    for i, v in enumerate(["x", "y", "psi"]):
        ln, = ax.plot(env.timeHistory["t"], env.timeHistory[f"x{i:d}"], label=v)
        ax.hlines(env.vehicle.setPoint[i], 0, env.timeHistory["t"].max(), color=ln.get_color(), linestyle="dashed")
    ax.legend()

    # Plot generalised control forces
    fig, ax  = plt.subplots()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Generalised control force or moment [N, Nm]")
    ax.set_xlim((0, env.timeHistory["t"].max()))
    for i, v in enumerate(["X", "Y", "N"]):
        ln, = ax.plot(env.timeHistory["t"], env.timeHistory[f"F{i:d}"], label=v)
        ax.hlines(env.vehicle.setPoint[i], 0, env.timeHistory["t"].max(), color=ln.get_color(), linestyle="dashed")
    ax.legend()

    # Plot control variables
    fig, ax  = plt.subplots()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Rudder angle [deg]")
    ax.set_xlim((0, env.timeHistory["t"].max()))
    ax.plot(env.timeHistory["t"], env.timeHistory["u1"]/np.pi*180, c="r", lw=2)
