# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 19:08:54 2023

@author: ALidtke
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import re
import gym
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate
from scipy.spatial.transform import Rotation

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

import resources

class BlueROV2Heavy6DoF:
    def __init__(self, setPoint):
        # Target set point (position and heading).
        # Should give reasonable behaviour - go to a point and
        # maintain a constant heading.
        self.setPoint = setPoint

        # Stuff for the PID controller.
        self.eOld = None
        self.eInt = np.zeros(6)
        self.tOld = 0.

        # ===
        # Physical properties
        self.rho_f = 1000.
        self.m = 11.4 # kg # +++
        self.dispVol = self.m / self.rho_f # For neutral buoyancy.
        self.Length = 0.457 # m (overall dimensions of the ROV) # +++
        self.Width = 0.338 # +++

        # Coordinate system centred on CB # +++
        self.CB = np.array([0., 0., 0.])
        self.CG = np.array([0., 0., 0.025]) # m (relative to the CB) # +++

        self.I = np.array([ # kg m2 # +++
            [0.16, 0.,  0.],
            [0., 0.16,  0.],
            [0., 0., 0.16],
        ])

        # CB offset - used for exporting data only. NED.
        self.x_origin = np.array([0., 0., 0.0375])

        # ===
        # Manoeuvring coefficients.

        # ---
        # Added mass and intertia
        self.Xudot = -5.5 # kg # +++
        self.Yvdot = -12.7 # +++

        # TODO check values in reference
        self.Zvdot = 0.
        self.Zwdot = -12.7
        self.Kpdot = -0.12
        self.Mqdot = -0.12

        self.Nrdot = -0.12 # kg m^2 / rad # +++
        # Cross-coupling terms.
        self.Yrdot = 0.
        self.Nvdot = 0.

        # ---
        # Quadratic terms.
        self.Xuu = -18.18 # kg/m # +++
        self.Yvv = -21.66 # +++
        self.Yrr = 0.
        self.Ypp = 0.

        # TODO check values in reference
        self.Zww = -21.0
        self.Zqq = 0.
        self.Kvv = 0.
        self.Kpp = 0.
        self.Krr = 0.
        self.Mww = -1.55
        self.Mqq = -1.55 # kg m^2 / rad^2

        self.Nvv = 0.0 # kg
        self.Nrr = -1.55 # kg m^2 / rad^2 # +++ # TODO maybe wrong units?
        self.Npp = 0.

        # ---
        # Linear damping terms - same as Kantapon but divided by velocity and density
        self.Xu = -4.03 # +++
        self.Yv = -6.22 # +++
        self.Yr = 0.
        self.Yp = 0.

        # TODO check values in reference
        self.Zw = 0.
        self.Zq = 0.
        self.Kv = 0.
        self.Kp = -0.07
        self.Kr = 0.
        self.Mw = 0.
        self.Mq = -0.07

        self.Nv = 0.
        self.Nr = -0.07 # +++
        self.Np = 0.

        # ===
        # Thruster.
        self.D_thruster = 0.1 # Diameter of thrusters # +++
        # Angle between each horizontal thruster axes and centreline (0 fwd)
        # Wu (2018) says it's 45 degrees but this doesn't match the geometry.
        self.alphaThruster = 33./180.*np.pi
        # These are actually further inwards than Wu says.
        # self.l_x = 0.156
        # self.l_y = 0.111
        self.l_x = 0.1475
        self.l_y = 0.101
        # 85 mm comes from Wu (2018) but doesn't match the real geometry
        # self.l_z = 0.085
        self.l_z = 0.068
        self.l_x_v = 0.120
        # self.l_y_v = 0.218
        self.l_y_v = 0.22
        self.l_z_v = 0.0  # irrelevant

        # Back-calculated from max rpm at 16 V from datasheet:
        # https://bluerobotics.com/store/thrusters/t100-t200-thrusters/t200-thruster/
        # And 40 N measured by Wu (2018) - pp. 49
        self.Kt_thruster = 40. / (1000. * (3500./60.)**2. * self.D_thruster**4.) # +++

        # Generalised control forces and moments - X Y Z K M N
        self.generalisedControlForces = np.zeros(6)

        # Motor rpms
        self.controlVector = np.zeros(8)

        # Thruster positions in the vehicle reference frame. Consistent with Wu (2018) fig 4.2
        self.thrusterPositions = np.array([
            [self.l_x, self.l_y, self.l_z],
            [self.l_x, -self.l_y, self.l_z],
            [-self.l_x, self.l_y, self.l_z],
            [-self.l_x, -self.l_y, self.l_z],
            [self.l_x_v, self.l_y_v, self.l_z_v],
            [self.l_x_v, -self.l_y_v, self.l_z_v],
            [-self.l_x_v, self.l_y_v, self.l_z_v],
            [-self.l_x_v, -self.l_y_v, self.l_z_v],
        ])
        self.thrusterNormals = np.array([
            [np.cos(self.alphaThruster), -np.sin(self.alphaThruster), 0.],
            [np.cos(self.alphaThruster), np.sin(self.alphaThruster), 0.],
            [-np.cos(self.alphaThruster), -np.sin(self.alphaThruster), 0.],
            [-np.cos(self.alphaThruster), np.sin(self.alphaThruster), 0.],
            [0., 0., -1.],
            [0., 0., 1.],
            [0., 0., 1.],
            [0., 0., -1.],
        ])

        # Use pseudo-inverse of the control allocation matrix in order to go from
        # desired generalised forces to actuator demands in rpm.
        # NOTE the original 3 DoF notation is inconsistent with page 48 in Wu (2018),
        # what follows is (or should be) the same. See Figure 4.2 and Eq. 4.62 in their work.
        self.A, self.Ainv = resources.computeThrustAllocation(self.thrusterPositions, self.thrusterNormals)

    def thrusterModel(self, rpm):
        # Force delivered when at rest.
        Fthruster = self.rho_f * (rpm/60.)**2. * np.sign(rpm) * self.D_thruster**4. * self.Kt_thruster
        return Fthruster

    def updateMovingCoordSystem(self, rotation_angles):
        # Store the current orientation.
        self.rotation_angles = rotation_angles
        # Create new vehicle axes from rotation angles (roll pitch yaw)
        self.iHat, self.jHat, self.kHat = Rotation.from_euler('XYZ', rotation_angles, degrees=False).as_matrix().T

    def globalToVehicle(self, vecGlobal):
        return np.array([
            np.dot(vecGlobal, self.iHat),
            np.dot(vecGlobal, self.jHat),
            np.dot(vecGlobal, self.kHat)])

    def vehicleToGlobal(self, vecVehicle):
        return vecVehicle[0]*self.iHat + vecVehicle[1]*self.jHat + vecVehicle[2]*self.kHat

    def forceModel(self, pos, vel, angles, rpms):
        phi, theta, psi = angles
        u, v, w, p, q, r = vel

        # TODO add a current model
        velCurrent = np.zeros(6)

        # Resolve the current into the vehicle reference frame.
        # TODO this routine should only accept things in the body reference frame.
        if np.linalg.norm(velCurrent) > 0.:
            velCurrent = self.globalToVehicle(velCurrent)

        # Apply saturation and deadband to model the discrepancy between required
        # and possible thruster output.
        def limit(x):
            r = max(-3500., min(3500., x))
            if np.abs(r) < 300:
                r = 0.
            return r

        # Hydrodynamic forces excluding added mass terms.
        H = np.zeros(6)
        for i in range(8):
            # Thruster model.
            H += self.thrusterModel(limit(rpms[i]))*self.A[:, i]

        # Relative fluid velocity. For added mass, assume rate of change of fluid
        # velocity is much smaller than that of the vehicle, hence d/dt(v-vc) = dv/dt.
        velRel = vel - velCurrent

        # ===
        # Total Mass Matrix (or inertia matrix), including added mass terms.
        Mrb = np.array([
            [self.m,                0.,                    0.,                    0.,                   self.m*self.CG[2],    -self.m*self.CG[1]],
            [0.,                    self.m,                0.,                    -self.m*self.CG[2],   0.,                    self.m*self.CG[0]],
            [0.,                    0.,                    self.m,                self.m*self.CG[1],    -self.m*self.CG[0],    0.],
            [0.,                    -self.m*self.CG[2],    self.m*self.CG[1],     0., 0., 0.],
            [self.m*self.CG[2],     0.,                    -self.m*self.CG[0],    0., 0., 0.],
            [-self.m*self.CG[1],    self.m*self.CG[0],     0.,                    0., 0., 0.],
        ])
        Mrb[3:,3:] = self.I

        Ma = -1. * np.diag([self.Xudot, self.Yvdot, self.Zvdot, self.Kpdot, self.Mqdot, self.Nrdot])

        M = Mrb + Ma

        # ===
        # Rigid-body accelerations.
        Crb = np.array([
            [0., 0., 0.,  self.m*(self.CG[1]*q + self.CG[2]*r),
                        -self.m*(self.CG[0]*q - w),
                        -self.m*(self.CG[0]*r + v)],
            [0., 0., 0., -self.m*(self.CG[1]*p + w),
                         self.m*(self.CG[2]*r + self.CG[0]*p),
                        -self.m*(self.CG[1]*r - u)],
            [0., 0., 0., -self.m*(self.CG[2]*p - v),
                        -self.m*(self.CG[2]*q + u),
                         self.m*(self.CG[0]*p + self.CG[1]*q)],
            [    -self.m*(self.CG[1]*q + self.CG[2]*r),
                 self.m*(self.CG[1]*p + w),
                 self.m*(self.CG[2]*p - v),
                0.,
                -self.I[1,2]*q - self.I[0,2]*p + self.I[2,2]*r,
                 self.I[1,2]*r + self.I[0,1]*p - self.I[1,1]*q],
            [     self.m*(self.CG[0]*q - w),
                -self.m*(self.CG[2]*r + self.CG[0]*p),
                 self.m*(self.CG[2]*q + u),
                 self.I[1,2]*q + self.I[0,2]*p - self.I[2,2]*r,
                0.,
                -self.I[0,2]*r - self.I[0,1]*q + self.I[0,0]*p],
            [     self.m*(self.CG[0]*r + v),
                 self.m*(self.CG[1]*r - u),
                -self.m*(self.CG[0]*p + self.CG[1]*q),
                -self.I[1,2]*r - self.I[0,1]*p + self.I[1,1]*q,
                 self.I[0,2]*r + self.I[0,1]*q - self.I[0,0]*q,
                0.],
        ])

        Ca = np.array([
            [0.,            0.,                0.,                0.,                -self.Zwdot*w,    self.Yvdot*v],
            [0.,            0.,                0.,                self.Zwdot*w,    0.,                -self.Xudot*u],
            [0.,            0.,                0.,                -self.Yvdot*v,    self.Xudot*u,    0.],
            [0.,            -self.Zwdot*w,    self.Yvdot*v,    0.,                -self.Nrdot*r,    self.Mqdot*q],
            [ self.Zwdot*w,    0.,                -self.Xudot*u,    self.Nrdot*r,    0.,                -self.Kpdot*p],
            [-self.Yvdot*v,     self.Xudot*u,    0.,                -self.Mqdot*q,    self.Kpdot*p,    0.],
        ])

        # ===
        # Fluid damping.
        Dl = -1. * np.array([
            [self.Xu,    0.,            0.,            0.,            0.,            0.],
            [0.,        self.Yv,    0.,            self.Yp,    0.,            self.Yr],
            [0.,        0.,            self.Zw,    0.,            self.Zq,    0.],
            [0.,        self.Kv,    0.,            self.Kp,    0.,            self.Kr],
            [0.,        0.,            self.Mw,    0.,            self.Mq,    0.],
            [0.,        self.Nv,    0.,            self.Np,    0.,            self.Nr],
        ])

        Dq = -1. * np.array([
            [self.Xuu,      0.,          0.,            0.,          0.,          0.],
            [0.,            self.Yvv,    0.,            self.Ypp,    0.,          self.Yrr],
            [0.,            0.,          self.Zww,      0.,          self.Zqq,    0.],
            [0.,            self.Kvv,    0.,            self.Kpp,    0.,          self.Krr],
            [0.,            0.,          self.Mww,      0.,          self.Mqq,    0.],
            [0.,            self.Nvv,    0.,            self.Npp,    0.,          self.Nrr],
        ]) * np.array([
            [np.abs(u),     0.,         0.,            0.,           0.,           0.],
            [0.,            np.abs(v),  0.,            np.abs(p),    0.,           np.abs(r)],
            [0.,            0.,         np.abs(w),     0.,           np.abs(q),    0.],
            [0.,            np.abs(v),  0.,            np.abs(p),    0.,           np.abs(r)],
            [0.,            0.,         np.abs(w),     0.,           np.abs(q),    0.],
            [0.,            np.abs(v),  0.,            np.abs(p),    0.,           np.abs(r)],
        ])

        D = Dl + Dq

        # ===
        # Hydrostatic forces and moments.
        W = self.m*9.81
        B = self.dispVol*self.rho_f*9.81
        # NOTE: G matrix is written as if it were on the LHS, consistently with the
        #    notaiton by Fossen (and Kantapon).
        G = np.array([
            (W - B) * np.sin(theta),
            -(W - B) * np.cos(theta)*np.sin(phi),
            -(W - B) * np.cos(theta)*np.cos(phi),
            -(self.CG[1]*W - self.CB[1]*B)*np.cos(theta)*np.cos(phi)
                + (self.CG[2]*W - self.CB[2]*B)*np.cos(theta)*np.sin(phi),
            (self.CG[2]*W - self.CB[2]*B)*np.sin(theta)
                + (self.CG[0]*W - self.CB[0]*B)*np.cos(theta)*np.cos(phi),
            -(self.CG[0]*W - self.CB[0]*B)*np.cos(theta)*np.sin(phi)
                - (self.CG[1]*W - self.CB[1]*B)*np.sin(theta)
        ])

        # ===
        # Externally applied forces and moments.
        E = np.zeros(6)

        # ===
        # Total forces and moments
        RHS = -np.dot(Crb, vel) - np.dot(Ca+D, velRel) - G + H + E

        return M, RHS

    def derivs(self, t, state):

        # Unpack the satate
        x, y, z, phi, theta, psi, u, v, w, p, q, r = state
        pos = np.array([x, y, z])
        angles = np.array([phi, theta, psi])
        vel = np.array([u, v, w, p, q, r])

        # Set the vehicle axes.
        self.updateMovingCoordSystem(angles)

        # Generalised forces and moments from the controller
        windup = np.array([2., 2., 2., 90./180.*np.pi, 90./180.*np.pi, 90./180.*np.pi])
        K_P = np.array([20., 20., 20., 1., 1., 1.])
        K_I = np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        K_D = np.array([5., 5., 5., 0.1, 0.1, 0.1])
        e = np.append(self.setPoint[:3] - np.array([x, y, z]),
                      [resources.angleError(self.setPoint[3], phi),
                      resources.angleError(self.setPoint[4], theta),
                      resources.angleError(self.setPoint[5], psi)])
        if self.eOld is None:
            self.eOld = e.copy()
        dedt = (e - self.eOld) / max(1e-9, t - self.tOld)
        self.eInt += 0.5*(self.eOld + e) * (t - self.tOld)
        self.eInt[np.where(np.abs(e) > windup)[0]] = 0.
        controlValues = K_P*e + K_D*dedt + K_I*self.eInt
        for i, m in enumerate([150., 150., 150., 100., 100., 100.]):
            controlValues[i] = max(-m, min(m, controlValues[i]))
        self.eOld = e
        self.tOld = t
        # controlValues = np.zeros(6)

        # Resolve into the vehicle reference frame before force allocation.
        self.generalisedControlForces = np.append(
            self.globalToVehicle(controlValues[:3]), self.globalToVehicle(controlValues[3:]))

        # Go from forces to rpm.
        cv = np.matmul(self.Ainv, self.generalisedControlForces)  # Newtons
        cv = np.sign(cv)*np.sqrt(np.abs(cv)/(self.rho_f*self.D_thruster**4.*self.Kt_thruster))*60.  # rpm
        self.controlVector = cv

        # Call the force model for the current state.
        M, RHS = self.forceModel(pos, vel, angles, cv)

        # Solve M*acc = F for accelerations
        # This is done in the vehicle reference frame.
        acc = np.linalg.solve(M, RHS)

        # Compute the coordinate transform to go from local to global coords.
        Jtransform = resources.coordinateTransform(phi, theta, psi, dof=6)
        # Apply a coordinate transformation to get velocities in the global coordinates.
        # After the integration this will yield displacements in the global coordinates.
        vel = np.dot(Jtransform, vel)

        # Return derivatives of the system along each degree of freedom. The first
        # part of the derivative vector are the rates of change of position in the
        # global reference frame; the second part are the accelerations,
        # i.e. force/moment divided by mass/inertia, including added mass effects,
        # in the body reference frame.
        return np.append(vel, acc)


class BlueROV2Heavy6DoFEnv(gym.Env):
    def __init__(self, seed=None, dt=0.2, maxSteps=250):
        # Call base class constructor.
        super(BlueROV2Heavy6DoFEnv, self).__init__()

        # Set common stuff.
        self.seed = seed
        self.dt = dt
        self._max_episode_steps = maxSteps

        # Define the action space.
        self.lenAction = 6
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.lenAction,), dtype=np.float32)

        # Define the observation space.
        self.lenObs = 9
        self.observation_space = gym.spaces.Box(
            -1*np.ones(self.lenObs, dtype=np.float32),
            np.ones(self.lenObs, dtype=np.float32),
            shape=(self.lenObs,))

    def dataToState(self, systemState):
        # Need to translate the system state (positions, velocities, etc.)
        # into the RL agent's state.
        return np.clip([
            # Dimensionless vector to target.
            (self.path[self.iWp, 0]-systemState[0]) / (self.vehicle.Length*3.),
            (self.path[self.iWp, 1]-systemState[1]) / (self.vehicle.Length*3.),
            (self.path[self.iWp, 2]-systemState[2]) / (self.vehicle.Length*3.),
            # To the following waypoint for path following
            (self.path[self.iWp+1, 0]-systemState[0]) / (self.vehicle.Length*3.),
            (self.path[self.iWp+1, 1]-systemState[1]) / (self.vehicle.Length*3.),
            (self.path[self.iWp+1, 2]-systemState[2]) / (self.vehicle.Length*3.),
            # Angle errors
            resources.angleError(self.vehicle.setPoint[3], systemState[3]) / (45./180.*np.pi),
            resources.angleError(self.vehicle.setPoint[4], systemState[4]) / (45./180.*np.pi),
            resources.angleError(self.vehicle.setPoint[5], systemState[5]) / (45./180.*np.pi),
        ], -1., 1.)

    def reset(self, initialSetpoint=None):
        if self.seed is not None:
            self._np_random, self.seed = gym.utils.seeding.np_random(self.seed)

        self.iStep = 0
        self.time = 0.

        if initialSetpoint is None:
            # Generate a path consisting of two-point segments
            # with a random heading for both waypoints.
            nWps = 2
            self.iWp = 0
            self.path = (np.random.rand(3*nWps).reshape((nWps, 3))-[0.5, 0.5])*[10., 10.]
            self.targetOrientation = np.random.rand(3)*2.*np.pi
            # The initial set point is the firs point on the path.
            sp = np.append(self.path[0, :], self.targetOrientation)
            self.fixedSp = False
        else:
            # Set up the vehicle with a fixed initial waypoint and desired heading.
            # Doing so should allow a dummy agent to be used such that simple A->B
            # behaviour could be simulated. This can be used for the tuning of the
            # controller and testing of system dynamics.
            sp = np.array(initialSetpoint)
            self.iWp = 0
            self.path = np.vstack([sp[:3], sp[:3]])
            self.targetOrientation = sp[3:]
            self.fixedSp = True

        # Create a vehicle instance with an initial set point.
        self.vehicle = BlueROV2Heavy6DoF(sp)

        # Set the initial conditions.
        self.systemState = np.array([
            0., 0., 0., 0./180.*np.pi, 0./180.*np.pi, 0./180.*np.pi,
            0., 0., 0., 0., 0., 0.])

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

        if self.fixedSp:
            # When a constant set point is given upon initialisation, just navigate
            # to it using the contorller built into the vehicle dynamics model.
            # This is mainly used for testing of the system dynamics and tuning
            # of the controller.
            pass
        else:
            # This is where the navigation algorithm has to provide a new set point
            # that will be passed to the vehicle.
            x_sp = action[0]*(2.*self.vehicle.Length) + self.systemState[0]
            y_sp = action[1]*(2.*self.vehicle.Length) + self.systemState[1]
            z_sp = action[2]*(2.*self.vehicle.Length) + self.systemState[2]
            # TODO this seems wrong?
            phi_d = action[3]*(45./180.*np.pi) + self.systemState[3]
            theta_d = action[4]*(45./180.*np.pi) + self.systemState[4]
            psi_d = action[5]*(45./180.*np.pi) + self.systemState[5]
            self.vehicle.setPoint = np.array([x_sp, y_sp, z_sp, phi_d, theta_d, psi_d])

        # Advance in time
        result_solve_ivp = scipy.integrate.solve_ivp(
            self.vehicle.derivs, (self.time-self.dt, self.time), self.systemState,
            method='RK45', t_eval=np.array([self.time]), max_step=self.dt, rtol=1e-3, atol=1e-3)

        # Sort out the computed angles.
        result_solve_ivp.y[3:6, :] = result_solve_ivp.y[3:6, :] % (2.*np.pi)

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
                columns=["t"]+["x", "y", "z", "phi", "theta", "psi"] + ["u", "v", "w", "p", "q", "r"]#[f"x{i:d}" for i in range(len(self.systemState))]
                    +[f"F{i:d}" for i in range(len(self.vehicle.generalisedControlForces))]
                    +[f"u{i:d}" for i in range(len(self.vehicle.controlVector))]
                    +["x_d", "y_d", "z_d", "phi_d", "theta_d", "psi_d"])

        if done:
            self.steps_beyond_done += 1
        else:
            self.steps_beyond_done = 0

        return self.state, reward, done, {}


def plotEpisodeDetail(env, title=""):
    # Plot trajectory
    fig  = plt.figure(figsize=(14, 6))
    plt.subplots_adjust(top=0.812, bottom=0.139, left=0.053, right=0.972, hspace=0.2, wspace=0.509)
    ax = [
        fig.add_subplot(1, 3, 1, projection='3d'),
        fig.add_subplot(1, 3, 2),
        fig.add_subplot(1, 3, 3),
    ]
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_zlabel("z")
    ax[0].set_aspect("equal")
    ax[0].invert_yaxis()  # NED and y +ve to stbd
    ax[0].invert_zaxis()
    ax[0].plot(env.timeHistory["x"], env.timeHistory["y"], env.timeHistory["z"], "r", label="Trajectory")
    ax[0].plot(env.path[:, 0], env.path[:, 1], env.path[:, 2], "m*--", ms=12, label="Waypoints")
    # Plot the coordinate systems throughout the episode.
    for i in range(0, env.timeHistory.shape[1], 5):
        env.vehicle.updateMovingCoordSystem(env.timeHistory.loc[i, ["phi", "theta", "psi"]])
        resources.plotCoordSystem(ax[0], env.vehicle.iHat, env.vehicle.jHat, env.vehicle.kHat,
            x0=env.timeHistory.loc[i, ["x", "y", "z"]].values, ls="--", ds=0.5)

    ax[0].set_aspect("equal")
    ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)

    # Plot individual DoFs
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Displacement [m, rad]")
    ax[1].set_xlim((0, env.timeHistory["t"].max()))
    for i, v in enumerate(["x", "y", "z", "phi", "theta", "psi"]):
        ln, = ax[1].plot(env.timeHistory["t"], env.timeHistory[v], label=v)
        ax[1].hlines(env.vehicle.setPoint[i], 0, env.timeHistory["t"].max(), color=ln.get_color(), linestyle="dashed")
    ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)

    # Plot generalised control forces
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Generalised control force or moment [N, Nm]")
    ax[2].set_xlim((0, env.timeHistory["t"].max()))
    for i, v in enumerate(["X", "Y", "Z", "K", "M", "N"]):
        ln, = ax[2].plot(env.timeHistory["t"], env.timeHistory[f"F{i:d}"], label=v)
        ax[2].hlines(env.vehicle.setPoint[i], 0, env.timeHistory["t"].max(), color=ln.get_color(), linestyle="dashed")
    ax[2].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)

    return fig, ax


if __name__ == "__main__":

    # === Test roll and pitch decay ===

    # Constants and initial conditions
    state0 = np.array([
        0., 0., 0., 0./180.*np.pi, 30./180.*np.pi, 0./180.*np.pi,
        0., 0., 0., 0., 0., 0.])
    tMax = 10.
    rov = BlueROV2Heavy6DoF(np.zeros(6))
    result_solve_ivp = scipy.integrate.solve_ivp(
        rov.derivs, (0, tMax), state0, method="DOP853",#'RK45',
        t_eval=np.arange(0, tMax+1e-3, 0.1), rtol=1e-3, atol=1e-3)

    fig, ax = plt.subplots()
    ax.set_xlim((0, tMax))
    for i, v in enumerate(["x", "y", "z", "theta", "phi", "psi"]):
        ln, = ax.plot(result_solve_ivp.t, result_solve_ivp.y[i, :], label=v)
        ax.hlines(rov.setPoint[i], result_solve_ivp.t[0], result_solve_ivp.t[-1],
                  color=ln.get_color(), linestyle="dashed")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)
    plt.show()

    # === Test the dynamics subject to simple control ===

    # # Constants and initial conditions
    # dt = 0.25
    # # state0 = np.array([
    # #     0., 0., 0., 0./180.*np.pi, 0./180.*np.pi, 0./180.*np.pi,
    # #     0., 0., 0., 0., 0., 0.])
    # state0 = np.array([
    #     0., 0., 0., 0./180.*np.pi, 0./180.*np.pi, 150./180.*np.pi,
    #     0., 0., 0., 0., 0., 0.])
    # # state0 = np.array([
    # #     0., 0., 0., 30./180.*np.pi, -40./180.*np.pi, 150./180.*np.pi,
    # #     0., 0., 0., 0., 0., 0.])
    # tMax = 15.
    # t = np.arange(0.0, tMax, dt)
    #
    # # Set up the vehicle with a single waypoint and desired attitude
    # rov = BlueROV2Heavy6DoF([1., -1., 0.5, -10./180.*np.pi, 10./180.*np.pi, 280./180.*np.pi])
    # # rov = BlueROV2Heavy6DoF([2., -4.5, 1.5, -10./180.*np.pi, 30./180.*np.pi, 280./180.*np.pi])
    #
    # # Advance in time
    # result_solve_ivp = scipy.integrate.solve_ivp(
    #     rov.derivs, (0, tMax), state0, method='RK45', t_eval=t, rtol=1e-3, atol=1e-3)
    #
    # # Sort out the computed angles.
    # # result_solve_ivp.y[3:, :] = result_solve_ivp.y[3:, :] % (2.*np.pi)
    # #
    # # Plot trajectory
    # fig  = plt.figure(figsize=(14, 8))
    # fig.canvas.manager.set_window_title('Test 1 - PID, vehicle class only')
    # ax = [
    #     fig.add_subplot(1, 2, 1, projection='3d'),
    #     fig.add_subplot(1, 2, 2),
    # ]
    # ax[0].set_xlabel("x")
    # ax[0].set_ylabel("y")
    # ax[0].set_zlabel("z")
    # ax[0].set_aspect("equal")
    # ax[0].invert_yaxis()  # NED and y +ve to stbd
    # ax[0].invert_zaxis()
    # ax[0].plot(result_solve_ivp.y[0, :], result_solve_ivp.y[1, :], result_solve_ivp.y[2, :], "r", label="Trajectory")
    # ax[0].plot(rov.setPoint[0], rov.setPoint[1], rov.setPoint[2], "ko", label="Waypoint")
    # # Plot the coordinate systems throughout the episode.
    # for i in range(0, result_solve_ivp.y.shape[1], 10):
    #     rov.updateMovingCoordSystem(result_solve_ivp.y[3:6, i])
    #     if i == 0:
    #         ls = "-"
    #     else:
    #         ls = "--"
    #     resources.plotCoordSystem(ax[0], rov.iHat, rov.jHat, rov.kHat, x0=result_solve_ivp.y[:3, i], ls=ls)
    # ax[0].set_aspect("equal")
    # ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    #
    # # Plot individual DoFs
    # ax[1].set_xlim((0, tMax))
    # for i, v in enumerate(["x", "y", "z", "theta", "phi", "psi"]):
    #     ln, = ax[1].plot(result_solve_ivp.t, result_solve_ivp.y[i, :], label=v)
    #     ax[1].hlines(rov.setPoint[i], 0, tMax, color=ln.get_color(), linestyle="dashed")
    #
    # ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)

    # # Save the trajectory.
    # with open("./tempData/trajectory.obj", "w") as outfile:
    #     for iTime in range(result_solve_ivp.t.shape[0]):
    #         outfile.write("v {:.6e} {:.6e} {:.6e}\n".format(result_solve_ivp.y[0, iTime],
    #             result_solve_ivp.y[1, iTime], result_solve_ivp.y[2, iTime]))
    #     for iTime in range(result_solve_ivp.t.shape[0]-1):
    #         outfile.write("l {:d} {:d}\n".format(iTime+1, iTime+2))

    # === Test the environment with a constant set point ===

    # env = BlueROV2Heavy6DoFEnv(maxSteps=100)
    # env.reset(initialSetpoint=[2., -4.5, 1.5, -10./180.*np.pi, 30./180.*np.pi, 280./180.*np.pi])
    # for i in range(100):
    #     env.step(np.zeros(6))
    # plotEpisodeDetail(env, title='Test 2 - PID, env class')

    plt.show()
