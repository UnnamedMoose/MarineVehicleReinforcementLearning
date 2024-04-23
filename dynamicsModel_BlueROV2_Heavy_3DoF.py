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
import scipy.integrate
import gym

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

from resources import headingError

class BlueROV2Heavy3DoF:
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
        self.m = 11.4 # kg # +++
        self.dispVol = self.m / self.rho_f # For neutral buoyancy.
        self.Length = 0.457 # m (overall dimensions of the ROV) # +++
        self.Width = 0.338 # +++

        self.CB = np.zeros(3) # Coordinate system centred on CB # +++
        self.CG = np.array([0., 0., 0.02]) # m (relative to the CB) # +++

        self.I = np.array([ # kg m2 # +++
            [0.16, 0.,  0.],
            [0., 0.16,  0.],
            [0., 0., 0.16],
        ])

        # ===
        # Manoeuvring coefficients.

        # ---
        # Added mass and intertia
        self.Xudot =  -5.5 # kg # +++
        self.Yvdot = -12.7 # +++
        self.Nrdot = -0.12 # kg m^2 / rad # +++
        # Cross-coupling terms.
        self.Yrdot =   0.
        self.Nvdot =   0.

        # ---
        # Quadratic terms.
        self.Xuu = -18.18 # kg/m # +++
        self.Yvv = -21.66 # +++
        self.Yrr =   0.
        self.Ypp =   0.
        self.Nvv =   0.0 # kg
        self.Nrr =  -1.55 # kg m^2 / rad^2 # +++ # TODO maybe wrong units?
        self.Npp =    0.

        # ---
        # Linear damping terms - same as Kantapon but divided by velocity and density
        self.Xu = -4.03 # +++
        self.Yv = -6.22 # +++
        self.Yr = 0.
        self.Yp = 0.
        self.Nv = 0.
        self.Nr = -0.07 # +++
        self.Np = 0.

        # ===
        # Thruster.
        self.D_thruster = 0.1 # Diameter of thrusters # +++
        self.alphaThruster = 45./180.*np.pi # Angle between thruster axes and centreline (0 fwd)
        self.l_x = 0.156 # Moment arms [m] # +++ # NOTE use these for model validation only.
        self.l_y = 0.111
        # Back-calculated from max rpm at 16 V from datasheet:
        # https://bluerobotics.com/store/thrusters/t100-t200-thrusters/t200-thruster/
        # And 40 N measured by Wu (2018) - pp. 49
        self.Kt_thruster = 40. / (1000. * (3500./60.)**2. * self.D_thruster**4.) # +++

        # Generalised control forces and moments - X Y N
        self.generalisedControlForces = np.zeros(3)
        # Motor rpms - FP AP FS AS
        self.controlVector = np.zeros(4)

        # Use pseudo-inverse of the control allocation matrix in order to go from
        # desired generalised forces to actuator demands in rpm.
        A = np.array([
            [1., 1., -1., -1.],
            [1., -1., 1., -1.],
            [1., 1., 1., 1.],
        ])
        A[0,:] = A[0,:]*np.cos(self.alphaThruster)
        A[1,:] = A[1,:]*np.sin(self.alphaThruster)
        A[2,:] = A[2,:]*np.sin(self.alphaThruster)*self.Length/2.
        self.Ainv = np.linalg.pinv(A)

    def thrusterModel(self, u, v, rpm):
        """ Compute the force delivered by a thruster. """

        # Force delivered when at rest.
        Fthruster = self.rho_f * (rpm/60.)**2. * np.sign(rpm) * self.D_thruster**4. * self.Kt_thruster

        # Jet velocity and associated drag augment.
        uJet = np.sqrt(np.abs(Fthruster) / (0.5*self.rho_f*np.pi*self.D_thruster**2))
        deltaCdFwd = 0.56599 * np.exp(-7.60891*np.abs(u)/max(1e-5, uJet)) \
            + 0.05654 * np.exp(-0.89679*np.abs(u)/max(1e-5, uJet))
        Xthruster = deltaCdFwd * -0.5*self.rho_f*np.abs(u)*u*self.dispVol**(2./3.)

        return Fthruster, Xthruster

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

        # Go from forces to rpm.
        cv = np.matmul(self.Ainv, np.array([Xd, Yd, Nd])) # N
        cv = np.sign(cv)*np.sqrt(np.abs(cv)/(self.rho_f*self.D_thruster**4.*self.Kt_thruster))*60. # rpm
        self.controlVector = cv

        # Apply saturation and deadband
        def limit(x):
            r = max(-3500., min(3500., x))
            if np.abs(r) < 300:
                r = 0.
            return r

        rpmFP = limit(cv[0])
        rpmAP = limit(cv[1])
        rpmFS = limit(cv[2])
        rpmAS = limit(cv[3])

        # TODO add a current model
        velCurrent = np.zeros(3)

        # Resolve the current into the vehicle reference frame.
        if np.linalg.norm(velCurrent) > 0.:
            invJtransform = np.linalg.pinv(Jtransform)
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
        # Divided by velocity to be made invariant of the current state.
        Dl = -1. * np.array([
            [self.Xu, 0., 0.],
            [0., self.Yv, self.Yr],
            [0., self.Nv, self.Nr],
        ])

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
        # Thruster model.
        F_FP, X_FP = self.thrusterModel(u, v, rpmFP)
        F_AP, X_AP = self.thrusterModel(u, v, rpmAP)
        F_FS, X_FS = self.thrusterModel(u, v, rpmFS)
        F_AS, X_AS = self.thrusterModel(u, v, rpmAS)

        # ===
        # Hydrodynamic forces excluding added mass terms.
        self.useTrueMomentArms = True
        self.useJetDragAugment = True

        if self.useJetDragAugment:
            Xh = X_FP + X_AP + X_FS + X_AS + (F_FP + F_AP - F_FS - F_AS)*np.cos(self.alphaThruster)
        else:
            Xh = (F_FP + F_AP - F_FS - F_AS)*np.cos(self.alphaThruster)

        Yh = (F_FP - F_AP + F_FS - F_AS)*np.sin(self.alphaThruster)

        if self.useTrueMomentArms:
            Nh = np.sqrt(self.l_x**2. + self.l_y**2.) * (F_FP + F_AP + F_FS + F_AS)
        else:
            Nh = (F_FP + F_AP + F_FS + F_AS)*np.sin(self.alphaThruster)*self.Length/2.

        H = np.array([Xh, Yh, Nh])

        # ===
        # Hydrostatic forces.
        G = np.zeros(3)

        # ===
        # Externally applied forces and moments.
        E = np.array([0., 0., 0.])

        # ===
        # Total forces and moments
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

        xyThrusters = np.array([
            [self.Length/2. - self.D_thruster/2.*np.sin(self.alphaThruster),
                -self.Width/2. + self.D_thruster/2.*np.cos(self.alphaThruster)],
            [-self.Length/2. + self.D_thruster/2.*np.sin(self.alphaThruster),
                -self.Width/2. + self.D_thruster/2.*np.cos(self.alphaThruster)],
            [self.Length/2. - self.D_thruster/2.*np.sin(self.alphaThruster),
                self.Width/2. - self.D_thruster/2.*np.cos(self.alphaThruster)],
            [-self.Length/2. + self.D_thruster/2.*np.sin(self.alphaThruster),
                self.Width/2. - self.D_thruster/2.*np.cos(self.alphaThruster)]
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

        F = np.array([
            [forces[0]*np.cos(self.alphaThruster), forces[0]*np.sin(self.alphaThruster)],
            [forces[1]*np.cos(self.alphaThruster), -forces[1]*np.sin(self.alphaThruster)],
            [-forces[2]*np.cos(self.alphaThruster), forces[2]*np.sin(self.alphaThruster)],
            [-forces[3]*np.cos(self.alphaThruster), -forces[3]*np.sin(self.alphaThruster)],
        ])

        def rotate(xy, psi):
            xyn = np.zeros(xy.shape)
            xyn[:,0] = np.cos(psi)*xy[:,0] - np.sin(psi)*xy[:,1]
            xyn[:,1] = np.sin(psi)*xy[:,0] + np.cos(psi)*xy[:,1]
            return xyn

        xyHull = rotate(xyHull*scale, psi) + x0
        xyThrusters = rotate(xyThrusters*scale, psi) + x0
        xyCentreline = rotate(xyCentreline*scale, psi) + x0
        xyDir = rotate(xyDir*scale, psi) + x0
        F = rotate(F, psi)

        objects = []
        objects += ax.fill(xyHull[:,1], xyHull[:,0], vehicleColour, alpha=0.5)
        objects += ax.plot(xyCentreline[:,1], xyCentreline[:,0], "k--", lw=2*markerSize)
        objects += ax.plot(xyDir[:,1], xyDir[:,0], "k-", lw=2*markerSize)
        objects += ax.plot(x0[1], x0[0], "ko", mew=3, mfc="None", ms=14*markerSize)
        objects += ax.plot(xyThrusters[:,1], xyThrusters[:,0], "k.", ms=8)

        for i in range(4):
            if np.abs(forces[i]) > 0:
                objects.append( ax.arrow(xyThrusters[i,1], xyThrusters[i,0], F[i,1]*scale, F[i,0]*scale, color="b",
                        width=0.002*self.Length*scale/2.*arrowSize, lw=2*arrowSize) )

        uD = np.array([np.sin(psiD), np.cos(psiD)])*self.Length/2.
        objects.append( ax.arrow(x0[1], x0[0], uD[0]*scale, uD[1]*scale, color="r",
                width=0.002*self.Length*scale/2.*arrowSize, lw=2*arrowSize) )

        return objects

class BlueROV2Heavy3DoFEnv(gym.Env):
    def __init__(self, seed=None, dt=0.2, maxSteps=250):
        # Call base class constructor.
        super(BlueROV2Heavy3DoFEnv, self).__init__()

        # Set common stuff.
        self.seed = seed
        self.dt = dt
        self._max_episode_steps = maxSteps

        # Define the action space.
        self.lenAction = 3
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
            (self.path[self.iWp, 0]-systemState[0]) / (self.vehicle.Length*3.),
            (self.path[self.iWp, 1]-systemState[1]) / (self.vehicle.Length*3.),
            # To the following waypoint for path following
            (self.path[self.iWp+1, 0]-systemState[0]) / (self.vehicle.Length*3.),
            (self.path[self.iWp+1, 1]-systemState[1]) / (self.vehicle.Length*3.),
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
            nWps = 2
            self.iWp = 0
            self.path = (np.random.rand(2*nWps).reshape((nWps, 2))-[0.5, 0.5])*[10., 10.]
            self.targetHeading = np.random.rand()*2.*np.pi
            # The initial set point is the firs point on the path.
            sp = np.append(self.path[0, :], self.targetHeading)
            self.fixedSp = False
        else:
            # Set up the vehicle with a fixed initial waypoint and desired heading.
            # Doing so should allow a dummy agent to be used such that simple A->B
            # behaviour could be simulated. This can be used for the tuning of the
            # controller and testing of system dynamics.
            sp = np.array(initialSetpoint)
            self.iWp = 0
            self.path = np.vstack([sp[:2], sp[:2]])
            self.targetHeading = sp[2]
            self.fixedSp = True

        # Create a vehicle instance with an initial set point.
        self.vehicle = BlueROV2Heavy3DoF(sp)

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
            psi_d = action[2]*(45./180.*np.pi) + self.systemState[2]
            self.vehicle.setPoint = np.array([x_sp, y_sp, psi_d])

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


def lineOfSight(p0, p1, Rnav):
    """ Using a line of sight approach, determine the point the vehicle should be
    navigating to, given its current position and a linear path segment starting
    at p0 and ending at p1. Waypoints should be given relative to the current
    vehicle position. For the best effect, non-dimensionalise the coordinate system
    using the distance to the furthest waypoint (this includes Rnav, too). """

    # Check if the second waypoint is within LOS. If yes, move to it directly.
    dToWp = np.sqrt(np.sum((p1)**2.))
    if dToWp < Rnav:
        targetPoint = p1

    else:
        # Check if and where LOS intersects the path segment.
        pathVec = p1 - p0
        pHat = pathVec / np.linalg.norm(pathVec)
        dSegment = np.sqrt(np.sum(pathVec**2.))
        determinant = p0[0]*p1[1] - p1[0]*p0[1]
        delta = Rnav**2.*dSegment**2. - determinant**2.

        if delta < 0:
            # Path segment is outside of the line of sight, follow at a direction perpendicular
            # to the path in order to get back in range.
            dAlongSegment = np.dot(-p0, pHat)
            if dAlongSegment > dSegment:
                targetPoint = p1
            elif dAlongSegment < 0:
                targetPoint = p0
            else:
                targetPoint = p0 + dAlongSegment*pHat

        if delta >= 0:
            # Compute the point on the line that's between the two waypoints and nearer to
            # the current one.
            sy = np.sign(pathVec[1])
            if np.abs(sy) < 1e-12:
                sy = 1.
            pp0 = np.array([
                (determinant*pathVec[1] + sy*pathVec[0]*np.sqrt(delta)) / max(1e-6, dSegment)**2.,
                (-determinant*pathVec[0] + np.abs(pathVec[1])*np.sqrt(delta)) / max(1e-6, dSegment)**2.,
            ])
            pp1 = np.array([
                (determinant*pathVec[1] - sy*pathVec[0]*np.sqrt(delta)) / max(1e-6, dSegment)**2.,
                (-determinant*pathVec[0] - np.abs(pathVec[1])*np.sqrt(delta)) / max(1e-6, dSegment)**2.,
            ])

            # Compute non-dimensional signed distance of each candidate point along
            # the path segment.
            s0 = np.dot(pHat, pp0 - p0) / max(1e-6, dSegment)
            s1 = np.dot(pHat, pp1 - p0) / max(1e-6, dSegment)

            # Pick values between 0 and 1 and closer to 1 (current objective).
            if (s0 >= 0.) and (s0 <= 1.) and (s0 > s1):
                targetPoint = pp0
            elif (s1 >= 0.) and (s1 <= 1.):
                targetPoint = pp1

            # If neither intersection point is on the line segment, use the nearest
            # of the two waypoints until the line segment gets in range.
            elif np.linalg.norm(p1) < np.linalg.norm(p0):
                targetPoint = p1
            else:
                targetPoint = p0

    return targetPoint


class LOSNavigation(object):
    def __init__(self):
        pass

    def predict(self, obs, deterministic=True):
        # NOTE deterministic is a dummy kwarg needed to make this function look
        # like a stable baselines equivalent
        states = obs

        # First two items in the state vector are the relative positions to the
        # path waypoints non-dimensionalised by a factor of the vehicle length and
        # truncated to <-1, 1>.
        p0 = states[:2]
        p1 = states[2:4]
        psi_e = states[4]
        Rnav = 0.5
        targetPoint = lineOfSight(p0, p1, Rnav)

        # Actions are the set point relative to the vehicle
        # (can be viewed as poistion error) and heading error.
        # The vehicle controller will act to minimise these.
        actions = np.array([targetPoint[0], targetPoint[1], psi_e])

        return actions, states


def plotEpisodeDetail(env, title=""):
    # Plot trajectory
    fig, ax  = plt.subplots(1, 3, figsize=(14, 6))
    plt.subplots_adjust(top=0.89, bottom=0.139, left=0.069, right=0.973, hspace=0.2, wspace=0.445)
    fig.canvas.manager.set_window_title(title)
    ax[0].set_xlabel("y [m, +ve east]")
    ax[0].set_ylabel("x [m, +ve north]")
    ax[0].plot(env.timeHistory["x1"], env.timeHistory["x0"], "r", label="Trajectory")
    ax[0].plot(env.path[:, 1], env.path[:, 0], "m*--", ms=12, label="Waypoints")
    ax[0].set_aspect("equal")
    for i in [0, 5, 10, 15]:
        env.vehicle.plot_horizontal(ax[0], env.timeHistory["x0"].values[i], env.timeHistory["x1"].values[i],
            env.timeHistory["x2"].values[i], env.vehicle.setPoint[2],
            np.array([env.timeHistory["u0"][i], env.timeHistory["u1"][i],
                      env.timeHistory["u2"][i], env.timeHistory["u3"][i]])/25000.,
            markerSize=0.5, arrowSize=1, scale=1)
    ax[0].legend(ncol=3, bbox_to_anchor=(0.5, 1.01), loc="lower center")

    # Plot individual DoFs
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Displacement [m, rad]")
    ax[1].set_xlim((0, env.timeHistory["t"].max()))
    for i, v in enumerate(["x", "y", "psi"]):
        ln, = ax[1].plot(env.timeHistory["t"], env.timeHistory[f"x{i:d}"], label=v)
        ax[1].hlines(env.vehicle.setPoint[i], 0, env.timeHistory["t"].max(), color=ln.get_color(), linestyle="dashed")
    ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)

    # Plot generalised control forces
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Generalised control force or moment [N, Nm]")
    ax[2].set_xlim((0, env.timeHistory["t"].max()))
    for i, v in enumerate(["X", "Y", "N"]):
        ln, = ax[2].plot(env.timeHistory["t"], env.timeHistory[f"F{i:d}"], label=v)
        ax[2].hlines(env.vehicle.setPoint[i], 0, env.timeHistory["t"].max(), color=ln.get_color(), linestyle="dashed")
    ax[2].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)

    return fig, ax


if __name__ == "__main__":

    # === Test the dynamics ===

    # Constants and initial conditions
    dt = 0.25
    state0 = np.array([0., 0., 0./180.*np.pi, 0., 0., 0.])
    tMax = 15.
    t = np.arange(0.0, tMax, dt)

    # Set up the vehicle with a single waypoint and desired heading
    rov = BlueROV2Heavy3DoF([1., -1., 280./180.*np.pi])

    # Advance in time
    result_solve_ivp = scipy.integrate.solve_ivp(
        rov.derivs, (0, tMax), state0, method='RK45', t_eval=t, rtol=1e-3, atol=1e-3)

    # Sort out the computed heading.
    result_solve_ivp.y[2, :] = result_solve_ivp.y[2, :] % (2.*np.pi)

    # Plot trajectory
    fig, ax  = plt.subplots(1, 2)
    fig.canvas.manager.set_window_title('Test 1 - PID, vehicle class only')
    ax[0].plot(result_solve_ivp.y[0, :], result_solve_ivp.y[1, :], "r", label="Trajectory")
    ax[0].plot(rov.setPoint[0], rov.setPoint[1], "ko", label="Waypoint")
    ax[0].set_aspect("equal")
    ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01))

    # Plot individual DoFs
    ax[1].set_xlim((0, tMax))
    for i, v in enumerate(["x", "y", "psi"]):
        ln, = ax[1].plot(result_solve_ivp.t, result_solve_ivp.y[i, :], label=v)
        ax[1].hlines(rov.setPoint[i], 0, tMax, color=ln.get_color(), linestyle="dashed")
    ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)

    # === Test the environment with a constant set point ===

    env = BlueROV2Heavy3DoFEnv(maxSteps=100)
    env.reset(initialSetpoint=[1., -1., 280./180.*np.pi])
    for i in range(100):
        env.step([0., 0., 0.])

    plotEpisodeDetail(env, title='Test 2 - PID, env class')

    # === Test the dummy agent that applies the LOS navigation algorithm ===

    num_steps = 100
    env = BlueROV2Heavy3DoFEnv(maxSteps=num_steps)
    agent = LOSNavigation()

    deterministic = True
    episode_rewards = []
    done = False

    obs = env.reset()
    for i in range(num_steps):
        action, _states = agent.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        episode_rewards.append(reward)

    plotEpisodeDetail(env, title='Test 3 - LOS navigation with the env class')

    # ===
    plt.show()
