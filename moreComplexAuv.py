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
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

from resources import headingError

# %% BlueROV 2 Heavy

class BlueROV2Heavy3DoF:
    def __init__(self, waypoint):

        # Target.
        self.waypoint = np.array(waypoint)

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
        e = np.append(self.waypoint[:2] - [x, y], headingError(self.waypoint[2], psi))
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

        # Go from forces to rpm.
        cv = np.matmul(self.Ainv, np.array([Xd, Yd, Nd])) # N
        cv = np.sign(cv)*np.sqrt(np.abs(cv)/(self.rho_f*self.D_thruster**4.*self.Kt_thruster))*60. # rpm

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
        # Divided by rho to correct for environment models c.f. Kantapon's thesis
        # and by velocity to be made invariant of the current state.
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
#        RHS = -np.dot(C+D, vel) - G + H + E
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


if __name__ == "__main__":
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
    fig, ax  = plt.subplots()
    ax.plot(result_solve_ivp.y[0, :], result_solve_ivp.y[1, :], "r", label="Trajectory")
    ax.plot(rov.waypoint[0], rov.waypoint[1], "ko", label="Waypoint")
    ax.set_aspect("equal")
    ax.legend()

    # Plot individual DoFs
    fig, ax  = plt.subplots()
    ax.set_xlim((0, tMax))
    for i, v in enumerate(["x", "y", "psi"]):
        ln, = ax.plot(result_solve_ivp.t, result_solve_ivp.y[i, :], label=v)
        ax.hlines(rov.waypoint[i], 0, tMax, color=ln.get_color(), linestyle="dashed")
    ax.legend()
