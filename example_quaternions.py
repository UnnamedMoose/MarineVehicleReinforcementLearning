# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
import os
import re
import sys
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate
from scipy.spatial.transform import Rotation

sys.path.append("D:/git/UnderwaterVehicleReinforcementLearning")
import resources

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

class RovTemp(object):
    def __init__(self):
        self.setPoint = np.array([-2., -1., 0.5, 47./180.*np.pi, -94./180.*np.pi, 172./180.*np.pi])

class Result(object):
    def __init__(self):
        self.t = []
        self.y = np.zeros((6, 0))

    def logNewState(self, tNew, yNew):
        self.t = np.append(self.t, tNew)
        self.y = np.append(self.y, np.array(yNew)[:, np.newaxis], axis=1)

# Run a mock simulation.
nSteps = 11
y0 = np.zeros(6)
result_solve_ivp = Result()
rov = RovTemp()
t = 0.
y = y0.copy()
result_solve_ivp.logNewState(t, y)
for i in range(nSteps):
    t = t + 0.1
    y = rov.setPoint*np.sin(0.5*np.pi*(i+1)/nSteps)
    result_solve_ivp.logNewState(t, y)

# Compute transformed coordinate system for each time step.
iHat0 = np.array([1, 0, 0])
jHat0 = np.array([0, 1, 0])
kHat0 = np.array([0, 0, 1])

# Combine the vectors into a 3x3 rotation matrix
initial_rotation_matrix = np.column_stack((iHat0, jHat0, kHat0))

coordSystems = [initial_rotation_matrix]
for i in range(nSteps):

    rotation_angles = result_solve_ivp.y[3:, i+1]

    # Create quaternion from rotation angles
    rotation_quaternion = Rotation.from_euler('xyz', rotation_angles, degrees=False).as_quat()

    # Convert quaternion to rotation matrix
    rotation_matrix = Rotation.from_quat(rotation_quaternion).as_matrix()

    # Apply rotation to the initial coordinate system
    rotated_matrix = rotation_matrix.dot(initial_rotation_matrix)

    # Extract the new coordinate system vectors
    new_axis_x, new_axis_y, new_axis_z = rotated_matrix.T
    print(i, np.dot(new_axis_x, new_axis_y), np.dot(new_axis_x, new_axis_z), np.dot(new_axis_y, new_axis_z))

    coordSystems.append(np.column_stack((new_axis_x, new_axis_y, new_axis_z)))

# Plot individual DoFs
fig, ax  = plt.subplots()
for i, v in enumerate(["x", "y", "z", "theta", "phi", "psi"]):
    ln, = ax.plot(result_solve_ivp.t, result_solve_ivp.y[i, :], label=v)
ax.legend()

# Plot trajectory.
fig  = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(result_solve_ivp.y[0, :], result_solve_ivp.y[1, :], result_solve_ivp.y[2, :], "k-")

for i, coordSystem in enumerate(coordSystems):
    x0 = result_solve_ivp.y[:3, i]
    ds = 0.5
    x1 = x0 + coordSystem[0, :]*ds
    x2 = x0 + coordSystem[1, :]*ds
    x3 = x0 + coordSystem[2, :]*ds
    ax.plot([x0[0], x1[0]], [x0[1], x1[1]], [x0[2], x1[2]], "r-", lw=2)
    ax.plot([x0[0], x2[0]], [x0[1], x2[1]], [x0[2], x2[2]], "g-", lw=2)
    ax.plot([x0[0], x3[0]], [x0[1], x3[1]], [x0[2], x3[2]], "b-", lw=2)

ax.set_aspect("equal")

# # Save the target location and orientation.
# resources.saveCoordSystem("./tempData/target.vtk", rov.setPoint[:3], rov.setPoint[3:6], L=0.4)
#
# # Save moving coordinate system and geometry.
# for iTime in range(result_solve_ivp.t.shape[0]):
#     filename = "./tempData/result_{:06d}.vtk".format(iTime)
#     resources.saveCoordSystem(filename, result_solve_ivp.y[:3, iTime], result_solve_ivp.y[3:6, iTime])

plt.show()
