# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import numpy as np
from scipy.spatial.transform import Rotation

class RovTemp(object):
    def __init__(self):
        # Main part - rotation matrix around the the global coordinate system axes.
        self.vehicleAxes = np.eye(3)
        # Current roll, pitch, yaw
        self.rotation_angles = np.zeros(3)
        # Unit vectors along the vehicle x, y, z axes unpacked from the aggregate
        # array for ease of use.
        self.iHat, self.jHat, self.kHat = self.getCoordSystem()

    def getCoordSystem(self):
        # iHat, jHat, kHat
        return self.vehicleAxes.T

    def computeRollPitchYaw(self):
        # Compute the global roll, pitch, and yaw angles
        roll = np.arctan2(self.kHat[1], self.kHat[2])
        pitch = np.arctan2(-self.kHat[0], np.sqrt(self.kHat[1]**2 + self.kHat[2]**2))
        yaw = np.arctan2(self.jHat[0], self.iHat[0])
        return np.array([roll, pitch, yaw])

    def updateMovingCoordSystem(self, rotation_angles):
        # Compute the change in the rotation angles compared to the previous time step.
        dRotAngles = rotation_angles - self.rotation_angles
        # Store the current orientation.
        self.rotation_angles = rotation_angles
        # Create quaternion from rotation angles from (roll pitch yaw)
        rotation_quaternion = Rotation.from_euler('xyz', dRotAngles, degrees=False).as_quat()
        # Convert quaternion to rotation matrix
        rotation_matrix = Rotation.from_quat(rotation_quaternion).as_matrix()
        # Apply rotation to the previous coordinate system.
        self.vehicleAxes = rotation_matrix.dot(self.vehicleAxes)
        # Extract the new coordinate system vectors
        self.iHat, self.jHat, self.kHat = self.getCoordSystem()

rov = RovTemp()

# Plot orientation.
fig  = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_aspect("equal")
plt.subplots_adjust(top=0.95, bottom=0.15)
lim = 0.5
ax.set_xlim((-lim, lim))
ax.set_ylim((-lim, lim))
ax.set_zlim((-lim, lim))

def plotCoordSystem(ax, iHat, jHat, kHat, x0=np.zeros(3), ds=0.45, ls="-"):
    x1 = x0 + iHat*ds
    x2 = x0 + jHat*ds
    x3 = x0 + kHat*ds
    lns = ax.plot([x0[0], x1[0]], [x0[1], x1[1]], [x0[2], x1[2]], "r", ls=ls, lw=2)
    lns += ax.plot([x0[0], x2[0]], [x0[1], x2[1]], [x0[2], x2[2]], "g", ls=ls, lw=2)
    lns += ax.plot([x0[0], x3[0]], [x0[1], x3[1]], [x0[2], x3[2]], "b", ls=ls, lw=2)
    return lns

# Plot twice - one plot will be updated, the other one will stay as reference.
plotCoordSystem(ax, rov.iHat, rov.jHat, rov.kHat, ls="--")
lns = plotCoordSystem(ax, rov.iHat, rov.jHat, rov.kHat)

sldr_ax1 = fig.add_axes([0.15, 0.01, 0.7, 0.025])
sldr_ax2 = fig.add_axes([0.15, 0.05, 0.7, 0.025])
sldr_ax3 = fig.add_axes([0.15, 0.09, 0.7, 0.025])
sldrLim = 180
sldr1 = Slider(sldr_ax1, 'phi', -sldrLim, sldrLim, valinit=0, valfmt="%.1f deg")
sldr2 = Slider(sldr_ax2, 'theta', -sldrLim, sldrLim, valinit=0, valfmt="%.1f deg")
sldr3 = Slider(sldr_ax3, 'psi', -sldrLim, sldrLim, valinit=0, valfmt="%.1f deg")

def onChanged(val):
    global rov, lns, ax
    angles = np.array([sldr1.val, sldr2.val, sldr3.val])/180.*np.pi
    rov.updateMovingCoordSystem(angles)
    for l in lns:
        l.remove()
    lns = plotCoordSystem(ax, rov.iHat, rov.jHat, rov.kHat)
    ax.set_title(
        "roll, pitch, yaw = " +", ".join(['{:.1f} deg'.format(v) for v in rov.computeRollPitchYaw()/np.pi*180.]))
    return lns

sldr1.on_changed(onChanged)
sldr2.on_changed(onChanged)
sldr3.on_changed(onChanged)

plt.show()
