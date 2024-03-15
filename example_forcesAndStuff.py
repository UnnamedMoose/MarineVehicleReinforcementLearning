# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from matplotlib.widgets import Slider

class RovTemp(object):
    def __init__(self):
        self.initial_rotation_matrix = np.eye(3)
        self.iHat, self.jHat, self.kHat = self.initial_rotation_matrix .T

    def updateMovingCoordSystem(self, rotation_angles):
        # Create quaternion from rotation angles from (roll pitch yaw)
        rotation_quaternion = Rotation.from_euler('xyz', rotation_angles, degrees=False).as_quat()
        # Convert quaternion to rotation matrix
        rotation_matrix = Rotation.from_quat(rotation_quaternion).as_matrix()
        # Apply rotation to the initial coordinate system
        rotated_matrix = rotation_matrix.dot(self.initial_rotation_matrix)
        # Extract the new coordinate system vectors
        self.iHat, self.jHat, self.kHat = rotated_matrix.T

rov = RovTemp()

# Plot orientation.
fig  = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_aspect("equal")
plt.subplots_adjust(top=1, bottom=0.15)
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
    global rov, lns
    angles = np.array([sldr1.val, sldr2.val, sldr3.val])/180.*np.pi
    rov.updateMovingCoordSystem(angles)
    for l in lns:
        l.remove()
    lns = plotCoordSystem(ax, rov.iHat, rov.jHat, rov.kHat)
    return lns

sldr1.on_changed(onChanged)
sldr2.on_changed(onChanged)
sldr3.on_changed(onChanged)

plt.show()
