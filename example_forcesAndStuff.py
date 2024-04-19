# -*- coding: utf-8 -*-

# TODO we want to express forces and moments in the body coordinate system and integrate there.
#   Then transform the displacements and orientation into the global reference frame.
#   This means we need to rotate the (iHat, jHat, kHat) coordinates around those axes
#   but and get their updated directions expressed in the global coordinate system.
# TODO Or use the old approach and compute the forces and integrate accelerations
#   in the local coordinate system but then convert the velocities to the global
#   reference frame before integrating.

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

        # Thruster geometry.
        self.alphaThruster = 45./180.*np.pi # Angle between each horizontal thruster axes and centreline (0 fwd)
        self.l_x = 0.156 # Moment arms [m] # +++
        self.l_y = 0.111
        self.l_z = 0.085
        self.l_x_v = 0.120
        self.l_y_v = 0.218
        self.l_z_v = 0.0  # irrelevant

        # Thrister positions in the vehicle reference frame. Consistent with Wu (2018) fig 4.2
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

        # Use pseudo-inverse of the control allocation matrix in order to go from
        # desired generalised forces to actuator demands in rpm.
        # NOTE the original 3 DoF notation is inconsistent with page 48 in Wu (2018),
        # what follows is (or should be) the same. See Figure 4.2 and Eq. 4.62 in their work.
        self.A = np.zeros((6, 8))
        self.A[:, 0] = [np.cos(self.alphaThruster), -np.sin(self.alphaThruster), 0.,
            np.sin(self.alphaThruster)*self.l_z, np.cos(self.alphaThruster)*self.l_z,
            -np.sin(self.alphaThruster)*self.l_x - np.cos(self.alphaThruster)*self.l_y]
        self.A[:, 1] = [np.cos(self.alphaThruster), np.sin(self.alphaThruster), 0.,
            -np.sin(self.alphaThruster)*self.l_z, np.cos(self.alphaThruster)*self.l_z,
            np.sin(self.alphaThruster)*self.l_x + np.cos(self.alphaThruster)*self.l_y]
        self.A[:, 2] = [-np.cos(self.alphaThruster), -np.sin(self.alphaThruster), 0.,
            np.sin(self.alphaThruster)*self.l_z, -np.cos(self.alphaThruster)*self.l_z,
            np.sin(self.alphaThruster)*self.l_x + np.cos(self.alphaThruster)*self.l_y]
        self.A[:, 3] = [-np.cos(self.alphaThruster), np.sin(self.alphaThruster), 0.,
            -np.sin(self.alphaThruster)*self.l_z, -np.cos(self.alphaThruster)*self.l_z,
            -np.sin(self.alphaThruster)*self.l_x - np.cos(self.alphaThruster)*self.l_y]
        self.A[:, 4] = [0., 0., -1., -self.l_y_v, self.l_x_v, 0.]
        self.A[:, 5] = [0., 0., 1., -self.l_y_v, -self.l_x_v, 0.]
        self.A[:, 6] = [0., 0., 1., self.l_y_v, self.l_x_v, 0.]
        self.A[:, 7] = [0., 0., -1., self.l_y_v, -self.l_x_v, 0.]
        self.Ainv = np.linalg.pinv(self.A)

    def getCoordSystem(self):
        # iHat, jHat, kHat
        return self.vehicleAxes.T

    def computeRollPitchYaw(self):
        # Compute the global roll, pitch, and yaw angles.
        # NOTE: These are not particularly safe and can be +/- pi away from the truth. Use with caution!
        roll = -np.arctan2(self.kHat[1], self.kHat[2])
        pitch = np.arctan2(self.kHat[0], self.kHat[2])
        # pitch = -np.arctan2(-self.kHat[0], np.sqrt(self.kHat[1]**2 + self.kHat[2]**2))
        yaw = -np.arctan2(self.jHat[0], self.iHat[0])
        return np.array([roll, pitch, yaw])

    def updateMovingCoordSystem(self, rotation_angles):
        # Store the current orientation.
        self.rotation_angles = rotation_angles
        # Create quaternion from rotation angles from (roll pitch yaw)
        self.vehicleAxes = Rotation.from_euler('XYZ', rotation_angles, degrees=False).as_matrix()
        # Extract the new coordinate system vectors
        self.iHat, self.jHat, self.kHat = self.getCoordSystem()

    def angularTransform(self, moments, rotation_angles, toWhatFrame):
        # TODO replace with scipy if possible to keep this clean.
        phi, theta, psi = rotation_angles

        cosThetaDenom = np.cos(theta)
        if np.abs(cosThetaDenom) < 1e-12:
            cosThetaDenom = 1e-6
        elif np.abs(cosThetaDenom) < 1e-6:
            cosThetaDenom = 1e-6 * np.sign(cosThetaDenom)

        J2 = np.array([
            [1., np.sin(phi)*np.sin(theta)/cosThetaDenom, np.cos(phi)*np.sin(theta)/cosThetaDenom],
            [0., np.cos(phi), -np.sin(phi)],
            [0., np.sin(phi)/cosThetaDenom, np.cos(phi)/cosThetaDenom],
        ])

        if toWhatFrame == "toVehicle":
            invJtransform = np.linalg.pinv(J2)
            return np.matmul(invJtransform, moments)
        elif toWhatFrame == "toGlobal":
            print(J2, moments)
            return np.dot(J2, moments)
        else:
            raise ValueError("what?")

    def globalToVehicle(self, vecGlobal):
        return np.array([
            np.dot(vecGlobal, self.iHat),
            np.dot(vecGlobal, self.jHat),
            np.dot(vecGlobal, self.kHat)])

    def vehicleToGlobal(self, vecVehicle):
        return vecVehicle[0]*self.iHat + vecVehicle[1]*self.jHat + vecVehicle[2]*self.kHat

rov = RovTemp()

# Plot orientation.
fig  = plt.figure(figsize=(8, 9))
ax = fig.add_subplot(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_aspect("equal")
plt.subplots_adjust(top=0.91, bottom=0.3)
lim = 0.5
ax.set_xlim((-lim, lim))
ax.set_ylim((-lim, lim))
ax.set_zlim((-lim, lim))
ax.invert_yaxis()  # NED and y +ve to stbd
ax.invert_zaxis()

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
texts = []

sldr_ax1 = fig.add_axes([0.1, 0.09, 0.3, 0.025])
sldr_ax2 = fig.add_axes([0.1, 0.05, 0.3, 0.025])
sldr_ax3 = fig.add_axes([0.1, 0.01, 0.3, 0.025])
sldrLim = 180
sldr1 = Slider(sldr_ax1, 'phi', -sldrLim, sldrLim, valinit=0, valfmt="%.1f deg")
sldr2 = Slider(sldr_ax2, 'theta', -sldrLim, sldrLim, valinit=0, valfmt="%.1f deg")
sldr3 = Slider(sldr_ax3, 'psi', -sldrLim, sldrLim, valinit=0, valfmt="%.1f deg")
sldr_ax4 = fig.add_axes([0.6, 0.09, 0.3, 0.025])
sldr_ax5 = fig.add_axes([0.6, 0.05, 0.3, 0.025])
sldr_ax6 = fig.add_axes([0.6, 0.01, 0.3, 0.025])
sldrLim = 1
sldr4 = Slider(sldr_ax4, 'Fg_x', -sldrLim, sldrLim, valinit=0., valfmt="%.2f")
sldr5 = Slider(sldr_ax5, 'Fg_y', -sldrLim, sldrLim, valinit=0., valfmt="%.2f")
sldr6 = Slider(sldr_ax6, 'Fg_z', -sldrLim, sldrLim, valinit=0., valfmt="%.2f")

sldr_ax10 = fig.add_axes([0.6, 0.21, 0.3, 0.025])
sldr_ax11 = fig.add_axes([0.6, 0.17, 0.3, 0.025])
sldr_ax12 = fig.add_axes([0.6, 0.13, 0.3, 0.025])
sldrLim = 0.2
sldr10 = Slider(sldr_ax10, 'Mg_x', -sldrLim, sldrLim, valinit=0., valfmt="%.2f")
sldr11 = Slider(sldr_ax11, 'Mg_y', -sldrLim, sldrLim, valinit=0., valfmt="%.2f")
sldr12 = Slider(sldr_ax12, 'Mg_z', -sldrLim, sldrLim, valinit=0., valfmt="%.2f")

# sldr_ax7 = fig.add_axes([0.1, 0.21, 0.3, 0.025])
# sldr_ax8 = fig.add_axes([0.1, 0.17, 0.3, 0.025])
# sldr_ax9 = fig.add_axes([0.1, 0.13, 0.3, 0.025])
# sldrLim = 0.5
# sldr7 = Slider(sldr_ax7, 'Fv_x', -sldrLim, sldrLim, valinit=0., valfmt="%.2f")
# sldr8 = Slider(sldr_ax8, 'Fv_y', -sldrLim, sldrLim, valinit=0., valfmt="%.2f")
# sldr9 = Slider(sldr_ax9, 'Fv_z', -sldrLim, sldrLim, valinit=0., valfmt="%.2f")

def onChanged(val):
    global rov, lns, texts, ax

    angles = np.array([sldr1.val, sldr2.val, sldr3.val])/180.*np.pi
    rov.updateMovingCoordSystem(angles)

    # Resolve force in the global coordinates to the vehicle axes.
    Fg = np.array([sldr4.val, sldr5.val, sldr6.val])
    Fglobal = rov.globalToVehicle(Fg)

    # Resolve moments to the vehicle axes.
    Mg = np.array([sldr10.val, sldr11.val, sldr12.val])
    Mglobal = rov.angularTransform(Mg, angles, "toVehicle")

    # Combine generalised control forces and moments into one vector.
    generalisedControlForces = np.append(Fglobal, Mglobal)

    # Translate into force demands for each thruster using the inverse of the thrust
    # allocation matrix.
    cv = np.matmul(rov.Ainv, generalisedControlForces)

    # # Resolve force in vehicle coordinates to the global coordinates.
    # Fv = np.array([sldr7.val, sldr8.val, sldr9.val])
    # Fvehicle = rov.vehicleToGlobal(Fv)

    # Plot everything.
    for l in lns:
        l.remove()
    for txt in texts:
        txt.remove()
    texts = []
    lns = plotCoordSystem(ax, rov.iHat, rov.jHat, rov.kHat)
    lns += ax.plot([0, Fg[0]], [0, Fg[1]], [0, Fg[2]], "m--", lw=2)
    lns += ax.plot([Fg[0]], [Fg[1]], [Fg[2]], "mo", ms=6)

    # Plot the thruster positions and actuation.
    H = np.zeros(6)
    for i in range(rov.thrusterPositions.shape[0]):
        xt = rov.vehicleToGlobal(rov.thrusterPositions[i, :])
        tVec = rov.vehicleToGlobal(rov.A[:3, i]*cv[i])
        H += rov.A[:, i]*cv[i]

        lns += ax.plot(xt[0], xt[1], xt[2], "ks", ms=5)
        texts.append(ax.text(xt[0], xt[1], xt[2], "{:d}".format(i+1)))
        lns += ax.plot([xt[0], xt[0]+tVec[0]], [xt[1], xt[1]+tVec[1]], [xt[2], xt[2]+tVec[2]], "k-", alpha=0.5, lw=2)

    # Compute net actuation in the global reference frame.
    Hglobal = np.append(
        rov.vehicleToGlobal(H[:3]),
        rov.angularTransform(H[3:], angles, "toGlobal"))

    # lns += ax.plot([0, Fvehicle[0]], [0, Fvehicle[1]], [0, Fvehicle[2]], "c--", lw=2)
    # lns += ax.plot([Fvehicle[0]], [Fvehicle[1]], [Fvehicle[2]], "co", ms=6)

    texts.append([fig.text(0.5, 0.975,
        "roll, pitch, yaw = " +", ".join(['{:.1f} deg'.format(v) for v in rov.computeRollPitchYaw()/np.pi*180.]),
        va="center", ha="center"
    )])
    texts.append(fig.text(0.5, 0.94,
        "Fg and Mg in vehicle reference frame = " +", ".join(['{:.2f}'.format(v) for v in np.append(Fglobal, Mglobal)]),
        va="center", ha="center"
    ))
    # texts.append(fig.text(0.5, 0.905,
    #     "Fv in global reference frame = " +", ".join(['{:.2f}'.format(v) for v in Fvehicle]),
    #     va="center", ha="center"
    # ))
    texts.append(fig.text(0.5, 0.905,
        "Total forces and moments in vehicle reference frame = " +", ".join(['{:.2f}'.format(v) for v in H]),
        va="center", ha="center"
    ))
    texts.append(fig.text(0.5, 0.87,
        "Total forces and moments in global reference frame = " +", ".join(['{:.2f}'.format(v) for v in Hglobal]),
        va="center", ha="center"
    ))

    return lns

sliders = [sldr1, sldr2, sldr3, sldr4, sldr5, sldr6,# sldr7, sldr8, sldr9,
    sldr10, sldr11, sldr12]
for sldr in sliders:
    sldr.on_changed(onChanged)

plt.show()
