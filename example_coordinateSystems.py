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
        # Set the coordinate transform matrix, its inverse, and vehicle axes unit vectors.
        self.updateMovingCoordSystem(np.zeros(3))

        # Temp in this example.
        self.pos = np.zeros(3)

        # CG offset - used for exporting data only. NED.
        self.xCG = np.array([0., 0., 0.0575])

        # Thruster geometry.
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

        # Thruster positions in the vehicle reference frame. Consistent with Wu (2018) fig 4.2
        self.thrusterNames = [
            "stbd_fwd_hor",
            "port_fwd_hor",
            "stbd_aft_hor",
            "port_aft_hor",
            "stbd_fwd_vert",
            "port_fwd_vert",
            "stbd_aft_vert",
            "port_aft_vert",
        ]
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

    def computeRollPitchYaw(self):
        # Compute the global roll, pitch, and yaw angles.
        # NOTE: These are not particularly safe and can be +/- pi away from the truth. Use with caution!
        roll = -np.arctan2(self.kHat[1], self.kHat[2])
        pitch = np.arctan2(self.kHat[0], self.kHat[2])
        yaw = -np.arctan2(self.jHat[0], self.iHat[0])
        return np.array([roll, pitch, yaw])

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

    def saveCoordSystem(self, filename, L=0.25):

        # Get vectors defining the body coordinate system.
        p0 = self.pos + self.xCG
        p1 = self.pos + self.xCG + L*self.iHat
        p2 = self.pos + self.xCG + L*self.jHat
        p3 = self.pos + self.xCG + L*self.kHat
        bodyCoords, bodyCoordPts = np.vstack([self.iHat, self.jHat, self.kHat]), np.vstack([p0, p1, p2, p3])

        # Mark thrusters.
        thrusterPts = []
        for i in range(self.thrusterPositions.shape[0]):
            xt = self.vehicleToGlobal(self.thrusterPositions[i, :] + self.xCG)
            tVec = rov.vehicleToGlobal(rov.A[:3, i]*L/2.)
            thrusterPts.append(xt)
            thrusterPts.append(xt+tVec)

        with open(filename, "w") as outfile:
            outfile.write("# vtk DataFile Version 3.0\n")
            outfile.write("vtk output\n")
            outfile.write("ASCII\n")
            outfile.write("DATASET POLYDATA\n")
            outfile.write("POINTS {:d} float\n".format(4 + len(thrusterPts)))
            for j in range(4):
                outfile.write("{:.5e} {:.5e} {:.5e}\n".format(
                    bodyCoordPts[j, 0], bodyCoordPts[j, 1], bodyCoordPts[j, 2]))
            for j in range(len(thrusterPts)):
                outfile.write("{:.5e} {:.5e} {:.5e}\n".format(
                    thrusterPts[j][0], thrusterPts[j][1], thrusterPts[j][2]))
            outfile.write("LINES {:d} {:d}\n".format(3+len(thrusterPts)//2, 3*(3+len(thrusterPts)//2)))
            outfile.write("2 0 1\n")
            outfile.write("2 0 2\n")
            outfile.write("2 0 3\n")
            for j in range(len(thrusterPts)//2):
                outfile.write("2 {:d} {:d}\n".format(4+j*2, 4+j*2+1))
            outfile.write("CELL_DATA {:d}\n".format(3+len(thrusterPts)//2))
            outfile.write("FIELD FieldData {:d}\n".format(1))
            outfile.write("\n")
            outfile.write("{} {:d} {:d} {}\n".format("iLine", 1, 3+len(thrusterPts)//2, "int"))
            outfile.write("1\n")
            outfile.write("2\n")
            outfile.write("3\n")
            for j in range(len(thrusterPts)//2):
                outfile.write("0\n")
            outfile.write("\n")

        return bodyCoords

    def makeCfdInputs(self):
        # Write the inverse of the thrust allocation matrix in Fortran.
        usercode = ""
        for i in range(len(self.thrusterNames)):
            usercode += 'thrusterNames({:d}) = "th_{:s}"\n'.format(i+1, self.thrusterNames[i])
        for i in range(len(self.thrusterNames)):
            usercode += "Ainv({:d},:) = (/".format(i+1) + ", ".join(["{:.6e}".format(v) for v in self.Ainv[i, :]]) + "/)\n"

        # Write ReFRESCO controls entries for all the thrusters
        controls = ""
        for i in range(len(self.thrusterNames)):
            if "vert" in self.thrusterNames[i]:
                v = [1., 0., 0.]
            else:
                v = [0., 0., 1.]
            controls += "\n".join((
            '<bodyForceModel name="th_{}">'.format(self.thrusterNames[i]),
            '    <PROPELLER>',
            '        <centreLocation>{:.6e} {:.6e} {:.6}</centreLocation>'.format(
                self.thrusterPositions[i, 0], self.thrusterPositions[i, 1], self.thrusterPositions[i, 2]),
            '        <propellerDiameter>0.077</propellerDiameter>',
            '        <hubDiameter>0.041</hubDiameter>',
            '        <axialVector>{:.6e} {:.6e} {:.6}</axialVector>'.format(
                self.A[0, i], self.A[1, i], self.A[2, i]),
            '        <upVector>{:.6e} {:.6e} {:.6}</upVector>'.format(v[0], v[1], v[2]),
            '        <referenceSystem>',
            '            <BODY_FIXED>',
            '                <bodyName>rov</bodyName>',
            '            </BODY_FIXED>',
            '        </referenceSystem>',
            '        <type>',
            '            <ACTUATOR_DISC>',
            '                <thickness>0.01</thickness>',
            '                <distributionType>',
            '                    <CUSTOM_PROPELLER>',
            '                        <ForceComponentAxial>',
            '                            <thrust>100.0</thrust>',
            '                            <Model>',
            '                                <ABMN>',
            '                                    <a>0.15</a>',
            '                                    <b>0.00</b>',
            '                                    <m>2.15</m>',
            '                                    <n>0.50</n>',
            '                                </ABMN>',
            '                            </Model>',
            '                        </ForceComponentAxial>',
            '                        <ForceComponentTangential>',
            '                            <torque>0.1</torque>',
            '                            <Model>',
            '                                <PDRATIO>',
            '                                    <value>1.0</value>',
            '                                </PDRATIO>',
            '                            </Model>',
            '                        </ForceComponentTangential>',
            '                    </CUSTOM_PROPELLER>',
            '                </distributionType>',
            '            </ACTUATOR_DISC>',
            '        </type>',
            '    </PROPELLER>',
            '</bodyForceModel>',
            '',
            ))

        return usercode, controls

rov = RovTemp()

usercode, controls = rov.makeCfdInputs()
# rov.saveCoordSystem("D:/_temp/rovCoords.vtk")

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

def onChanged(val):
    global rov, lns, texts, ax

    # Update the orientation of the vehicle.
    angles = np.array([sldr1.val, sldr2.val, sldr3.val])/180.*np.pi
    rov.updateMovingCoordSystem(angles)

    # Get the force and moment demands.
    Fg = np.array([sldr4.val, sldr5.val, sldr6.val])
    Mg = np.array([sldr10.val, sldr11.val, sldr12.val])
    # Resolve force in the vehicle axes.
    Fglobal = rov.globalToVehicle(Fg)
    Mglobal = rov.globalToVehicle(Mg)

    # Combine generalised control forces and moments into one vector.
    generalisedControlForces = np.append(Fglobal, Mglobal)
    print("===")
    print(np.append(Fg, Mg))
    print(generalisedControlForces)

    # Translate into force demands for each thruster using the inverse of the thrust
    # allocation matrix.
    cv = np.matmul(rov.Ainv, generalisedControlForces)

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
    Fh, Mh = np.zeros(3), np.zeros(3)
    for i in range(rov.thrusterPositions.shape[0]):
        xt = rov.vehicleToGlobal(rov.thrusterPositions[i, :])
        tVec = rov.vehicleToGlobal(rov.A[:3, i]*cv[i])
        Fh += rov.A[:3, i]*cv[i]
        Mh += rov.A[3:, i]*cv[i]

        lns += ax.plot(xt[0], xt[1], xt[2], "ks", ms=5)
        texts.append(ax.text(xt[0], xt[1], xt[2], "{:d}".format(i+1)))
        lns += ax.plot([xt[0], xt[0]+tVec[0]], [xt[1], xt[1]+tVec[1]], [xt[2], xt[2]+tVec[2]], "k-", alpha=0.5, lw=2)

    # Compute net actuation in the global reference frame.
    Fhglobal = rov.vehicleToGlobal(Fh)
    Mhglobal = rov.vehicleToGlobal(Mh)

    texts.append(fig.text(0.5, 0.975,
        "roll, pitch, yaw = " +", ".join(['{:.1f} deg'.format(v) for v in rov.computeRollPitchYaw()/np.pi*180.]),
        va="center", ha="center"
    ))
    texts.append(fig.text(0.5, 0.94,
        "Target forces and moments in vehicle reference frame = " +", ".join(['{:.2f}'.format(v) for v in generalisedControlForces]),
        va="center", ha="center"
    ))
    texts.append(fig.text(0.5, 0.905,
        "Total forces and moments in global reference frame = " +", ".join(['{:.2f}'.format(v) for v in np.append(Fhglobal, Mhglobal)]),
        va="center", ha="center"
    ))
    texts.append(fig.text(0.5, 0.87,
        "Total forces and moments in vehicle reference frame = " +", ".join(['{:.2f}'.format(v) for v in np.append(Fh, Mh)]),
        va="center", ha="center"
    ))

    return lns

sliders = [sldr1, sldr2, sldr3, sldr4, sldr5, sldr6, sldr10, sldr11, sldr12]
for sldr in sliders:
    sldr.on_changed(onChanged)

plt.show()
