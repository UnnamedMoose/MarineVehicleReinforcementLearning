# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 22:24:51 2022

@author: ALidtke
"""

import numpy as np
import os
import yaml
import matplotlib.pyplot as plt

class ReconstructedFlow(object):
    def __init__(self, dataDir):
        # Read the coefficients computed with pySPOD
        self.coeffs = np.load(os.path.join(dataDir, 'coeffs.npy'))
        self.modes = np.load(os.path.join(dataDir, 'modes_r.npy'))
        self.lt_mean = np.load(os.path.join(dataDir, 'ltm.npy'))

        # Reconstruct flow data from modes and amplitudes.
        self.flowData = np.zeros((self.coeffs.shape[1], self.modes.shape[0], self.modes.shape[1], self.modes.shape[2]))
        for iTime in range(self.flowData.shape[0]):
            self.flowData[iTime, :, :, :] = np.real(np.matmul(self.modes, self.coeffs[:, iTime])) + self.lt_mean

        # Compute turbulence intensity on the plane.
        self.TI = np.sqrt(0.5*(
            np.sqrt(np.sum((self.flowData[:, :, :, 0] - 1.)**2., axis=0)/self.flowData.shape[0])
            + np.sqrt(np.sum((self.flowData[:, :, :, 1] - 0.)**2., axis=0)/self.flowData.shape[0])))

        # Read the time step size and coordinates of input data.
        with open(os.path.join(dataDir, "params_coeffs.yaml"), "r") as infile:
            params = yaml.safe_load(infile)
        self.baseDt = params["time_step"]
        self.baseTime = np.array([i*params["time_step"] for i in range(self.flowData.shape[0])])
        self.baseCoords = np.load(os.path.join(dataDir, "turbulence_coords.npy"))
        self.baseTI = self.TI[self.TI.shape[0]//2, self.TI.shape[1]//2]

        # Check source grid spacing. It should be uniform in both directions.
        # Note that the flow data is stored in (y, x) orientation, following
        #   the convention used by matplotlib.
        self.dx = self.baseCoords[0, 1:, 0] - self.baseCoords[0, :-1, 0]
        self.dy = self.baseCoords[1:, 0, 1] - self.baseCoords[:-1, 0, 1]
        if not np.all(np.abs(self.dx - self.dx[0]) < 1e-6):
            raise ValueError("Non-uniform input grid spacing in the x-direction")
        if not np.all(np.abs(self.dy - self.dy[0]) < 1e-6):
            raise ValueError("Non-uniform input grid spacing in the y-direction")
        self.dx = self.dx[0]
        self.dy = self.dy[0]

    def interp(self, time, xy):
        """
        Fast linear interpolation on a Cartesian grid uniform along x, y and t but
        not necessarily with uniform grid spacings along the two spatial dimensions.
        Much faster than any scipy alternatives.

        Parameters
        ----------
        time : float
            Time value to interpolate at.
        xy : float list or tuple of shape 2
            Coordinates at which to interpolate.

        Returns
        -------
        Interpolated flow values as a np.array of shape equal to last dimension
        of self.flowData array.

        """

        # Interpolation coordinate as a function of grid spacing and time step.
        tt = time / self.baseDt
        xx = xy[0] / self.dx
        yy = xy[1] / self.dy
        # Index of interpolation point below the interpolation coordinate.
        kk = min(self.baseTime.shape[0]-2, max(0, int(np.floor(tt))))
        ii = min(self.baseCoords.shape[1]-2, max(0, int(np.floor(xx))))
        jj = min(self.baseCoords.shape[0]-2, max(0, int(np.floor(yy))))
        # Weights along the x and y directions.
        tt = np.array([1. - (tt-kk), tt-kk])
        xx = np.array([1. - (xx-ii), xx-ii])
        yy = np.array([1. - (yy-jj), yy-jj])

        # Sum the weighted data.
        res = np.zeros(self.flowData.shape[3])
        for k in range(res.shape[0]):
            res[k] = np.matmul(yy.T, np.matmul(self.flowData[kk, jj:jj+2, ii:ii+2, k], xx))*tt[0] \
                + np.matmul(yy.T, np.matmul(self.flowData[kk+1, jj:jj+2, ii:ii+2, k], xx))*tt[1]

        return res


# %% Check
if __name__ == "__main__":
    dataDir = "./turbulenceData"
    flow = ReconstructedFlow(dataDir)

    # Plot turbulence intensity levels.
    fig, ax = plt.subplots()
    cs = plt.contourf(flow.baseCoords[:, :, 0], flow.baseCoords[:, :, 1], flow.TI)
    plt.colorbar(cs)
    ax.set_aspect("equal")

    # Test interpolation in time
    plt.subplots()
    plt.plot(flow.baseTime, flow.flowData[:, 21, 21, 0], "b-", lw=3)
    ti = np.linspace(0, flow.baseTime[-1], flow.baseTime.shape[0]*4)
    u = [flow.interp(t, [0.105, 0.105])[0] for t in ti]
    plt.plot(ti, u, "r--")

    # Test interpolation along x.
    plt.subplots()
    u = flow.flowData[155, 13, :, 0]
    plt.plot(flow.baseCoords[13, :, 0], u, "b-", lw=3)
    u = [flow.interp(flow.baseTime[155], [flow.baseCoords[13, i, 0], 0.065])[0] for i in range(flow.baseCoords.shape[1])]
    x = np.array([flow.baseCoords[13, i, :] for i in range(flow.baseCoords.shape[1])])
    plt.plot(flow.baseCoords[13, :, 0], u, "r--")

    # Test interpolation along x and y
    testData = []
    iData = []
    for i in range(np.min(flow.baseCoords.shape[:2])):
        testData.append([flow.baseCoords[i, i, 0], flow.baseCoords[i, i, 1], flow.flowData[1505, i, i, 1]])
        iData.append(flow.interp(flow.baseTime[1505], testData[-1][:2])[1])
    plt.subplots()
    plt.plot(np.array(testData)[:,2], "b-", lw=3)
    plt.plot(iData, "r--")

    # Plot a snapshot of the flow
    fig, ax = plt.subplots()
    plt.contour(flow.baseCoords[:, :, 0], flow.baseCoords[:, :, 1], flow.flowData[0, :, :, 0],
                levels=np.linspace(0.75, 1.25, 15), cmap=plt.cm.jet)
    ax.set_aspect("equal")
    ax.plot(x[:, 0], x[:, 1], "k.")
    ax.plot(x[:, 1], x[:, 0], "r.")
