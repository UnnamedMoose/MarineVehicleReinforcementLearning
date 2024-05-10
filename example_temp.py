from scipy.interpolate import CubicSpline
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas
import scipy.integrate

from dynamicsModel_BlueROV2_Heavy_6DoF import BlueROV2Heavy6DoF
import resources

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

RHS = np.array([-1.366025e+01, 3.660254e+00, 5.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00])
acc = np.array([-8.224159e-01, 1.537282e-01, 4.385965e-01, 1.564733e-01, 8.371019e-01, 0.000000e+00])
M = np.array([
    [1.690000e+01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 2.850000e-01, -0.000000e+00],
    [0.000000e+00, 2.410000e+01, 0.000000e+00, -2.850000e-01, 0.000000e+00, 0.000000e+00],
    [0.000000e+00, 0.000000e+00, 1.140000e+01, 0.000000e+00, -0.000000e+00, 0.000000e+00],
    [0.000000e+00, -2.850000e-01, 0.000000e+00, 2.800000e-01, 0.000000e+00, 0.000000e+00],
    [2.850000e-01, 0.000000e+00, -0.000000e+00, 0.000000e+00, 2.800000e-01, 0.000000e+00],
    [-0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 2.800000e-01],
])
