# Underwater Vehicle Reinforcement Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7981976.svg)](https://doi.org/10.5281/zenodo.7981976)

## Overview

Simple example of how to manoeuvre an underwater vehicle using reinforcement learning.

The problem is posed by placing an underwater vehicle at a random position and
heading inside the domain. It then needs to make its way to the origin and adopt
a specific heading. There is a background turbulent flow along the x-axis with
creating time-varying disturbances moving the vehicle away from the goal point.

The manoeuvring model is deliberately simplified
in order to keep the simulations fast. Specifically, added mass and inertia have
been ignored, rigid body accelerations are not taken into account, and cross-coupling
hydrodynamic coefficients are set to zero. Only three degrees of freedom (surge,
sway, yaw) are considered and time integration is performed using the Euler method.
All of this is done in order to make the solution procedure as fast as possible
and allow rapid training of RL agents rather than prioritising accuracy of the
physics model.

In addition to an RL agent, a simple proportional-derivative (PD)
controller is used as a benchmark.

![Alt text](Figures/episodeAnim_RL_control.gif?raw=true "Example episode.")

The environment can be run in parallel using the `SubprocVecEnv` structure provided
by [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3). The figure
below shows the speedup seen on my machine with 8 physical cores.

![Alt text](Figures/scalingTest.png?raw=true "Parallel environment speed-up vs no. cores.")

## How to use

The code is divided into three parts:

- Numerical core:

    * The core of the implementation resides in `verySimpleAuv.py`, this is where the environment is implemented.

    * Turbulent flow is being generated using pre-computed spectral POD from [pySPOD](https://github.com/MathEXLab/PySPOD), which is implemented in `flowGenerator.py`. Data has been generated using [ReFRESCO](https://www.marin.nl/en/facilities-and-tools/software/refresco) CFD code building on the synthetic turbulence generation technique described in [Lidtke et al.](https://doi.org/10.3390/jmse9111274).

    * Helper functions for plotting, training and evaluating RL agents are provided in `resources.py`.

- Implementations of complete training and evaluation pipelines are given in
scripts with names starting with `main_`. Specifically:

    * `main_00_SAC_stable_baselines.py` - simplest possible soft-actor critic (SAC) approach using brute-force training.

    * `main_01_SAC_sbl_customInit.py` - SAC supported by pretraining using the [imitation](https://github.com/HumanCompatibleAI/imitation) library. **UNDER CONSTRUCTION**

- Loose scripts used for testing and visualising data

    * `script_0_checkScaling.py` - used to check parallel scaling of the environment.

    * `script_1_compareTraining.py` - used to compare training histories of different models trained independently.

    * `script_2_testImitation.py` - minimum working example for the imitation library.

## How to cite?

This code can be cited with

```
@software{Lidtke_2023_7981976,
  author       = {Lidtke, Artur K.},
  title        = {Underwater Vehicle Reinforcement Learning},
  month        = May,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v1.0},
  doi          = {10.5281/zenodo.7981976},
  url          = {https://doi.org/10.5281/zenodo.7981976}
}
```
