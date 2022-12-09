# UnderwaterVehicleReinforcementLearning
Simple example of how to manoeuvre an underwater vehicle using RL.

To improve performance, try adjusting the variables placed in the state vector,
tweaking the reward function and tuning the hyperparameters. All of these have
been marked with TODO's.

The problem is posed by placing an underwater vehicle at a random position and
heading inside the domain. It then needs to make its way to the origin and adopt
a specific heading. In addition to an RL agent, a simple proportional-derivative (PD)
controller is used as a benchmark. The manoeuvring model is deliberately simplified
in order to keep the simulations fast. Specifically, added mass and inertia have
been ignored, rigid body accelerations are not taken into account, and cross-coupling
hydrodynamic coefficients are set to zero. Only three degrees of freedom (surge,
sway, yaw) are considered and time integration is performed using the Euler method.

![Alt text](exampleEpisode.png?raw=true "Example trajectory of the vehicle.")
