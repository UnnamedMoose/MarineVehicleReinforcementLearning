# UnderwaterVehicleReinforcementLearning

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
