"""Couette flow simulation"""
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import lbm

from tqdm import trange
from typing import Optional
from globals import *


def couette_flow(
        proba_density: np.array,
        wall_velocity: np.array,
        omega: Optional[float] = 0.8,
        steps: Optional[int] = 10000,
        annotate: Optional[bool] = False,
        ) -> None:
    """
    Simulate Couette flow, i.e. fluid flow between a moving wall at the top
        and a rigid wall at the bottom.
    Args:
            proba_density (np.ndarray): Probability density function of Lattice.
                Boltzmann Equation.
            wall_velocity (np.ndarray): The velocity vector of the moving boundary.
                Currently, a 2D-ndarray containing one value for the x-component
                and one value for the y-component of the velocity.
            steps (Optional[int]): Simulation steps.
            omega (Optional[float]): Collision frequency.
            annotate (Optional[bool]): Indicates whether the velocity profile plot will include
                rigid lines for the moving boundaries and an arrow indicating
                the direction of the moving boundary.
    Returns:
        None.
    """
    # Get the dimensions
    _, x_shape, y_shape = proba_density.shape

    # Choose initial value for density: ρ(0) = 1.0 at time t=0
    density = np.ones((x_shape, y_shape), dtype=float)
    # Choose initial value for velocity u(0) = 0.0 at time t=0
    velocity = np.zeros((2, x_shape, y_shape), dtype=float)
    # Initialize probability density based on equilibrium
    proba_density = lbm.calculate_equilibrium_distro(density, velocity)

    # Initialize dictionary to hold x-velocity values at different timesteps
    # and keep the initial velocity
    vx_dict = {0: np.moveaxis(velocity[0], 0, 1)[:, x_shape // 2]}

    # For each iteration step
    for step in trange(steps, desc="Couette flow"):

        # Calculate density
        density = lbm.calculate_density(proba_density)
        # Calculate velocity
        velocity = lbm.calculate_velocity(proba_density)
        # Perform collision/relaxation
        lbm.collision_relaxation(proba_density, velocity, density, omega=omega)
        # Keep the probability density function pre-streaming
        pre_stream_proba_density = proba_density.copy()
        # Streaming
        lbm.streaming(proba_density)
        # Apply boundary conditions on the bottom rigid wall
        lbm.rigid_wall(proba_density, pre_stream_proba_density, "lower")
        # Apply boundary condition on the top moving wall
        lbm.moving_wall(proba_density, pre_stream_proba_density,
                        wall_velocity, density, "upper")

        if step % 500 == 0:
            # Keep the velocity in a slice on the axis that is perpendicular to the moving
            # boundary, the shape of vx_dict[<step>] is (lattice_size,)
            vx_dict[step] = np.moveaxis(velocity[0], 0, 1)[:, x_shape // 2]

    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 10))

    # Figure title
    fig.suptitle("Couette Flow with moving boundary on the vertical axis",
                 fontsize=16)

    # Create plot of the evolution of velocity profile from start to
    # end of experiment
    for step in vx_dict.keys():
        _ = ax[1].plot(vx_dict[step], label="Step {}".format(step))

    _ = ax[1].set_title("Evolution of the velocity profile from start\n of experiment"
                        " to the steady-state")
    _ = ax[1].set_ylabel("PBC axis")
    _ = ax[1].set_xlabel("Solid boundary axis")
    _ = ax[1].legend()

    # Velocity field streamplot
    lbm.plot_velocity_field(velocity, fig, ax[0])

    # Extra annotation might overcrowd the plot
    # adding a command line argument to turn them on/off
    if annotate:
        # Draw the moving boundary on top, add +0.5 for viewing
        _ = ax[0].axhline(y_shape - 0.5,
                          color="red",
                          linewidth=3)

        # Draw the rigid boundary on the bottom, add -0.5 for viewing
        _ = ax[0].axhline(0 - 0.5,
                          color="black",
                          linewidth=3)
        _ = ax[0].arrow(40, 103, 20, 0,
                        width=0.5,
                        color="red",
                        edgecolor=None,
                        label="Boundary Velocity")

    plt.legend()

    # Save plot
    print("Saving graph under: /data")
    path_exists = os.path.exists("data")
    if not path_exists:
        # Create path if it does not exist
        os.makedirs("data")
    plt.savefig('data/couette_flow')

    # Display plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Coutte Flow",
        description="Run Coutte Flow experiment."
    )

    parser.add_argument(
        "-a",
        "--annotate",
        help="If provided the generated plot for the velocity profile"
             "will include the boundaries and an arrow indicating the"
             "direction of the velocity of the moving boundary. "
             "Default value is False.",
        default=False,
        action="store_true"
    )

    parser.add_argument(
        '-v',
        "--velocity",
        help="The velocity of the moving boundary. "
             "Currently only top moving boundary is supported.",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "-o",
        "--omega",
        help="The collision frequency. Default value is 0.8.",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "-s",
        "--steps",
        help="The simulation steps. Default value is 10000.",
        type=int,
        default=10000,
    )

    parser.add_argument('-g',
                        '--grid_size',
                        nargs='+',
                        help='Space separated list of grid size (dim_0, dim_1). For '
                             'example: -g 50 50',
                        type=int,
                        default=(50, 50))

    args = parser.parse_args()

    print("Running Couette flow simulation for omega {}:".format(args.omega))

    # Grid dimensions
    dim_0, dim_1 = args.grid_size
    # Initialize the grid
    p = np.zeros((9, dim_0, dim_1))
    print("Grid size: {} x {}".format(dim_0, dim_1))

    # Define wall velocity
    w = np.array([args.velocity, 0.])
    # Run simulation
    couette_flow(p, w,
                 omega=args.omega,
                 steps=args.steps,
                 annotate=args.annotate
                 )
    print("Simulation completed.")
