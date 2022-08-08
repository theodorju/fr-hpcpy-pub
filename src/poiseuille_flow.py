"""Poiseuille flow simulation"""
import os
import numpy as np
import argparse
import lbm
import matplotlib.pyplot as plt

from tqdm import trange
from scipy.integrate import simpson
from typing import Optional
from utils import theoretical_poiseuille_flow, theoretical_viscosity


def poiseuille_flow(
        proba_density: np.array,
        omega: Optional[float] = 0.8,
        steps: Optional[int] = 10000,
        density_steady_state: Optional[float] = 1.0,
        density_increase_percentage: Optional[float] = 0.01,
        density_decrease_percentage: Optional[float] = 0.01
) -> None:
    """
    Simulate Poiseuille flow, i.e. fluid flow in a pipe between two rigid
        walls (upper and lower boundaries) driven by the pressure difference
        between inlet (P_in) and outlet (P_out).

    Args:
        proba_density (np.ndarray): Probability density function of Lattice.
                Boltzmann Equation.
        omega (Optional[float]): Collision frequency.
        steps (Optional[int]): Simulation steps.
        density_steady_state (Optional[float]): The density at each point on the grid at steady
            state. Current implementation assumes that all points of the grid have the
            same density at the start of the experiment. Default = 1.
        density_increase_percentage (Optional[float]): The percentage of increase in density
            value at the input. Default = 0.01.
        density_decrease_percentage (Optional[float]): The percentage of decrease in density
            value at the output. Default = 0.01.

    Returns: None.
    """
    # Get the dimensions
    _, x_shape, y_shape = proba_density.shape

    # Choose initial value for density ρ(0) = 1.0 at time t=0
    density = np.ones((x_shape, y_shape), dtype=float)
    # Calculate increase of density at the input
    density_input = density_steady_state + density_steady_state * density_increase_percentage
    # Calculate decrease of density at the output
    density_output = density_steady_state - density_steady_state * density_decrease_percentage
    # Choose initial value for velocity u(0) = 0.0 at time t=0
    velocity = np.zeros((2, x_shape, y_shape), dtype=float)
    # Initialize probability density based on the equilibrium
    proba_density = lbm.calculate_equilibrium_distro(density, velocity)

    # Initialize dictionary to hold x-velocity values at different timesteps
    # start by keeping the initial velocity
    vx_dict = {0: np.moveaxis(velocity[0], 0, 1)[:, x_shape // 2]}

    for step in trange(steps + 1, desc="Poiseuille flow"):

        # Calculate density
        density = lbm.calculate_density(proba_density)
        # Calculate velocity
        velocity = lbm.calculate_velocity(proba_density)
        # Perform collision/relaxation
        lbm.collision_relaxation(proba_density, velocity, density, omega=omega)
        # Keep the probability density function pre-streaming
        pre_streaming_proba_density = proba_density.copy()
        # Apply periodic boundary conditions with pressure gradient
        lbm.pressure_gradient(proba_density,
                              density,
                              velocity,
                              density_input,
                              density_output,
                              flow="left_to_right")
        # Streaming
        lbm.streaming(proba_density)
        # Apply boundary conditions on the bottom wall
        lbm.rigid_wall(proba_density, pre_streaming_proba_density, "lower")
        # Apply boundary conditions on the top wall
        lbm.rigid_wall(proba_density, pre_streaming_proba_density, "upper")

        # Keep velocity for plotting
        if step % 1000 == 0:
            # Keep the velocity in a slice on the axis that is perpendicular to the moving
            # boundary, the shape of vx_dict[<step>] is (lattice_size,)
            vx_dict[step] = np.moveaxis(velocity[0], 0, 1)[:, x_shape // 2]

    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))

    # Figure title
    fig.suptitle("Poiseuille Flow along the vertical axis", fontsize=16)

    # Create plot of the evolution of velocity profile from start to
    # end of experiment
    for step in vx_dict.keys():
        _ = ax.plot(vx_dict[step], label="Step {}".format(step))

    _ = ax.set_title("Evolution of the velocity profile from start\n of experiment"
                     " to the steady-state")
    _ = ax.set_ylabel("Velocity (Solid boundary axis)")
    _ = ax.set_xlabel("Coordinate along the PBC with pressure gradient axis")
    _ = ax.legend()

    # Save plot
    print("Saving graph under: /data")
    path_exists = os.path.exists("data")
    if not path_exists:
        # Create path if it does not exist
        os.makedirs("data")
    plt.savefig('data/poiseuille_flow')

    # Display plot
    plt.show()

    # Reinitialize plots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
    # Plot velocity
    lbm.plot_velocity_field(velocity, fig, ax,
                            title="Poiseuille ")

    # Save plot
    print("Saving graph under: /data")
    path_exists = os.path.exists("data")
    if not path_exists:
        # Create path if it does not exist
        os.makedirs("data")
    plt.savefig('data/poiseuille_velocity_flow')

    # Display plot
    plt.show()

    # Calculate theoretical velocity field
    theoretical_velocity_profile = \
        theoretical_poiseuille_flow(density,
                                    omega,
                                    density_input,
                                    density_output
                                    )
    # Reinitialize plots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
    _ = ax.set_title("Comparison of theoretical and experimental velocity"
                     "profiles on Poiseuille flow")
    _ = ax.set_ylabel("Velocity (Solid boundary axis)")
    _ = ax.set_xlabel("Coordinate along PBC with pressure gradient axis")
    # Add theoretical velocity profile
    _ = ax.plot(theoretical_velocity_profile, label="Theoretical result")
    # Add experimental velocity profile from last simulation step
    _ = ax.plot(vx_dict[steps], label="Experimental result")
    _ = ax.legend()

    # Save plot
    print("Saving graph under: /data")
    path_exists = os.path.exists("data")
    if not path_exists:
        # Create path if it does not exist
        os.makedirs("data")
    plt.savefig('data/poiseuille_theoretical_experimental')

    # Display plot
    plt.show()

    # Steady state velocity in the middle of the pipe
    v_middle = vx_dict[steps]
    # Steady state velocity in the inlet
    v_inlet = np.moveaxis(velocity[0], 0, 1)[:, 0]
    # Steady state velocity in the outlet
    v_outlet = np.moveaxis(velocity[0], 0, 1)[:, -1]

    # Reinitialize plots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
    _ = ax.set_title("Velocities in the inlet, middle and outlet of the pipe")
    _ = ax.set_ylabel("Velocity (Solid boundary axis)")
    _ = ax.set_xlabel("Coordinate along PBC with pressure gradient axis")
    _ = ax.plot(v_inlet, label="inlet")
    _ = ax.plot(v_outlet, label="outlet", marker="*")
    _ = ax.plot(v_middle, label="middle")
    _ = ax.legend()

    # Save plot
    print("Saving graph under: /data")
    path_exists = os.path.exists("data")
    if not path_exists:
        # Create path if it does not exist
        os.makedirs("data")
    plt.savefig('data/poiseuille_velocities_comparison')

    # Display plot
    plt.show()

    print("======================================================")
    print("Area under curve in the inlet: {}".format(simpson(v_inlet)))
    print("Area under curve in the middle: {}".format(simpson(v_middle)))
    print("Area under curve in the outlet: {}".format(simpson(v_outlet)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Poiseuille Flow",
        description="Run Poiseuille Flow experiment."
    )

    parser.add_argument(
        "-o",
        "--omega",
        help="The collision frequency. Default value is 0.8.",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "-dss",
        "--density_steady_state",
        help="The density at each point on the grid at steady"
             "state. Current implementation assumes that all points of the grid have the"
             "same density at the start of the experiment. Default = 1.",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "-din",
        "--density_input",
        help="The percentage of increase in density value at the input. "
             "Default value is 1%% increase of the steady state density.",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "-dout",
        "--density_output",
        help="The percentage of increase in density value at the input. "
             "Default value is 1%% decrease of the steady state density.",
        type=float,
        default=0.01
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

    # Grid dimensions
    dim_0, dim_1 = args.grid_size
    # Initialize the grid
    p = np.zeros((9, dim_0, dim_1))

    viscosity = theoretical_viscosity(args.omega)

    print("Running Poiseuille flow simulation with the following setup:"
          f"\n\tGrid size: \t\t\t{dim_0} x {dim_1}"
          f"\n\tCollision Frequency: \t\t{args.omega}"
          f"\n\tViscosity: \t\t\t{viscosity:.4f}"
          f"\n\tSteady State Density: \t\t{args.density_steady_state}"
          f"\n\tInput Density increased by\t{args.density_input}%"
          f"\n\tOutput Density decreased by \t{args.density_output}%"
          f"\n\tSimulation steps: \t\t{args.steps}"
          )

    # Run simulation
    poiseuille_flow(p, omega=args.omega, steps=args.steps, density_increase_percentage=args.density_input,
                    density_decrease_percentage=args.density_output)

    print("Simulation completed.")
