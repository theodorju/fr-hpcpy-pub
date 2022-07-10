"""Poiseuille flow simulation"""
import numpy as np
import argparse
import lbm

from tqdm import trange
from typing import Optional


def poiseuille_flow(
        proba_density: np.array,
        omega: Optional[float] = 0.8,
        steps: Optional[int] = 10000,
        steady_state_density: Optional[float] = 1.0,
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
        steady_state_density (Optional[float]): The density at each point on the grid at steady
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
    density_input = steady_state_density + steady_state_density * density_increase_percentage
    # Calculate decrease of density at the output
    density_output = steady_state_density - steady_state_density * density_decrease_percentage
    # Choose initial value for velocity u(0) = 0.0 at time t=0
    velocity = np.zeros((2, x_shape, y_shape), dtype=float)
    # Initialize probability density based on the equilibrium
    proba_density = lbm.calculate_equilibrium_distro(density, velocity)

    # Initialize dictionary to hold x-velocity values at different timesteps
    # start by keeping the initial velocity
    vx_dict = {0: np.moveaxis(velocity[0], 0, 1)[:, x_shape // 2]}

    for step in trange(steps, desc="Poiseuille flow"):

        # Calculate density
        density = lbm.calculate_density(proba_density)
        # Calculate velocity
        velocity = lbm.calculate_velocity(proba_density)
        # Keep the probability density function pre-streaming
        pre_streaming_proba_density = proba_density.copy()
        # Apply periodic boundary conditions with pressure gradient
        lbm.pressure_gradient(proba_density,
                              pre_streaming_proba_density,
                              density,
                              velocity,
                              density_input,
                              density_output)
        # Perform collision/relaxation
        proba_density = \
            lbm.collision_relaxation(proba_density, velocity, density, omega=omega)
        # Keep the probability density function before the streaming and after collision/relaxation
        pre_streaming_proba_density = proba_density.copy()
        # Streaming
        lbm.streaming(proba_density)
        # Apply boundary conditions on the bottom wall
        lbm.rigid_wall(proba_density, pre_streaming_proba_density, "lower")
        # Apply boundary conditions on the top wall
        lbm.rigid_wall(proba_density, pre_streaming_proba_density, "upper")


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
             "Default value is 1% increase of the steady state density.",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "-dout",
        "--density_output",
        help="The percentage of increase in density value at the input. "
             "Default value is 1% decrease of the steady state density.",
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

    print("Running Poiseuille flow simulation for omega {}:".format(args.omega))

    # Grid dimensions
    dim_0, dim_1 = args.grid_size
    # Initialize the grid
    p = np.zeros((9, dim_0, dim_1))
    print("Grid size: {} x {}".format(dim_0, dim_1))

    # Run simulation
    poiseuille_flow(p,
                    omega=args.omega,
                    steps=args.steps,
                    steady_state_density=density_steady_state,
                    density_increase_percentage=args.density_input,
                    density_decrease_percentage=args.density_output
                    )

    print("Simulation completed.")
