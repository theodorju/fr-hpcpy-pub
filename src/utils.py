"""Utilities and side functions for main calculations"""
import numpy as np

from enum import IntEnum
from globals import c_s_squared

class InitMode(IntEnum):
    DEV = 1
    RAND = 2
    EQUILIBRIUM = 3


def theoretical_viscosity(omega: float = 1.) -> float:
    """
    Analytical calculation of viscosity.
    Args:
        omega: collision frequency (inverse of relaxation time).
    Returns:
        Fluid viscosity.
    """
    return (1/omega - 0.5) / 3


def theoretical_decay_calcs(initial_amp: float, omega: float, length: float,
                            time: list) -> np.array:
    """
    Calculate the theoretical decay.
        initial_amp: Sinusoidal wave amplitude in the initial conditions
        omega: omega: collision frequency (inverse of relaxation time).
        length: lattice length on the y-coordinates
        time:
    Returns:
        Theoretically calculated decay.

    """
    # Calculate viscosity
    viscosity = theoretical_viscosity(omega)

    # Theoretical calculation of the exponential decay
    return initial_amp * np.exp(-viscosity * (2 * np.pi / length) ** 2 *
                                np.array(time))


def theoretical_poiseuille_flow(
        density: np.array,
        omega: float,
        density_input: float,
        density_output: float
) -> np.array:
    """
    Calculate the analytical solution of the velocity field for a
        Hagen-Poiseuille flow in a pipe.

    Args:
        density (np.ndarray): Mass density at each position of the grid.
        omega (float): Collision frequency
        density_input (float): Density value at the input.
        density_output (float): Density value at the output.

    Returns:
        The velocity field as a np.ndarray.
    """
    # Get dimensions
    X, Y = density.shape

    # Create points to be returned
    points = np.linspace(0, Y, Y)

    # Calculate the mean value of density
    mean_density = np.mean(density)
    # Calculate viscosity
    viscosity = theoretical_viscosity(omega)
    # Calculate dynamic viscosity
    dynamic_viscosity = mean_density * viscosity
    # Calculate derivative nominator as the difference in derivatives
    derivative_nom = density_output - density_input

    # Calculate velocity field
    return -(derivative_nom * points * (Y - points)) / (2 * X * dynamic_viscosity) * c_s_squared
