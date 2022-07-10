"""Utilities and side functions for main calculations"""
import numpy as np

from enum import IntEnum


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
