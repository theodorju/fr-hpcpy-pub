import numpy as np

from src.utils import InitMode
from src.lbm import streaming, init_proba_density


def test_mass_conservation():
    """
    Tests that the mass is conserved after on streaming step.
    """
    # Randomly initialize the grid
    proba = init_proba_density(mode=InitMode.RAND, seed=True)

    # Keep probability distribution function before streaming
    proba_before = proba.copy()

    # Calculate the total mass of the system before streaming
    mass_before = proba.sum()

    # Perform one streaming step
    streaming(proba)

    # Calculate the total mass of the system after streaming
    mass_after = proba.sum()

    # Assert correct behavior
    np.testing.assert_equal(mass_before, mass_after)
    np.testing.assert_equal(np.any(np.not_equal(proba_before, proba)), True)
