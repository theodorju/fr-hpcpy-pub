"""Global variables"""
import numpy as np

# Hardcoded velocity channels, shape (9x2)
velocity_channels = \
    np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
              [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T

# Small positive constant to avoid division by zero
epsilon = 1e-16

# Equilibrium occupation numbers
weights = np.array([4. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 9,
                    1. / 36, 1. / 36, 1. / 36, 1. / 36])

# Shear wave decay constants
SW_EPSILON = 0.05

# D2Q9 channels and anti-channels
channels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
anti_channels = [0, 3, 4, 1, 2, 7, 8, 5, 6]

# Useful channels for boundary conditions. Ordering matters.
up_in_channels = (4, 7, 8)
up_out_channels = (2, 5, 6)
down_in_channels = (2, 5, 6)
down_out_channels = (4, 7, 8)
inlet_in_channels = (1, 5, 8)
inlet_out_channels = (3, 6, 7)
outlet_in_channels = (3, 6, 7)
outlet_out_channels = (1, 5, 8)

# Speed of sound squared
c_s_squared = 1 / 3.
