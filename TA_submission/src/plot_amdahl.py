"""
Generate benchmarking plots.
All timings are from this google sheet:
    https://docs.google.com/spreadsheets/d/1kO4LfrUYbkT5su9cKMI7Ee0WcFy7c6IN3ehic8QqP48/edit#gid=0
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

# Matplotlib properties
font = {'size': 15}

matplotlib.rc('font', **font)

mpi_size_300 = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
mlups_300 = \
    np.array([10491589.24, 18433557.27, 25951557.09, 30977868.03, 35370406.76,
              49494060.71, 42003080.23, 34782608.7, 34159486.85, 35330140.54])

mpi_size_1000 = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
mlups_1000 = np.array([7435978.088, 11712015.59, 15826741.5, 30353527.54, 75570938.44,
                       188058298.1, 267572846.7, 326179137.6, 358345875.4, 292800046.8])

mpi_size_3000 = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
mlups_3000 = np.array([5532921.467, 9628542.595, 13442661, 23557761.14, 44573454.33, 93483474.72,
                       214218388.5, 451587329.5, 1046402121, 1258459645])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
_ = ax.set_ylabel("MLUPS")
_ = ax.set_xlabel("Number of processors used by MPI")
_ = ax.set_yscale("log")
_ = ax.set_xscale("log")

_ = ax.plot(mpi_size_300, mlups_300, marker='o', label="Grid size: 300 x 300")
_ = ax.plot(mpi_size_1000, mlups_1000, marker='o', label="Grid size: 1000 x 1000")
_ = ax.plot(mpi_size_3000, mlups_3000, marker='o', label="Grid size: 3000 x 3000")

_ = ax.legend(prop={'size': 20})
_ = ax.grid(True, which="both", ls="-")

# Save plot
print("Saving graph under: /data")
path_exists = os.path.exists("data")
if not path_exists:
    # Create path if it does not exist
    os.makedirs("data")
plt.savefig('data/amdahl_law')

# Display plot
plt.show()
