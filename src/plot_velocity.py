"""
Hardcoded plot of parallel execution with:
- 4 processes
- grid size 300 x 300
- collision frequency = 1.7
"""
import numpy as np
import matplotlib.pyplot as plt

velocity_0 = np.load("data/parallel_velocity/velocity_0.npy")
velocity_1 = np.load("data/parallel_velocity/velocity_1.npy")
velocity_2 = np.load("data/parallel_velocity/velocity_2.npy")
velocity_3 = np.load("data/parallel_velocity/velocity_3.npy")

velocity_x0 = np.moveaxis(velocity_0[0], 0, 1)
velocity_x1 = np.moveaxis(velocity_1[0], 0, 1)
velocity_x2 = np.moveaxis(velocity_2[0], 0, 1)
velocity_x3 = np.moveaxis(velocity_3[0], 0, 1)

velocity_y0 = np.moveaxis(velocity_0[1], 0, 1)
velocity_y1 = np.moveaxis(velocity_1[1], 0, 1)
velocity_y2 = np.moveaxis(velocity_2[1], 0, 1)
velocity_y3 = np.moveaxis(velocity_3[1], 0, 1)

velocity_x = np.zeros((300, 300), dtype=float)
velocity_y = np.zeros((300, 300), dtype=float)

velocity_x[:150, :150] = velocity_x0
velocity_x[150:, :150] = velocity_x1
velocity_x[:150, 150:] = velocity_x2
velocity_x[150:, 150:] = velocity_x3

velocity_y[:150, :150] = velocity_y0
velocity_y[150:, :150] = velocity_y1
velocity_y[:150, 150:] = velocity_y2
velocity_y[150:, 150:] = velocity_y3

x, y = np.meshgrid(range(300), range(300))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
streamplot = ax.streamplot(x, y, velocity_x, velocity_y, color=velocity_x, density=3)
plt.legend()
plt.show()
