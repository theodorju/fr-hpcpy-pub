"""Script taken from ILIAS. Added minor modifications to also print colorbar"""
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))

# Figure title
fig.suptitle("Sliding lid experiment - Parallel Execution", fontsize=16)

ux_kl = np.load('ux.npy')
uy_kl = np.load('uy.npy')

nx, ny = ux_kl.shape

column_labels = list(range(nx))
row_labels = list(range(ny))

# Set up columns and labels
xticks = column_labels[::10] + [column_labels[-1]]
yticks = row_labels[::5] + [row_labels[-1]]

_ = ax.set_xticks(xticks, minor=False)
_ = ax.set_yticks(yticks, minor=False)
_ = ax.tick_params(axis="x", labelrotation=60)
max_velocity = np.max(ux_kl)
min_velocity = np.min(uy_kl)

# Create colorbar ticks
cbar_ticks = [np.around(i, 2) for i in np.linspace(max_velocity, min_velocity)]

x_k = np.arange(nx)
y_l = np.arange(ny)
streamplot = ax.streamplot(x_k, y_l, ux_kl.T, uy_kl.T, color=ux_kl.T)

# Add colorbar
fig.colorbar(streamplot.lines, ax=ax, ticks=cbar_ticks)
_ = ax.set_xticklabels(xticks, minor=False)
_ = ax.set_yticklabels(yticks, minor=False)

# Add labels and title
_ = ax.set_ylabel("y-coordinate")
_ = ax.set_xlabel("x-coordinate")
_ = ax.set_title("Velocity Field")

plt.show()
