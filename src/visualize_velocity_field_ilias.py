import numpy as np
import matplotlib.pyplot as plt

ux_kl = np.load('ux.npy')
uy_kl = np.load('uy.npy')

nx, ny = ux_kl.shape

plt.figure()
x_k = np.arange(nx)
y_l = np.arange(ny)
plt.streamplot(x_k, y_l, ux_kl.T, uy_kl.T)
plt.show()
