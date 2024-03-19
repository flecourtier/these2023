import numpy as np
import matplotlib.pyplot as plt

x0,y0 = (0,0)
r = 1

def disk_to_potato(x):
    x1, x2 = x.T
    x = x1 - 0.5 * x2**2 + 0.3 * np.sin(x2)
    y = x2 + 0.1 * x1 + 0.12 * np.cos(x1)
    print(x.shape, y.shape)
    print(np.array([x, y]).shape)
    return np.array([x, y]).T

def sampling_in_disk(n):
    r = np.sqrt(np.random.rand(n))
    theta = 2 * np.pi * np.random.rand(n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y]).T

N = 5000
xy = sampling_in_disk(N)
xy_p = disk_to_potato(xy)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(xy[:, 0], xy[:, 1], s=1)
ax[0].set_title('Disk')
ax[1].scatter(xy_p[:, 0], xy_p[:, 1], s=1)
ax[1].set_title('Potato')

plt.show()