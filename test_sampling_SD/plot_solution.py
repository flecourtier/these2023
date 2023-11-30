import numpy as np
import matplotlib.pyplot as plt

from modules.Problem import *

circle = Circle_Solution2()

x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
X,Y = np.meshgrid(x,y)

mask = circle.Omega_bool(X,Y)
mask = np.logical_not(mask)

####
# plot !
####

fig, axs = plt.subplots(2,2,figsize=(10,10))

# plot f
Z = circle.u_ex_prime2(np,[X,Y],None)[0]

ax = axs[0,0]
c = ax.contourf(X,Y,Z,30)
circle_patch = plt.Circle((0.5, 0.5), np.sqrt(2)/4, color='r', fill = False)
ax.add_patch(circle_patch)
plt.colorbar(c)
ax.set_title("f")

# plot f with mask
Z[mask] = np.nan

ax = axs[0,1]
c = ax.contourf(X,Y,Z,30)
circle_patch = plt.Circle((0.5, 0.5), np.sqrt(2)/4, color='r', fill = False)
ax.add_patch(circle_patch)
plt.colorbar(c)
ax.set_title("f with mask")

# plot solution
Z = circle.u_ex(np,[X,Y],None)

ax = axs[1,0]
c = ax.contourf(X,Y,Z,30)
circle_patch = plt.Circle((0.5, 0.5), np.sqrt(2)/4, color='r', fill = False)
ax.add_patch(circle_patch)
plt.colorbar(c)
ax.set_title("u_ex")

# plot solution with mask

Z[mask] = np.nan

ax = axs[1,1]
c = ax.contourf(X,Y,Z,30)
circle_patch = plt.Circle((0.5, 0.5), np.sqrt(2)/4, color='r', fill = False)
ax.add_patch(circle_patch)
plt.colorbar(c)
ax.set_title("u_ex with mask")

plt.show()

