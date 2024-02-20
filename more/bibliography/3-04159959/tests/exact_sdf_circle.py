import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

r = 1.0

# cone : base unit circle, height 1
def phi(x,y):
    return r-np.sqrt(x**2+y**2)

# Discrétisation des angles
n_pts = 100
x = np.linspace(-r, r, n_pts)
y = np.linspace(-r, r, n_pts)
x, y = np.meshgrid(x, y)
z = phi(x,y)

for i in range(n_pts):
    for j in range(n_pts):
        if z[i,j] < 0:
            z[i,j] = None

fig,ax = plt.subplots()
ax = fig.add_subplot(projection='3d')

ax.scatter(x,y,z,marker="+")

# plt.contourf(x,y,z)
# plt.colorbar()

plt.show()

# create an xyz file
# with open("sphere.xyz","w") as f:
#     for i in range(len(x)):
#         f.write(f"{x[i]} {y[i]} {z[i]} {normal[i,0]} {normal[i,1]} {normal[i,2]}\n")
#     f.close()
