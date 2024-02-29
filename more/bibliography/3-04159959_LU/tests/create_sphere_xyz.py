import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

X0 = (0.,0.,0.)
x0,y0,z0 = X0
r = 1

# paramteric curve for sphere
def c(theta,phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x,y,z

def exact_normal(x,y,z):
    return np.array([x,y,z]).T

def norm(v):
    x,y,z = v
    return np.sqrt(x**2+y**2+z**2)

# circle = patches.Circle(X0,r,fill=None,edgecolor="red")

# Discrétisation des angles
n_pts = 100
theta = np.linspace(0, np.pi, n_pts)
phi = np.linspace(0, 2 * np.pi, n_pts)
theta, phi = np.meshgrid(theta, phi)
theta = theta.flatten()
phi = phi.flatten()

x,y,z = c(theta,phi)
normal = exact_normal(x,y,z)

# fig,ax = plt.subplots()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(x,y,z,marker="+")
# ax.quiver(x,y,z,normal[:,0],normal[:,1],normal[:,2],length=0.1,color="red")

# plt.show()

# create an xyz file
with open("sphere.xyz","w") as f:
    for i in range(len(x)):
        f.write(f"{x[i]} {y[i]} {z[i]} {normal[i,0]} {normal[i,1]} {normal[i,2]}\n")
    f.close()
