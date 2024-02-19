import numpy as np
import matplotlib.pyplot as plt

X0 = (0.,0.)
x0,y0 = X0
r = 1

# paramteric curve for circle
def c(theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x,y

def exact_normal(x,y):
    return np.array([x,y]).T

def norm(v):
    x,y = v
    return np.sqrt(x**2+y**2)

# Discrétisation des angles
n_pts = 1000
theta = np.linspace(0, 2*np.pi, n_pts)

x,y = c(theta)
normal = exact_normal(x,y)

fig,ax = plt.subplots()

ax.scatter(x,y,marker="+")
# ax.quiver(x,y,normal[:,0],normal[:,1],color="red")

plt.show()

# create an xyz file
# with open("circle.xyz","w") as f:
#     for i in range(len(x)):
#         f.write(f"{x[i]} {y[i]} {normal[i,0]} {normal[i,1]}\n")
#     f.close()
