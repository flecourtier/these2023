import numpy as np
import matplotlib.pyplot as plt

from modules.problem.Problem import *
from modules.problem.Case import *

cas = Case("case.json")

problem_considered = cas.Problem


a0,b0 = problem_considered.domain_O[0]
a1,b1 = problem_considered.domain_O[1]

x = np.linspace(a0,b0,100)
y = np.linspace(a1,b1,100)
X,Y = np.meshgrid(x,y)

mask = problem_considered.Omega_bool(X,Y)
mask = np.logical_not(mask)

def get_patch():
    if isinstance(problem_considered, Circle):
        x0,y0 = problem_considered.x0, problem_considered.y0
        r = problem_considered.r
        patch = plt.Circle((x0, y0), r, color='r', fill = False)
    elif isinstance(problem_considered, Square):
        a0,b0 = [0,1]
        a1,b1 = [0,1]
        patch = plt.Rectangle((a0,a1),b0-a0,b1-a1, color='r', fill = False)
    else:
        raise Exception("Problem not recognized")
    return patch

####
# plot !
####

params = [0.5, 1, 0.]

fig, axs = plt.subplots(2,2,figsize=(10,10))

# plot f
Z = problem_considered.u_ex_prime2(np,[X,Y],params)[0]

ax = axs[0,0]
c = ax.contourf(X,Y,Z,30)
patch = get_patch()
ax.add_patch(patch)
plt.colorbar(c)
ax.set_title("f")

# plot f with mask
Z[mask] = np.nan

ax = axs[0,1]
c = ax.contourf(X,Y,Z,30)
patch = get_patch()
ax.add_patch(patch)
plt.colorbar(c)
ax.set_title("f with mask")

# plot solution
Z = problem_considered.u_ex(np,[X,Y],params)

ax = axs[1,0]
c = ax.contourf(X,Y,Z,30)
patch = get_patch()
ax.add_patch(patch)
plt.colorbar(c)
ax.set_title("u_ex")

# plot solution with mask

Z[mask] = np.nan

ax = axs[1,1]
c = ax.contourf(X,Y,Z,30)
patch = get_patch()
ax.add_patch(patch)
plt.colorbar(c)
ax.set_title("u_ex with mask")

plt.show()

