import numpy as np
import matplotlib.pyplot as plt
import abc
import forms

classform = forms.Bean
plot = False

nameform = classform.__name__
nameform = nameform.lower()
form = classform()

# create set of points
if plot:
    n_pts = 100
else:
    n_pts = 1000

t = np.linspace(0,1,n_pts)

c_t = form.c(t)

grad_c_t = form.c_prime_rot(t,theta=-np.pi/2)
grad_c_t_norm = np.linalg.norm(grad_c_t,axis=0)
normals = grad_c_t/grad_c_t_norm

# check if normals is unitary
assert np.allclose(np.linalg.norm(normals,axis=0),1)

if plot:
    fig, ax = plt.subplots()

    form.plot_curve(color="black")
    ax.quiver(c_t[0],c_t[1],normals[0],normals[1],color="red")

    plt.xlim(form.bord_a,form.bord_b)
    plt.ylim(form.bord_a2,form.bord_b2)

    plt.show()
else:
    # create an xyz file
    x,y = c_t
    normals = normals.T
    with open("xyzfiles/"+nameform+".xyz","w") as f:
        for i in range(len(x)):
            f.write(f"{x[i]} {y[i]} {normals[i,0]} {normals[i,1]}\n")
        f.close()