import abc
import numpy as np
import matplotlib.pyplot as plt
import torch

# pas aboutit - problèmes

class ParametricCurves3D(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def c(self,theta,phi):
        pass

    @abc.abstractmethod
    def c_prime(self,theta,phi):
        pass

    def orthogonal(self,theta,phi):
        c_t = self.c(theta,phi)
        c_prime_theta,c_prime_phi = self.c_prime(theta,phi)

        # produit vectoriel
        prod = torch.cross(c_prime_theta,c_prime_phi)

        # sens de la normale
        # si le produit scalaire est négatif, on inverse le vecteur
        prod_scal = (prod*c_t).sum(axis = 0)
        prod = torch.where(prod_scal<0,-prod,prod)
        
        # afficher dernier (cas où x,y,z = (0,0,-1))
        print(c_t[:,-1])
        print(c_prime_theta[:,-1])
        print(c_prime_phi[:,-1])
        print(prod[:,-1])


        return prod
        
    # def plot_curve(self,color="white"):
    #     t = np.linspace(0,1,100)
    #     c_t = self.c(t)
    #     # ajout du point de départ
    #     c_t = np.concatenate([c_t,np.array([c_t[:,0]]).T],axis=1)
    #     for i in range(100):
    #         pt1 = c_t[:,i]
    #         pt2 = c_t[:,i+1]
    #         plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],color,linewidth=3)
    
class Sphere(ParametricCurves3D):
    def __init__(self):
        self.name = self.__class__.__name__
        self.bord_a,self.bord_b = (-1.,1.)
        self.bord_a2,self.bord_b2 = (-1.,1.)
        self.bord_a3,self.bord_b3 = (-1.,1.)

        self.x0,self.y0,self.z0 = (0.0,0.0,0.0) # centre de la sphère
        self.r = 1. # rayon de la sphère

    def c(self,theta,phi): # theta \in [0,1] , phi \in [0,1]
        # conversion : theta \in [0,pi] , phi \in [0,2pi]
        theta = np.pi*theta
        phi = 2*np.pi*phi
        x = self.x0 + self.r * np.sin(theta) * np.cos(phi)
        y = self.y0 + self.r * np.sin(theta) * np.sin(phi)
        z = self.z0 + self.r * np.cos(theta)
        return torch.Tensor(np.array([x,y,z]))
    
    def c_prime(self,theta,phi): # theta \in [0,1] , phi \in [0,1]
        # conversion : theta \in [0,pi] , phi \in [0,2pi]
        theta = np.pi*theta
        phi = 2*np.pi*phi

        x_theta = np.pi * self.r * np.cos(theta) * np.cos(phi)
        y_theta = np.pi * self.r * np.cos(theta) * np.sin(phi)
        z_theta = -np.pi * self.r * np.sin(theta)

        x_phi = -2*np.pi * self.r * np.sin(theta) * np.sin(phi)
        y_phi = 2*np.pi * self.r * np.sin(theta) * np.cos(phi)
        z_phi = np.zeros_like(x_phi)

        return torch.Tensor(np.array([x_theta,y_theta,z_theta])), torch.Tensor(np.array([x_phi,y_phi,z_phi]))

form = Sphere()

n_bc_points = 10
theta = np.linspace(0, 1.0, n_bc_points)
phi = np.linspace(0, 1.0, n_bc_points)
theta, phi = np.meshgrid(theta, phi)
theta = theta.flatten()
phi = phi.flatten()

c_t = form.c(theta,phi).numpy()
x,y,z = c_t

ortho = form.orthogonal(theta,phi).numpy()
ortho_norm = np.linalg.norm(ortho,axis=0)
normals = ortho/ortho_norm

# get index where normals is nan
# nan_idx = np.where(np.isnan(normals[0,:]))[0]
# print(nan_idx)
# print(theta[nan_idx])
# print(phi[nan_idx])
# print(x[nan_idx],y[nan_idx],z[nan_idx])

fig,ax = plt.subplots()
ax = fig.add_subplot(projection='3d')

ax.scatter(x,y,z,marker="+")
ax.quiver(x,y,z,normals[0,:],normals[1,:],normals[2,:],length=0.1,color="red")

plt.show()

assert np.allclose(np.linalg.norm(normals,axis=0),1)