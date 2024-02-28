import abc
import numpy as np
import matplotlib.pyplot as plt
import torch

class ParametricCurves(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def c(self,t):
        pass

    @abc.abstractmethod
    def c_prime(self,t):
        pass

    def c_prime_rot(self,t,theta=np.pi/2):
        rot_mat = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        c_prime = self.c_prime(t).detach().numpy()
        return torch.Tensor(rot_mat @ c_prime)
        
    def plot_curve(self,color="white"):
        t = np.linspace(0,1,100)
        c_t = self.c(t)
        # ajout du point de départ
        c_t = np.concatenate([c_t,np.array([c_t[:,0]]).T],axis=1)
        for i in range(100):
            pt1 = c_t[:,i]
            pt2 = c_t[:,i+1]
            plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],color,linewidth=3)
    
class Circle(ParametricCurves):
    def __init__(self):
        self.name = self.__class__.__name__
        self.bord_a,self.bord_b = (0.,1.)
        self.bord_a2,self.bord_b2 = (0.,1.)

        self.x0,self.y0 = (0.5,0.5) # centre du cercle
        self.r = np.sqrt(2.)/4. # rayon du cercle

    def c(self,t): # t \in [0,1]
        x = self.x0 + self.r*np.cos(2*np.pi*t)
        y = self.y0 + self.r*np.sin(2*np.pi*t)
        # return np.array([x,y])
        return torch.Tensor(np.array([x,y]))
    
    def c_prime(self,t):
        x = -2.*np.pi*self.r*np.sin(2*np.pi*t)
        y = 2.*np.pi*self.r*np.cos(2*np.pi*t)
        # return np.array([x,y])
        return torch.Tensor(np.array([x,y]))
    
form = Circle()

n_bc_points = 100
t = np.linspace(0,1,n_bc_points)
c_t = form.c(t)
grad_c_t = form.c_prime_rot(t,theta=-np.pi/2)
print(grad_c_t.shape)
grad_c_t_norm = np.linalg.norm(grad_c_t,axis=0)
print(grad_c_t_norm.shape)
normals = grad_c_t/grad_c_t_norm
print(normals.shape)

fig, ax = plt.subplots()

form.plot_curve(color="black")
ax.quiver(c_t[0,::10],c_t[1,::10],normals[0,::10],normals[1,::10],color="red")

plt.xlim(form.bord_a,form.bord_b)
plt.ylim(form.bord_a2,form.bord_b2)

plt.show()

assert np.allclose(np.linalg.norm(normals,axis=0),1)