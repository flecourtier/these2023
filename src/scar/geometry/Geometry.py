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

    def c_prime_rot(self,t): # theta = 90°
        x,y = self.c_prime(t)
        return torch.Tensor(np.array([-y.cpu().numpy(),x.cpu().numpy()]))
        
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

class Astroid(ParametricCurves):
    def __init__(self):
        self.name = self.__class__.__name__
        self.bord_a,self.bord_b = (-1,1)
        self.bord_a2,self.bord_b2 = (-1,1)

    def c(self,t): # t \in [0,1]
        x = np.cos(2*np.pi*t)**3
        y = np.sin(2*np.pi*t)**3
        # return np.array([x,y])
        return torch.Tensor([x,y])

    def c_prime(self,t):
        x = -6*np.pi*np.cos(2*np.pi*t)**2*np.sin(2*np.pi*t)
        y = 6*np.pi*np.cos(2*np.pi*t)*np.sin(2*np.pi*t)**2
        # return np.array([x,y])
        return torch.Tensor([x,y])
    
class SmoothAstroid(ParametricCurves):
    def __init__(self):
        self.name = self.__class__.__name__
        self.bord_a,self.bord_b = (-1,1)
        self.bord_a2,self.bord_b2 = (-1,1)
        self.r = 0.4

    def c(self,t): # t \in [0,1]
        x = self.r * np.cos(2*np.pi*t) + 0.5*np.cos(2*np.pi*t)**3
        y = self.r * np.sin(2*np.pi*t) + 0.5*np.sin(2*np.pi*t)**3
        # return np.array([x,y])
        return torch.Tensor([x,y])

    def c_prime(self,t):
        x = -2*np.pi*self.r*np.sin(2*np.pi*t) - 3*np.pi*np.sin(2*np.pi*t)*np.cos(2*np.pi*t)**2
        y = 2*np.pi*self.r*np.cos(2*np.pi*t) + 3*np.pi*np.cos(2*np.pi*t)*np.sin(2*np.pi*t)**2
        # return np.array([x,y])
        return torch.Tensor([x,y])
    
class SmoothCardioid(ParametricCurves):
    def __init__(self):
        self.name = self.__class__.__name__
        self.bord_a,self.bord_b = (-1,1)
        self.bord_a2,self.bord_b2 = (-1,1)
        self.r = 0.3

    def c(self,t):
        x = self.r * ( 2*np.cos(2*np.pi*t) - 0.7*np.cos(4*np.pi*t) )
        y = self.r * ( 2*np.sin(2*np.pi*t) - 0.7*np.sin(4*np.pi*t) )
        # return np.array([x,y])
        return torch.Tensor([x,y])
    
    def c_prime(self, t):
        x = self.r * ( -4*np.pi*np.sin(2*np.pi*t) + 2.8*np.pi*np.sin(4*np.pi*t) )
        y = self.r * ( 4*np.pi*np.cos(2*np.pi*t) - 2.8*np.pi*np.cos(4*np.pi*t) )
        # return np.array([x,y])
        return torch.Tensor([x,y])
    
class Pumpkin(ParametricCurves):
    def __init__(self):
        self.name = self.__class__.__name__
        self.bord_a,self.bord_b = (-1.5,1.5)
        self.bord_a2,self.bord_b2 = (-1.5,1.5)

    def c(self,t):
        x = np.cos(2*np.pi*t)+0.3*np.cos(6*np.pi*t)+0.1*np.cos(10*np.pi*t)
        y = np.sin(2*np.pi*t)+0.3*np.sin(6*np.pi*t)+0.1*np.sin(10*np.pi*t)
        # return np.array([x,y])
        return torch.Tensor([x,y])
    
    def c_prime(self, t):
        x = -2*np.pi*np.sin(2*np.pi*t) - 1.8*np.pi*np.sin(6*np.pi*t) - np.pi*np.sin(10*np.pi*t)
        y = 2*np.pi*np.cos(2*np.pi*t) + 1.8*np.pi*np.cos(6*np.pi*t) + np.pi*np.cos(10*np.pi*t)
        # return np.array([x,y])
        return torch.Tensor([x,y])
    
class Bean(ParametricCurves):
    def __init__(self,a=3,b=5):
        self.name = self.__class__.__name__
        self.bord_a,self.bord_b = (-0.5,1.5)
        self.bord_a2,self.bord_b2 = (-1.5,0.5)
        self.a = a
        self.b = b
        self.theta = -np.pi/2
    
    def R(self):
        # rot = np.array([[np.cos(self.theta),-np.sin(self.theta)],[np.sin(self.theta),np.cos(self.theta)]])
        rot = torch.Tensor([[np.cos(self.theta),-np.sin(self.theta)],[np.sin(self.theta),np.cos(self.theta)]])
        return rot
    
    def c(self,t):
        x = (np.sin(2*np.pi*t)**self.a+np.cos(2*np.pi*t)**self.b)*np.cos(2*np.pi*t)
        y = (np.sin(2*np.pi*t)**self.a+np.cos(2*np.pi*t)**self.b)*np.sin(2*np.pi*t)

        # return self.R() @ np.array([x,y])
        return self.R()@torch.Tensor([x,y])
    
    def c_prime(self, t):
        x = (2*np.pi*self.a*np.sin(2*np.pi*t)**(self.a-1)*np.cos(2*np.pi*t) - 2*np.pi*self.b*np.sin(2*np.pi*t)*np.cos(2*np.pi*t)**(self.b-1))*np.cos(2*np.pi*t) - 2*np.pi*(np.sin(2*np.pi*t)**self.a + np.cos(2*np.pi*t)**self.b)*np.sin(2*np.pi*t)

        y = (2*np.pi*self.a*np.sin(2*np.pi*t)**(self.a-1)*np.cos(2*np.pi*t) - 2*np.pi*self.b*np.sin(2*np.pi*t)*np.cos(2*np.pi*t)**(self.b-1))*np.sin(2*np.pi*t) + 2*np.pi*(np.sin(2*np.pi*t)**self.a + np.cos(2*np.pi*t)**self.b)*np.cos(2*np.pi*t)

        # return self.R() @ np.array([x,y])
        return self.R()@torch.Tensor([x,y])