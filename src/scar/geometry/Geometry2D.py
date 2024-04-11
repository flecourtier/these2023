import abc
import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

class ParametricCurves(abc.ABC):
    def __init__(self):
        self.name = self.__class__.__name__
        self.theta_normals = -np.pi/2
        self.bound_box = [[self.bord_a,self.bord_b],[self.bord_a2,self.bord_b2]]

    @abc.abstractmethod
    def c(self,t):
        pass

    @abc.abstractmethod
    def c_prime(self,t):
        pass

    def c_prime_rot(self,t,theta=np.pi/2):
        rot_mat = torch.Tensor(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        ).to(device)
        # rot_mat = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        c_prime = self.c_prime(t)#.detach().numpy()
        return rot_mat @ c_prime
        
    def plot_curve(self,color="white",n_pts=100):
        t = np.linspace(0,1,n_pts)
        c_t = self.c(t).cpu().detach().numpy()
        # ajout du point de départ
        c_t = np.concatenate([c_t,np.array([c_t[:,0]]).T],axis=1)
        for i in range(n_pts):
            pt1 = c_t[:,i]
            pt2 = c_t[:,i+1]
            plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],color,linewidth=3)
    
class Circle(ParametricCurves):
    def __init__(self):
        self.bord_a,self.bord_b = (0.,1.)
        self.bord_a2,self.bord_b2 = (0.,1.)
        super().__init__()
        
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
        self.bord_a,self.bord_b = (-1,1)
        self.bord_a2,self.bord_b2 = (-1,1)
        super().__init__()

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
        self.bord_a,self.bord_b = (-1,1)
        self.bord_a2,self.bord_b2 = (-1,1)
        self.r = 0.4        
        super().__init__()

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
        self.bord_a,self.bord_b = (-1,1)
        self.bord_a2,self.bord_b2 = (-1,1)
        self.r = 0.3        
        super().__init__()

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
        self.bord_a,self.bord_b = (-1.5,1.5)
        self.bord_a2,self.bord_b2 = (-1.5,1.5)
        super().__init__()

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
        self.bord_a,self.bord_b = (-0.5,1.5)
        self.bord_a2,self.bord_b2 = (-1.5,0.5)
        super().__init__()

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
    
PI = 3.14159265358979323846
class Cat(ParametricCurves):
    def __init__(self,a=3,b=5):
        self.bord_a,self.bord_b = (-1.0,1.0)
        self.bord_a2,self.bord_b2 = (-1.0,1.0)
        super().__init__()

        self.theta_normals = np.pi/2

    def c(self,t):
        t = 2 * PI * torch.tensor(t)[None, :]
        t = t.to(device)
        
        x = -(721 * torch.sin(t))/4 + 196/3 * torch.sin(2 * t) - 86/3 * torch.sin(3 * t) - 131/2 * torch.sin(4 * t)+477/14 * torch.sin(5 * t)+27 * torch.sin(6 * t)-29/2 * torch.sin(7 * t)+68/5 * torch.sin(8 * t)+1/10 * torch.sin(9 * t)+23/4 * torch.sin(10 * t)-19/2 * torch.sin(12 * t)-85/21 * torch.sin(13 * t)+2/3 * torch.sin(14 * t)+27/5 * torch.sin(15 * t)+7/4 * torch.sin(16 * t)+17/9 * torch.sin(17 * t)-4 * torch.sin(18 * t)-1/2 * torch.sin(19 * t)+1/6 * torch.sin(20 * t)+6/7 * torch.sin(21 * t)-1/8 * torch.sin(22 * t)+1/3 * torch.sin(23 * t)+3/2 * torch.sin(24 * t)+13/5 * torch.sin(25 * t)+ torch.sin(26 * t)-2 * torch.sin(27 * t)+3/5 * torch.sin(28 * t)-1/5 * torch.sin(29 * t)+1/5 * torch.sin(30 * t)+(2337 * torch.cos(t))/8-43/5 * torch.cos(2 * t)+322/5 * torch.cos(3 * t)-117/5 * torch.cos(4 * t)-26/5 * torch.cos(5 * t)-23/3 * torch.cos(6 * t)+143/4 * torch.cos(7 * t)-11/4 * torch.cos(8 * t)-31/3 * torch.cos(9 * t)-13/4 * torch.cos(10 * t)-9/2 * torch.cos(11 * t)+41/20 * torch.cos(12 * t)+8 * torch.cos(13 * t)+2/3 * torch.cos(14 * t)+6 * torch.cos(15 * t)+17/4 * torch.cos(16 * t)-3/2 * torch.cos(17 * t)-29/10 * torch.cos(18 * t)+11/6 * torch.cos(19 * t)+12/5 * torch.cos(20 * t)+3/2 * torch.cos(21 * t)+11/12 * torch.cos(22 * t)-4/5 * torch.cos(23 * t)+ torch.cos(24 * t)+17/8 * torch.cos(25 * t)-7/2 * torch.cos(26 * t)-5/6 * torch.cos(27 * t)-11/10 * torch.cos(28 * t)+1/2 * torch.cos(29 * t)-1/5 * torch.cos(30 * t)
        
        y = -(637 * torch.sin(t))/2-188/5 * torch.sin(2 * t)-11/7 * torch.sin(3 * t)-12/5 * torch.sin(4 * t)+11/3 * torch.sin(5 * t)-37/4 * torch.sin(6 * t)+8/3 * torch.sin(7 * t)+65/6 * torch.sin(8 * t)-32/5 * torch.sin(9 * t)-41/4 * torch.sin(10 * t)-38/3 * torch.sin(11 * t)-47/8 * torch.sin(12 * t)+5/4 * torch.sin(13 * t)-41/7 * torch.sin(14 * t)-7/3 * torch.sin(15 * t)-13/7 * torch.sin(16 * t)+17/4 * torch.sin(17 * t)-9/4 * torch.sin(18 * t)+8/9 * torch.sin(19 * t)+3/5 * torch.sin(20 * t)-2/5 * torch.sin(21 * t)+4/3 * torch.sin(22 * t)+1/3 * torch.sin(23 * t)+3/5 * torch.sin(24 * t)-3/5 * torch.sin(25 * t)+6/5 * torch.sin(26 * t)-1/5 * torch.sin(27 * t)+10/9 * torch.sin(28 * t)+1/3 * torch.sin(29 * t)-3/4 * torch.sin(30 * t)-(125 * torch.cos(t))/2-521/9 * torch.cos(2 * t)-359/3 * torch.cos(3 * t)+47/3 * torch.cos(4 * t)-33/2 * torch.cos(5 * t)-5/4 * torch.cos(6 * t)+31/8 * torch.cos(7 * t)+9/10 * torch.cos(8 * t)-119/4 * torch.cos(9 * t)-17/2 * torch.cos(10 * t)+22/3 * torch.cos(11 * t)+15/4 * torch.cos(12 * t)-5/2 * torch.cos(13 * t)+19/6 * torch.cos(14 * t)+7/4 * torch.cos(15 * t)+31/4 * torch.cos(16 * t)- torch.cos(17 * t)+11/10 * torch.cos(18 * t)-2/3 * torch.cos(19 * t)+13/3 * torch.cos(20 * t)-5/4 * torch.cos(21 * t)+2/3 * torch.cos(22 * t)+1/4 * torch.cos(23 * t)+5/6 * torch.cos(24 * t)+3/4 * torch.cos(26 * t)-1/2 * torch.cos(27 * t)-1/10 * torch.cos(28 * t)-1/3 * torch.cos(29 * t)-1/19 * torch.cos(30 * t)

        x = x/500
        y = y/500

        return torch.cat((x, y), 0)

    def c_prime(self, t):
        t = 2 * PI * torch.tensor(t)[None, :]
        t = t.to(device)

        x = -2337*PI*torch.sin(t)/4 + 34.4*PI*torch.sin(2*t) - 386.4*PI*torch.sin(3*t) + 187.2*PI*torch.sin(4*t) + 52.0*PI*torch.sin(5*t) + 92.0*PI*torch.sin(6*t) - 500.5*PI*torch.sin(7*t) + 44.0*PI*torch.sin(8*t) + 186.0*PI*torch.sin(9*t) + 65.0*PI*torch.sin(10*t) + 99.0*PI*torch.sin(11*t) - 49.2*PI*torch.sin(12*t) - 208*PI*torch.sin(13*t) - 18.6666666666667*PI*torch.sin(14*t) - 180*PI*torch.sin(15*t) - 136.0*PI*torch.sin(16*t) + 51.0*PI*torch.sin(17*t) + 104.4*PI*torch.sin(18*t) - 69.6666666666667*PI*torch.sin(19*t) - 96.0*PI*torch.sin(20*t) - 63.0*PI*torch.sin(21*t) - 40.3333333333333*PI*torch.sin(22*t) + 36.8*PI*torch.sin(23*t) - 48*PI*torch.sin(24*t) - 106.25*PI*torch.sin(25*t) + 182.0*PI*torch.sin(26*t) + 45.0*PI*torch.sin(27*t) + 61.6*PI*torch.sin(28*t) - 29.0*PI*torch.sin(29*t) + 12.0*PI*torch.sin(30*t) - 721*PI*torch.cos(t)/2 + 261.333333333333*PI*torch.cos(2*t) - 172.0*PI*torch.cos(3*t) - 524.0*PI*torch.cos(4*t) + 340.714285714286*PI*torch.cos(5*t) + 324*PI*torch.cos(6*t) - 203.0*PI*torch.cos(7*t) + 217.6*PI*torch.cos(8*t) + 1.8*PI*torch.cos(9*t) + 115.0*PI*torch.cos(10*t) - 228.0*PI*torch.cos(12*t) - 105.238095238095*PI*torch.cos(13*t) + 18.6666666666667*PI*torch.cos(14*t) + 162.0*PI*torch.cos(15*t) + 56.0*PI*torch.cos(16*t) + 64.2222222222222*PI*torch.cos(17*t) - 144*PI*torch.cos(18*t) - 19.0*PI*torch.cos(19*t) + 6.66666666666667*PI*torch.cos(20*t) + 36.0*PI*torch.cos(21*t) - 5.5*PI*torch.cos(22*t) + 15.3333333333333*PI*torch.cos(23*t) + 72.0*PI*torch.cos(24*t) + 130.0*PI*torch.cos(25*t) + 52*PI*torch.cos(26*t) - 108*PI*torch.cos(27*t) + 33.6*PI*torch.cos(28*t) - 11.6*PI*torch.cos(29*t) + 12.0*PI*torch.cos(30*t)

        y = 125*PI*torch.sin(t) + 231.555555555556*PI*torch.sin(2*t) + 718.0*PI*torch.sin(3*t) - 125.333333333333*PI*torch.sin(4*t) + 165.0*PI*torch.sin(5*t) + 15.0*PI*torch.sin(6*t) - 54.25*PI*torch.sin(7*t) - 14.4*PI*torch.sin(8*t) + 535.5*PI*torch.sin(9*t) + 170.0*PI*torch.sin(10*t) - 161.333333333333*PI*torch.sin(11*t) - 90.0*PI*torch.sin(12*t) + 65.0*PI*torch.sin(13*t) - 88.6666666666667*PI*torch.sin(14*t) - 52.5*PI*torch.sin(15*t) - 248.0*PI*torch.sin(16*t) + 34*PI*torch.sin(17*t) - 39.6*PI*torch.sin(18*t) + 25.3333333333333*PI*torch.sin(19*t) - 173.333333333333*PI*torch.sin(20*t) + 52.5*PI*torch.sin(21*t) - 29.3333333333333*PI*torch.sin(22*t) - 11.5*PI*torch.sin(23*t) - 40.0*PI*torch.sin(24*t) - 39.0*PI*torch.sin(26*t) + 27.0*PI*torch.sin(27*t) + 5.6*PI*torch.sin(28*t) + 19.3333333333333*PI*torch.sin(29*t) + 3.15789473684211*PI*torch.sin(30*t) - 637*PI*torch.cos(t) - 150.4*PI*torch.cos(2*t) - 9.42857142857143*PI*torch.cos(3*t) - 19.2*PI*torch.cos(4*t) + 36.6666666666667*PI*torch.cos(5*t) - 111.0*PI*torch.cos(6*t) + 37.3333333333333*PI*torch.cos(7*t) + 173.333333333333*PI*torch.cos(8*t) - 115.2*PI*torch.cos(9*t) - 205.0*PI*torch.cos(10*t) - 278.666666666667*PI*torch.cos(11*t) - 141.0*PI*torch.cos(12*t) + 32.5*PI*torch.cos(13*t) - 164.0*PI*torch.cos(14*t) - 70.0*PI*torch.cos(15*t) - 59.4285714285714*PI*torch.cos(16*t) + 144.5*PI*torch.cos(17*t) - 81.0*PI*torch.cos(18*t) + 33.7777777777778*PI*torch.cos(19*t) + 24.0*PI*torch.cos(20*t) - 16.8*PI*torch.cos(21*t) + 58.6666666666667*PI*torch.cos(22*t) + 15.3333333333333*PI*torch.cos(23*t) + 28.8*PI*torch.cos(24*t) - 30.0*PI*torch.cos(25*t) + 62.4*PI*torch.cos(26*t) - 10.8*PI*torch.cos(27*t) + 62.2222222222222*PI*torch.cos(28*t) + 19.3333333333333*PI*torch.cos(29*t) - 45.0*PI*torch.cos(30*t)

        return torch.cat((x, y), 0)