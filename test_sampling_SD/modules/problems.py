import numpy as np
import torch
from modules.MVP import MVP

class Circle():
    def __init__(self):
        self.x0,self.y0 = (0.5,0.5)
        self.r = np.sqrt(2)/4
        self.eps = 0.5-self.r
        self.domain_O = [[self.x0-self.r-self.eps,self.x0+self.r+self.eps],[self.y0-self.r-self.eps,self.y0+self.r+self.eps]]

    def phi(self, pre, xy):
        x,y=xy
        return -self.r**2+(x-self.x0)**2+(y-self.y0)**2

    def levelset(self,X):
        return self.phi(torch, X.T)
    
    def phi_construct(self, pre, xy):
        return self.phi(pre, xy)
    
    def call_Omega(self, pre, xy):
        return self.phi_construct(pre,xy)<0
    
    def Omega_bool(self, x, y):
        xy = (x,y)
        return self.call_Omega(None, xy)

    def u_ex(self, pre, xy, S, f, p):
        x,y=xy
        return S * pre.sin(1/(self.r**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2))

    def u_ex_prime(self, pre, xy, S ,f, p):
        x,y=xy
        du_dx = pre.pi*S*(2*x - 2*self.x0)*pre.cos(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**2
        du_dy = pre.pi*S*(2*y - 2*self.y0)*pre.cos(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**2
        return du_dx,du_dy

    def u_ex_prime2(self, pre, xy, S ,f, p):
        x,y=xy
        du_dxx = 2*pre.pi*S*pre.cos(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**2 - pre.pi**2*S*(2*x - 2*self.x0)**2*pre.sin(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**4
        du_dyy = 2*pre.pi*S*pre.cos(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**2 - pre.pi**2*S*(2*y - 2*self.y0)**2*pre.sin(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**4
        return du_dxx,du_dyy

    def f(self, pre, xy, S, f, p):
        x,y=xy
        return 4/(self.r**4)*pre.pi**2*S*((x-self.x0)**2+(y-self.y0)**2)*pre.sin(1/(self.r**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2)) - \
                4/(self.r**2)*pre.pi*S*pre.cos(1/(self.r**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2))

    def g(self, pre, xy, S, f, p):
        return 0*torch.ones_like(xy[0])
    
class Circle2(): # non homogène
    def __init__(self):
        self.x0,self.y0 = (0.5,0.5)
        self.r = np.sqrt(2)/4
        self.eps = 0.5-self.r
        self.domain_O = [[self.x0-self.r-self.eps,self.x0+self.r+self.eps],[self.y0-self.r-self.eps,self.y0+self.r+self.eps]]

    def phi(self, pre, xy):
        x,y=xy
        return -self.r**2+(x-self.x0)**2+(y-self.y0)**2

    def levelset(self,X):
        return self.phi(torch, X.T)
    
    def phi_construct(self, pre, xy):
        return self.phi(pre, xy)
    
    def call_Omega(self, pre, xy):
        return self.phi_construct(pre,xy)<0

    def u_ex(self, pre, xy, S, f, p):
        x,y=xy
        return S * pre.sin(1/(self.r**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2))+3*(x+y)

    def f(self, pre, xy, S, f, p):
        x,y=xy
        return 4/(self.r**4)*pre.pi**2*S*((x-self.x0)**2+(y-self.y0)**2)*pre.sin(1/(self.r**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2)) - \
                4/(self.r**2)*pre.pi*S*pre.cos(1/(self.r**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2))

    def g(self, pre, xy, S, f, p):
        val_u_ex = self.u_ex(pre, xy, S, f, p)
        val_phi = self.phi(pre, xy)
        return val_u_ex*(1.0+val_phi)
    
class RandomPolygon():
    def __init__(self,nb_pts=40,ADF=MVP):
        print("Initiate RandomPolygon")
        self.x0,self.y0 = (0.0,0.0)
        self.eps = 1e-3
        self.r_max = 2.0-self.eps
        self.r_min = 1.4
        
        self.domain_O = [[self.x0-self.r_max-self.eps,self.x0+self.r_max+self.eps],[self.y0-self.r_max-self.eps,self.y0+self.r_max+self.eps]]

        self.nb_pts = nb_pts
        self.polygon = self.create_polygon(self.nb_pts+1)

        self.ADF = ADF(self.polygon)

    def create_polygon(self, n_points):
        r = np.random.uniform(self.r_min, self.r_max, n_points-1)
        theta = np.linspace(0, 2*np.pi, n_points)
        theta = theta[:-1]
        x = self.x0 + r * np.cos(theta)
        y = self.y0 + r * np.sin(theta)

        data = np.concatenate([x[:, None], y[:, None]], axis=1)
        # data = np.concatenate([data, [data[0]]])

        return torch.Tensor(data)

    def phi(self, pre, xy):
        return self.ADF(xy)
    
    def levelset(self,X):
        return self.phi(torch, X.T)

    def phi_construct(self, pre, xy):
        return self.phi(pre,xy)

    def call_Omega(self, pre, xy):
        return self.phi_construct(pre,xy)<0
    
    def u_ex(self, pre, xy, S, f, p):
        x,y=xy
        return S * pre.sin(1/(self.r_max**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2))

    def f(self, pre, xy, S, f, p):
        x,y=xy
        return 4/(self.r_max**4)*pre.pi**2*S*((x-self.x0)**2+(y-self.y0)**2)*pre.sin(1/(self.r_max**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2)) - \
                4/(self.r_max**2)*pre.pi*S*pre.cos(1/(self.r_max**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2))

    def g(self, pre, xy, S, f, p):
        val_u_ex = self.u_ex(pre, xy, S, f, p)
        val_phi = self.phi(pre, xy)
        return val_u_ex*(1.0+val_phi)


class SquareMVP():
    def __init__(self,nb_pts=40,ADF=MVP):
        print("Initiate RandomPolygon")
        self.x0,self.y0 = (0.0,0.0)
        self.eps = 1e-3
        self.r_max = 2.0-self.eps
        self.r_min = 1.4
        
        self.domain_O = [[self.x0-self.r_max-self.eps,self.x0+self.r_max+self.eps],[self.y0-self.r_max-self.eps,self.y0+self.r_max+self.eps]]

        self.nb_pts = 4
        self.polygon = torch.Tensor(np.array([[-1,-1],[1,-1],[1,1],[-1,1]]))

        self.ADF = ADF(self.polygon)

    def create_polygon(self, n_points):
        r = np.random.uniform(self.r_min, self.r_max, n_points-1)
        theta = np.linspace(0, 2*np.pi, n_points)
        theta = theta[:-1]
        x = self.x0 + r * np.cos(theta)
        y = self.y0 + r * np.sin(theta)

        data = np.concatenate([x[:, None], y[:, None]], axis=1)
        # data = np.concatenate([data, [data[0]]])

        return torch.Tensor(data)

    def phi(self, pre, xy):
        return self.ADF(xy)
    
    def levelset(self,X):
        return self.phi(torch, X.T)

    def phi_construct(self, pre, xy):
        return self.phi(pre,xy)

    def call_Omega(self, pre, xy):
        return self.phi_construct(pre,xy)<0
    
    def u_ex(self, pre, xy, S, f, p):
        x,y=xy
        return S * pre.sin(1/(self.r_max**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2))

    def f(self, pre, xy, S, f, p):
        x,y=xy
        return 4/(self.r_max**4)*pre.pi**2*S*((x-self.x0)**2+(y-self.y0)**2)*pre.sin(1/(self.r_max**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2)) - \
                4/(self.r_max**2)*pre.pi*S*pre.cos(1/(self.r_max**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2))

    def g(self, pre, xy, S, f, p):
        val_u_ex = self.u_ex(pre, xy, S, f, p)
        val_phi = self.phi(pre, xy)
        return val_u_ex*(1.0+val_phi)
     
# class RandomPolygon_boucle():
#     def __init__(self,polygon,ADF=MVP):
#         self.polygon = polygon
#         self.ADF = ADF(self.polygon)

#     def create_polygon(self, n_points):
#         r = np.random.uniform(self.r_min, self.r_max, n_points-1)
#         theta = np.linspace(0, 2*np.pi, n_points)
#         theta = theta[:-1]
#         x = self.x0 + r * np.cos(theta)
#         y = self.y0 + r * np.sin(theta)

#         data = np.concatenate([x[:, None], y[:, None]], axis=1)
#         # data = np.concatenate([data, [data[0]]])

#         return torch.Tensor(data)

#     def phi(self, pre, xy):
#         return self.ADF(xy,vectorized=False)
    
#     def levelset(self,X):
#         return self.phi(torch, X.T)

#     def phi_construct(self, pre, xy):
#         return self.phi(pre,xy)

#     def call_Omega(self, pre, xy):
#         return self.phi_construct(pre,xy)<0
    
#     def u_ex(self, pre, xy, S, f, p):
#         x,y=xy
#         return S * pre.sin(1/(self.r_max**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2))

#     def f(self, pre, xy, S, f, p):
#         x,y=xy
#         return 4/(self.r_max**4)*pre.pi**2*S*((x-self.x0)**2+(y-self.y0)**2)*pre.sin(1/(self.r_max**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2)) - \
#                 4/(self.r_max**2)*pre.pi*S*pre.cos(1/(self.r_max**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2))

#     def g(self, pre, xy, S, f, p):
#         val_u_ex = self.u_ex(pre, xy, S, f, p)
#         val_phi = self.phi(pre, xy)
#         print(val_u_ex.shape,val_phi.shape)
#         return val_u_ex*(1.0+val_phi)