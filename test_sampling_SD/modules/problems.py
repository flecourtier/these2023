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

    def phi_construct(self, pre, xy):
        return self.phi(pre, xy)
    
    def levelset(self,X):
        return self.phi_construct(torch, X.T)
    
    def call_Omega(self, pre, xy):
        return self.phi_construct(pre,xy)<0
    
    def Omega_bool(self, x, y):
        xy = (x,y)
        return self.call_Omega(None, xy)

    def u_ex(self, pre, xy, mu):
        x,y=xy
        S = mu
        return S * pre.sin(1/(self.r**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2))

    def u_ex_prime(self, pre, xy, mu):
        x,y=xy
        S = mu
        du_dx = pre.pi*S*(2*x - 2*self.x0)*pre.cos(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**2
        du_dy = pre.pi*S*(2*y - 2*self.y0)*pre.cos(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**2
        return du_dx,du_dy

    def u_ex_prime2(self, pre, xy, mu):
        x,y=xy
        S = mu
        du_dxx = 2*pre.pi*S*pre.cos(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**2 - pre.pi**2*S*(2*x - 2*self.x0)**2*pre.sin(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**4
        du_dyy = 2*pre.pi*S*pre.cos(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**2 - pre.pi**2*S*(2*y - 2*self.y0)**2*pre.sin(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**4
        return du_dxx,du_dyy

    def f(self, pre, xy, mu):
        x,y=xy
        S = mu
        return 4/(self.r**4)*pre.pi**2*S*((x-self.x0)**2+(y-self.y0)**2)*pre.sin(1/(self.r**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2)) - \
                4/(self.r**2)*pre.pi*S*pre.cos(1/(self.r**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2))

    def g(self, pre, xy, mu):
        return 0*torch.ones_like(xy[0])

class Square():
    def __init__(self):
        self.eps = 0.5
        self.domain_O = [[0-self.eps,1+self.eps],[0-self.eps,1+self.eps]]

    def phi(self, pre, xy):
        x,y=xy
        return x*(1-x)*y*(1-y)
    
    def phi_construct(self, pre, xy):
        # return np.linalg.norm(np.array(xy)-0.5,np.inf,axis=0)-0.5
        return torch.linalg.norm(xy-0.5,float('inf'),axis=0)-0.5

    def levelset(self,X):
        return self.phi_construct(torch, X.T)
    
    def call_Omega(self, pre, xy):
        return self.phi_construct(pre,xy)<0
    
    def Omega_bool(self, x, y):
        xy = (x,y)
        return self.call_Omega(None, xy)

    def u_ex(self, pre, xy, mu):
        x,y=xy
        S, f, p = mu
        return S*pre.sin(2*pre.pi*f*x + p) * pre.sin(2*pre.pi*f*y + p)

    def u_ex_prime(self, pre, xy, mu):
        x,y=xy
        S,f,p = mu
        du_dx = 2*pre.pi*S*f*pre.sin(2*pre.pi*f*y + p)*pre.cos(2*pre.pi*f*x + p)
        du_dy = pre.pi*S*(2*y - 2*self.y0)*pre.cos(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**2
        return du_dx,du_dy

    def u_ex_prime2(self, pre, xy, mu):
        x,y=xy
        S,f,p = mu
        du_dxx = -4*pre.pi**2*S*f**2*pre.sin(2*pre.pi*f*x + p)*pre.sin(2*pre.pi*f*y + p)
        du_dyy = -4*pre.pi**2*S*f**2*pre.sin(2*pre.pi*f*x + p)*pre.sin(2*pre.pi*f*y + p)
        return du_dxx,du_dyy

    def f(self, pre, xy, mu):
        x,y=xy
        S, f, p = mu
        return 8*pre.pi**2*S*f**2*pre.sin(2*pre.pi*f*x + p)*pre.sin(2*pre.pi*f*y + p)
    
    def g(self, pre, xy, mu):
        return 0*torch.ones_like(xy[0])
