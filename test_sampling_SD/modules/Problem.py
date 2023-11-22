import numpy as np
import torch

class Circle():
    def __init__(self):
        """Circle domain centered at (x_0,y_0) with radius r
        """
        self.x0,self.y0 = (0.5,0.5)
        self.r = np.sqrt(2)/4
        self.eps = 0.5-self.r
        self.domain_O = [[self.x0-self.r-self.eps,self.x0+self.r+self.eps],[self.y0-self.r-self.eps,self.y0+self.r+self.eps]]

    def phi(self, pre, xy):
        """Level set function for the circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        x,y=xy
        return -self.r**2+(x-self.x0)**2+(y-self.y0)**2

    def phi_construct(self, pre, xy):
        """Level set function for the circle domain (for creation of the Omega domain)

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        return self.phi(pre, xy)
    
    def levelset(self,X):
        """Level set function for the circle domain

        :param X: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        return self.phi(torch, X.T)
    
    def levelset_construct(self,X):
        """Level set function for the circle domain (for creation of the Omega domain)

        :param X: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        return self.phi_construct(torch, X.T)
    
    def call_Omega(self, pre, xy):
        """Returns True if (x,y) is in the Omega domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :return: True if (x,y) is in the Omega domain
        """
        return self.phi_construct(pre,xy)<0
    
    def Omega_bool(self, x, y):
        """Returns True if (x,y) is in the Omega domain

        :param x: x coordinate
        :param y: y coordinate
        :return: True if (x,y) is in the Omega domain
        """
        xy = (x,y)
        return self.call_Omega(None, xy)

    def u_ex(self, pre, xy, mu):
        """Analytical solution for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Analytical solution evaluated at (x,y)
        """
        x,y=xy
        S,f,p = mu
        print("mu :",mu)
        return S * pre.sin(1/(self.r**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2))

    def u_ex_prime(self, pre, xy, mu):
        """First derivative of the analytical solution for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: First derivative of the analytical solution evaluated at (x,y)
        """
        x,y=xy
        S,f,p = mu
        du_dx = pre.pi*S*(2*x - 2*self.x0)*pre.cos(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**2
        du_dy = pre.pi*S*(2*y - 2*self.y0)*pre.cos(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**2
        return du_dx,du_dy

    def u_ex_prime2(self, pre, xy, mu):
        """Second derivative of the analytical solution for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Second derivative of the analytical solution evaluated at (x,y)
        """
        x,y=xy
        S,f,p = mu
        du_dxx = 2*pre.pi*S*pre.cos(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**2 - pre.pi**2*S*(2*x - 2*self.x0)**2*pre.sin(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**4
        du_dyy = 2*pre.pi*S*pre.cos(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**2 - pre.pi**2*S*(2*y - 2*self.y0)**2*pre.sin(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**4
        return du_dxx,du_dyy

    def f(self, pre, xy, mu):
        """Right hand side of the PDE for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Right hand side of the PDE evaluated at (x,y)
        """
        x,y=xy
        S,f,p = mu
        return 4/(self.r**4)*pre.pi**2*S*((x-self.x0)**2+(y-self.y0)**2)*pre.sin(1/(self.r**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2)) - \
                4/(self.r**2)*pre.pi*S*pre.cos(1/(self.r**2)*pre.pi*((x-self.x0)**2+(y-self.y0)**2))

    def g(self, pre, xy, mu):
        """Boundary condition for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Boundary condition evaluated at (x,y)
        """
        return 0*torch.ones_like(xy[0])

class Square():
    def __init__(self):
        """Square domain with side length 1
        """
        self.eps = 0.5
        self.domain_O = [[0-self.eps,1+self.eps],[0-self.eps,1+self.eps]]

    def phi(self, pre, xy):
        """Level set function for the square domain

        :param pre: Preconditioner 
        :param xy: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        x,y=xy
        return x*(1-x)*y*(1-y)
    
    def phi_construct(self, pre, xy):
        """Level set function for the square domain (for creation of the Omega domain)

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        # return np.linalg.norm(np.array(xy)-0.5,np.inf,axis=0)-0.5
        return torch.linalg.norm(xy-0.5,float('inf'),axis=0)-0.5

    def levelset(self,X):
        """Level set function for the square domain

        :param X: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        return self.phi(torch, X.T)
    
    def levelset_construct(self,X):
        """Level set function for the square domain (for creation of the Omega domain)

        :param X: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        return self.phi_construct(torch, X.T)
    
    def call_Omega(self, pre, xy):
        """Returns True if (x,y) is in the Omega domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :return: True if (x,y) is in the Omega domain
        """
        return self.phi_construct(pre,xy)<0
    
    def Omega_bool(self, x, y):
        """Returns True if (x,y) is in the Omega domain

        :param x: x coordinate
        :param y: y coordinate
        :return: True if (x,y) is in the Omega domain
        """
        xy = (x,y)
        return self.call_Omega(None, xy)

    def u_ex(self, pre, xy, mu):
        """Analytical solution for the Square domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S,f,p) parameters
        :return: Analytical solution evaluated at (x,y)
        """
        x,y=xy
        S, f, p = mu
        return S*pre.sin(2*pre.pi*f*x + p) * pre.sin(2*pre.pi*f*y + p)

    def u_ex_prime(self, pre, xy, mu):
        """First derivative of the analytical solution for the Square domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S,f,p) parameters
        :return: First derivative of the analytical solution evaluated at (x,y)
        """
        x,y=xy
        S,f,p = mu
        du_dx = 2*pre.pi*S*f*pre.sin(2*pre.pi*f*y + p)*pre.cos(2*pre.pi*f*x + p)
        du_dy = pre.pi*S*(2*y - 2*self.y0)*pre.cos(pre.pi*((x - self.x0)**2 + (y - self.y0)**2)/self.r**2)/self.r**2
        return du_dx,du_dy

    def u_ex_prime2(self, pre, xy, mu):
        """Second derivative of the analytical solution for the Square domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S,f,p) parameters
        :return: Second derivative of the analytical solution evaluated at (x,y)
        """
        x,y=xy
        S,f,p = mu
        du_dxx = -4*pre.pi**2*S*f**2*pre.sin(2*pre.pi*f*x + p)*pre.sin(2*pre.pi*f*y + p)
        du_dyy = -4*pre.pi**2*S*f**2*pre.sin(2*pre.pi*f*x + p)*pre.sin(2*pre.pi*f*y + p)
        return du_dxx,du_dyy

    def f(self, pre, xy, mu):
        """Right hand side of the PDE for the Square domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S,f,p) parameters
        :return: Right hand side of the PDE evaluated at (x,y)
        """
        x,y=xy
        S, f, p = mu
        return 8*pre.pi**2*S*f**2*pre.sin(2*pre.pi*f*x + p)*pre.sin(2*pre.pi*f*y + p)
    
    def g(self, pre, xy, mu):
        """Boundary condition for the Square domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S,f,p) parameters
        :return: Boundary condition evaluated at (x,y)
        """
        return 0*torch.ones_like(xy[0])
