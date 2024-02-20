import numpy as np
import torch
from modules.problem import Geometry
from modules.problem.SDFunction import SDCircle

class TrigSolOnCircle:
    def __init__(self,circle:Geometry.Circle):
        self.x0,self.y0 = circle.x0,circle.y0
        self.r = circle.r

        S, f, p = (0.5, 1, 0.)
        eps = 0.000001
        self.parameter_domain = [[S, S+eps],[f, f+eps],[p, p+eps]]

    def u_ex(self, pre, xy, mu):
        """Analytical solution for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Analytical solution evaluated at (x,y)
        """
        x,y=xy
        S,f,p = mu
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

class WSolOnCircle:
    def __init__(self,circle:Geometry.Circle,sdf=SDCircle):
        self.sdf = sdf(circle)
        self.x0,self.y0 = circle.x0,circle.y0
        self.r = circle.r
        
        self.parameter_domain = []

    def u_ex(self, pre, xy, mu):
        """Analytical solution for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Analytical solution evaluated at (x,y)
        """
        x,y=xy
        return self.sdf.phi(pre,xy)*pre.sin(x)*pre.exp(y)

    def u_ex_prime(self, pre, xy, mu):
        """First derivative of the analytical solution for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: First derivative of the analytical solution evaluated at (x,y)
        """
        x,y=xy
        du_dx = (2*x - 2*self.x0)*pre.exp(y)*pre.sin(x) + (-self.r**2 + (x - self.x0)**2 + (y - self.y0)**2)*pre.exp(y)*pre.cos(x)
        du_dy = (2*y - 2*self.y0)*pre.exp(y)*pre.sin(x) + (-self.r**2 + (x - self.x0)**2 + (y - self.y0)**2)*pre.exp(y)*pre.sin(x)
        return du_dx,du_dy

    def u_ex_prime2(self, pre, xy, mu):
        """Second derivative of the analytical solution for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Second derivative of the analytical solution evaluated at (x,y)
        """
        x,y=xy
        du_dxx = 2*(2*x - 2*self.x0)*pre.exp(y)*pre.cos(x) - (-self.r**2 + (x - self.x0)**2 + (y - self.y0)**2)*pre.exp(y)*pre.sin(x) + 2*pre.exp(y)*pre.sin(x)
        du_dyy = 2*(2*y - 2*self.y0)*pre.exp(y)*pre.sin(x) + (-self.r**2 + (x - self.x0)**2 + (y - self.y0)**2)*pre.exp(y)*pre.sin(x) + 2*pre.exp(y)*pre.sin(x)
        return du_dxx,du_dyy

    def f(self, pre, xy, mu):
        """Right hand side of the PDE for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Right hand side of the PDE evaluated at (x,y)
        """
        x,y=xy
        return -2*(2*x - 2*self.x0)*pre.exp(y)*pre.cos(x) - 2*(2*y - 2*self.y0)*pre.exp(y)*pre.sin(x) - 4*pre.exp(y)*pre.sin(x)
    
    def g(self, pre, xy, mu):
        """Boundary condition for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Boundary condition evaluated at (x,y)
        """
        return 0*torch.ones_like(xy[0])


class UnknownSolForMVP:
    def __init__(self, form:Geometry.ParametricCurves):
        self.parameter_domain = []

    def f(self, pre, xy, mu):
        return torch.ones_like(xy[0])

    def g(self, pre, xy, mu):
        """Boundary condition for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Boundary condition evaluated at (x,y)
        """
        return 0*torch.ones_like(xy[0])