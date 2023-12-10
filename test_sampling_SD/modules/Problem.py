import numpy as np
import torch
from modules.Geometry import *

class Circle_Solution1(Circle):
    def __init__(self):
        print(self.__class__.__name__)

        super().__init__()

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

class Circle_Solution2(Circle):
    def __init__(self):
        print(self.__class__.__name__)
        super().__init__()

    def u_ex(self, pre, xy, mu):
        """Analytical solution for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Analytical solution evaluated at (x,y)
        """
        x,y=xy
        return self.phi(pre,xy)*pre.sin(x)*pre.exp(y)

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

class Square_Solution1(Square):
    def __init__(self):
        print(self.__class__.__name__)
        super().__init__()
        
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
        du_dy = 2*pre.pi*S*f*pre.sin(2*pre.pi*f*x + p)*pre.cos(2*pre.pi*f*y + p)
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


class Random_domain_Solution1(Random_domain):
    def __init__(self):
        print(self.__class__.__name__)
        super().__init__()

    def u_ex(self, pre, xy, mu):
        """Analytical solution for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Analytical solution evaluated at (x,y)
        """
        x,y=xy
        return self.phi(pre,xy)*pre.sin(x)*pre.exp(y)

    # def u_ex_prime(self, pre, xy, mu):
    #     """First derivative of the analytical solution for the Circle domain

    #     :param pre: Preconditioner
    #     :param xy: (x,y) coordinates
    #     :param mu: (S) parameter
    #     :return: First derivative of the analytical solution evaluated at (x,y)
    #     """
    #     x,y=xy
    #     du_dx = (2*x - 2*self.x0)*pre.exp(y)*pre.sin(x) + (-self.r**2 + (x - self.x0)**2 + (y - self.y0)**2)*pre.exp(y)*pre.cos(x)
    #     du_dy = (2*y - 2*self.y0)*pre.exp(y)*pre.sin(x) + (-self.r**2 + (x - self.x0)**2 + (y - self.y0)**2)*pre.exp(y)*pre.sin(x)
    #     return du_dx,du_dy

    # def u_ex_prime2(self, pre, xy, mu):
    #     """Second derivative of the analytical solution for the Circle domain

    #     :param pre: Preconditioner
    #     :param xy: (x,y) coordinates
    #     :param mu: (S) parameter
    #     :return: Second derivative of the analytical solution evaluated at (x,y)
    #     """
    #     x,y=xy
    #     du_dxx = 2*(2*x - 2*self.x0)*pre.exp(y)*pre.cos(x) - (-self.r**2 + (x - self.x0)**2 + (y - self.y0)**2)*pre.exp(y)*pre.sin(x) + 2*pre.exp(y)*pre.sin(x)
    #     du_dyy = 2*(2*y - 2*self.y0)*pre.exp(y)*pre.sin(x) + (-self.r**2 + (x - self.x0)**2 + (y - self.y0)**2)*pre.exp(y)*pre.sin(x) + 2*pre.exp(y)*pre.sin(x)
    #     return du_dxx,du_dyy

    # def f(self, pre, xy, mu):
    #     """Right hand side of the PDE for the Circle domain

    #     :param pre: Preconditioner
    #     :param xy: (x,y) coordinates
    #     :param mu: (S) parameter
    #     :return: Right hand side of the PDE evaluated at (x,y)
    #     """
    #     x,y=xy
    #     return -2*(2*x - 2*self.x0)*pre.exp(y)*pre.cos(x) - 2*(2*y - 2*self.y0)*pre.exp(y)*pre.sin(x) - 4*pre.exp(y)*pre.sin(x)
    
    # def g(self, pre, xy, mu):
    #     """Boundary condition for the Circle domain

    #     :param pre: Preconditioner
    #     :param xy: (x,y) coordinates
    #     :param mu: (S) parameter
    #     :return: Boundary condition evaluated at (x,y)
    #     """
    #     return 0*torch.ones_like(xy[0])
