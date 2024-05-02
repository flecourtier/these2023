import numpy as np
import torch
import mshr
import dolfin as df
from dolfin import *

from scar.geometry import Geometry2D
from scar.geometry.SDFunction import SDCircle
from scar.geometry.StandardMesh import overrefined_mesh
# from scar.geometry.PolygonalMesh import create_domain

class TrigSolOnCircle:
    def __init__(self,circle:Geometry2D.Circle):
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
    def __init__(self,circle:Geometry2D.Circle,sdf=SDCircle):
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
    def __init__(self, form:Geometry2D.ParametricCurves):
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
    
class ConstantForce:
    def __init__(self,form:Geometry2D.ParametricCurves):
        self.form = form
        self.parameter_domain = []

        if isinstance(self.form,Geometry2D.Circle):
            self.analytical_sol = True
            self.x0,self.y0 = self.form.x0,self.form.y0
            self.r = self.form.r
        else:
            self.analytical_sol = False

    def u_ex(self, pre, xy, mu):
        """Analytical solution for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Analytical solution evaluated at (x,y)
        """
        if self.analytical_sol:
            x,y=xy
            return -1./4.*((x-self.form.x0)**2+(y-self.form.y0)**2-self.form.r**2)
        else:
            return torch.ones_like(xy[0])

    def get_u_ref(self,mesh_form):
        V = FunctionSpace(mesh_form, "CG", 1)
        dx = Measure("dx", domain=mesh_form)

        g = Constant("0.0")
        bc = DirichletBC(V, g, "on_boundary")

        f_expr = Constant("1.0")

        u = TrialFunction(V)
        v = TestFunction(V)

        # Resolution of the variationnal problem
        a = inner(grad(u), grad(v)) * dx
        l = f_expr * v * dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)

        sol = Function(V)
        solve(A,sol.vector(),L)

        return sol

    # def u_ref(self):
    #     if not self.analytical_sol:
    #         domain = create_domain(self.form, n_bc_points=1000)
    #         mesh_form = mshr.generate_mesh(domain, 50)

    #         return self.get_u_ref(mesh_form)
    #     else:
    #         pass

    def u_ref(self,trainer,mesh_dir):
        # if not self.analytical_sol:
        mesh_ex = overrefined_mesh(self.form,trainer,mesh_dir)
        V_ex = FunctionSpace(mesh_ex, "CG", 1)
        u_ref = self.get_u_ref(mesh_ex)

        return mesh_ex,V_ex,u_ref
        # else:
        #     pass

    def u_ex_prime(self, pre, xy, mu):
        """First derivative of the analytical solution for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: First derivative of the analytical solution evaluated at (x,y)
        """
        pass

    def u_ex_prime2(self, pre, xy, mu):
        """Second derivative of the analytical solution for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Second derivative of the analytical solution evaluated at (x,y)
        """
        pass

    def f(self, pre, xy, mu):
        """Right hand side of the PDE for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Right hand side of the PDE evaluated at (x,y)
        """
        # return torch.ones_like(xy[0])
        return 1.0

    def g(self, pre, xy, mu):
        """Boundary condition for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Boundary condition evaluated at (x,y)
        """
        return 0*torch.ones_like(xy[0])