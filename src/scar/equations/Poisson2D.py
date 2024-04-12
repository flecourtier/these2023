from scar.problem.Problem import *

import torch
# from scimba.pinns import domain
from scimba.equations import domain, pdes

# from scimba.pinns.pdes import AbstractPDEx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Poisson2D(pdes.AbstractPDEx):
    def __init__(self, space_domain, problem): # space_domain : SignedDistanceBasedDomain
        """Represents the Poisson equation in 2D

        :param problem: Problem Considered - domain/solution considered (see modules.problems)
        :param parameter_domain: Problem parameters
        :param sampling_on: Collocation points are sampled on the Omega domain or on the O_cal domain, defaults to "Omega"
        :param impose_exact_bc: Choice to learn u (False) or u=phi*w (True), , defaults to False
        """
        nb_parameters = len(problem.parameter_domain)
        super().__init__(
            nb_unknowns=1,
            space_domain=space_domain,
            nb_parameters=nb_parameters,
            parameter_domain=problem.parameter_domain,
        )
        self.problem = problem

        self.first_derivative = True
        self.second_derivative = True

    def make_data(self, n_data):
        pass

    def bc_residual(self, w, x, mu, **kwargs):
        """Boundary conditions residual (Dirichlet)

        :param w: Solution
        :param x: Coordinates of the points
        :param mu: Parameters
        :return: Boundary conditions residual (Dirichlet)
        """
        u = self.get_variables(w)
        return u

    def residual(self, w, x, mu, **kwargs):
        """Residual of the PDE (Poisson equation)

        :param w: Solution
        :param x: Coordinates of the points
        :param mu: Parameters
        :return: Residual of the PDE (Poisson equation)
        """
        x1, x2 = x.get_coordinates()
        X = torch.stack([x1,x2])
        alpha = self.get_parameters(mu)
        u_xx = self.get_variables(w, "w_xx")
        u_yy = self.get_variables(w, "w_yy")
        f = self.problem.f(torch, X, alpha)
        # print("u_xx", u_xx.shape)
        # print("u_yy", u_yy.shape)
        # print("f", f.shape)
        return u_xx + u_yy + f
    
    def post_processing(self, x, mu, w):
        x1, x2 = x.get_coordinates()
        mul = self.space_domain.large_domain.sdf(x)
        
        X = torch.stack([x1,x2])
        alpha = self.get_parameters(mu)
        add = self.problem.g(torch, X, alpha)
        
        return mul * w + add
    
    def get_mul(self,x):
        return self.space_domain.sdf(x)

    def reference_solution(self, x, mu):
        """Reference solution (can be exact solution)

        :param x: Coordinates of the points
        :param mu: Parameters
        :return: Reference solution
        """
        # si self.problem.u_ex existe
        if hasattr(self.problem, "u_ex"):
            x1, x2 = x.get_coordinates()
            X = torch.stack([x1,x2])
            alpha = self.get_parameters(mu)
            return self.problem.u_ex(torch, X, alpha)
        else:
            return x