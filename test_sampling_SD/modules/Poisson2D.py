import torch
from scimba.pinns import domain
from scimba.pinns.pdes import AbstractPDEx
from modules.problems import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Poisson2D(AbstractPDEx):
    def __init__(self, problem, parameter_domain, use_levelset=False, space_domain=None):
        if space_domain is None:
            space_domain = domain.SignedDistanceBasedDomain(2, problem.domain_O, problem.levelset)
        super().__init__(
            nb_unknowns=1,
            space_domain=space_domain,
            nb_parameters=1,
            parameter_domain=parameter_domain,
        )

        self.problem = problem
        self.first_derivative = True
        self.second_derivative = True
        self.use_levelset = use_levelset

    def make_data(self, n_data):
        pass

    def bc_residual(self, w, x, mu, **kwargs):
        u = self.get_variables(w)
        return u

    def residual(self, w, x, mu, **kwargs):
        x1, x2 = self.get_coordinates(x)
        X = torch.stack([x1,x2])
        alpha = self.get_parameters(mu)
        u_xx = self.get_variables(w, "w_xx")
        u_yy = self.get_variables(w, "w_yy")
        f = self.problem.f(torch, X, alpha, None, None)
        return u_xx + u_yy + f

    def bc_mul(self, x, mu):
        if self.use_levelset:
            x1, x2 = self.get_coordinates(x)
            X = torch.stack([x1,x2])
            return self.problem.phi(torch, X)
        else:
            return 1. 
    
    def bc_add(self, x, mu, w):
        x1, x2 = self.get_coordinates(x)
        X = torch.stack([x1,x2])
        alpha = self.get_parameters(mu)
        return self.problem.g(torch, X, alpha, None, None)

    def reference_solution(self, x, mu):
        x1, x2 = self.get_coordinates(x)
        X = torch.stack([x1,x2])
        alpha = self.get_parameters(mu)
        return self.problem.u_ex(torch, X, alpha, None, None)

class Poisson2D_fixed(Poisson2D):
    def __init__(self, problem, use_levelset=False):
        S = 0.5
        parameter_domain = [[S, S+0.000001]]
        super().__init__(problem, parameter_domain, use_levelset)

class Poisson2D_f(Poisson2D):
    def __init__(self, problem, use_levelset=False):
        parameter_domain = [[0.1, 1.0]]
        super().__init__(problem, parameter_domain, use_levelset)

class Poisson2D_fixed_carre(Poisson2D):
    def __init__(self, problem, use_levelset=False):
        S = 0.5
        parameter_domain = [[S, S+0.000001]]
        space_domain = domain.SquareDomain(2, problem.domain_O)
        super().__init__(problem, parameter_domain, use_levelset, space_domain)
