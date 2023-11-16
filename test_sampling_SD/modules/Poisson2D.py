import torch
from scimba.pinns import domain
from scimba.pinns.pdes import AbstractPDEx
from modules.problems import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Poisson2D(AbstractPDEx):
    def __init__(self, problem, parameter_domain, nb_parameters=1, use_levelset=False, space_domain=None):
        if space_domain is None:
            space_domain = domain.SignedDistanceBasedDomain(2, problem.domain_O, problem.levelset)
        super().__init__(
            nb_unknowns=1,
            space_domain=space_domain,
            nb_parameters=nb_parameters,
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
        f = self.problem.f(torch, X, alpha)
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
        return self.problem.g(torch, X, alpha)

    def reference_solution(self, x, mu):
        x1, x2 = self.get_coordinates(x)
        X = torch.stack([x1,x2])
        alpha = self.get_parameters(mu)
        return self.problem.u_ex(torch, X, alpha)

#####
# Omega = Circle
#####

# on fixe S=0.5
class Poisson2D_fixed(Poisson2D):
    def __init__(self, problem, use_levelset=False):
        S = 0.5
        parameter_domain = [[S, S+0.000001]]
        super().__init__(problem, parameter_domain, use_levelset)

# on fait varier S entre 0.1 et 1
class Poisson2D_f(Poisson2D):
    def __init__(self, problem, use_levelset=False):
        parameter_domain = [[0.1, 1.0]]
        super().__init__(problem, parameter_domain, use_levelset)

# on fixe S=0.5 et on entraîne sur un carré
class Poisson2D_fixed_carre(Poisson2D):
    def __init__(self, problem, use_levelset=False):
        S = 0.5
        parameter_domain = [[S, S+0.000001]]
        space_domain = domain.SquareDomain(2, problem.domain_O)
        super().__init__(problem, parameter_domain, use_levelset, space_domain)

#####
# Omega = Square
#####

# on fixe S=0.5, f=1, p=0
class Poisson2D_fixed2(Poisson2D):
    def __init__(self, problem, use_levelset=False):
        S = 0.5
        f = 1
        p = 0.
        parameter_domain = [[S, S+0.000001],[f, f+0.000001],[p, p+0.000001]]
        super().__init__(problem, parameter_domain, use_levelset=use_levelset, nb_parameters=3)

# on fixe S=0.5, f=1, p=0
class Poisson2D_fixed2_carre(Poisson2D):
    def __init__(self, problem, use_levelset=False):
        S = 0.5
        f = 1
        p = 0.
        parameter_domain = [[S, S+0.000001],[f, f+0.000001],[p, p+0.000001]]
        space_domain = domain.SquareDomain(2, problem.domain_O)
        print("domain_O", problem.domain_O)
        super().__init__(problem, parameter_domain, use_levelset=use_levelset, nb_parameters=3, space_domain=space_domain)