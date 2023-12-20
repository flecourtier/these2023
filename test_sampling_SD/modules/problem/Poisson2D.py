from modules.problem.Problem import *

import torch
from scimba.pinns import domain
from scimba.pinns.pdes import AbstractPDEx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Poisson2D(AbstractPDEx):
    def __init__(self, problem, parameter_domain, sampling_on="Omega", impose_exact_bc=True):
        """Represents the Poisson equation in 2D

        :param problem: Problem Considered - domain/solution considered (see modules.problems)
        :param parameter_domain: Problem parameters
        :param sampling_on: Collocation points are sampled on the Omega domain or on the O_cal domain, defaults to "Omega"
        :param impose_exact_bc: Choice to learn u (False) or u=phi*w (True), , defaults to False
        """
        if sampling_on=="Omega":
            space_domain = domain.SignedDistanceBasedDomain(2, problem.domain_O, problem.levelset_construct)
        elif sampling_on=="O_cal":
            space_domain = domain.SquareDomain(2, problem.domain_O)
        else:
            raise ValueError("sampling_on must be either 'Omega' or 'O_cal'")
        
        nb_parameters = len(parameter_domain)

        super().__init__(
            nb_unknowns=1,
            space_domain=space_domain,
            nb_parameters=nb_parameters,
            parameter_domain=parameter_domain,
        )

        self.problem = problem
        self.impose_exact_bc = impose_exact_bc

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
        x1, x2 = self.get_coordinates(x)
        X = torch.stack([x1,x2])
        alpha = self.get_parameters(mu)
        u_xx = self.get_variables(w, "w_xx")
        u_yy = self.get_variables(w, "w_yy")
        f = self.problem.f(torch, X, alpha)
        return u_xx + u_yy + f

    def bc_mul(self, x, mu):
        """Multiplicator term of the solution (phi in the the impose_exact_bc case)

        :param x: Coordinates of the points
        :param mu: Parameters
        :return: Multiplicator term of the solution
        """
        if self.impose_exact_bc:
            x1, x2 = self.get_coordinates(x)
            X = torch.stack([x1,x2])
            return self.problem.phi(torch, X)
        else:
            return 1. 
    
    def bc_add(self, x, mu, w):
        """Additive term of the solution

        :param x: Coordinates of the points
        :param mu: Parameters
        :param w: Parameters
        :return: Additive term of the solution
        """
        x1, x2 = self.get_coordinates(x)
        X = torch.stack([x1,x2])
        alpha = self.get_parameters(mu)
        return self.problem.g(torch, X, alpha)

    def reference_solution(self, x, mu):
        """Reference solution (can be exact solution)

        :param x: Coordinates of the points
        :param mu: Parameters
        :return: Reference solution
        """
        x1, x2 = self.get_coordinates(x)
        X = torch.stack([x1,x2])
        alpha = self.get_parameters(mu)
        return self.problem.u_ex(torch, X, alpha)
 
#####
# SingleProblem
#####

class SingleProblem(Poisson2D):
    def __init__(self, problem, sampling_on="Omega", impose_exact_bc=False):
        """Represents the Poisson equation in 2D with a single problem (fixed domain and fixed solution)

        :param problem: Problem Considered - domain/solution considered (see modules.problems)
        :param sampling_on: Collocation points are sampled on the Omega domain or on the O_cal domain, defaults to "Omega"
        :param impose_exact_bc: Choice to learn u (False) or u=phi*w (True), , defaults to False
        """
        S, f, p = (0.5, 1, 0.)
        eps = 0.000001
        parameter_domain = [[S, S+eps],[f, f+eps],[p, p+eps]]

        if isinstance(problem, Circle_Solution2) or isinstance(problem, Random_domain_Solution1):
            parameter_domain = []

        super().__init__(problem, parameter_domain, sampling_on=sampling_on, impose_exact_bc=impose_exact_bc)

#####
# VariedSolution (fixed domain, varied solution)
#####

class VariedSolution_S(Poisson2D):
    def __init__(self, problem, sampling_on="Omega", impose_exact_bc=False):
        """Represents the Poisson equation in 2D with a single problem (fixed domain and varied solution)

        :param problem: Problem Considered - domain/solution considered (see modules.problems)
        :param sampling_on: Collocation points are sampled on the Omega domain or on the O_cal domain, defaults to "Omega"
        :param impose_exact_bc: Choice to learn u (False) or u=phi*w (True), , defaults to False
        """
        S_ = [0.1, 1.0]
        f, p = (1,0.)
        parameter_domain = [S_,[f, f+0.000001],[p, p+0.000001]]
        super().__init__(problem, parameter_domain, sampling_on=sampling_on, impose_exact_bc=impose_exact_bc)