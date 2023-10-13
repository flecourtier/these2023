import torch
from scimba.pinns.pdes import AbstractPDEx
from modules.data import *

x0,y0,r,domain_O = domain_parameters()
S = 0.5

class Poisson2D(AbstractPDEx):
    def __init__(self, space_domain, use_levelset=False):
        super().__init__(
            nb_unknowns=1,
            space_domain=space_domain,
            nb_parameters=1,
            parameter_domain=[[S, S+0.000001]],
        )

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
        # f = 4/(r**4)*torch.pi**2*alpha*((x1-x0)**2+(x2-y0)**2)*torch.sin(1/(r**2)*torch.pi*((x1-x0)**2+(x2-y0)**2)) - \
        #     4/(r**2)*torch.pi*alpha*torch.cos(1/(r**2)*torch.pi*((x1-x0)**2+(x2-y0)**2))
        f = call_F(torch, X, alpha, None, None)
        return u_xx + u_yy + f

    def bc_mul(self, x, mu):
        if self.use_levelset:
            x1, x2 = self.get_coordinates(x)
            X = torch.stack([x1,x2])
            return call_phi(torch, X)
        else:
            return 1. 

    def reference_solution(self, x, mu):
        x1, x2 = self.get_coordinates(x)
        X = torch.stack([x1,x2])
        alpha = self.get_parameters(mu)
        # return alpha * torch.sin(1/(r**2)*torch.pi*((x1-x0)**2+(x2-y0)**2))
        return call_Y_true(torch, X, alpha, None, None)