from modules.data import *
from solver import *

x0,y0,r,domain_O = domain_parameters()
homogeneous = True

from dolfin import *
import dolfin as dol
from dolfin.function.expression import (
    BaseExpression,
    _select_element,
    _InterfaceExpression,
)
import multiphenics as mph
import mshr

parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["representation"] = "uflacs"

#################
# Solver PhiFEM #
#################

polV=1
polPhi=2
sigma_stab=20.0
gamma=1.0 

class PhiFemSolver(Solver):
    def __init__(self, nb_cell, params):
        super().__init__(nb_cell, params)
    
    # method direct in the non-homogeneous case
    def fem(self, i):
        # parameter of the ghost penalty
        sigma_stab = 1.

        params = self.params[i]
        f_expr = FExpr(params, degree=6, domain=self.mesh)
        y_true = UexExpr(params, degree=8, domain=self.mesh)

        # V_phi = FunctionSpace(self.mesh, "CG", polPhi)
        phi = PhiExpr(degree=polPhi, domain=self.mesh)
        # phi = interpolate(phi, V_phi)         
 
        a = (
            inner(grad(phi * self.u), grad(phi * self.v)) * self.dx
            - dot(inner(grad(phi * self.u), self.n), phi * self.v) * self.ds
            # stab terms
            + sigma_stab
            * avg(self.h)
            * dot(
                jump(grad(phi * self.u), self.n),
                jump(grad(phi * self.v), self.n),
            )
            * self.dS(1) #facets ghost
            + sigma_stab
            * self.h**2
            * inner(
                div(grad(phi * self.u)),
                div(grad(phi * self.v)),
            )
            * self.dx(1) #cells ghost
        )

        L = (
            f_expr * self.v * phi * self.dx
            # stab terms
            - sigma_stab
            * self.h**2
            * inner(f_expr, div(grad(phi * self.v)))
            * self.dx(1) #cells ghost
        )

        # Define solution function
        w = Function(self.V)
        solve(a == L, w)  # , solver_parameters={'linear_solver': 'mumps'})

        sol = phi * w
        
        norm_L2 = (assemble((((y_true - sol)) ** 2) * self.dx) ** (0.5)) / (assemble((((y_true)) ** 2) * self.dx) ** (0.5))

        return sol,norm_L2
    
    def fem_several(self):
        sols = []
        normes = []
        nb = len(self.params)
        for i in range(nb):
            print(f"{i}/{nb}:", end="")
            sol,norm = self.fem(i)
            sols.append(sol)
            normes.append(norm)

        return sols,normes

    def corr_mult(self, i, phi_tild):
        """To solve the Laplace Problem for one parameters with the correction by multiplication.
            We consider the problem : phi_tild*C

        :param i: Index of the parameter.
        :param phi_tild: FEniCS expression for the disturbed solution.
        :return: L2 norm of the error.
        """    
        g = Constant("0.0")    
        params = self.params[i]
        f_expr = FExpr(params, degree=6, domain=self.mesh)
        u_ex = UexExpr(params, degree=10, domain=self.mesh)

        a = (
            inner(grad((phi_tild-g) * self.u), grad((phi_tild-g) * self.v)) * self.dx
            - dot(inner(grad((phi_tild-g) * self.u), self.n), (phi_tild-g) * self.v) * self.ds
            # stab terms
            + sigma_stab
            * avg(self.h)
            * dot(
                jump(grad((phi_tild-g) * self.u), self.n),
                jump(grad((phi_tild-g) * self.v), self.n),
            )
            * self.dS(1) #facets ghost
            + sigma_stab
            * self.h**2
            * inner(
                div(grad((phi_tild-g) * self.u)),
                div(grad((phi_tild-g) * self.v)),
            )
            * self.dx(1) #cells ghost
        )

        L = (
            f_expr * self.v * (phi_tild-g) * self.dx
            # stab terms
            - sigma_stab
            * self.h**2
            * inner(f_expr, div(grad((phi_tild-g) * self.v)))
            * self.dx(1) #cells ghost
        )

        # Define solution function
        C_h = Function(self.V)
        solve(a == L, C_h)  # , solver_parameters={'linear_solver': 'mumps'})

        sol = (phi_tild-g) * C_h 

        norm_L2 = (assemble((((u_ex - sol)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))

        return sol, C_h, norm_L2
    
    def corr_add(self, i, phi_tild):
        """To solve the Laplace Problem for one parameters with the correction by addition.
            We consider the problem : phi_tild+C

        :param i: Index of the parameter.
        :param phi_tild: FEniCS expression for the disturbed solution.
        :return: L2 norm of the error.
        """     

        params = self.params[i]
        f_expr = FExpr(params, degree=6, domain=self.mesh)
        u_ex = UexExpr(params, degree=10, domain=self.mesh)
        phi = PhiExpr(degree=6, domain=self.mesh)
       
        f_tild = f_expr + div(grad(phi_tild))

        a = (
            inner(grad(phi * self.u), grad(phi * self.v)) * self.dx
            - dot(inner(grad(phi * self.u), self.n), phi * self.v) * self.ds
            # stab terms
            + sigma_stab
            * avg(self.h)
            * dot(
                jump(grad(phi * self.u), self.n),
                jump(grad(phi * self.v), self.n),
            )
            * self.dS(1) #facets ghost
            + sigma_stab
            * self.h**2
            * inner(
                div(grad(phi * self.u)),
                div(grad(phi * self.v)),
            )
            * self.dx(1) #cells ghost
        )

        L = (
            f_tild * self.v * phi * self.dx
            # stab terms
            - sigma_stab
            * self.h**2
            * inner(f_tild, div(grad(phi * self.v)))
            * self.dx(1) #cells ghost
        )

        # Define solution function
        C_h = Function(self.V)
        solve(a == L, C_h)  # , solver_parameters={'linear_solver': 'mumps'})

        C_tild = phi*C_h
        sol = C_tild + phi_tild

        norm_L2 = (assemble((((u_ex - sol)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))
        
        return sol, C_h, norm_L2
    
    def corr_add_IPP(self, i, phi_tild):
        """To solve the Laplace Problem for one parameters with the correction by addition.
            We consider the problem : phi_tild+C

        :param i: Index of the parameter.
        :param phi_tild: FEniCS expression for the disturbed solution.
        :return: L2 norm of the error.
        """   
        params = self.params[i]   
        f_expr = FExpr(params, degree=6, domain=self.mesh)
        u_ex = UexExpr(params, degree=10, domain=self.mesh)
        phi = PhiExpr(degree=6, domain=self.mesh)

        f_tild = f_expr + div(grad(phi_tild))

        a = (
            inner(grad(phi * self.u), grad(phi * self.v)) * self.dx
            - dot(inner(grad(phi * self.u), self.n), phi * self.v) * self.ds
            # stab terms
            + sigma_stab
            * avg(self.h)
            * dot(
                jump(grad(phi * self.u), self.n),
                jump(grad(phi * self.v), self.n),
            )
            * self.dS(1) #facets ghost
            + sigma_stab
            * self.h**2
            * inner(
                div(grad(phi * self.u)),
                div(grad(phi * self.v)),
            )
            * self.dx(1) #cells ghost
        )

        L = (
            f_expr * self.v * phi * self.dx
            - inner(grad(phi_tild), grad(phi * self.v)) * self.dx
            + dot(inner(grad(phi_tild), self.n), phi * self.v) * self.ds
            # stab terms
            - sigma_stab
            * self.h**2
            * inner(f_tild, div(grad(phi * self.v)))
            * self.dx(1) #cells ghost
        )

        # Define solution function
        C_h = Function(self.V)
        solve(a == L, C_h)  # , solver_parameters={'linear_solver': 'mumps'})

        C_tild = phi*C_h
        sol = C_tild + phi_tild

        norm_L2 = (assemble((((u_ex - sol)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))

        return sol, C_h, norm_L2