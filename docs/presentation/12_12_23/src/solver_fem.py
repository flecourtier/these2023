from utils import *

config = read_json_file()

geometry = config["geometry"]
if geometry == "circle":
    from data_circle import *

    int_a = 0.0
    int_b = 1.0
elif geometry == "square":
    from data_square import *

    int_a = -0.5
    int_b = 1.5
else:
    raise ValueError("Geometry not recognized")

homogeneous = config["homogeneous"]
if homogeneous:
    cd="homo"
else:
    cd="non homo"

from solver import *

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

# #######
# # FEM #
# #######

class FEMSolver():
    def __init__(self,nb_cell,params,params_pert):
        self.N = nb_cell
        self.params = params
        self.params_pert = params_pert
        self.mesh,self.V,self.dx = self.__create_FEM_domain()

    def make_matrix(self,expr):
        nb_vert = self.N+1
        expr = expr.compute_vertex_values(self.mesh)
        expr = np.reshape(expr, [nb_vert, nb_vert])
        return expr

    def __create_FEM_domain(self):
        nb_vert = self.N+1
        if geometry=="circle":
            domain = mshr.Circle(Point(0.5, 0.5), sqrt(2.)/4.)
        elif geometry=="square":
            domain = mshr.Rectangle(Point(0., 0.), Point(1., 1.))

        mesh_macro = RectangleMesh(Point(int_a, int_a), Point(int_b, int_b), nb_vert - 1, nb_vert - 1)
        h_macro = mesh_macro.hmax()
        H = int(nb_vert/3)
        mesh = mshr.generate_mesh(domain,H)
        h = mesh.hmax()
        while h > h_macro:
            H += 1
            mesh = mshr.generate_mesh(domain,H)
            h = mesh.hmax()

        V = FunctionSpace(mesh, "CG", 1)
        dx = Measure("dx", domain=mesh)

        return mesh, V, dx

    def fem(self, i):
        boundary = "on_boundary"

        params = self.params[i]
        f_expr = FExpr(params, degree=10, domain=self.mesh)
        u_ex = UexExpr(params, degree=10, domain=self.mesh)

        phi = PhiExpr(degree=10,domain=self.mesh)
        phi_tild = Constant("1.0")
        
        if cd=="homo":
            g = Constant("0.0")
        else:
            g = u_ex*(Constant("1.0")+phi)
        bc = DirichletBC(self.V, g, boundary)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        a = inner(grad(phi_tild * u), grad(phi_tild * v)) * self.dx
        l = f_expr * phi_tild * v * self.dx

        C = Function(self.V)

        solve(a==l, C, bcs=bc)
        
        u_Corr = phi_tild * C

        norme_L2 = (assemble((((u_ex - u_Corr)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))

        return u_Corr,norme_L2
    
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
        
    def corr_mult(self, i, eps=0., m=0.):
        boundary = "on_boundary"

        params = self.params[i]
        params_pert = self.params_pert[i]
        f_expr = FExpr(params, degree=10, domain=self.mesh)
        u_ex = UexExpr(params, degree=10, domain=self.mesh)
        pert = PertExpr(params_pert, degree=10, domain=self.mesh)
        phi_tild = u_ex + eps*pert + m
        
        g = Constant("1.0")
        bc = DirichletBC(self.V, g, boundary)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        a = inner(grad(phi_tild * u), grad(phi_tild * v)) * self.dx
        l = f_expr * phi_tild * v * self.dx
        C = Function(self.V)

        solve(a==l, C, bcs=bc)
        
        u_Corr = phi_tild * C - m

        norme_L2 = (assemble((((u_ex - u_Corr)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))
        
        return u_Corr,C,norme_L2

    def corr_add(self, i, eps=0.):
        boundary = "on_boundary"

        params = self.params[i]
        params_pert = self.params_pert[i]
        f_expr = FExpr(params, degree=10, domain=self.mesh)
        u_ex = UexExpr(params, degree=10, domain=self.mesh)
        pert = PertExpr(params_pert, degree=10, domain=self.mesh)
        # u_ex = UexExpr(params, degree=4, domain=self.mesh)
        # pert = PertExpr(params_pert, degree=4, domain=self.mesh)
        phi_tild = u_ex + eps*pert
        f_tild = f_expr + div(grad(phi_tild))

        norm_phi_tild = (assemble((((u_ex - phi_tild)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))
        print(norm_phi_tild)

        g = Constant(0.0)
        bc = DirichletBC(self.V, g, boundary)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        a = inner(grad(u), grad(v)) * self.dx
        l = f_tild * v * self.dx
        C = Function(self.V)

        solve(a==l, C, bcs=bc)
        
        u_Corr = C + phi_tild

        norme_L2 = (assemble((((u_ex - u_Corr)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))
        
        return u_Corr,C,norme_L2
        # return norme_L2,norm_phi_tild

    def corr_add_IPP(self, i, eps=0.):
        boundary = "on_boundary"

        params = self.params[i]
        params_pert = self.params_pert[i]
        f_expr = FExpr(params, degree=10, domain=self.mesh)
        u_ex = UexExpr(params, degree=10, domain=self.mesh)
        pert = PertExpr(params_pert, degree=10, domain=self.mesh)
        phi_tild = u_ex + eps*pert
        f_tild = f_expr + div(grad(phi_tild))

        g = Constant(0.0)
        bc = DirichletBC(self.V, g, boundary)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        a = inner(grad(u), grad(v)) * self.dx
        l = f_expr * v * self.dx - inner(grad(phi_tild), grad(v)) * self.dx
        C = Function(self.V)

        solve(a==l, C, bcs=bc)
        
        u_Corr = C + phi_tild

        norme_L2 = (assemble((((u_ex - u_Corr)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))
        
        return u_Corr,C,norme_L2
