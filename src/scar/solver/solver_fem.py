# homogeneous = True
cd = "homo"
print_time=False

###########
# Imports #
###########

from scar.solver.fenics_expressions import *
from scar.geometry import Geometry

from dolfin import *
import dolfin as df
import mshr
import time
import numpy as np

parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["representation"] = "uflacs"
# parameters["form_compiler"]["quadrature_degree"] = 10

#######
# FEM #
#######

class FEMSolver():
    def __init__(self,nb_cell,params,cas):
        self.N = nb_cell
        self.params = params
        self.pb_considered = cas.problem
        self.sdf_considered = cas.sd_function
        self.form_considered = cas.form

        self.times_fem = {}
        self.times_corr_add = {}
        self.mesh,self.V,self.dx = self.__create_FEM_domain()

    def __create_FEM_domain(self):
        # check if pb_considered is instance of Circle class
        if isinstance(self.form_considered, Geometry.Circle):
            domain = mshr.Circle(Point(self.pb_considered.x0, self.pb_considered.y0), self.pb_considered.r)
        else:
            raise Exception("Problem not implemented")

        nb_vert = self.N+1
            
        domain_O = np.array(self.sdf_considered.bound_box)
        mesh_macro = RectangleMesh(Point(domain_O[0,0], domain_O[1,0]), Point(domain_O[0,1], domain_O[1,1]), nb_vert - 1, nb_vert - 1)
        h_macro = mesh_macro.hmax()
        H = int(nb_vert/3)
        mesh = mshr.generate_mesh(domain,H)
        h = mesh.hmax()
        while h > h_macro:
            H += 1
            start = time.time()
            mesh = mshr.generate_mesh(domain,H)
            end = time.time()
            h = mesh.hmax()

        if print_time:
            print("Time to generate mesh: ", end-start)
        self.times_fem["mesh"] = end-start
        self.times_corr_add["mesh"] = end-start
        
        V = FunctionSpace(mesh, "CG", 1)
        dx = Measure("dx", domain=mesh)

        return mesh, V, dx

    def fem(self, i):
        boundary = "on_boundary"

        params = self.params[i]
        f_expr = FExpr(params, degree=10, domain=self.mesh, pb_considered=self.pb_considered)
        u_ex = UexExpr(params, degree=10, domain=self.mesh, pb_considered=self.pb_considered)
        # phi = PhiExpr(degree=10,domain=self.mesh, sdf_considered=self.sdf_considered)
        
        if cd=="homo":
            g = Constant("0.0")
        # else:
        #     g = u_ex*(Constant("1.0")+phi)
        bc = DirichletBC(self.V, g, boundary)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        
        start = time.time()

        a = inner(grad(u), grad(v)) * self.dx
        l = f_expr * v * self.dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)

        end = time.time()

        if print_time:
            print("Time to assemble the matrix : ",end-start)
        self.times_fem["assemble"] = end-start

        sol = Function(self.V)

        start = time.time()
        solve(A,sol.vector(),L)
        # solve(a==l, sol, bcs=bc)
        end = time.time()

        if print_time:
            print("Time to solve the system :",end-start)
        self.times_fem["solve"] = end-start

        norme_L2 = (assemble((((u_ex - sol)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))

        return sol,norme_L2

    def corr_add(self, i, phi_tild):
        boundary = "on_boundary"

        params = self.params[i]
        f_expr = FExpr(params, degree=10, domain=self.mesh, pb_considered=self.pb_considered)
        u_ex = UexExpr(params, degree=10, domain=self.mesh, pb_considered=self.pb_considered)

        f_tild = f_expr + div(grad(phi_tild))

        g = Constant(0.0)
        bc = DirichletBC(self.V, g, boundary)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        start = time.time()
        a = inner(grad(u), grad(v)) * self.dx
        l = f_tild * v * self.dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)

        end = time.time()

        if print_time:
            print("Time to assemble the matrix : ",end-start)
        self.times_corr_add["assemble"] = end-start

        C_tild = Function(self.V)

        start = time.time()
        solve(A,C_tild.vector(),L)
        end = time.time()

        if print_time:
            print("Time to solve the system :",end-start)
        self.times_corr_add["solve"] = end-start

        sol = C_tild + phi_tild

        norme_L2 = (assemble((((u_ex - sol)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))
        
        return sol,C_tild,norme_L2