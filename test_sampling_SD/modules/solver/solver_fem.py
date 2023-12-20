from modules.solver.fenics_expressions import *

from modules.problem.Case import *
cas = Case("case.json")
pb_considered = cas.Problem

print_time = False

# homogeneous = True
cd = "homo"

from dolfin import *
import dolfin as df
import mshr
import time

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
    def __init__(self,nb_cell,params):
        self.N = nb_cell
        self.params = params
        self.times_fem = {}
        self.times_corr_add = {}
        self.mesh,self.V,self.dx = self.__create_FEM_domain()

    def __create_FEM_domain(self):
        # check if pb_considered is instance of Circle class
        if isinstance(pb_considered, Circle):
            domain = mshr.Circle(Point(pb_considered.x0, pb_considered.y0), pb_considered.r)
        elif isinstance(pb_considered, Square):
            domain = mshr.Rectangle(Point(0, 0), Point(1, 1))
        else:
            raise Exception("Problem not implemented")

        nb_vert = self.N+1
            
        domain_O = np.array(pb_considered.domain_O)
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
        f_expr = FExpr(params, degree=10, domain=self.mesh)
        u_ex = UexExpr(params, degree=10, domain=self.mesh)

        # phi = PhiExpr(degree=10,domain=self.mesh)
        
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
        f_expr = FExpr(params, degree=10, domain=self.mesh)
        u_ex = UexExpr(params, degree=10, domain=self.mesh)

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
        # solve(a==l, C_tild, bcs=bc)
        end = time.time()

        if print_time:
            print("Time to solve the system :",end-start)
        self.times_corr_add["solve"] = end-start

        sol = C_tild + phi_tild

        norme_L2 = (assemble((((u_ex - sol)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))
        
        return sol,C_tild,norme_L2
    
    # def corr_add_IPP(self, i, phi_tild):
    #     boundary = "on_boundary"

    #     params = self.params[i]
    #     f_expr = FExpr(params, degree=10, domain=self.mesh)
    #     u_ex = UexExpr(params, degree=10, domain=self.mesh)
        
    #     # f_tild = f_expr + div(grad(phi_tild))

    #     g = Constant(0.0)
    #     bc = DirichletBC(self.V, g, boundary)

    #     u = TrialFunction(self.V)
    #     v = TestFunction(self.V)
        
    #     # Resolution of the variationnal problem
    #     a = inner(grad(u), grad(v)) * self.dx
    #     l = f_expr * v * self.dx - inner(grad(phi_tild), grad(v)) * self.dx
    #     C = Function(self.V)

    #     solve(a==l, C, bcs=bc)
        
    #     u_Corr = C + phi_tild

    #     norme_L2 = (assemble((((u_ex - u_Corr)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))
        
    #     return u_Corr,C,norme_L2
    
    # def corr_mult(self, i, phi_tild, m=0.):
    #     boundary = "on_boundary"

    #     params = self.params[i]
    #     f_expr = FExpr(params, degree=10, domain=self.mesh)
    #     u_ex = UexExpr(params, degree=10, domain=self.mesh)

    #     phi_hat = phi_tild+m

    #     g = Constant("1.0")
    #     bc = DirichletBC(self.V, g, boundary)

    #     u = TrialFunction(self.V)
    #     v = TestFunction(self.V)
        
    #     # Resolution of the variationnal problem
    #     a = inner(grad(phi_hat * u), grad(phi_hat * v)) * self.dx
    #     l = f_expr * phi_hat * v * self.dx
    #     C = Function(self.V)

    #     solve(a==l, C, bcs=bc)
        
    #     u_Corr = phi_hat * C - m

    #     norme_L2 = (assemble((((u_ex - u_Corr)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))
        
    #     return u_Corr,C,norme_L2