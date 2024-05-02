# homogeneous = True
cd = "homo"
print_time=False

###########
# Imports #
###########

from scar.solver.fenics_expressions import *
from scar.geometry import Geometry2D
from scar.geometry.StandardMesh import *
# from scar.geometry.PolygonalMesh import create_domain

from dolfin import *
import dolfin as df
import mshr
import time
import numpy as np
from pathlib import Path

parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["representation"] = "uflacs"
# parameters["form_compiler"]["quadrature_degree"] = 10

current = Path(__file__).parent.parent.parent.parent

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

    # def __create_FEM_domain(self):
    #     # check if pb_considered is instance of Circle class
    #     if isinstance(self.form_considered, Geometry2D.Circle):
    #         domain = mshr.Circle(Point(self.pb_considered.x0, self.pb_considered.y0), self.pb_considered.r)
    #     else:
    #         domain = create_domain(self.form_considered, n_bc_points=200)
    #         # raise Exception("Problem not implemented")

    #     nb_vert = self.N+1
            
    #     domain_O = np.array(self.sdf_considered.bound_box)
    #     mesh_macro = RectangleMesh(Point(domain_O[0,0], domain_O[1,0]), Point(domain_O[0,1], domain_O[1,1]), nb_vert - 1, nb_vert - 1)
    #     h_macro = mesh_macro.hmax()
    #     H = int(nb_vert/3)
    #     mesh = mshr.generate_mesh(domain,H)
    #     h = mesh.hmax()
    #     while h > h_macro:
    #         H += 1
    #         start = time.time()
    #         mesh = mshr.generate_mesh(domain,H)
    #         end = time.time()
    #         h = mesh.hmax()

    #     if print_time:
    #         print("Time to generate mesh: ", end-start)
    #     self.times_fem["mesh"] = end-start
    #     self.times_corr_add["mesh"] = end-start
        
    #     V = FunctionSpace(mesh, "CG", 1)
    #     dx = Measure("dx", domain=mesh)

    #     return mesh, V, dx

    def __create_FEM_domain(self):
        nb_vert = self.N+1
                
        domain_O = np.array(self.sdf_considered.bound_box)
        mesh_macro = RectangleMesh(Point(domain_O[0,0], domain_O[1,0]), Point(domain_O[0,1], domain_O[1,1]), nb_vert - 1, nb_vert - 1)
        h_macro = mesh_macro.hmax()

        # check if pb_considered is instance of Circle class
        if isinstance(self.form_considered, Geometry2D.Circle):
            domain = mshr.Circle(Point(self.pb_considered.x0, self.pb_considered.y0), self.pb_considered.r)
            
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
        else:
            form_trainer = self.sdf_considered.form_trainer
            form_dir_name = current / "networks" / "EikonalLap2D" / self.form_considered.__class__.__name__ / "meshes"
            # generate equivalent mesh
            
            # import matplotlib.pyplot as plt
            # mesh = overrefined_mesh(self.form_considered,form_trainer,str(form_dir_name)+"/",new_mesh=True)
            # print("hmax:",mesh.hmax())
            # plt.figure()
            # plot(mesh)
            # plt.show()
            
            # print("Standard mesh")
            
            # mesh = standard_mesh(self.form_considered,form_trainer,str(form_dir_name)+"/",0.01,0.05,n=101,new_mesh=True)
            # print("hmax:",mesh.hmax())
            # plt.figure()
            # plot(mesh)
            # plt.show()
            
            # print("AUTREEEEEEEEEEE")
            
            # mesh = standard_mesh(self.form_considered,form_trainer,str(form_dir_name)+"/",0.001,0.005,n=101,new_mesh=True)
            # print("hmax:",mesh.hmax())
            # plt.figure()
            # plot(mesh)
            # plt.show()
            
            hmin = h_macro/5
            hmax = h_macro
            h = 10*h_macro

            while h>h_macro:
                mesh = standard_mesh(self.form_considered,form_trainer,str(form_dir_name)+"/",hmin,hmax,n=101,new_mesh=True)
                h = mesh.hmax()
                hmin = hmin / 2
                hmax = hmax / 2
            print("hmin,hmax :",hmin,hmax)     
            print("h,h_macro :",h,h_macro)
        
        V = FunctionSpace(mesh, "CG", 1)
        dx = Measure("dx", domain=mesh)

        return mesh, V, dx
    

    def fem(self, i, get_error=True, analytical_sol=True):
        boundary = "on_boundary"

        params = self.params[i]
        f_expr = FExpr(params, degree=10, domain=self.mesh, pb_considered=self.pb_considered)
        if get_error:
            if analytical_sol:
                u_ex = UexExpr(params, degree=10, domain=self.mesh, pb_considered=self.pb_considered)
            else:
                u_ex = self.pb_considered.u_ref()
            
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

        norme_L2 = None
        if get_error:
            norme_L2 = (assemble((((u_ex - sol)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))

        return sol,norme_L2

    def corr_add(self, i, phi_tild, get_error=True, analytical_sol=True):
        boundary = "on_boundary"

        params = self.params[i]
        f_expr = FExpr(params, degree=10, domain=self.mesh, pb_considered=self.pb_considered)
        # f_expr = Constant(1.0)
        if get_error:
            if analytical_sol:
                u_ex = UexExpr(params, degree=10, domain=self.mesh, pb_considered=self.pb_considered)
            else:
                u_ex = self.pb_considered.u_ref()
        f_tild = f_expr + div(grad(phi_tild))
        
        # V_phi = phi_tild.function_space()
        # print(self.V==V_phi)
        
        
        # import matplotlib.pyplot as plt
        # plt.figure()
        # c = plot(f_tild, cmap="gist_ncar")
        # plt.colorbar(c)
        # plt.show()

        g = Constant(0.0)
        
        g = Function(self.V)
        phi_tild_inter = interpolate(phi_tild, self.V)
        g.vector()[:] = (phi_tild_inter.vector()[:])
        
        import matplotlib.pyplot as plt
        plt.figure()
        c = plot(g, cmap="gist_ncar")
        plt.colorbar(c)
        
        # g = -phi_tild
        
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

        norme_L2 = None
        if get_error:
            norme_L2 = (assemble((((u_ex - sol)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))
        
        return sol,C_tild,norme_L2
    
    def corr_add_IPP(self, i, phi_tild, get_error=True, analytical_sol=True):
        print("ICIIIIIII : corr_add_IPP")
        boundary = "on_boundary"

        params = self.params[i]
        # f_expr = FExpr(params, degree=10, domain=self.mesh, pb_considered=self.pb_considered)
        f_expr = Constant(1.0)
        if get_error:
            if analytical_sol:
                u_ex = UexExpr(params, degree=10, domain=self.mesh, pb_considered=self.pb_considered)
            else:
                u_ex = self.pb_considered.u_ref()
        # f_tild = f_expr + div(grad(phi_tild))

        g = Constant(0.0)
        # g = -phi_tild
        bc = DirichletBC(self.V, g, boundary)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        start = time.time()
        a = inner(grad(u), grad(v)) * self.dx
        # l = f_tild * v * self.dx
        l = f_expr * v * self.dx - inner(grad(phi_tild),grad(v))*self.dx

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

        norme_L2 = None
        if get_error:
            norme_L2 = (assemble((((u_ex - sol)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))
        
        return sol,C_tild,norme_L2