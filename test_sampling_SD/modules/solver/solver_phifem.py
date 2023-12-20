from modules.solver.fenics_expressions import *
from modules.problem.Case import *

cas = Case("case.json")
pb_considered = cas.Problem

print_time = False

homogeneous = True
cd = "homo"

from dolfin import *
import dolfin as df
import multiphenics as mph
import mshr
import time

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
# sigma_stab=1.0
# gamma=1.0 

class PhiFemSolver:
    def __init__(self, nb_cell, params):
        """To initialize an instance of the Solver class.

        :param nb_cell: Number of cells.
        :param params: Parameters.
        :param Y_test: Reference solution (over-refined solution calculated by standard FEM).
        :param V_ex: FEniCS Function Space on the over-refined mesh.
        :param dx_ex: FEniCS Measure on the over-refined domain.
        """

        self.N = nb_cell
        self.params = params

        self.times_fem = {}
        self.times_corr_add = {}

        domain_O = np.array(pb_considered.domain_O)
        self.mesh_macro = RectangleMesh(Point(domain_O[0,0], domain_O[1,0]), Point(domain_O[0,1], domain_O[1,1]), self.N, self.N)
        self.V_macro = FunctionSpace(self.mesh_macro, "CG", polV)

        domains = MeshFunction(
            "size_t", self.mesh_macro, self.mesh_macro.topology().dim()
        )
        domains.set_all(0)

        start = time.time()
        for ind in range(self.mesh_macro.num_cells()):
            mycell = Cell(self.mesh_macro, ind)
            v1x, v1y, v2x, v2y, v3x, v3y = mycell.get_vertex_coordinates()
            if (pb_considered.Omega_bool(v1x, v1y) or pb_considered.Omega_bool(v2x, v2y) or pb_considered.Omega_bool(v3x, v3y)):
                domains[ind] = 1
        self.mesh = SubMesh(self.mesh_macro, domains, 1)
        end = time.time()

        if print_time:
            print("Time to generate Omega_h: ", end-start)
        self.times_fem["Omega_h"] = end-start
        self.times_corr_add["Omega_h"] = end-start

        self.V = FunctionSpace(self.mesh, "CG", polV)
        self.V_phi = FunctionSpace(self.mesh, "CG", polPhi)

        self.phi_Omega = PhiConstructExpr(degree=polPhi, domain=self.mesh)
        self.phi_Omega = interpolate(self.phi_Omega, self.V_phi)

        # Facets and cells where we apply the ghost penalty
        self.mesh.init(1, 2)
        facet_ghost = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        cell_ghost = MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
        # cell_sub = MeshFunction("bool", self.mesh, self.mesh.topology().dim())
        # facet_sub = MeshFunction("bool", self.mesh, self.mesh.topology().dim() - 1)
        # vertices_sub = MeshFunction("bool", self.mesh, self.mesh.topology().dim() - 2)
        facet_ghost.set_all(0)
        cell_ghost.set_all(0)
        # cell_sub.set_all(0)
        # facet_sub.set_all(0)
        # vertices_sub.set_all(0)
        count_cell_ghost = 0

        start = time.time()
        for mycell in cells(self.mesh):
            for myfacet in facets(mycell):
                v1, v2 = vertices(myfacet)
                if (
                    self.phi_Omega(v1.point().x(), v1.point().y())
                    * self.phi_Omega(v2.point().x(), v2.point().y())
                    < 1e-10
                ):
                    cell_ghost[mycell] = 1
                    # cell_sub[mycell] = 1
                    for myfacet2 in facets(mycell):
                        facet_ghost[myfacet2] = 1
                        # facet_sub[myfacet2] = 1
                        v1, v2 = vertices(myfacet2)
                        # vertices_sub[v1], vertices_sub[v2] = 1,1

        for mycell in cells(self.mesh):
            if cell_ghost[mycell] == 1:
                count_cell_ghost += 1

        end = time.time()
        if print_time:
            print("Time to generate cells/facets : ", end-start)
        self.times_fem["cells-facets"] = end-start
        self.times_corr_add["cells-facets"] = end-start

        print("num of cell in the ghost penalty:", count_cell_ghost)

        # File2 = File("sub.rtc.xml/mesh_function_2.xml")
        # File2 << cell_sub
        # File1 = File("sub.rtc.xml/mesh_function_1.xml")
        # File1 << facet_sub
        # File0 = File("sub.rtc.xml/mesh_function_0.xml")
        # File0 << vertices_sub

        # self.yp_res = mph.MeshRestriction(self.mesh,"sub.rtc.xml")

        # Initialize cell function for domains
        self.dx = Measure("dx")(domain=self.mesh, subdomain_data=cell_ghost)
        self.ds = Measure("ds")(domain=self.mesh)
        self.dS = Measure("dS")(domain=self.mesh, subdomain_data=facet_ghost)  

        # Resolution
        self.n = FacetNormal(self.mesh)
        self.h = CellDiameter(self.mesh)
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        self.mesh_ex,self.V_ex,self.dx_ex = self.__create_FEM_domain()

    def __create_FEM_domain(self):
        # check if problem_considered is instance of Circle class
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
            mesh = mshr.generate_mesh(domain,H)
            h = mesh.hmax()

        V = FunctionSpace(mesh, "CG", 1)
        dx = Measure("dx", domain=mesh)

        return mesh, V, dx

    # Phi-FEM Poisson solver
    def fem(self, i, sigma_stab=1.0):
        params = self.params[i]
        f_expr = FExpr(params, degree=6, domain=self.mesh)
        u_ex = UexExpr(params, degree=8, domain=self.mesh)

        phi = PhiExpr(degree=polPhi, domain=self.mesh)
 
        start = time.time()
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

        l = (
            f_expr * self.v * phi * self.dx
            # stab terms
            - sigma_stab
            * self.h**2
            * inner(f_expr, div(grad(phi * self.v)))
            * self.dx(1) #cells ghost
        )

        A = df.assemble(a)
        L = df.assemble(l)

        end = time.time()

        if print_time:
            print("Time to assemble the matrix : ",end-start)
        self.times_fem["assemble"] = end-start

        # Define solution function
        w = Function(self.V)

        start = time.time()
        solve(A,w.vector(),L)
        # solve(a == L, w)  # , solver_parameters={'linear_solver': 'mumps'})
        sol = phi * w
        end = time.time()

        if print_time:
            print("Time to solve the system :",end-start)
        self.times_fem["solve"] = end-start
        
        norm_L2 = (assemble((((u_ex - sol)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))

        project_on_Omega = True
        if project_on_Omega:
            sol_ = project(sol, self.V)
            sol_Omega = project(sol_, self.V_ex)

            u_ex_Omega = UexExpr(params, degree=10, domain=self.mesh_ex)
            u_ex_Omega = project(u_ex_Omega, self.V)
            u_ex_Omega = project(u_ex_Omega, self.V_ex)
            
            norm_L2_Omega = (assemble((((u_ex_Omega - sol_Omega)) ** 2) * self.dx_ex) ** (0.5)) / (assemble((((u_ex_Omega)) ** 2) * self.dx_ex) ** (0.5))
            
            return sol_Omega, norm_L2_Omega

        return sol,norm_L2
    
    def corr_add(self, i, phi_tild, sigma_stab=1.0):
        """To solve the Laplace Problem for one parameters with the correction by addition.
            We consider the problem : phi_tild + phi*C

        :param i: Index of the parameter.
        :param phi_tild: FEniCS expression for the disturbed solution.
        :return: L2 norm of the error.
        """     

        params = self.params[i]
        f_expr = FExpr(params, degree=10, domain=self.mesh)
        u_ex = UexExpr(params, degree=10, domain=self.mesh)
        phi = PhiExpr(degree=10, domain=self.mesh)

        f_tild = f_expr + div(grad(phi_tild))

        start = time.time()
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

        l = (
            f_tild * self.v * phi * self.dx
            # stab terms
            - sigma_stab
            * self.h**2
            * inner(f_tild, div(grad(phi * self.v)))
            * self.dx(1) #cells ghost
        )

        A = df.assemble(a)
        L = df.assemble(l)

        end = time.time()

        if print_time:
            print("Time to assemble the matrix : ",end-start)
        self.times_corr_add["assemble"] = end-start

        # Define solution function
        C_h = Function(self.V)

        start = time.time()
        solve(A,C_h.vector(),L)
        # solve(a == L, C_h)
        C_tild = phi*C_h
        sol = C_tild + phi_tild
        end = time.time()

        if print_time:
            print("Time to solve the system :",end-start)
        self.times_corr_add["solve"] = end-start

        norm_L2 = (assemble((((u_ex - sol)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))
    
        project_on_Omega = True
        if project_on_Omega:
            sol_ = project(sol, self.V)
            sol_Omega = project(sol_, self.V_ex)

            u_ex_Omega = UexExpr(params, degree=10, domain=self.mesh_ex)
            u_ex_Omega = project(u_ex_Omega, self.V)
            u_ex_Omega = project(u_ex_Omega, self.V_ex)
            
            norm_L2_Omega = (assemble((((u_ex_Omega - sol_Omega)) ** 2) * self.dx_ex) ** (0.5)) / (assemble((((u_ex_Omega)) ** 2) * self.dx_ex) ** (0.5))
            
            return sol_Omega, C_tild, norm_L2_Omega

        return sol, C_tild, norm_L2

    # def corr_add_IPP(self, i, phi_tild):
    #     """To solve the Laplace Problem for one parameters with the correction by addition.
    #         We consider the problem : phi_tild+C

    #     :param i: Index of the parameter.
    #     :param phi_tild: FEniCS expression for the disturbed solution.
    #     :return: L2 norm of the error.
    #     """   
    #     params = self.params[i]   
    #     f_expr = FExpr(params, degree=6, domain=self.mesh)
    #     u_ex = UexExpr(params, degree=10, domain=self.mesh)
    #     phi = PhiExpr(degree=6, domain=self.mesh)

    #     f_tild = f_expr + div(grad(phi_tild))

    #     a = (
    #         inner(grad(phi * self.u), grad(phi * self.v)) * self.dx
    #         - dot(inner(grad(phi * self.u), self.n), phi * self.v) * self.ds
    #         # stab terms
    #         + sigma_stab
    #         * avg(self.h)
    #         * dot(
    #             jump(grad(phi * self.u), self.n),
    #             jump(grad(phi * self.v), self.n),
    #         )
    #         * self.dS(1) #facets ghost
    #         + sigma_stab
    #         * self.h**2
    #         * inner(
    #             div(grad(phi * self.u)),
    #             div(grad(phi * self.v)),
    #         )
    #         * self.dx(1) #cells ghost
    #     )

    #     L = (
    #         f_expr * self.v * phi * self.dx
    #         - inner(grad(phi_tild), grad(phi * self.v)) * self.dx
    #         + dot(inner(grad(phi_tild), self.n), phi * self.v) * self.ds
    #         # stab terms
    #         - sigma_stab
    #         * self.h**2
    #         * inner(f_tild, div(grad(phi * self.v)))
    #         * self.dx(1) #cells ghost
    #     )

    #     # Define solution function
    #     C_h = Function(self.V)
    #     solve(a == L, C_h)  # , solver_parameters={'linear_solver': 'mumps'})

    #     C_tild = phi*C_h
    #     sol = C_tild + phi_tild

    #     norm_L2 = (assemble((((u_ex - sol)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))

    #     return sol, C_h, norm_L2

    # def corr_mult(self, i, phi_tild):
    #     """To solve the Laplace Problem for one parameters with the correction by multiplication.
    #         We consider the problem : phi_tild*C

    #     :param i: Index of the parameter.
    #     :param phi_tild: FEniCS expression for the disturbed solution.
    #     :return: L2 norm of the error.
    #     """    
    #     g = Constant("0.0")    
    #     params = self.params[i]
    #     f_expr = FExpr(params, degree=6, domain=self.mesh)
    #     u_ex = UexExpr(params, degree=10, domain=self.mesh)

    #     a = (
    #         inner(grad((phi_tild-g) * self.u), grad((phi_tild-g) * self.v)) * self.dx
    #         - dot(inner(grad((phi_tild-g) * self.u), self.n), (phi_tild-g) * self.v) * self.ds
    #         # stab terms
    #         + sigma_stab
    #         * avg(self.h)
    #         * dot(
    #             jump(grad((phi_tild-g) * self.u), self.n),
    #             jump(grad((phi_tild-g) * self.v), self.n),
    #         )
    #         * self.dS(1) #facets ghost
    #         + sigma_stab
    #         * self.h**2
    #         * inner(
    #             div(grad((phi_tild-g) * self.u)),
    #             div(grad((phi_tild-g) * self.v)),
    #         )
    #         * self.dx(1) #cells ghost
    #     )

    #     L = (
    #         f_expr * self.v * (phi_tild-g) * self.dx
    #         # stab terms
    #         - sigma_stab
    #         * self.h**2
    #         * inner(f_expr, div(grad((phi_tild-g) * self.v)))
    #         * self.dx(1) #cells ghost
    #     )

    #     # Define solution function
    #     C_h = Function(self.V)
    #     solve(a == L, C_h)  # , solver_parameters={'linear_solver': 'mumps'})

    #     sol = (phi_tild-g) * C_h 

    #     norm_L2 = (assemble((((u_ex - sol)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))

    #     return sol, C_h, norm_L2