from modules.data import *

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

###
# Expression
###

# noinspection PyAbstractClass
class MyUserExpression(BaseExpression):
    """JIT Expressions"""

    def __init__(self, degree, domain):
        cell = domain.ufl_cell()
        element = _select_element(
            family=None, cell=cell, degree=degree, value_shape=()
        )  # modifier value_shape si Expression non scalaire

        self._cpp_object = _InterfaceExpression(self, ())

        BaseExpression.__init__(
            self,
            cell=cell,
            element=element,
            domain=domain,
            name=None,
            label=None,
        )

class PhiConstructExpr(MyUserExpression):
    def __init__(self, degree, domain):
        super().__init__(degree, domain)

    def eval(self, value, x):
        value[0] = call_phi_construct(dol, x)

class PhiExpr(MyUserExpression):
    def __init__(self, degree, domain):
        super().__init__(degree, domain)

    def eval(self, value, x):
        value[0] = call_phi(dol, x)

class UexExpr(MyUserExpression):
    def __init__(self, params, degree, domain):
        super().__init__(degree, domain)
        self.S = params[0]
        self.f = params[1]
        self.p = params[2]
    
    def eval(self, value, x):
        value[0] = call_Y_true(dol, x, self.S, self.f, self.p)
                               
class FExpr(MyUserExpression):
    def __init__(self, params, degree, domain):
        super().__init__(degree, domain)
        self.S = params[0]
        self.f = params[1]
        self.p = params[2]

    def eval(self, value, x):
        value[0] = call_F(dol, x, self.S, self.f, self.p)

###
# Solver
###
class Solver:
    """To solver Laplacian problem.
    """
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

        self.mesh_macro = RectangleMesh(Point(domain_O[0,0], domain_O[0,1]), Point(domain_O[1,0], domain_O[1,1]), self.N, self.N)
        self.V_macro = FunctionSpace(self.mesh_macro, "CG", polV)

        domains = MeshFunction(
            "size_t", self.mesh_macro, self.mesh_macro.topology().dim()
        )
        domains.set_all(0)

        for ind in range(self.mesh_macro.num_cells()):
            mycell = Cell(self.mesh_macro, ind)
            v1x, v1y, v2x, v2y, v3x, v3y = mycell.get_vertex_coordinates()
            if (Omega_bool(v1x, v1y) or Omega_bool(v2x, v2y) or Omega_bool(v3x, v3y)):
                domains[ind] = 1

        self.mesh = SubMesh(self.mesh_macro, domains, 1)
        self.V = FunctionSpace(self.mesh, "CG", polV)
        self.V_phi = FunctionSpace(self.mesh, "CG", polPhi)

        self.phi_Omega = PhiConstructExpr(degree=polPhi, domain=self.mesh)
        self.phi_Omega = interpolate(self.phi_Omega, self.V_phi)

        # Facets and cells where we apply the ghost penalty
                # Facets and cells where we apply the ghost penalty
        self.mesh.init(1, 2)
        facet_ghost = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        cell_ghost = MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
        cell_sub = MeshFunction("bool", self.mesh, self.mesh.topology().dim())
        facet_sub = MeshFunction("bool", self.mesh, self.mesh.topology().dim() - 1)
        vertices_sub = MeshFunction("bool", self.mesh, self.mesh.topology().dim() - 2)
        facet_ghost.set_all(0)
        cell_ghost.set_all(0)
        cell_sub.set_all(0)
        facet_sub.set_all(0)
        vertices_sub.set_all(0)
        count_cell_ghost = 0

        for mycell in cells(self.mesh):
            for myfacet in facets(mycell):
                v1, v2 = vertices(myfacet)
                if (
                    self.phi_Omega(v1.point().x(), v1.point().y())
                    * self.phi_Omega(v2.point().x(), v2.point().y())
                    < 1e-10
                ):
                    cell_ghost[mycell] = 1
                    cell_sub[mycell] = 1
                    for myfacet2 in facets(mycell):
                        facet_ghost[myfacet2] = 1
                        facet_sub[myfacet2] = 1
                        v1, v2 = vertices(myfacet2)
                        vertices_sub[v1], vertices_sub[v2] = 1,1

        for mycell in cells(self.mesh):
            if cell_ghost[mycell] == 1:
                count_cell_ghost += 1
        print("num of cell in the ghost penalty:", count_cell_ghost)

        File2 = File("sub.rtc.xml/mesh_function_2.xml")
        File2 << cell_sub
        File1 = File("sub.rtc.xml/mesh_function_1.xml")
        File1 << facet_sub
        File0 = File("sub.rtc.xml/mesh_function_0.xml")
        File0 << vertices_sub

        self.yp_res = mph.MeshRestriction(self.mesh,"sub.rtc.xml")

        # Initialize cell function for domains
        self.dx = Measure("dx")(domain=self.mesh, subdomain_data=cell_ghost)
        self.ds = Measure("ds")(domain=self.mesh)
        self.dS = Measure("dS")(domain=self.mesh, subdomain_data=facet_ghost)  

        # Resolution
        self.n = FacetNormal(self.mesh)
        self.h = CellDiameter(self.mesh)
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

    def make_matrix(self, expr):
        """To convert a FEniCS Expression into a Numpy matrix.

        :param expr: FEniCS expression to convert 
        :return: Numpy Matrix.
        """
        expr = project(expr)
        expr = interpolate(expr, self.V_macro)
        expr = expr.compute_vertex_values(self.mesh_macro)
        expr = np.reshape(expr, [self.N + 1, self.N + 1])
        return expr