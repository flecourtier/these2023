
from dolfin import *
import dolfin as dol
from dolfin.function.expression import (
    BaseExpression,
    _select_element,
    _InterfaceExpression,
)
import multiphenics as mph

from modules.problems import *
from modules.Poisson2D import *

class Problem:
    def __init__(self,class_pb_considered=Square,pde_considered=Poisson2D_fixed2_carre):
        self.class_pb_considered = class_pb_considered
        self.pb_considered = self.class_pb_considered()
        self.pde_considered = pde_considered
        self.name_problem_considered = self.class_pb_considered.__name__+"/"
        self.name_pde_considered = self.pde_considered.__name__+"/"
        self.dir_name = "networks/"+self.name_problem_considered+self.name_pde_considered

###
# Expression
###

pb_considered = Problem().pb_considered

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
        value[0] = pb_considered.phi_construct(dol, x)

class PhiExpr(MyUserExpression):
    def __init__(self, degree, domain):
        super().__init__(degree, domain)

    def eval(self, value, x):
        value[0] = pb_considered.phi(dol, x)

class UexExpr(MyUserExpression):
    def __init__(self, params, degree, domain):
        super().__init__(degree, domain)
        self.S = params[0]
        self.f = params[1]
        self.p = params[2]
        self.mu = params
    
    def eval(self, value, x):
        value[0] = pb_considered.u_ex(dol, x, self.mu)
                               
class FExpr(MyUserExpression):
    def __init__(self, params, degree, domain):
        super().__init__(degree, domain)
        self.S = params[0]
        self.f = params[1]
        self.p = params[2]
        self.mu = params

    def eval(self, value, x):
        value[0] = pb_considered.f(dol, x, self.mu)