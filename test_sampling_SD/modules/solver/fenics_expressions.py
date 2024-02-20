###########
# Imports #
###########

from dolfin import *
import dolfin as dol
from dolfin.function.expression import (
    BaseExpression,
    _select_element,
    _InterfaceExpression,
)

###############
# Expressions #
###############

from modules.problem.Case import *
cas = Case("case.json")
pb_considered = cas.problem
sdf_considered = cas.sd_function

# noinspection PyAbstractClass
class MyUserExpression(BaseExpression):
    """JIT Expressions"""

    def __init__(self, degree, domain):
        cell = domain.ufl_cell()
        element = _select_element(
            family=None, cell=cell, degree=degree, value_shape=()
        )

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
        value[0] = sdf_considered.phi(dol, x)

class PhiExpr(MyUserExpression):
    def __init__(self, degree, domain):
        super().__init__(degree, domain)

    def eval(self, value, x):
        value[0] = sdf_considered.phi(dol, x)

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