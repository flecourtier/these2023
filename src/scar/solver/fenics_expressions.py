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
# from scar.problem.Case import *

######################
# FENICS Expressions #
######################

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
    def __init__(self, degree, domain, sdf_considered):
        super().__init__(degree, domain)
        self.sdf_considered = sdf_considered

    def eval(self, value, x):
        value[0] = self.sdf_considered.phi(dol, x)

class PhiExpr(MyUserExpression):
    def __init__(self, degree, domain, sdf_considered):
        super().__init__(degree, domain)
        self.sdf_considered = sdf_considered

    def eval(self, value, x):
        value[0] = self.sdf_considered.phi(dol, x)

class UexExpr(MyUserExpression):
    def __init__(self, params, degree, domain, pb_considered):
        super().__init__(degree, domain)
        self.S = params[0]
        self.f = params[1]
        self.p = params[2]
        self.mu = params
        self.pb_considered = pb_considered
    
    def eval(self, value, x):
        value[0] = self.pb_considered.u_ex(dol, x, self.mu)
                               
class FExpr(MyUserExpression):
    def __init__(self, params, degree, domain, pb_considered):
        super().__init__(degree, domain)
        self.S = params[0]
        self.f = params[1]
        self.p = params[2]
        self.mu = params
        self.pb_considered = pb_considered

    def eval(self, value, x):
        value[0] = self.pb_considered.f(dol, x, self.mu)