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
# sigma_stab=20.0
# gamma=1.0 

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

class PertExpr(MyUserExpression):
    def __init__(self, params, degree, domain):
        super().__init__(degree, domain)
        self.S = params[0]
        self.f = params[1]
        self.p = params[2]
    
    def eval(self, value, x):
        value[0] = call_P(dol, x, self.S, self.f, self.p)
                               
class FExpr(MyUserExpression):
    def __init__(self, params, degree, domain):
        super().__init__(degree, domain)
        self.S = params[0]
        self.f = params[1]
        self.p = params[2]

    def eval(self, value, x):
        value[0] = call_F(dol, x, self.S, self.f, self.p)

