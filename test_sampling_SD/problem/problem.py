# from modules.Problem import Circle
import sympy

x, y = sympy.symbols('x y')
S, f, p = sympy.symbols('S f p')

##########
# Circle #
##########

# x0, y0 = sympy.symbols('x0 y0')
# r = sympy.symbols('r')

##
# Solution 1
##

# u_ex = S * sympy.sin(1/(r**2)*sympy.pi*((x-x0)**2+(y-y0)**2))

##
# Solution 2
##

# phi = -r**2+(x-x0)**2+(y-y0)**2
# u_ex = phi*sympy.sin(x)*sympy.exp(y)

##########
# Square #
##########

##
# Solution 1
##

u_ex = S*sympy.sin(2*sympy.pi*f*x + p) * sympy.sin(2*sympy.pi*f*y + p)

print("Dérivées premières : ")
grad_x = sympy.diff(u_ex, x)
print("grad_x : ",grad_x)
grad_y = sympy.diff(u_ex, y)
print("grad_y : ",grad_y)

print("Dérivées secondes : ")
grad_xx = sympy.diff(grad_x, x)
print("grad_xx : ",grad_xx)
grad_yy = sympy.diff(grad_y, y)
print("grad_yy : ",grad_yy)

print("-Laplacien : ")
f = -(sympy.diff(grad_x,x)+sympy.diff(grad_y,y))
print("f : ",f)
