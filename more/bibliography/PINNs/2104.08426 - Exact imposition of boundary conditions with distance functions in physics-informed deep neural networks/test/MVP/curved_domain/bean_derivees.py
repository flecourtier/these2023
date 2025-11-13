import sympy

t = sympy.Symbol('t')
a = sympy.Symbol('a')
b = sympy.Symbol('b')

x = ( sympy.sin(2*sympy.pi*t)**a + sympy.cos(2*sympy.pi*t)**b ) * sympy.cos(2*sympy.pi*t)
y = ( sympy.sin(2*sympy.pi*t)**a + sympy.cos(2*sympy.pi*t)**b ) * sympy.sin(2*sympy.pi*t)

# print(x.diff(t))
# print(y.diff(t))

print(x)
# dx/dt
print(sympy.diff(y,t))

