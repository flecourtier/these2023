import numpy as np

##############
# DOMAIN
##############

def domain_parameters():
    x0,y0 = (0.5,0.5)
    r = np.sqrt(2)/4
    eps = 0.5-r
    domain_O = [[x0-r-eps,x0+r+eps],[y0-r-eps,y0+r+eps]]
    return x0,y0,r,domain_O

x0,y0,r,domain_O = domain_parameters()

def call_phi(pre, xy):
    x,y=xy
    return -r**2+(x-x0)**2+(y-y0)**2

def call_phi_construct(pre, xy):
    return call_phi(pre,xy)

def call_Omega(pre, xy):
    return call_phi_construct(pre,xy)<0

def Omega_bool(x, y):
    xy = (x, y)
    return call_Omega(None, xy)

def get_vert_coord(nb_vert):
    x = np.linspace(domain_O[0,0], domain_O[0,1], nb_vert)
    y = np.linspace(domain_O[1,0], domain_O[1,1], nb_vert)
    XX, YY = np.meshgrid(x, y)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])
    return XXYY

def omega_mask(nb_vert):
    XXYY = get_vert_coord(nb_vert)

    mask = call_Omega(None,XXYY)
    mask = np.reshape(mask, [nb_vert, nb_vert])

    return mask.astype(np.float64)

##############
# PROBLEM
##############

def call_Y_true(pre, xy, S, f, p):
    x,y=xy
    return S * pre.sin(1/(r**2)*pre.pi*((x-x0)**2+(y-y0)**2))

def call_F(pre, xy, S, f, p):
    x,y=xy
    return 4/(r**4)*pre.pi**2*S*((x-x0)**2+(y-y0)**2)*pre.sin(1/(r**2)*pre.pi*((x-x0)**2+(y-y0)**2)) - \
            4/(r**2)*pre.pi*S*pre.cos(1/(r**2)*pre.pi*((x-x0)**2+(y-y0)**2))



# import sympy
# x, y = sympy.symbols('x y')
# x0,y0,r,S = sympy.symbols('x0 y0 r S')
# u_ex = S*sympy.sin(1./(r**2)*sympy.pi*((x-x0)**2+(y-y0)**2))

# grad_x = sympy.diff(u_ex, x)
# print("grad_x : ",grad_x)
# grad_y = sympy.diff(u_ex, y)
# print("grad_y : ",grad_y)
# f = -(sympy.diff(grad_x,x)+sympy.diff(grad_y,y))
# print("f : ",f)



# def call_Y_true_prime(pre, xy, S ,f, p):
#     return du_dx,du_dy

# def call_Y_true_prime2(pre, xy, S ,f, p):
#     return du_dxx,du_dyy

# def call_P(pre, xy, S, f, p):
#     return call_Y_true(pre,xy,S,f,p)