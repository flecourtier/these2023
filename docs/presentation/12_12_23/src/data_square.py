import numpy as np

def call_phi(pre, xy):
    return xy[0]*(1-xy[0])*xy[1]*(1-xy[1])

def call_phi_construct(pre, xy):
    return np.linalg.norm(np.array(xy)-0.5,np.inf,axis=0)-0.5

def call_Omega(pre, xy):
    # return call_phi(pre,xy)<0
    return call_phi_construct(pre,xy)<0

def Omega_bool(x, y):
    xy = (x, y)
    return call_Omega(None, xy)

def get_vert_coord(nb_vert):
    xy = np.linspace(-0.5, 1.5, nb_vert)
    XX, YY = np.meshgrid(xy, xy)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])
    return XXYY

def omega_mask(nb_vert):
    XXYY = get_vert_coord(nb_vert)

    mask = call_Omega(None,XXYY)
    mask = np.reshape(mask, [nb_vert, nb_vert])

    return mask.astype(np.float64)

def call_Y_true(pre, xy, S, f, p):
    x,y=xy
    return S*pre.sin(2*pre.pi*f*x + p) * pre.sin(2*pre.pi*f*y + p)

def call_F(pre, xy, S, f, p):
    x,y=xy
    return 8*pre.pi**2*S*f**2*pre.sin(2*pre.pi*f*x + p)*pre.sin(2*pre.pi*f*y + p)

def call_P(pre, xy, S, f, p):
    return call_Y_true(pre,xy,S,f,p)