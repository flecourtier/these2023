import numpy as np

def call_phi(pre, xy):
    return -0.125 + (xy[0] - 0.5)**2 + (xy[1] - 0.5)**2

def call_phi_construct(pre, xy):
    return -0.125 + (xy[0] - 0.5)**2 + (xy[1] - 0.5)**2

def call_Omega(pre, xy):
    # return call_phi(pre,xy)<0
    return call_phi_construct(pre,xy)<0

def Omega_bool(x, y):
    xy = (x, y)
    return call_Omega(None, xy)

def get_vert_coord(nb_vert):
    xy = np.linspace(0, 1, nb_vert)
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
    # return S * pre.sin(8*2*pre.pi*f*((x-0.5)**2+(y-0.5)**2)+p)
    return S * pre.sin(8*pre.pi*f*((x-0.5)**2+(y-0.5)**2)+p)

def call_F(pre, xy, S, f, p):
    x,y=xy
    pi = pre.pi
    # return 256*pi**2*S*f**2*(4*(x - 0.5)**2)*pre.sin(16*pi*f*((x - 0.5)**2 + (y - 0.5)**2) + p) + 256*pi**2*S*f**2*(4*(y - 0.5)**2)*pre.sin(16*pi*f*((x - 0.5)**2 + (y - 0.5)**2) + p) - 64*pi*S*f*pre.cos(16*pi*f*((x - 0.5)**2 + (y - 0.5)**2) + p)
    return 64*pi**2*S*f**2*(4*(x - 0.5)**2)*pre.sin(8*pi*f*((x - 0.5)**2 + (y - 0.5)**2) + p) + 64*pi**2*S*f**2*(4*(y - 0.5)**2)*pre.sin(8*pi*f*((x - 0.5)**2 + (y - 0.5)**2) + p) - 32*pi*S*f*pre.cos(8*pi*f*((x - 0.5)**2 + (y - 0.5)**2) + p)

def call_P(pre, xy, S, f, p):
    return call_Y_true(pre,xy,S,f,p)