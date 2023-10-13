import numpy as np
import matplotlib.pyplot as plt

# on définit notre ellipse
def get_coord(theta):
    return np.array([a*np.cos(theta),b*np.sin(theta)])

def create_poly(nb_pts):
    tab_theta = np.linspace(0,2*np.pi,nb_pts)
    partial_S = get_coord(tab_theta).T
    return partial_S

a,b = (0.5,0.3)
nb_pts = 50
partial_S = create_poly(nb_pts)
partial_S_plus = np.concatenate([partial_S,[partial_S[0]]])

def norm(X):
    x,y = X
    return np.sqrt(x**2 + y**2)

# on récupère les longueurs des segments et les points milieux
tab_L = []
tab_Xc = []
for i in range(nb_pts):
    pt1 = partial_S_plus[i]
    pt2 = partial_S_plus[i+1]
    tab_L.append(norm(pt2-pt1)**2)
    tab_Xc.append((pt1+pt2)/2)


def f_i(X,i):
    x,y=X
    X1,X2,L = (partial_S_plus[i],partial_S_plus[i+1],tab_L[i])
    x1,y1 = X1
    x2,y2 = X2
    return ((x-x1)*(y2-y1)-(y-y1)*(x2-x1))/L

def t_i(X,i):
    x,y=X
    Xc,L = (tab_Xc[i],tab_L[i])
    return 1/L*((L/2)**2-((x-Xc[0])**2+(y-Xc[1])**2))

def phi_i(X,i):
    varphi = lambda X : np.sqrt(t_i(X,i)**2+f_i(X,i)**4)
    return np.sqrt(f_i(X,i)**2+((varphi(X)-t_i(X,i))/2)**2)

def phi(X,m):
    den = np.zeros_like(X[0])
    for i in range(nb_pts):
        den += 1./phi_i(X,i)**m
    return 1./den**(1/m)

bord_a,bord_b = (-a-0.5,a+0.5)
bord_a2,bord_b2 = (-b-0.5,b+0.5)
N = 201

lin = np.linspace(bord_a,bord_b,N)
lin2 = np.linspace(bord_a2,bord_b2,N)

XX,YY = np.meshgrid(lin,lin2)

plt.figure()

m = 1
val = phi(np.array([XX,YY]),m)
plt.contourf(XX,YY,val,levels=15)#,cmap="hot")
for i in range(nb_pts):
    pt1 = partial_S_plus[i]
    pt2 = partial_S_plus[i+1]
    plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],"white",linewidth=3)
plt.title("phi")
plt.colorbar()

plt.savefig("REQ/REQ_ellipse.png")
plt.show()