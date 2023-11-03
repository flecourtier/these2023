import numpy as np
import matplotlib.pyplot as plt

def norm(X):
    x,y = X
    return np.sqrt(x**2 + y**2)

# on définit notre triangle
nb_pts = 3
X1 = np.array([-0.5,-0.5])
X2 = np.array([0.5,-0.5])
X3 = np.array([0,0.5])
partial_S = [X1,X2,X3]
partial_S_plus = np.concatenate([partial_S,[partial_S[0]]])

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

a,b = (-1,1)
a2,b2 = (-1,1)
N = 200

lin = np.linspace(a,b,N)
lin2 = np.linspace(a2,b2,N)

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

plt.savefig("REQ/REQ_triangle.png")
plt.show()