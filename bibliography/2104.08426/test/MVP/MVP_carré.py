import numpy as np
import matplotlib.pyplot as plt

# ATTENTION : sur le bord du domaine W s'annule !

def det(X,Xp):
    x,y = X
    xp,yp = Xp
    return x*yp-y*xp 

def prod_scal(X,Xp):
    x,y = X
    xp,yp = Xp
    return x*xp+y*yp

def norm(X):
    return np.sqrt(prod_scal(X,X))

# on définit notre carré
nb_pts = 4
X1 = np.array([0.,0.])
X2 = np.array([1,0])
X3 = np.array([1,1])
X4 = np.array([0,1])
partial_S = [X1,X2,X3,X4]
partial_S_plus = np.concatenate([partial_S,[partial_S[0]]])

def get_tj(X,j):
    Xj = partial_S_plus[j] 
    Rj = Xj[:,None,None]-X
    rj = norm(Rj)
    Xjp = partial_S_plus[j+1]
    Rjp = Xjp[:,None,None]-X
    rjp = norm(Rjp)
    return det(Rj,Rjp)/(rj*rjp+prod_scal(Rj,Rjp))

def W(X):
    val = np.zeros_like(X[0])
    for j in range(nb_pts):
        Xj = partial_S_plus[j]
        rj = norm(Xj[:,None,None]-X)
        Xjp = partial_S_plus[j+1]
        rjp = norm(Xjp[:,None,None]-X)
        tj = get_tj(X,j)
        val += (1/rj+1/rjp)*tj

    return val

def phi(X):
    return 2./W(X)

a,b = (0,1)
a2,b2 = (0,1)
N = 200

lin = np.linspace(a,b,N)
lin2 = np.linspace(a2,b2,N)

XX,YY = np.meshgrid(lin,lin2)

plt.figure()

val = phi(np.array([XX,YY]))
plt.contourf(XX,YY,val,levels=15)#,cmap="hot")
for i in range(nb_pts):
    pt1 = partial_S_plus[i]
    pt2 = partial_S_plus[i+1]
    plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],"white",linewidth=3)
plt.title("phi")
plt.colorbar()

plt.savefig("MVP/MVP_carré.png")
plt.show()