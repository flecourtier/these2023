import numpy as np
import matplotlib.pyplot as plt

# on définit notre ellipse
a,b = (0.5,0.2)

def get_coord(theta):
    return np.array([a*np.cos(theta),b*np.sin(theta)])

def create_poly(nb_pts):
    tab_theta = np.linspace(0,2*np.pi,nb_pts)
    partial_S = get_coord(tab_theta).T
    return partial_S

nb_pts = 50
partial_S = create_poly(nb_pts)
partial_S_plus = np.concatenate([partial_S,[partial_S[0]]])

min_x = np.min(partial_S[:,0])
max_x = np.max(partial_S[:,0])
min_y = np.min(partial_S[:,1])
max_y = np.max(partial_S[:,1])

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
    val = 2./W(X)
    val[np.isnan(val)] = 0 # X on \partial\Omega => phi=0
    return val

bord_a,bord_b = (min_x-0.1,max_x+0.1)
bord_a2,bord_b2 = (min_y-0.1,max_y+0.1)
N = 201

lin = np.linspace(bord_a,bord_b,N)
lin2 = np.linspace(bord_a2,bord_b2,N)

XX,YY = np.meshgrid(lin,lin2)

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
val = phi(np.array([XX,YY]))
plt.contourf(XX,YY,val,levels=100)#,cmap="hot")
for i in range(nb_pts):
    pt1 = partial_S_plus[i]
    pt2 = partial_S_plus[i+1]
    plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],"white",linewidth=3)
plt.title("phi")
plt.colorbar()

plt.subplot(1,2,2)
neg_val = val<0
val[neg_val] = 0.
plt.contourf(XX,YY,val,levels=100)#,cmap="hot")
for i in range(nb_pts):
    pt1 = partial_S_plus[i]
    pt2 = partial_S_plus[i+1]
    plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],"white",linewidth=3)
plt.title("phi (zero outside)")
plt.colorbar()

plt.savefig("MVP/MVP_ellipse.png")
plt.show()