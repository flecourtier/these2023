import numpy as np
import matplotlib.pyplot as plt

# on définit notre polygône aléatoire
def generate_data(n_points):
    x0,x1 = (0.0,0.0)

    r_max = 2.0
    r_min = 1.4

    r = np.random.uniform(r_min, r_max, n_points-1)
    theta = np.linspace(0, 2*np.pi, n_points)
    theta = theta[:-1]
    x = x0 + r * np.cos(theta)
    y = x1 + r * np.sin(theta)

    data = np.concatenate([x[:, None], y[:, None]], axis=1)
    data = np.concatenate([data, [data[0]]])

    return data

nb_pts = 40
partial_S_plus = generate_data(nb_pts+1)

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
    print(val.shape)
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

bord_a,bord_b = (-2,2)
bord_a2,bord_b2 = (-2,2)
N = 201

lin = np.linspace(bord_a,bord_b,N)
lin2 = np.linspace(bord_a2,bord_b2,N)

XX,YY = np.meshgrid(lin,lin2)

plt.figure(figsize=(15,5))

print(np.array([XX,YY]).shape)

plt.subplot(1,2,1)
val = phi(np.array([XX,YY]))
plt.contourf(XX,YY,val,levels=100)#,cmap="hot")
for i in range(nb_pts):
    pt1 = partial_S_plus[i]
    pt2 = partial_S_plus[i+1]
    plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],"white",linewidth=3)
plt.title("phi")
plt.colorbar()

# plt.subplot(1,2,2)
# neg_val = val<0
# val[neg_val] = 0.

# plt.contourf(XX,YY,val,levels=100)#,cmap="hot")
# for i in range(nb_pts):
#     pt1 = partial_S_plus[i]
#     pt2 = partial_S_plus[i+1]
#     plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],"white",linewidth=3)
# plt.title("phi (zero outside)")
# plt.colorbar()


plt.subplot(1,2,2)
neg_val = val>0
data_inside = np.array([XX,YY])[:,neg_val]
val_inside = val[neg_val] 

plt.tricontourf(data_inside[0,:],data_inside[1,:],val_inside,levels=100)#,cmap="hot")
for i in range(nb_pts):
    pt1 = partial_S_plus[i]
    pt2 = partial_S_plus[i+1]
    plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],"white",linewidth=3)
plt.title("phi (zero outside)")
plt.colorbar()

plt.savefig("MVP_random_polygon.png")
plt.show()