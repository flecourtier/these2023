import numpy as np
import matplotlib.pyplot as plt

x1,y1=(-0.5,0)
x2,y2 = (0.5,0)

X1 = np.array([x1,y1])
X2 = np.array([x2,y2])
L = (x2-x1)**2+(y2-y1)**2

Xc = (X1+X2)/2


def f(X):
    x,y=X
    return ((x-x1)*(y2-y1)-(y-y1)*(x2-x1))/L

def t(X):
    x,y=X
    return 1/L*((L/2)**2-((x-Xc[0])**2+(y-Xc[1])**2))

def phi(X):
    varphi = lambda X : np.sqrt(t(X)**2+f(X)**4)

    return np.sqrt(f(X)**2+((varphi(X)-t(X))/2)**2)

a,b = (-1,1)
a2,b2 = (-0.5,0.5)
nb_pts = 20

lin = np.linspace(a,b,nb_pts)
lin2 = np.linspace(a2,b2,nb_pts)

XX,YY = np.meshgrid(lin,lin2)

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
val = f(np.array([XX,YY]))
plt.contourf(XX,YY,val)#,cmap="hot")
plt.plot([x1,x2],[y1,y2],"white",linewidth=3)
plt.title("f")
plt.colorbar()

plt.subplot(1,3,2)
val = t(np.array([XX,YY]))
plt.contourf(XX,YY,val)#,cmap="hot")
plt.plot([x1,x2],[y1,y2],"white",linewidth=3)
plt.title("t")
plt.colorbar()

plt.subplot(1,3,3)
val = phi(np.array([XX,YY]))
plt.contourf(XX,YY,val)#,cmap="hot")
plt.plot([x1,x2],[y1,y2],"white",linewidth=3)
plt.title("phi")
plt.colorbar()

plt.savefig("REQ/REQ_segment.png")
plt.show()