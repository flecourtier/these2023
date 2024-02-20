import numpy as np
import matplotlib.pyplot as plt

def dist(A,B):
    return np.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2)

a = 0.6
b = 1.2
d = 0.4

F = (d,0)
F_prime = (-d,0)
R = 2*a
R_prime = 2*b

assert R > dist(F, F_prime)
assert R_prime > dist(F, F_prime)

# affichage du cercle intérieur
def plot_circle_in():
    t = np.linspace(0,2*np.pi,100)
    x = d + 2*a*np.cos(t)
    y = 2*a*np.sin(t)
    plt.plot(d,0,"red",marker="+",markersize=5)
    plt.text(d,0,"F",color="red",fontsize=15)
    plt.plot(x,y,"red",linewidth=1)

def plot_circle_out():
    t = np.linspace(0,2*np.pi,100)
    x = -d + 2*b*np.cos(t)
    y = 2*b*np.sin(t)
    plt.plot(-d,0,"blue",marker="+",markersize=5)
    plt.text(-d,0,"F'",color="blue",fontsize=15)
    plt.plot(x,y,"blue",linewidth=1)

# paramétrisation
def c_in(t): # t \in [0,1]
    t = 2*np.pi*t

    r1 = np.sqrt(a**2-d**2*np.sin(t)**2)
    r2 = np.sqrt(b**2-d**2*np.sin(t)**2)
    
    x = d**2*np.sin(t)*np.sin(2*t)+d*(r2-r1)*np.cos(2*t)+2*r1*r2*np.cos(t)
    y = 2*np.sin(t)*(-d**2*np.cos(t)+d*(r2-r1)*np.cos(t)+r1*r2)

    return np.array([x,y])/(r1+r2)

def c_out(t): # t \in [0,1]
    t = 2*np.pi*t

    r1 = np.sqrt(a**2-d**2*np.sin(t)**2)
    r2 = np.sqrt(b**2-d**2*np.sin(t)**2)
    
    x = d**2*np.sin(t)*np.sin(2*t)+d*(r1+r2)*np.cos(2*t)-2*r1*r2*np.cos(t)
    y = 2*np.sin(t)*(-d**2*np.cos(t)+d*(r1+r2)*np.cos(t)-r1*r2)

    return np.array([x,y])/(r2-r1)

def plot_form(c,color):
    t = np.linspace(0,1,100)
    c_t = c(t)
    # ajout du point de départ
    c_t = np.concatenate([c_t,np.array([c_t[:,0]]).T],axis=1)
    for i in range(100):
        pt1 = c_t[:,i]
        pt2 = c_t[:,i+1]

        plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],color,linewidth=3)

plt.figure(figsize=(10,10))

plot_circle_in()
plot_circle_out()

plot_form(c_in,"green")
plot_form(c_out,"black")

plt.title("phi")

plt.show()