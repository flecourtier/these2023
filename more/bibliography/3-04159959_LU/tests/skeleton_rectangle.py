import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

width = 1.5
height = 1.

def get_zone(i):
    rect = [[0,0],[0,height],[width,height],[width,0]]
    pt1 = [height/2.,height/2.]
    pt2 = [width-height/2.,height/2.]

    # zone 0 : left triangle 
    if i == 0:
        return [rect[0],rect[1],pt1]
    # zone 1 : polygon up
    elif i == 1:
        return [rect[1],rect[2],pt2,pt1]
    # zone 2 : right triangle     
    elif i == 2:
        return [rect[2],pt2,rect[3]]
    # zone 4 : polygon down
    elif i == 3:
        return [rect[3],rect[0],pt1,pt2]
    
def get_zones():
    zones = []
    for i in range(4):
        zones.append(get_zone(i))
    return zones

def point_inside_polygon(x,y,poly):
    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def inside_zone(x,y):
    zones = get_zones()
    for i in range(len(zones)):
        if point_inside_polygon(x,y,zones[i]):
            return i
    return None 

def phi(x,y):
    zone = inside_zone(x,y)
    if zone == 0:
        return x
    elif zone == 1:
        return height-y
    elif zone == 2:
        return width-x
    elif zone == 3:
        return y
    else:
        return None
    
def grad_phi(x,y):
    zone = inside_zone(x,y)
    if zone == 0:
        return (1,0)
    elif zone == 1:
        return (0,-1)
    elif zone == 2:
        return (-1,0)
    elif zone == 3:
        return (0,1)
    else:
        return (0,0)

# plot
fig, ax = plt.subplots()
plt.xlim(-0.2,width+0.2)
plt.ylim(-0.2,height+0.2)

n_pts = 100
x = np.linspace(0,width,n_pts)
y = np.linspace(0,height,n_pts)
XX,YY = np.meshgrid(x,y)
XX,YY = XX.flatten(),YY.flatten()

sdf = []
u = []
v = []
for i in range(len(XX)):
    x,y = XX[i],YY[i]
    sdf.append(phi(x,y))
    grad = grad_phi(x,y)
    u.append(grad[0])
    v.append(grad[1])

plt.scatter(XX,YY,c=sdf,linewidths=0.5)
plt.colorbar()

n_pts = 20
x = np.linspace(0,width,n_pts)
y = np.linspace(0,height,n_pts)
XX,YY = np.meshgrid(x,y)
XX,YY = XX.flatten(),YY.flatten()

u = []
v = []
for i in range(len(XX)):
    x,y = XX[i],YY[i]
    grad = grad_phi(x,y)
    u.append(grad[0])
    v.append(grad[1])

plt.quiver(XX,YY,u,v)

rect = patches.Rectangle((0, 0), width, height, fill=None, edgecolor='white')
ax.add_patch(rect)

zones = get_zones()
for i in range(len(zones)):
    poly = patches.Polygon(zones[i],closed=True, fill=None, edgecolor='white')
    ax.add_patch(poly)

plt.show()

