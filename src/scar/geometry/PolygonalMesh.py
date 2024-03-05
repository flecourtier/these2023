import numpy as np
import matplotlib.pyplot as plt
import dolfin as df
import mshr

def dist(pt1,pt2):
    return np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)

def tri(c_t):
    n_bc_points = c_t.shape[0]
    c_t_tri = [c_t[0,:]]
    c_t = np.delete(c_t,0,axis=0)
    for _ in range(1,n_bc_points):
        pt1 = c_t_tri[-1]
        # point le plus proche de pt1
        dist_pts = [dist(pt1,c_t[i,:]) for i in range(len(c_t))]
        i_min = np.argmin(dist_pts)
        c_t_tri.append(c_t[i_min,:])
        c_t = np.delete(c_t,i_min,axis=0)
    return np.array(c_t_tri)

def create_domain(form,n_bc_points=200):
    t = np.linspace(0,1,n_bc_points)
    c_t = form.c(t)
    c_t = np.array(c_t.T)
    # si le premier point et le derni point sont pareil on enleve le dernier point
    if np.abs(np.all(c_t[0,:]-c_t[-1,:]))<1e-10:
        c_t = c_t[:-1,:]

    c_t_tri = tri(c_t)
    c_t_tri = np.concatenate([c_t_tri,np.array([c_t_tri[0]])],axis=0)

    domain_vertices = [df.Point(c_t_tri[i,0],c_t_tri[i,1]) for i in range(n_bc_points)]
    domain = mshr.Polygon(domain_vertices)

    return domain
    # plot polygon (sorted and unsorted)
    # fig,axs = plt.subplots(1,2,figsize=(10,5))
    # for i in range(n_bc_points-1):
    #     axs[0].plot([c_t[i,0],c_t[i+1,0]],[c_t[i,1],c_t[i+1,1]],"black",linewidth=3)
    #     axs[1].plot([c_t_tri[i,0],c_t_tri[i+1,0]],[c_t_tri[i,1],c_t_tri[i+1,1]],"black",linewidth=3)
    # axs[1].plot([c_t_tri[-2,0],c_t_tri[-1,0]],[c_t_tri[-2,1],c_t_tri[-1,1]],"black",linewidth=3)