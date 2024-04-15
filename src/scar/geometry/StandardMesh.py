import numpy as np
import torch
import os
import dolfin as df
from dolfin import *
from pymedit import (
    P1Function,
    square,
    mmg2d,
    trunc,
)
import shutil 
from scimba.equations.domain import SpaceTensor
from scar.utils import create_tree

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_XY(bound_box,n=101):
    a,b = bound_box[0]
    a2,b2 = bound_box[1]
    X,Y = np.meshgrid(np.linspace(a, b, n),np.linspace(a2, b2, n))
    X = X.flatten()
    Y = Y.flatten()
    XY = np.array([X,Y]).T

    return XY

def create_test_sample(XY,parameter_domain,):
    X_test = torch.tensor(np.array(XY), dtype=torch.float32) 
    X_test = SpaceTensor(X_test,torch.zeros_like(X_test,dtype=int))

    nb_params = len(parameter_domain)
    shape = (XY.shape[0],nb_params)
    if shape[1] == 0:
        mu_test = torch.zeros(shape)
    else:
        ones = torch.ones(shape)
        mu_test = (torch.mean(parameter_domain, axis=1) * ones).to(device)

    return X_test,mu_test

def get_levelset(trainer,X_test,mu_test,n=101):
    pred = trainer.network.setup_w_dict(X_test, mu_test)
    phi = pred["w"][:,0].cpu().detach().numpy()
    phi = phi.reshape(n,n)

    n = np.shape(phi)[0]
    M = square(n - 1, n - 1)
    M.debug = 4 
    phi = phi.flatten("F")
    phiP1 = P1Function(M, phi) # Setting a P1 level set function

    return M,phiP1

def construct_mesh(M,phiP1,hmin,hmax,filename):
    print("hmin = ", hmin)
    print("hmax = ", hmax)
    newM = mmg2d(
        M,
        hmax=hmax,
        hmin=hmin,
        hgrad=None,
        sol=phiP1,
        ls=True,
        verb=0,
    )

    Mf = trunc(newM, 3) # Trunc the negative subdomain of the level set
    Mf.save(filename+".mesh")  # Saving in binary format
    print("CONVERTTTT")
    command = "meshio convert "+filename+".mesh "+filename+".xml"+" >> output.txt"
    os.system(command) # Convert and save in xml format

def get_df_mesh(bound_box,filename):
    if os.path.isfile(filename+"_new.xml"):
        print("Reading new mesh from file")
        mesh = df.Mesh(filename+"_new.xml")
    elif os.path.isfile(filename+".xml"):
        print("Reading mesh from file")
        mesh = df.Mesh(filename+".xml") # Read the mesh with FEniCS

        # dilat mesh in the correct bound_box
        mesh_coord = mesh.coordinates()
        a,b = bound_box[0]
        a2,b2 = bound_box[1]
        mesh_coord[:,0] = (b-a)*mesh_coord[:,0]+a
        mesh_coord[:,1] = (b2-a2)*mesh_coord[:,1]+a2
        mesh.coordinates()[:] = mesh_coord
        df.File(filename+"_new.xml") << mesh
    else:
        raise ValueError("File not found")

    return mesh

def get_boundary_vertices(mesh):
    # Mark a CG1 Function with ones on the boundary
    V = FunctionSpace(mesh, "CG", 1)
    bc = DirichletBC(V, Constant("1.0"), "on_boundary")
    u = Function(V)
    bc.apply(u.vector())

    # Get vertices sitting on boundary
    d2v = dof_to_vertex_map(V)
    vertices_on_boundary = d2v[u.vector() == 1.0]
    xy = mesh.coordinates()
    xy_boundary = xy[vertices_on_boundary]

    return xy_boundary

def overrefined_mesh(form,trainer,dir_name,n=101,hmin=0.001,hmax=0.005,new_mesh=False):
    dir_name += "overrefined_mesh/"
    if not os.path.exists(dir_name):
        create_tree(str(dir_name))
    if new_mesh:
        shutil.rmtree(dir_name)
        create_tree(str(dir_name))
        
    parameter_domain = trainer.pde.parameter_domain
    XY = get_XY(form.bound_box,n)
    X_test,mu_test = create_test_sample(XY,parameter_domain)
    M,phiP1 = get_levelset(trainer,X_test,mu_test,n)

    filename = dir_name+"overrefined_mesh"
    if not os.path.exists(filename+".xml"):
        construct_mesh(M,phiP1,hmin,hmax,filename)
    mesh = get_df_mesh(form.bound_box,filename)

    return mesh

def standard_mesh(form,trainer,dir_name,hmin,hmax,n=101,new_mesh=False):
    #create dir
    dir_name += "standard_mesh/"
    if not os.path.exists(dir_name):
        create_tree(str(dir_name))
    if new_mesh:
        shutil.rmtree(dir_name)
        create_tree(str(dir_name))
        
    parameter_domain = trainer.pde.parameter_domain
    XY = get_XY(form.bound_box,n)
    X_test,mu_test = create_test_sample(XY,parameter_domain)
    M,phiP1 = get_levelset(trainer,X_test,mu_test,n)

    filename = dir_name+"standard_mesh"
    if not os.path.exists(filename+".xml"):
        construct_mesh(M,phiP1,hmin,hmax,filename)
    mesh = get_df_mesh(form.bound_box,filename)

    return mesh