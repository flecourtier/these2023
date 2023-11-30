###########
# Imports #
###########

import argparse

from modules.Case import *
from modules.utils import read_config,create_tree
from modules.run_laplacian import *

from modules.solver.solver_fem import FEMSolver
from modules.solver.solver_phifem import PhiFemSolver
from create_xlsx_file import create_xlsx_file

import torch
from torch.autograd import grad as grad_torch
import dolfin as df
from dolfin import FunctionSpace,Function

import matplotlib.pyplot as plt

# from dolfin import parameters
# parameters["form_compiler"]["quadrature_degree"] = 10

###############
# Define case #
###############

cas = Case("case.json")

impose_exact_bc = cas.impose_exact_bc
problem_considered = cas.Problem
pde_considered = cas.PDE

dir_name = cas.dir_name
models_dir = dir_name+"models/"
derivees_dir = dir_name+"derivees/"
create_tree(derivees_dir)

#############
# Arguments #
#############

# Récupération des arguments lors de l'exécution du script python
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", help="Index of configuration file.", type=int, default=0)
    parser.add_argument("--domain", help="Domain on which we compute the derivatives (\"Omega\" or \"Omega_h\").", type=str, default="Omega")
    parser.add_argument("--derive", help="Order of the derivatives (1 or 2) .", type=int, default=1)
    parser.add_argument("--direction", help="Direction of the derivatives (x or y) .", type=str, default="x")

    args = parser.parse_args()

    return args, parser

args, parser = get_args()

##########################
# Récupération du modèle #
##########################

config = args.config

config_filename = models_dir+"config_"+str(config)+".json"
model_filename = models_dir+"model_"+str(config)+".pth"
print("### Config file : ",config_filename)
print("### Model file : ",model_filename)

dict = read_config(config_filename)
print("### Config ", config, " : ", dict)

trainer = test_laplacian_2d(cas,config,dict)

############
# Dérivées #
############

def derivees_exactes(pre,XXYY,mu):
    if args.derive == 1:
        du_true_dx,du_true_dy = problem_considered.u_ex_prime(pre,XXYY,mu)
        if args.direction == "x":
            return du_true_dx
        elif args.direction == "y":
            return du_true_dy
        else:
            ValueError("direction = 'x' or 'y'")
    elif args.derive == 2: 
        du_true_dxx,du_true_dyy = problem_considered.u_ex_prime2(pre,XXYY,mu)
        if args.direction == "x":
            return du_true_dxx
        elif args.direction == "y":
            return du_true_dyy
        else:
            ValueError("direction = 'x' or 'y'")
    else:
        ValueError("derive = 1 or 2")

def derivees_torch(u_pred,x):
    first_derivatives = grad_torch(u_pred["w"].sum(), x, create_graph=True)[0]
    du_dx = first_derivatives[:, 0]
    du_dy = first_derivatives[:, 1]
    if args.derive == 1:
        if args.direction == "x":
            return du_dx
        elif args.direction == "y":
            return du_dy
        else:
            ValueError("direction = 'x' or 'y'")
    elif args.derive == 2: 
        if args.direction == "x":
            second_derivatives_x = grad_torch(du_dx.sum(), x, create_graph=True)[0]
            return second_derivatives_x[:, 0]
        elif args.direction == "y":
            second_derivatives_y = grad_torch(du_dy.sum(), x, create_graph=True)[0]
            return second_derivatives_y[:, 1]
        else:
            ValueError("direction = 'x' or 'y'")
    else:
        ValueError("derive = 1 or 2")

def derivees_FEniCS(phi_tild_FEniCS,V_phi):
    du_dx,du_dy = df.grad(phi_tild_FEniCS)
    if args.derive == 1:
        if args.direction == "x":
            return df.project(du_dx,V_phi)
        elif args.direction == "y":
            return df.project(du_dy,V_phi)
        else:
            ValueError("direction = 'x' or 'y'")
    elif args.derive == 2: 
        if args.direction == "x":
            du_dxx,du_dxy = df.grad(du_dx)
            return df.project(du_dxx,V_phi)
        elif args.direction == "y":
            du_dxy,du_dyy = df.grad(du_dy)
            return df.project(du_dyy,V_phi)
        else:
            ValueError("direction = 'x' or 'y'")
    else:
        ValueError("derive = 1 or 2")


if trainer.pde.nb_parameters == 0:
    params = [[None,None,None]]
else:
    mu = torch.mean(trainer.pde.parameter_domain, axis=1)
    S,f,p = mu.numpy()
    params = [[S,f,p]]

nb_vert = 32
deg_phi = 10

# si FEM => dérivées sur Omega
if args.domain == "Omega":
    print("### Dérivées sur Omega")

    #####
    # Get the PINNs prediction
    #####

    solver = FEMSolver(nb_cell=nb_vert-1, params=params)
    
    # get coordinates of the dof
    V_phi = FunctionSpace(solver.mesh,"CG",deg_phi)
    XXYY = V_phi.tabulate_dof_coordinates()
    X_test = torch.tensor(XXYY,requires_grad=True)

    # get parameters
    shape = (XXYY.shape[0],trainer.pde.nb_parameters)
    if shape[1] == 0:
        mu_test = torch.zeros(shape)
    else:
        ones = torch.ones(shape)
        mu_test = (torch.mean(trainer.pde.parameter_domain, axis=1) * ones).to(device)

    # get u_PINNs
    pred = trainer.network.setup_w_dict(X_test, mu_test)
    phi_tild = pred["w"][:,0].cpu().detach().numpy()
    u_PINNs = Function(V_phi)
    u_PINNs.vector()[:] = phi_tild.copy()
    
    #####
    # Compute derivatives !
    #####

    # Dérivées exactes
    du_ex = derivees_exactes(np,XXYY.T,params[0])

    # Dérivées Torch
    du_pytorch = derivees_torch(pred,X_test)

    # Dérivées FEniCS
    du_fenics = derivees_FEniCS(u_PINNs,V_phi)

    #####
    # Plot !
    #####

    vmin = np.min(du_ex)
    vmax = np.max(du_ex)
    ticks = np.linspace(vmin,vmax,10)

    plt.figure(figsize=(15,10))
    if args.derive == 1:
        fig_title = "Dérivées premières selon "
    elif args.derive == 2:
        fig_title = "Dérivées secondes selon "
    plt.suptitle(fig_title+args.direction,fontsize=30)

    den = args.direction*args.derive

    plt.subplot(2,3,1)
    du_ex_FE = Function(V_phi)
    du_ex_FE.vector()[:] = du_ex.copy()
    c = df.plot(du_ex_FE, mode="color", vmin=vmin, vmax=vmax, title="du_ex_d"+den)
    plt.colorbar(c, ticks=ticks)
    plt.xlim(problem_considered.domain_O[0])
    plt.ylim(problem_considered.domain_O[1])

    #####
    # Pytorch
    #####

    plt.subplot(2,3,2)
    du_pytorch_FE = Function(V_phi)
    du_pytorch_FE.vector()[:] = du_pytorch.detach().numpy().copy()
    c = df.plot(du_pytorch_FE, mode="color", vmin=vmin, vmax=vmax, title="du_d"+den)
    plt.xlim(problem_considered.domain_O[0])
    plt.ylim(problem_considered.domain_O[1])
    plt.colorbar(c, ticks=ticks)
    plt.ylabel("PyTorch",fontsize=20)

    plt.subplot(2,3,3)
    error = abs(du_ex_FE-du_pytorch_FE)
    c = df.plot(error, title="|du_ex_d"+den+"-du_pytorch_d"+den+"|")
    plt.xlim(problem_considered.domain_O[0])
    plt.ylim(problem_considered.domain_O[1])
    plt.colorbar(c)

    # print("shape du_true_dx : ",du_true_dx.shape)
    # print("shape u_x : ",u_x.detach().numpy().shape)

    # print("Exact-Pytorch :")
    # print("max|du_true_dx-u_x| : ",np.max(np.abs(du_true_dx-u_x.detach().numpy())))

    # print("Exact - passage FEniCS :")
    # print("max|du_true_dx-du_true_dx_FE| :",np.max(np.abs(du_true_dx-du_true_dx_FE.vector()[:])))

    # print("Pytorch - passage FEniCS :")
    # print("max(u_x-u_x_FE) :",np.max(np.abs(u_x.detach().numpy()-u_x_FE.vector()[:])))

    #####
    # FEniCS
    #####

    plt.subplot(2,3,5)
    c = df.plot(du_fenics, mode="color", vmin=vmin, vmax=vmax, title="du_d"+den)
    plt.xlim(problem_considered.domain_O[0])
    plt.ylim(problem_considered.domain_O[1])
    plt.colorbar(c, ticks=ticks)
    plt.ylabel("FEniCS",fontsize=20)

    # print(du_ex_FE.vector()[:][0],du_fenics.vector()[0])
    # diff_ = (du_ex_FE.vector()[:]-du_fenics.vector()[:]).copy()
    # print(np.max(np.abs(diff_)))
    # argmax_ = np.argmax(np.abs(diff_))
    # print(argmax_,diff_[argmax_])
    # print(diff_)

    plt.subplot(2,3,6)
    error = abs(du_ex_FE-du_fenics)
    c = df.plot(error, title="|du_ex_d"+den+"-du_fenics_d"+den+"|")
    plt.xlim(problem_considered.domain_O[0])
    plt.ylim(problem_considered.domain_O[1])
    plt.colorbar(c)

    config_dir = derivees_dir+"config_"+str(config)+"/"
    create_tree(config_dir)
    plt.savefig(config_dir+"derivees_Omega_"+den+".png")
    # plt.show()


# si PhiFEM => dérivées sur Omega_h
elif args.domain=="Omega_h":
    print("### Dérivées sur Omega_h")

    #####
    # Get the PINNs prediction
    #####

    solver = PhiFemSolver(nb_cell=nb_vert-1, params=params)

# get coordinates of the dof
    V_phi = FunctionSpace(solver.mesh,"CG",deg_phi)
    XXYY = V_phi.tabulate_dof_coordinates()
    X_test = torch.tensor(XXYY,requires_grad=True)

    # get parameters
    shape = (XXYY.shape[0],trainer.pde.nb_parameters)
    if shape[1] == 0:
        mu_test = torch.zeros(shape)
    else:
        ones = torch.ones(shape)
        mu_test = (torch.mean(trainer.pde.parameter_domain, axis=1) * ones).to(device)

    # get u_PINNs
    pred = trainer.network.setup_w_dict(X_test, mu_test)
    phi_tild = pred["w"][:,0].cpu().detach().numpy()
    u_PINNs = Function(V_phi)
    u_PINNs.vector()[:] = phi_tild.copy()
    
    #####
    # Compute derivatives !
    #####

    # Dérivées exactes
    du_ex = derivees_exactes(np,XXYY.T,params[0])

    # Dérivées Torch
    du_pytorch = derivees_torch(pred,X_test)

    # Dérivées FEniCS
    du_fenics = derivees_FEniCS(u_PINNs,V_phi)

    #####
    # Plot !
    #####

    vmin = np.min(du_ex)
    vmax = np.max(du_ex)
    ticks = np.linspace(vmin,vmax,10)

    plt.figure(figsize=(15,10))
    if args.derive == 1:
        fig_title = "Dérivées premières selon "
    elif args.derive == 2:
        fig_title = "Dérivées secondes selon "
    plt.suptitle(fig_title+args.direction,fontsize=30)

    den = args.direction*args.derive

    plt.subplot(2,3,1)
    du_ex_FE = Function(V_phi)
    du_ex_FE.vector()[:] = du_ex.copy()
    c = df.plot(du_ex_FE, mode="color", vmin=vmin, vmax=vmax, title="du_ex_d"+den)
    plt.colorbar(c, ticks=ticks)
    plt.xlim(problem_considered.domain_O[0])
    plt.ylim(problem_considered.domain_O[1])

    #####
    # Pytorch
    #####

    plt.subplot(2,3,2)
    du_pytorch_FE = Function(V_phi)
    du_pytorch_FE.vector()[:] = du_pytorch.detach().numpy().copy()
    c = df.plot(du_pytorch_FE, mode="color", vmin=vmin, vmax=vmax, title="du_d"+den)
    plt.xlim(problem_considered.domain_O[0])
    plt.ylim(problem_considered.domain_O[1])
    plt.colorbar(c, ticks=ticks)
    plt.ylabel("PyTorch",fontsize=20)

    plt.subplot(2,3,3)
    error = abs(du_ex_FE-du_pytorch_FE)
    c = df.plot(error, title="|du_ex_d"+den+"-du_pytorch_d"+den+"|")
    plt.xlim(problem_considered.domain_O[0])
    plt.ylim(problem_considered.domain_O[1])
    plt.colorbar(c)

    # print("shape du_true_dx : ",du_true_dx.shape)
    # print("shape u_x : ",u_x.detach().numpy().shape)

    # print("Exact-Pytorch :")
    # print("max|du_true_dx-u_x| : ",np.max(np.abs(du_true_dx-u_x.detach().numpy())))

    # print("Exact - passage FEniCS :")
    # print("max|du_true_dx-du_true_dx_FE| :",np.max(np.abs(du_true_dx-du_true_dx_FE.vector()[:])))

    # print("Pytorch - passage FEniCS :")
    # print("max(u_x-u_x_FE) :",np.max(np.abs(u_x.detach().numpy()-u_x_FE.vector()[:])))

    #####
    # FEniCS
    #####

    plt.subplot(2,3,5)
    c = df.plot(du_fenics, mode="color", vmin=vmin, vmax=vmax, title="du_d"+den)
    plt.xlim(problem_considered.domain_O[0])
    plt.ylim(problem_considered.domain_O[1])
    plt.colorbar(c, ticks=ticks)
    plt.ylabel("FEniCS",fontsize=20)

    # print(du_true_dx_FE.vector()[:][0],du_fenics_FE_proj.vector()[0])
    # diff_ = (du_true_dx_FE.vector()[:]-du_fenics_FE.vector()[:]).copy()
    # print(np.max(np.abs(diff_)))
    # argmax_ = np.argmax(np.abs(diff_))
    # print(argmax_,diff_[argmax_])
    # print(diff_)

    plt.subplot(2,3,6)
    error = abs(du_ex_FE-du_fenics)
    c = df.plot(error, title="|du_ex_d"+den+"-du_fenics_d"+den+"|")
    plt.xlim(problem_considered.domain_O[0])
    plt.ylim(problem_considered.domain_O[1])
    plt.colorbar(c)

    config_dir = derivees_dir+"config_"+str(config)+"/"
    create_tree(config_dir)
    plt.savefig(config_dir+"derivees_Omega_h_"+den+".png")
    # plt.show()

create_xlsx_file(cas)