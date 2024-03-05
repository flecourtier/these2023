###########
# Imports #
###########

import os
from scar.equations.Poisson2D import *
from scar.equations.run_Poisson2D import *

from scar.solver.solver_fem import *
from scar.solver.solver_phifem import *

####################
# PINNs prediction #
####################

# create set of point in P_{deg_corr} to evaluate the PINNs
def get_test_sample(solver,parameter_domain,deg_corr):
    # get coordinates of the dof
    V_phi = FunctionSpace(solver.mesh,"CG",deg_corr)
    XXYY = V_phi.tabulate_dof_coordinates()
    X_test = torch.tensor(XXYY,requires_grad=True)

    # get parameters
    nb_params = len(parameter_domain)
    shape = (XXYY.shape[0],nb_params)
    if shape[1] == 0:
        mu_test = torch.zeros(shape)
    else:
        ones = torch.ones(shape)
        mu_test = (torch.mean(parameter_domain, axis=1) * ones).to(device)

    return V_phi,X_test,mu_test

# get the PINNs prediction for the set of points (fenics function)
def get_u_PINNs(trainer,solver,deg_corr,get_error=False,analytical_sol=True):
    parameter_domain = trainer.pde.parameter_domain
    V_phi,X_test,mu_test = get_test_sample(solver,parameter_domain,deg_corr)

    pred = trainer.network.setup_w_dict(X_test, mu_test)
    phi_tild = pred["w"][:,0].cpu().detach().numpy()
    u_PINNs = Function(V_phi)
    u_PINNs.vector()[:] = phi_tild.copy()

    norm_L2_PINNs = None
    if get_error:
        if analytical_sol:
            u_ex = UexExpr(solver.params[0], degree=deg_corr, domain=solver.mesh, pb_considered=solver.pb_considered)
        else:
            u_ex = solver.pb_considered.u_ref()
        
        norm_L2_PINNs = (assemble((((u_ex - u_PINNs)) ** 2) * solver.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * solver.dx) ** (0.5))

    return u_PINNs, norm_L2_PINNs

######################
# Correct prediction #
###################### 

def correct_pred(solver,u_PINNs,corr_type,analytical_sol=True):
    # get u_Corr
    if corr_type == "add":
        u_Corr,C,norm_L2_Corr = solver.corr_add(0,u_PINNs,analytical_sol=analytical_sol)
    # elif corr_type == "mult":
    #     u_Corr,C,norm_L2_Corr = solver.corr_mult(0,u_PINNs)
    else:
        raise ValueError("corr_type not recognized")
    
    return u_Corr, C, norm_L2_Corr

################
# Plot results #
################

def plot_sol(corr_dir,config,solver_type,u_ex,C,solutions,normes,project_on,project_on_Omega=None):
    assert solver_type in ["FEM","PhiFEM"]

    if project_on_Omega:
        nb_row = 2
        u_Corr,u_FEM = solutions
        norm_L2_Corr,norm_L2_FEM = normes
    else:
        nb_row = 3
        u_PINNs,u_Corr,u_FEM = solutions
        norm_L2_PINNs,norm_L2_Corr,norm_L2_FEM = normes     

    plt.figure(figsize=(15,nb_row*5))
    
    plt.subplot(nb_row,3,1)
    if project_on_Omega:
        plt.text(0.2,0.5,solver_type+"/Corr = {:.2f}".format(norm_L2_FEM/norm_L2_Corr),fontsize=15)
    else:
        plt.text(0.2,0.75,solver_type+"/PINNs = {:.2f}".format(norm_L2_FEM/norm_L2_PINNs),fontsize=15)
        plt.text(0.2,0.5,solver_type+"/Corr = {:.2f}".format(norm_L2_FEM/norm_L2_Corr),fontsize=15)
        plt.text(0.2,0.25,"PINNs/Corr = {:.2f}".format(norm_L2_PINNs/norm_L2_Corr),fontsize=15)
    plt.axis('off')

    # FEM
    plt.subplot(nb_row,3,2)
    c = plot(u_FEM, title="u_"+solver_type)
    plt.ylabel(solver_type, fontsize=20)
    plt.colorbar(c)

    plt.subplot(nb_row,3,3)
    u_FEM_proj = project(u_FEM,project_on)
    error = abs(u_ex-u_FEM_proj)
    # error = project(error,project_on)
    c = plot(error, title="||u_ex-u_"+solver_type+"||_L2 = {:.2e}".format(norm_L2_FEM))
    plt.colorbar(c)

    count=4

    # PINNs
    if not project_on_Omega:
        plt.subplot(nb_row,3,count)
        u_ex_proj = project(u_ex,project_on)
        c = plot(u_ex_proj, title="u_ex")
        plt.colorbar(c)

        plt.subplot(nb_row,3,count+1)
        c = plot(u_PINNs, title="u_PINNs")
        plt.ylabel("PINNs", fontsize=20)
        plt.colorbar(c)

        plt.subplot(nb_row,3,count+2)
        error = abs(u_ex-project(u_PINNs,project_on))
        # error = project(error,project_on)
        c = plot(error, title="||u_ex-u_PINNs||_L2 : {:.2e}".format(norm_L2_PINNs))
        plt.colorbar(c)

        count+=3

    # Corr
    plt.subplot(nb_row,3,count)
    C_proj = project(C,project_on)
    c = plot(C_proj, title="C_tild")
    plt.ylabel("Corr", fontsize=20)
    plt.colorbar(c) 

    plt.subplot(nb_row,3,count+1)
    u_Corr_proj = project(u_Corr,project_on)
    c = plot(u_Corr_proj, title="u_Corr")
    plt.colorbar(c)

    plt.subplot(nb_row,3,count+2)
    error = abs(u_ex-project(u_Corr,project_on))
    # error = project(error,project_on)
    c = plot(error, title="||u_ex-u_Corr||_L2 : {:.2e}".format(norm_L2_Corr))
    plt.colorbar(c)

    FEM_dir = corr_dir+solver_type+"/"
    if not os.path.exists(FEM_dir):
        os.makedirs(FEM_dir)
    if project_on_Omega==None:
        plt.savefig(FEM_dir+"corr_"+solver_type+"_"+str(config)+".png")
    else:
        assert solver_type == "PhiFEM"
        plt.savefig(FEM_dir+"corr_"+solver_type+"_"+str(config)+"_Omega"+".png")