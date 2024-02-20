# -----------------------------------------------------------------------------#
# -------- Fichier non trié - étude en fonction du paramètre sigma ! ----------#
# -----------------------------------------------------------------------------#


###########
# Imports #
###########

import argparse

from modules.problem.Case import *
from modules.problem.Poisson2D import *
from modules.utils import read_config,create_tree
from modules.run_laplacian import *

from modules.solver.solver_fem import *
from modules.solver.solver_phifem import *

from create_xlsx_file import *

###############
# Define case #
###############

cas = Case("case.json")

# impose_exact_bc = cas.impose_exact_bc
problem_considered = cas.problem
pde_considered = cas.pde

dir_name = "../"+cas.dir_name
models_dir = dir_name+"models/"
corr_type = cas.corr_type
corr_dir = cas.corr_dir_name
create_tree(corr_dir)

#############
# Arguments #
#############

# Récupération des arguments lors de l'exécution du script python
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--type", help="0 - all ; 1 - add ; 2 - add IPP ; 3 - mult", type=int, default=1)
    parser.add_argument("--config", help="Index of configuration file.", type=int, default=0)
    parser.add_argument("--fem", help="0 - both ; 1 - use FEM ; 2 - use PhiFEM.", type=int, default=0)

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

##############
# Correction #
##############

def get_test_sample(solver,parameter_domain,deg_corr):
    # get coordinates of the dof
    V_phi = FunctionSpace(solver.mesh,"CG",deg_corr)
    XXYY = V_phi.tabulate_dof_coordinates()
    X_test = torch.tensor(XXYY)

    # get parameters
    nb_params = len(parameter_domain)
    shape = (XXYY.shape[0],nb_params)
    if shape[1] == 0:
        mu_test = torch.zeros(shape)
    else:
        ones = torch.ones(shape)
        mu_test = (torch.mean(parameter_domain, axis=1) * ones).to(device)

    return V_phi,X_test,mu_test

def get_u_PINNs(solver,parameter_domain,deg_corr):
    V_phi,X_test,mu_test = get_test_sample(solver,parameter_domain,deg_corr)

    pred = trainer.network.setup_w_dict(X_test, mu_test)
    phi_tild = pred["w"][:,0].cpu().detach().numpy()
    u_PINNs = Function(V_phi)
    u_PINNs.vector()[:] = phi_tild

    return u_PINNs

def plot_sol(corr_type,solver_type,u_ex,solutions,normes,project_on,project_on_Omega=None):
    assert corr_type in ["add","mult"]
    solver_type = "PhiFEM"

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
    c = plot(error, title="||u_ex-u_"+solver_type+"||_L2 = {:.2e}".format(norm_L2_FEM))
    plt.colorbar(c)

    count=4

    # PINNs
    if not project_on_Omega:
        plt.subplot(nb_row,3,count)
        c = plot(u_ex, title="u_ex")
        plt.colorbar(c)

        plt.subplot(nb_row,3,count+1)
        c = plot(u_PINNs, title="u_PINNs")
        plt.ylabel("PINNs", fontsize=20)
        plt.colorbar(c)

        plt.subplot(nb_row,3,count+2)
        error = abs(u_ex-project(u_PINNs,project_on))
        c = plot(error, title="||u_ex-u_PINNs||_L2 : {:.2e}".format(norm_L2_PINNs))
        plt.colorbar(c)

        count+=3

    # Corr
    plt.subplot(nb_row,3,count)
    c = plot(C, title="C_tild")
    plt.ylabel("Corr", fontsize=20)
    plt.colorbar(c) 

    plt.subplot(nb_row,3,count+1)
    c = plot(u_Corr, title="u_Corr")
    plt.colorbar(c)

    plt.subplot(nb_row,3,count+2)
    error = abs(u_ex-project(u_Corr,project_on))
    c = plot(error, title="||u_ex-u_Corr||_L2 : {:.2e}".format(norm_L2_Corr))
    plt.colorbar(c)

    FEM_dir = corr_dir+solver_type+"/"
    if not os.path.exists(FEM_dir):
        os.makedirs(FEM_dir)
    if project_on_Omega==None:
        plt.savefig(FEM_dir+"corr_"+solver_type+"_"+str(config)+"_0.1.png")
    else:
        assert solver_type == "PhiFEM"
        plt.savefig(FEM_dir+"corr_"+solver_type+"_"+str(config)+"_Omega"+"_0.1.png")

if trainer.pde.nb_parameters == 0:
    params = [[None,None,None]]
else:
    mu = torch.mean(trainer.pde.parameter_domain, axis=1)
    S,f,p = mu.numpy()
    params = [[S,f,p]]

nb_vert = 32
deg_corr = 10

tab_sigma = [0.01,0.1,0.5,1.0,2.0,5.0,10.0,20.0,50.0,100.0]

print("### Correction par addition avec PhiFEM")

#####
# Compute !
#####

solver = PhiFemSolver(nb_cell=nb_vert-1, params=params)

# get u_ex
u_ex = UexExpr(params[0], degree=deg_corr, domain=solver.mesh)

# get u_FEM
u_FEM,norm_L2_FEM = solver.fem(0)
print("# Error L2 - FEM : ",norm_L2_FEM)

# get u_PINNs
u_PINNs = get_u_PINNs(solver,trainer.pde.parameter_domain,deg_corr)
norm_L2_PINNs = (assemble((((u_ex - u_PINNs)) ** 2) * solver.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * solver.dx) ** (0.5))
print("# Error L2 - PINNs : ",norm_L2_PINNs)

# get u_Corr
tab_norm_L2_Corr = []
tab_facteurs = []
for sigma in tab_sigma:
    print("### Sigma = ",sigma)
    u_Corr,C,norm_L2_Corr = solver.corr_add(0,u_PINNs,sigma)
    print("# Error L2 - Corr ",norm_L2_Corr)
    facteur = norm_L2_FEM/norm_L2_Corr
    print("# Facteur : ", facteur)
    tab_norm_L2_Corr.append(norm_L2_Corr)
    tab_facteurs.append(facteur)

fig,axs = plt.subplots(1,2,figsize=(10,5))

axs[0].plot(tab_sigma,tab_norm_L2_Corr)
axs[0].set_xlabel("Sigma")
axs[0].set_ylabel("Error L2 - Corr")

axs[1].plot(tab_sigma,tab_facteurs)
axs[1].set_xlabel("Sigma")
axs[1].set_ylabel("Facteur")

plt.savefig(corr_dir+"sigma_study.png")
plt.show()


# #####
# # Plot !
# #####

# u_ex = project(u_ex, solver.V)

# solutions = [u_PINNs,u_Corr,u_FEM]
# normes = [norm_L2_PINNs,norm_L2_Corr,norm_L2_FEM]
# plot_sol(corr_type,u_ex,solutions,normes,solver.V)

# print("### Correction par addition avec PhiFEM - Projection sur Omega")

# #####
# # Compute !
# #####

# # project u_ex on Omega
# u_ex_Omega = project(u_ex, solver.V_ex)

# # project u_Corr on Omega
# u_Corr_ = project(u_Corr, solver.V)
# u_Corr_Omega = project(u_Corr_, solver.V_ex)    
# norm_L2_Corr_Omega = (assemble((((u_ex_Omega - u_Corr_Omega)) ** 2) * solver.dx_ex) ** (0.5)) / (assemble((((u_ex_Omega)) ** 2) * solver.dx_ex) ** (0.5))


# # project u_FEM on Omega
# u_FEM_ = project(u_FEM, solver.V)
# u_FEM_Omega = project(u_FEM_, solver.V_ex)

# norm_L2_FEM_Omega = (assemble((((u_ex_Omega - u_FEM_Omega)) ** 2) * solver.dx_ex) ** (0.5)) / (assemble((((u_ex_Omega)) ** 2) * solver.dx_ex) ** (0.5))


# #####
# # Plot !
# #####

# solutions = [u_Corr_Omega,u_FEM_Omega]
# normes = [norm_L2_Corr_Omega,norm_L2_FEM_Omega]
# plot_sol(corr_type,u_ex_Omega,solutions,normes,solver.V_ex,project_on_Omega=True)

