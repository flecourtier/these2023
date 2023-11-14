import argparse
import json
import os

from modules.run_laplacian import *
from modules.Poisson2D import *

from modules.solver.solver_phifem import *
from modules.solver.solver_fem import *
from create_xlsx_file import *

classe_ = Problem()

problem_considered = classe_.class_pb_considered
pde_considered = classe_.pde_considered
name_problem_considered = classe_.name_problem_considered
name_pde_considered = classe_.name_pde_considered
dir_name = classe_.dir_name

result_dir = dir_name+"results/corr/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# lire un fichier de configuration
def read_config(filename):
    with open(filename) as f:
        raw_config = f.read()
        dict = json.loads(raw_config)
    return  dict

###
# Récupération des arguments lors de l'exécution du script python
###

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", help="Index of configuration file.", type=int, default=0)
    parser.add_argument("--bc", "--boundary_conditions", help="Don't use the signed distance function to impose exact boundary conditions.", action='store_false')
    parser.add_argument("--fem", help="0 - use FEM : 1 - use PhiFEM ; 2 - both.", type=int, default=2)

    args = parser.parse_args()

    return args, parser


args, parser = get_args()

# imposition des conditions exactes au bord ?
if args.bc:
    end = "_exact_bc"
else:
    end = ""

config = args.config


config_filename = dir_name+"configs/config_"+str(config)+".json"
model_filename = dir_name+"models/model_"+str(config)+end+".pth"
print("### Config file : ",config_filename)
print("### Model file : ",model_filename)

dict = read_config(config_filename)
print("### Config ", config, " : ", dict)

trainer = test_laplacian_2d(problem_considered,pde_considered,config,dict,args.bc)

S,f,p = (0.5,1,0)
nb_vert = 32
params = [[S,f,p]]

deg_corr = 10

# si FEM ou Both
if args.fem != 1:
    print("### Correction par addition avec FEM Standard")

    #####
    # Compute !
    #####

    solver = FEMSolver(nb_cell=nb_vert-1, params=params)

    # u_ex = UexExpr(params[0], degree=deg_corr, domain=solver.mesh)
    
    # get coordinates of the dof
    V_phi = FunctionSpace(solver.mesh,"CG",deg_corr)
    XXYY = V_phi.tabulate_dof_coordinates()
    X_test = torch.tensor(XXYY)
    mu = torch.ones(XXYY.shape[0],trainer.pde.nb_parameters)*S

    # get u_PINNs
    pred = trainer.network.setup_w_dict(X_test, mu)
    phi_tild = pred["w"][:,0].cpu().detach().numpy()
    u_PINNs = Function(V_phi)
    u_PINNs.vector()[:] = phi_tild

    assert np.max(np.abs(phi_tild-u_PINNs.vector()[:]))<1e-10

    # norm_L2_PINNs = (assemble((((u_ex - u_PINNs)) ** 2) * solver.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * solver.dx) ** (0.5))

    # get u_corr
    u_Corr,C,norm_L2_Corr = solver.corr_add(0,u_PINNs)
    print("# Error L2 - Corr ",norm_L2_Corr)

    # get u_FEM
    u_FEM,norm_L2_FEM = solver.fem(0)
    print("# Error L2 - FEM : ",norm_L2_FEM)
    print("# Facteur : ", norm_L2_FEM/norm_L2_Corr)

    u_ex = UexExpr(params[0], degree=1, domain=solver.mesh)
    u_ex = project(u_ex, solver.V)
    print(project(u_ex-u_FEM,solver.V).vector()[:].shape)
    print(np.max(np.abs(project(u_ex-u_FEM,solver.V).vector()[:])))

    
    # #####
    # # Plot !
    # #####
    
    u_ex = project(u_ex, solver.V)

    plt.figure(figsize=(20,15))
    
    plt.subplot(3,3,1)
    plt.text(0.2,0.75,"FEM/PINNs = {:.2f}".format(norm_L2_FEM/norm_L2_PINNs),fontsize=15)
    plt.text(0.2,0.5,"FEM/Corr = {:.2f}".format(norm_L2_FEM/norm_L2_Corr),fontsize=15)
    plt.text(0.2,0.25,"PINNs/Corr = {:.2f}".format(norm_L2_PINNs/norm_L2_Corr),fontsize=15)
    plt.axis('off')

    # FEM
    plt.subplot(3,3,2)
    c = plot(u_FEM, title="u_FEM")
    plt.ylabel("FEM", fontsize=20)
    plt.colorbar(c)

    plt.subplot(3,3,3)
    error = abs(u_ex-project(u_FEM,solver.V))
    c = plot(error, title="||u_ex-u_FEM||_L2 = {:.2e}".format(norm_L2_FEM))
    plt.colorbar(c)

    # PINNs
    plt.subplot(3,3,4)
    c = plot(u_ex, title="u_ex")
    plt.colorbar(c)

    plt.subplot(3,3,5)
    c = plot(u_PINNs, title="u_PINNs")
    plt.ylabel("PINNs", fontsize=20)
    plt.colorbar(c)

    plt.subplot(3,3,6)
    error = abs(u_ex-project(u_PINNs,solver.V))
    c = plot(error, title="||u_ex-u_PINNs||_L2 : {:.2e}".format(norm_L2_PINNs))
    plt.colorbar(c)

    # Corr
    plt.subplot(3,3,7)
    c = plot(C, title="C_tild")
    plt.ylabel("Corr", fontsize=20)
    plt.colorbar(c) 

    plt.subplot(3,3,8)
    c = plot(u_Corr, title="u_Corr")
    plt.colorbar(c)

    plt.subplot(3,3,9)
    error = abs(u_ex-project(u_Corr,solver.V))
    c = plot(error, title="||u_ex-u_Corr||_L2 : {:.2e}".format(norm_L2_Corr))
    plt.colorbar(c)

    plt.savefig(result_dir+"corr_fem_"+str(config)+end+".png")

# si PhiFEM ou Both
on_Omega = False
if args.fem != 0:
    if not on_Omega:
        print("### Correction par addition avec PhiFEM")

        #####
        # Compute !
        #####

        solver = PhiFemSolver(nb_cell=nb_vert-1, params=params)

        u_ex = UexExpr(params[0], degree=deg_corr, domain=solver.mesh)

        # get coordinates of the dof
        V_phi = FunctionSpace(solver.mesh,"CG",deg_corr)
        XXYY = V_phi.tabulate_dof_coordinates()
        X_test = torch.tensor(XXYY)
        mu = torch.ones(XXYY.shape[0],trainer.pde.nb_parameters)*S

        # get u_PINNs
        pred = trainer.network.setup_w_dict(X_test, mu)
        phi_tild = pred["w"][:,0].cpu().detach().numpy()
        u_PINNs = Function(V_phi)
        u_PINNs.vector()[np.arange(0,phi_tild.shape[0],1)] = phi_tild
        norm_L2_PINNs = (assemble((((u_ex - u_PINNs)) ** 2) * solver.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * solver.dx) ** (0.5))

        # get u_corr
        u_Corr,C,norm_L2_Corr = solver.corr_add(0,u_PINNs)
        print("# Error L2 - Corr ",norm_L2_Corr)

        # get u_PhiFEM
        u_PhiFEM,norm_L2_PhiFEM = solver.fem(0)
        print("# Error L2 - PhiFEM : ",norm_L2_PhiFEM)
        print("# Facteur : ", norm_L2_PhiFEM/norm_L2_Corr)

        
        #####
        # Plot !
        #####
        
        u_ex = project(u_ex, solver.V)

        plt.figure(figsize=(20,15))
        
        plt.subplot(3,3,1)
        plt.text(0.2,0.75,"PhiFEM/PINNs = {:.2f}".format(norm_L2_PhiFEM/norm_L2_PINNs),fontsize=15)
        plt.text(0.2,0.5,"PhiFEM/Corr = {:.2f}".format(norm_L2_PhiFEM/norm_L2_Corr),fontsize=15)
        plt.text(0.2,0.25,"PINNs/Corr = {:.2f}".format(norm_L2_PINNs/norm_L2_Corr),fontsize=15)
        plt.axis('off')

        # FEM
        plt.subplot(3,3,2)
        c = plot(u_PhiFEM, title="u_PhiFEM")
        plt.ylabel("PhiFEM", fontsize=20)
        plt.colorbar(c)

        plt.subplot(3,3,3)
        error = abs(u_ex-project(u_PhiFEM,solver.V))
        c = plot(error, title="||u_ex-u_PhiFEM||_L2 = {:.2e}".format(norm_L2_PhiFEM))
        plt.colorbar(c)

        # PINNs
        plt.subplot(3,3,4)
        c = plot(u_ex, title="u_ex")
        plt.colorbar(c)

        plt.subplot(3,3,5)
        c = plot(u_PINNs, title="u_PINNs")
        plt.ylabel("PINNs", fontsize=20)
        plt.colorbar(c)

        plt.subplot(3,3,6)
        error = abs(u_ex-project(u_PINNs,solver.V))
        c = plot(error, title="||u_ex-u_PINNs||_L2 : {:.2e}".format(norm_L2_PINNs))
        plt.colorbar(c)

        # Corr
        plt.subplot(3,3,7)
        c = plot(C, title="C_tild")
        plt.ylabel("Corr", fontsize=20)
        plt.colorbar(c) 

        plt.subplot(3,3,8)
        c = plot(u_Corr, title="u_Corr")
        plt.colorbar(c)

        plt.subplot(3,3,9)
        error = abs(u_ex-project(u_Corr,solver.V))
        c = plot(error, title="||u_ex-u_Corr||_L2 : {:.2e}".format(norm_L2_Corr))
        plt.colorbar(c)

        plt.savefig(result_dir+"corr_phifem_"+str(config)+end+".png")
    else:
        print("### Correction par addition avec PhiFEM - projection sur Omega")

        #####
        # Compute !
        #####

        solver = PhiFemSolver(nb_cell=nb_vert-1, params=params)

        # get coordinates of the dof
        V_phi = FunctionSpace(solver.mesh,"CG",deg_corr)
        XXYY = V_phi.tabulate_dof_coordinates()
        X_test = torch.tensor(XXYY)
        mu = torch.ones(XXYY.shape[0],trainer.pde.nb_parameters)*S

        # get u_PINNs
        pred = trainer.network.setup_w_dict(X_test, mu)
        phi_tild = pred["w"][:,0].cpu().detach().numpy()
        u_PINNs = Function(V_phi)
        u_PINNs.vector()[np.arange(0,phi_tild.shape[0],1)] = phi_tild

        # get u_corr
        u_Corr_Omega,C,norm_L2_Corr_Omega = solver.corr_add(0,u_PINNs,on_Omega)
        print("# Error L2 Omega - Corr ",norm_L2_Corr_Omega)

        # get u_PhiFEM
        u_PhiFEM_Omega,norm_L2_PhiFEM_Omega = solver.fem(0,on_Omega)
        print("# Error L2 Omega - PhiFEM : ",norm_L2_PhiFEM_Omega)
        print("# Facteur : ", norm_L2_PhiFEM_Omega/norm_L2_Corr_Omega)

        
        #####
        # Plot !
        #####
        
        u_ex = UexExpr(params[0], degree=deg_corr, domain=solver.mesh_ex)
        u_ex = project(u_ex, solver.V)
        u_ex = project(u_ex, solver.V_ex)

        plt.figure(figsize=(20,15))
        
        plt.subplot(3,3,1)
        plt.text(0.2,0.5,"PhiFEM/Corr = {:.2f}".format(norm_L2_PhiFEM_Omega/norm_L2_Corr_Omega),fontsize=15)
        plt.axis('off')

        # FEM
        plt.subplot(2,3,2)
        c = plot(u_PhiFEM_Omega, title="u_PhiFEM")
        plt.ylabel("PhiFEM", fontsize=20)
        plt.colorbar(c)

        plt.subplot(2,3,3)
        error = abs(u_ex-project(u_PhiFEM_Omega,solver.V_ex))
        c = plot(error, title="||u_ex-u_PhiFEM||_L2 = {:.2e}".format(norm_L2_PhiFEM_Omega))
        plt.colorbar(c)

        # Corr
        plt.subplot(2,3,4)
        c = plot(u_ex, title="u_ex")
        plt.colorbar(c)
        
        plt.subplot(2,3,5)
        c = plot(u_Corr_Omega, title="u_Corr")
        plt.ylabel("Corr", fontsize=20)
        plt.colorbar(c)

        plt.subplot(2,3,6)
        error = abs(u_ex-project(u_Corr_Omega,solver.V_ex))
        c = plot(error, title="||u_ex-u_Corr||_L2 : {:.2e}".format(norm_L2_Corr_Omega))
        plt.colorbar(c)

        plt.savefig(result_dir+"corr_phifem_"+str(config)+"_Omega"+end+".png")

create_xlsx_file(problem_considered,pde_considered)