###########
# Imports #
###########

import matplotlib.pyplot as plt
import argparse
import os

from scar.problem.Case import *
from scar.utils import read_config
from scar.equations.run_Poisson2D import *
from scar.solver.solver_fem import *
from scar.solver.solver_phifem import *
import pandas as pd

#############
# Arguments #
#############

# Récupération des arguments lors de l'exécution du script python
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", help="Index of configuration file.", type=int, default=0)
    parser.add_argument("--casefile", help="Path to the case file.", type=str, default="case.json")
    
    parser.add_argument("--fem", help="0 - False ; 1 - True", type=int, default=1)
    parser.add_argument("--phifem", help="0 - False ; 1 - True", type=int, default=1)
    parser.add_argument("--pinns", help="0 - False ; 1 - True", type=int, default=1)
    parser.add_argument("--corr_fem", help="0 - False ; 1 - True", type=int, default=1)
    parser.add_argument("--corr_phifem", help="0 - False ; 1 - True", type=int, default=1)
    parser.add_argument("--load", help="If activate, just try to load data", action='store_true')

    args = parser.parse_args()

    return args, parser

args, parser = get_args()

###############
# Define case #
###############

casefile = args.casefile
cas = Case("../testcases/"+casefile)

problem_considered = cas.problem
pde_considered = cas.pde

dir_name = "../"+cas.dir_name
models_dir = dir_name+"models/"

################
# Define paths #
################

if args.load:
    args.fem = 0
    args.phifem = 0
    args.pinns = 0
    args.corr_fem = 0
    args.corr_phifem = 0

deg_corr = 10
result_dir = dir_name+"time_precision/"+"config"+str(args.config)+"_P"+str(deg_corr)+"/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
print(result_dir)

##############
# LOAD MODEL #
##############

config = args.config

config_filename = models_dir+"config_"+str(config)+".json"
model_filename = models_dir+"model_"+str(config)+".pth"
print("### Config file : ",config_filename)
print("### Model file : ",model_filename)

dict = read_config(config_filename)
print("### Config ", config, " : ", dict)

# Load model #

trainer = test_laplacian_2d(cas,config,dict)

# Get test sample #

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

# Get u_PINNs #

def get_u_PINNs(solver,parameter_domain,deg_corr):
    V_phi,X_test,mu_test = get_test_sample(solver,parameter_domain,deg_corr)

    pred = trainer.network.setup_w_dict(X_test, mu_test)
    phi_tild = pred["w"][:,0].cpu().detach().numpy()
    u_PINNs = Function(V_phi)
    u_PINNs.vector()[:] = phi_tild

    return u_PINNs

##################
# TIME PRECISION #
##################

params = [[0.5,1.0,0.]]

tab_nb_vert = [8*2**i for i in range(5)]
tab_nb_vert_corr = [5*i for i in range(1,7)]
np.save(result_dir+"tab_nb_vert_corr.npy",tab_nb_vert_corr)
np.save(result_dir+"tab_nb_vert.npy",tab_nb_vert)

def get_time_precision(class_solver,result_subdir):
    subtimes = {}
    times = []
    norms = []
    for nb_vert in tab_nb_vert:
        print("# nb_vert : ", nb_vert)
        solver = class_solver(nb_cell=nb_vert-1, params=params, cas=cas)
        sol,norm_L2 = solver.fem(0)
        times_fem = solver.times_fem
        for key in times_fem:
            if key in subtimes:
                subtimes[key].append(times_fem[key])
            else:
                subtimes[key] = [times_fem[key]]
        times.append(sum(times_fem.values()))
        norms.append(norm_L2)
        
    np.save(result_subdir+"norms.npy",norms)
    np.save(result_subdir+"times.npy",times)
    np.save(result_subdir+"subtimes.npy",subtimes)

    return subtimes, times, norms

def get_time_precision_corr(class_solver,result_subdir,get_pinns=False):
    subtimes = {}
    times = []
    norms = []

    times_pinns = []
    norms_pinns = []

    for nb_vert in tab_nb_vert_corr:
        print("# nb_vert : ", nb_vert)

        solver = class_solver(nb_cell=nb_vert-1, params=params, cas=cas)

        ###
        # PINNs
        ###
        
        start = time.time()
        u_PINNs = get_u_PINNs(solver,trainer.pde.parameter_domain,deg_corr)
        end = time.time()
        
        time_get_u_PINNs = end-start
        print("Time to get u_PINNs : ",time_get_u_PINNs)

        if get_pinns:
            subtimes_pinns = solver.times_corr_add.copy()
            subtimes_pinns["get_u_PINNs"] = time_get_u_PINNs
            times_pinns.append(sum(subtimes_pinns.values()))

            u_ex = UexExpr(params[0], degree=deg_corr, domain=solver.mesh, pb_considered=solver.pb_considered)
            norm_L2_PINNs = (assemble((((u_ex - u_PINNs)) ** 2) * solver.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * solver.dx) ** (0.5))
            norms_pinns.append(norm_L2_PINNs)

        ###
        # Correction
        ###

        u_Corr,C,norm_L2_Corr = solver.corr_add(0,u_PINNs)

        times_corr_add = solver.times_corr_add
        times_corr_add["get_u_PINNs"] = time_get_u_PINNs
        for key in times_corr_add:
            if key in subtimes:
                subtimes[key].append(times_corr_add[key])
            else:
                subtimes[key] = [times_corr_add[key]]

        times.append(sum(times_corr_add.values()))
        norms.append(norm_L2_Corr)

    np.save(result_subdir+"norms_pinns.npy",norms_pinns)
    np.save(result_subdir+"times_pinns.npy",times_pinns)
    np.save(result_subdir+"norms.npy",norms)
    np.save(result_subdir+"times.npy",times)
    np.save(result_subdir+"subtimes.npy",subtimes)

    return subtimes, times, norms, times_pinns, norms_pinns

# COMPUTE FEM

print("## FEM")

result_subdir = result_dir+"fem/"
if not os.path.exists(result_subdir):
    os.makedirs(result_subdir)

if args.fem:
    subtimes_fem, times_fem, norms_fem = get_time_precision(FEMSolver,result_subdir)
else:
    try:
        norms_fem = np.load(result_subdir+"norms.npy")
        times_fem = np.load(result_subdir+"times.npy")
        subtimes_fem = np.load(result_subdir+"subtimes.npy",allow_pickle=True).item()
    except:
        print("Please run the script with run=True to compute the results")

print("subtimes_fem : ", subtimes_fem)
print("times_fem : ", times_fem)
print("norms_fem : ", norms_fem)

# COMPUTE PHIFEM

print("## PHIFEM")

result_subdir = result_dir+"phifem/"
if not os.path.exists(result_subdir):
    os.makedirs(result_subdir)

if args.phifem:
    subtimes_phifem, times_phifem, norms_phifem = get_time_precision(PhiFemSolver,result_subdir)
else:
    try:
        norms_phifem = np.load(result_subdir+"norms.npy")
        times_phifem = np.load(result_subdir+"times.npy")
        subtimes_phifem = np.load(result_subdir+"subtimes.npy",allow_pickle=True).item()
    except:
        print("Please run the script with run=True to compute the results")

print("subtimes_phifem : ", subtimes_phifem)

# COMPUTE CORR_FEM

print("## Corr_add_FEM")

result_subdir = result_dir+"corr_fem/"
if not os.path.exists(result_subdir):
    os.makedirs(result_subdir)

if args.corr_fem:
    subtimes_corr_add_fem, times_corr_add_fem, norms_corr_add_fem, times_pinns, norms_pinns = get_time_precision_corr(FEMSolver,result_subdir,get_pinns=True)
else:
    try:
        norms_pinns = np.load(result_subdir+"norms_pinns.npy")
        times_pinns = np.load(result_subdir+"times_pinns.npy")

        norms_corr_add_fem = np.load(result_subdir+"norms.npy")
        times_corr_add_fem = np.load(result_subdir+"times.npy")
        subtimes_corr_add_fem = np.load(result_subdir+"subtimes.npy",allow_pickle=True).item()
    except:
        print("Please run the script with run=True to compute the results")

# COMPUTE CORR_PHIFEM

print("Corr_add_PhiFEM")

result_subdir = result_dir+"corr_phifem/"
if not os.path.exists(result_subdir):
    os.makedirs(result_subdir)

if args.corr_phifem:
    subtimes_corr_add_phifem, times_corr_add_phifem, norms_corr_add_phifem, _, _ = get_time_precision_corr(PhiFemSolver,result_subdir)
else:
    try:
        norms_corr_add_phifem = np.load(result_subdir+"norms.npy")
        times_corr_add_phifem = np.load(result_subdir+"times.npy")
        subtimes_corr_add_phifem = np.load(result_subdir+"subtimes.npy",allow_pickle=True).item()
    except:
        print("Please run the script with run=True to compute the results")
        
#############
# SAVE DATA #
#############

save_data = True

if save_data:

    # FEMs

    methods = ["FEM","PHIFEM"]
    values_names = ["norms","times"]

    tab = pd.MultiIndex.from_product([methods, values_names])
    tab = tab.insert(0,('','n_vert'))

    data = [tab_nb_vert, norms_fem, times_fem, norms_phifem, times_phifem]

    df = pd.DataFrame(data, index=tab)
    print(df)

    df.to_excel(result_dir+'results_FEMs.xlsx', header=False)

    # Corr_add

    methods = ["Corr_add_FEM","Corr_add_PHIFEM"]
    values_names = ["norms","times"]

    tab = pd.MultiIndex.from_product([methods, values_names])
    tab = tab.insert(0,('','n_vert'))

    data = [tab_nb_vert_corr, norms_corr_add_fem, times_corr_add_fem, norms_corr_add_phifem, times_corr_add_phifem]

    df = pd.DataFrame(data, index=tab)
    print(df)

    df.to_excel(result_dir+'results_Corr.xlsx', header=False)

#############
# PLOT DATA #
###########

plot_data = True

import seaborn as sns
sns.set_theme(style="whitegrid")
sns.despine(left=True)

if plot_data:
    colors = ["tab:blue","tab:red","tab:cyan","tab:orange"]

    plt.figure()

    def plot_values(times,norms,color,label):
        plt.loglog(times,norms,"+-",color=color,label=label,markersize=10,markeredgewidth=2)

    plot_values(times_fem,norms_fem,colors[0],"FEM")
    plot_values(times_phifem,norms_phifem,colors[1],"PHIFEM")

    plot_values(times_corr_add_fem,norms_corr_add_fem,colors[2],"Corr_add_FEM")
    plot_values(times_corr_add_phifem,norms_corr_add_phifem,colors[3],"Corr_add_PHIFEM")

    plt.xlabel("Time")
    plt.ylabel("L2 norm")

    plt.legend()
    plt.savefig(result_dir+"time_precision.png")
    # plt.show()

###############
# TIMES TABLE #
###############
    
times_table = False

if times_table:

    # SUM Omega_h and cells-facets in mesh -> mesh for PhiFEM
        # => ils ont tous les mêmes clés
        
    subtimes_phifem["mesh"] = list(np.array(subtimes_phifem["Omega_h"])+np.array(subtimes_phifem["cells-facets"]))
    del subtimes_phifem["Omega_h"]
    del subtimes_phifem["cells-facets"]

    subtimes_corr_add_phifem["mesh"] = list(np.array(subtimes_corr_add_phifem["Omega_h"])+np.array(subtimes_corr_add_phifem["cells-facets"]))
    del subtimes_corr_add_phifem["Omega_h"]
    del subtimes_corr_add_phifem["cells-facets"]

    # Times for get_u_PINNs with FEM and PhiFEM = None
    
    subtimes_fem["get_u_PINNs"] = None
    subtimes_phifem["get_u_PINNs"] = None

    # Create dict of times and norms

    methods = ["FEM","PHIFEM","Corr_FEM","Corr_PHIFEM"]
    steps = ["mesh","get_u_PINNs","assemble","solve","TOTAL"]

    norms = {"FEM":norms_fem,"PHIFEM":norms_phifem,"Corr_FEM":norms_corr_add_fem,"Corr_PHIFEM":norms_corr_add_phifem}
    subtimes = {"FEM":subtimes_fem,"PHIFEM":subtimes_phifem,"Corr_FEM":subtimes_corr_add_fem,"Corr_PHIFEM":subtimes_corr_add_phifem}
    times = {"FEM":times_fem,"PHIFEM":times_phifem,"Corr_FEM":times_corr_add_fem,"Corr_PHIFEM":times_corr_add_phifem}

    for method in methods:
        subtimes[method]["TOTAL"] = times[method]

    ###
    # Interpolation
    ###

    def get_index(norms,given_precision):
        if given_precision < norms[-1]:
            index = len(norms)-1
        elif given_precision > norms[0]:
            index = 0
        else:           
            index = np.where(norms < given_precision)[0][0]
        
        return index
    
    def linear_interpolation(norms,times,index):
        norm = norms[index-1:index+1]
        time = times[index-1:index+1]

        t_inter = time[0]+(time[1]-time[0])/(norm[1]-norm[0])*(given_precision-norm[0])
        
        return t_inter
    
    given_precision = 1e-4
    
    times_inter = {}
    for method in methods:
        times_inter[method] = {}
        for key in steps:
            if not subtimes[method][key] is None:
                index = get_index(norms[method],given_precision)
                t_inter = linear_interpolation(norms[method],subtimes[method][key],index)
                times_inter[method][key] = t_inter
            else:
                times_inter[method][key] = None
    
    # Create a dataframe with the times interpolated

    df = pd.DataFrame.from_dict(times_inter).T
    # réorganiser lignes et colonnes par clés
    df = df.reindex(methods)
    df = df[steps]

    # Create an excel file with the times

    df.to_excel(result_dir+'times_table.xlsx')