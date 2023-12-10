import matplotlib.pyplot as plt

from modules.solver.solver_fem_time import *
from modules.solver.solver_phifem_time import *

import os
result_dir = "time_precision/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

run = True
params = [[None,None,None]]

tab_nb_vert = [8*2**i for i in range(5)]

# COMPUTE FEM

print("FEM")

if run:
    subtimes_fem = {}
    times_fem = []
    norms_fem = []
    for nb_vert in tab_nb_vert:
        print("###### nb_vert : ", nb_vert)
        solver = FEMSolver(nb_cell=nb_vert-1,params=params)
        sol,norm_L2 = solver.fem(0)
        subtimes = solver.times_fem
        for key in subtimes:
            if key in subtimes_fem:
                subtimes_fem[key].append(subtimes[key])
            else:
                subtimes_fem[key] = [subtimes[key]]
        times_fem.append(sum(subtimes.values()))
        norms_fem.append(norm_L2)

    print("subtimes_fem : ", subtimes_fem)
    print("times_fem : ", times_fem)

    mean_subtimes_fem = {}
    for key in subtimes_fem:
        mean_subtimes_fem[key] = np.mean(subtimes_fem[key])

    print("mean_subtimes_fem : ", mean_subtimes_fem)

    # # Calcul des proportions
    # proportions = list(mean_subtimes_fem.values())

    # # Labels des clés
    # labels = list(mean_subtimes_fem.keys())

    # proportions = [0.05,0.05,0.9]

    # # Définir les couleurs pour chaque partie de la barre
    # couleurs = ['red', 'green', 'blue']

    # # Tracer la barre avec les proportions
    # # Calculer les sommets cumulatives
    # valeurs_cumulatives = np.cumsum(proportions)

    # # Tracer la barre cumulée
    # plt.bar([0], valeurs_cumulatives[0], color=couleurs[0])
    # plt.bar([0], valeurs_cumulatives[1], bottom=valeurs_cumulatives[0], color=couleurs[1])
    # plt.bar([0], valeurs_cumulatives[2], bottom=valeurs_cumulatives[1], color=couleurs[2])

    # # Affichage de la barre en échelle logarithmique
    # # plt.bar(labels, proportions[0], width=0.5, label='Proportion 1')
    # # plt.bar(labels, proportions[1], width=0.5, bottom=proportions[0], label='Proportion 2')
    # # plt.bar(labels, proportions[2], width=0.5, bottom=[proportions[0][i] + proportions[1][i] for i in range(len(labels))], label='Proportion 3')
    # plt.legend()
    # plt.yscale('log')
    # # fix scale to 1
    # plt.ylim(0,1)

    # # Ajout des labels
    # for i in range(len(labels)):
    #     plt.text(i, proportions[i], str(round(proportions[i], 2)), ha='center')

    # # Affichage du graphique
    # plt.show()
    
    # np.save(result_dir+"tab_nb_vert.npy",tab_nb_vert)
    # np.save(result_dir+"norms_fem.npy",norms_fem)
    # np.save(result_dir+"times_fem.npy",times_fem)
    # np.save(result_dir+"subtimes_fem.npy",subtimes_fem)
    # np.save(result_dir+"mean_subtimes_fem.npy",mean_subtimes_fem)
else:
    try:
        tab_nb_vert = np.load(result_dir+"tab_nb_vert.npy")
        norms_fem = np.load(result_dir+"norms_fem.npy")
        times_fem = np.load(result_dir+"times_fem.npy")
        assert len(tab_nb_vert) == len(norms_fem)
        assert len(tab_nb_vert) == len(times_fem)
        subtimes_fem = np.load(result_dir+"subtimes_fem.npy",allow_pickle=True).item()
        mean_subtimes_fem = np.load(result_dir+"mean_subtimes_fem.npy",allow_pickle=True).item()
    except:
        print("Please run the script with run=True to compute the results")

# COMPUTE PHIFEM

run = False

print("PHIFEM")

if run:
    subtimes_phifem = {}
    times_phifem = []
    norms_phifem = []
    for nb_vert in tab_nb_vert:
        print("###### nb_vert : ", nb_vert)
        solver = PhiFemSolver(nb_cell=nb_vert-1,params=params)
        sol,norm_L2 = solver.fem(0)
        subtimes = solver.times_fem
        for key in subtimes:
            if key in subtimes_phifem:
                subtimes_phifem[key].append(subtimes[key])
            else:
                subtimes_phifem[key] = [subtimes[key]]
        times_phifem.append(sum(subtimes.values()))
        norms_phifem.append(norm_L2)

    print("norms_phifem : ", norms_phifem)
    print("times_fem : ", times_phifem)

    mean_subtimes_phifem = {}
    for key in subtimes_phifem:
        mean_subtimes_phifem[key] = np.mean(subtimes_phifem[key])

    print("mean_subtimes_phifem : ", mean_subtimes_phifem)

    np.save(result_dir+"norms_phifem.npy",norms_phifem)
    np.save(result_dir+"times_phifem.npy",times_phifem)
    np.save(result_dir+"subtimes_phifem.npy",subtimes_phifem)
    np.save(result_dir+"mean_subtimes_phifem.npy",mean_subtimes_phifem)
else:
    try:
        norms_phifem = np.load(result_dir+"norms_phifem.npy")
        times_phifem = np.load(result_dir+"times_phifem.npy")
        assert len(norms_phifem) == len(tab_nb_vert)
        assert len(times_phifem) == len(tab_nb_vert)
        subtimes_phifem = np.load(result_dir+"subtimes_phifem.npy",allow_pickle=True).item()
        mean_subtimes_phifem = np.load(result_dir+"mean_subtimes_phifem.npy",allow_pickle=True).item()
    except:
        print("Please run the script with run=True to compute the results")

# Correct PINNs prediction

# Récupération du modèle #

from modules.Case import *
from modules.utils import read_config
from modules.run_laplacian import *

# Define case #

cas = Case("case.json")

impose_exact_bc = cas.impose_exact_bc
problem_considered = cas.Problem
pde_considered = cas.PDE

dir_name = cas.dir_name
models_dir = dir_name+"models/"

# Choose config #

config = 0

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
        print("mu_test : ",mu_test.numpy())
    return V_phi,X_test,mu_test

# Get u_PINNs #

def get_u_PINNs(solver,parameter_domain,deg_corr):
    V_phi,X_test,mu_test = get_test_sample(solver,parameter_domain,deg_corr)

    pred = trainer.network.setup_w_dict(X_test, mu_test)
    phi_tild = pred["w"][:,0].cpu().detach().numpy()
    u_PINNs = Function(V_phi)
    u_PINNs.vector()[:] = phi_tild

    return u_PINNs

deg_corr = 10
tab_nb_vert_corr = [5*i for i in range(1,7)]
np.save(result_dir+"tab_nb_vert_corr.npy",tab_nb_vert_corr)

# Correction with FEM

print("Corr_add_FEM")

if run:
    times_pinns = []
    norms_pinns = []

    subtimes_corr_add_fem = {}
    times_corr_add_fem = []
    norms_corr_add_fem = []
    for nb_vert in tab_nb_vert_corr:
        print("###### nb_vert : ", nb_vert)
        solver = FEMSolver(nb_cell=nb_vert-1, params=params)

        # get u_ex
        u_ex = UexExpr(params[0], degree=deg_corr, domain=solver.mesh)

        # get u_PINNs
        start = time.time()
        u_PINNs = get_u_PINNs(solver,trainer.pde.parameter_domain,deg_corr)
        end = time.time()
        time_get_u_PINNs = end-start

        print("Time to get u_PINNs : ",time_get_u_PINNs)

        subtimes = solver.times_corr_add
        subtimes["get_u_PINNs"] = time_get_u_PINNs

        norm_L2_PINNs = (assemble((((u_ex - u_PINNs)) ** 2) * solver.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * solver.dx) ** (0.5))
        print("# Error L2 - PINNs : ",norm_L2_PINNs)

        times_pinns.append(sum(subtimes.values()))
        norms_pinns.append(norm_L2_PINNs)

        u_Corr,C,norm_L2_Corr = solver.corr_add(0,u_PINNs)

        subtimes = solver.times_corr_add
        subtimes["get_u_PINNs"] = time_get_u_PINNs
        for key in subtimes:
            if key in subtimes_corr_add_fem:
                subtimes_corr_add_fem[key].append(subtimes[key])
            else:
                subtimes_corr_add_fem[key] = [subtimes[key]]

        times_corr_add_fem.append(sum(subtimes.values()))
        norms_corr_add_fem.append(norm_L2_Corr)

    mean_subtimes_corr_add_fem = {}
    for key in subtimes_corr_add_fem:
        mean_subtimes_corr_add_fem[key] = np.mean(subtimes_corr_add_fem[key])
    
    print("mean_subtimes_corr_add_fem : ", mean_subtimes_corr_add_fem)

    np.save(result_dir+"norms_pinns.npy",norms_pinns)
    np.save(result_dir+"times_pinns.npy",times_pinns)
    np.save(result_dir+"norms_corr_add_fem.npy",norms_corr_add_fem)
    np.save(result_dir+"times_corr_add_fem.npy",times_corr_add_fem)
    np.save(result_dir+"subtimes_corr_add_fem.npy",subtimes_corr_add_fem)
    np.save(result_dir+"mean_subtimes_corr_add_fem.npy",mean_subtimes_corr_add_fem)
else:
    try:
        norms_pinns = np.load(result_dir+"norms_pinns.npy")
        times_pinns = np.load(result_dir+"times_pinns.npy")
        norms_corr_add_fem = np.load(result_dir+"norms_corr_add_fem.npy")
        times_corr_add_fem = np.load(result_dir+"times_corr_add_fem.npy")
        assert len(norms_pinns) == len(tab_nb_vert)
        assert len(times_pinns) == len(tab_nb_vert)
        assert len(norms_corr_add_fem) == len(tab_nb_vert)
        assert len(times_corr_add_fem) == len(tab_nb_vert)
        subtimes_corr_add_fem = np.load(result_dir+"subtimes_corr_add_fem.npy",allow_pickle=True).item()
        mean_subtimes_corr_add_fem = np.load(result_dir+"mean_subtimes_corr_add_fem.npy",allow_pickle=True).item()
    except:
        print("Please run the script with run=True to compute the results")

# Correction with PHIFEM

print("Corr_add_PhiFEM")

if run:
    subtimes_corr_add_phifem = {}
    times_corr_add_phifem = []
    norms_corr_add_phifem = []
    for nb_vert in tab_nb_vert_corr:
        print("###### nb_vert : ", nb_vert)
        solver = PhiFemSolver(nb_cell=nb_vert-1, params=params)

        # get u_ex
        u_ex = UexExpr(params[0], degree=deg_corr, domain=solver.mesh)

        # get u_PINNs
        start = time.time()
        u_PINNs = get_u_PINNs(solver,trainer.pde.parameter_domain,deg_corr)
        end = time.time()
        time_get_u_PINNs = end-start

        print("Time to get u_PINNs : ",time_get_u_PINNs)

        norm_L2_PINNs = (assemble((((u_ex - u_PINNs)) ** 2) * solver.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * solver.dx) ** (0.5))
        print("# Error L2 - PINNs : ",norm_L2_PINNs)

        u_Corr,C,norm_L2_Corr = solver.corr_add(0,u_PINNs)

        subtimes = solver.times_corr_add
        subtimes["get_u_PINNs"] = time_get_u_PINNs
        for key in subtimes:
            if key in subtimes_corr_add_phifem:
                subtimes_corr_add_phifem[key].append(subtimes[key])
            else:
                subtimes_corr_add_phifem[key] = [subtimes[key]]

        times_corr_add_phifem.append(sum(subtimes.values()))        

        norms_corr_add_phifem.append(norm_L2_Corr)

    mean_subtimes_corr_add_phifem = {}
    for key in subtimes_corr_add_phifem:
        mean_subtimes_corr_add_phifem[key] = np.mean(subtimes_corr_add_phifem[key])
    
    print("mean_subtimes_corr_add_phifem : ", mean_subtimes_corr_add_phifem)

    np.save(result_dir+"norms_corr_add_phifem.npy",norms_corr_add_phifem)
    np.save(result_dir+"times_corr_add_phifem.npy",times_corr_add_phifem)
    np.save(result_dir+"subtimes_corr_add_phifem.npy",subtimes_corr_add_phifem)
    np.save(result_dir+"mean_subtimes_corr_add_phifem.npy",mean_subtimes_corr_add_phifem)
else:
    try:
        norms_corr_add_phifem = np.load(result_dir+"norms_corr_add_phifem.npy")
        times_corr_add_phifem = np.load(result_dir+"times_corr_add_phifem.npy")
        assert len(norms_corr_add_phifem) == len(tab_nb_vert)
        assert len(times_corr_add_phifem) == len(tab_nb_vert)
        subtimes_corr_add_phifem = np.load(result_dir+"subtimes_corr_add_phifem.npy",allow_pickle=True).item()
        mean_subtimes_corr_add_phifem = np.load(result_dir+"mean_subtimes_corr_add_phifem.npy",allow_pickle=True).item()
    except:
        print("Please run the script with run=True to compute the results")

print("tab_nb_vert : ", tab_nb_vert)
print("tab_nb_vert_corr : ", tab_nb_vert_corr)

print("norms_fem : ", norms_fem)
print("times_fem : ", times_fem)
print("mean_subtimes_fem : ", mean_subtimes_fem)

print("norms_phifem : ", norms_phifem)
print("times_phifem : ", times_phifem)
print("mean_subtimes_phifem : ", mean_subtimes_phifem)

print("norms_pinns : ", norms_pinns)
print("times_pinns : ", times_pinns)

print("norms_corr_add_fem : ", norms_corr_add_fem)
print("times_corr_add_fem : ", times_corr_add_fem)
print("mean_subtimes_corr_add_fem : ", mean_subtimes_corr_add_fem)

print("norms_corr_add_phifem : ", norms_corr_add_phifem)
print("times_corr_add_phifem : ", times_corr_add_phifem)
print("mean_subtimes_corr_add_phifem : ", mean_subtimes_corr_add_phifem)

plt.figure()
plt.loglog(times_fem,norms_fem,"+-",label="FEM")
# for i in range(len(tab_nb_vert)):
#     plt.text(times_fem[i],norms_fem[i],str(tab_nb_vert[i]),horizontalalignment='left',verticalalignment='bottom',fontsize=15)
plt.loglog(times_phifem,norms_phifem,"+-",label="PHIFEM")
plt.loglog(times_pinns,norms_pinns,"+-",label="PINNs")
plt.loglog(times_corr_add_fem,norms_corr_add_fem,"+-",label="Corr_add_FEM")
plt.loglog(times_corr_add_phifem,norms_corr_add_phifem,"+-",label="Corr_add_PHIFEM")
plt.xlabel("Time")
plt.ylabel("L2 norm")
plt.legend()
plt.savefig(result_dir+"time_precision.png")
plt.show()


# [0.13407755 0.30069423 0.48627949 0.96344686 1.39226723 2.01996827]