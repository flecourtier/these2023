import time
import numpy as np
import os
import torch
from dolfin import *

from scar.correction.correct_pred import get_u_PINNs
from scar.solver.fenics_expressions import UexExpr

def get_params(trainer):
    if trainer.pde.nb_parameters == 0:
        params = [[None,None,None]]
    else:
        mu = torch.mean(trainer.pde.parameter_domain, axis=1)
        S,f,p = mu.numpy()
        params = [[S,f,p]]
    return params

def get_time_precision(class_solver,trainer,result_subdir,tab_nb_vert,cas):
    params = get_params(trainer)
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

def get_time_precision_corr(class_solver,trainer,result_subdir,tab_nb_vert_corr,cas,deg_corr,get_pinns=False):
    params = get_params(trainer)
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
        u_PINNs,_ = get_u_PINNs(trainer,solver,deg_corr)
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
