###########
# Imports #
###########

import matplotlib.pyplot as plt
from pathlib import Path

from scimba.nets import mlp
from scimba.pinns import pinn_x, training_x
from scimba.sampling import sampling_pde, sampling_parameters, uniform_sampling

from modules.problem.Poisson2D import *

###############
# Define case #
###############

current = Path(__file__).parent.parent

def plot_sampling(bornes,data,name):
    plt.plot(data[:,0],data[:,1],"+")
    plt.xlim(bornes[0][0],bornes[0][1])
    plt.ylim(bornes[1][0],bornes[1][1])
    plt.title(name)

def plot_solution(cas,sampling_on,trainer,fig_path):
    pde = cas.class_PDE(cas.Problem, sampling_on=sampling_on, impose_exact_bc=cas.impose_exact_bc)
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(sampler=uniform_sampling.UniformSampling, model=pde)
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)
    
    trainer.plot(50000,filename=fig_path,sampler=sampler)

def test_laplacian_2d(cas, num_config, dict, save_sampling = False, new_training = False):
    ###
    # Sampler
    ###

    impose_exact_bc = cas.impose_exact_bc
    pde = cas.PDE
    dir_name = current / cas.dir_name


    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    if save_sampling:
        if cas.sampling_on=="Omega":
            bornes = pde.space_domain.surrounding_domain.bound
        elif cas.sampling_on=="O_cal":
            bornes = pde.space_domain.bound
        else:
            raise ValueError("Sampling_on must be either 'Omega' or 'O_cal'")
        
        if impose_exact_bc:
            data_inside = sampler.sampling(dict["n_collocations"])[0].detach().numpy()
            plt.figure(figsize=(5,5))
            plot_sampling(bornes,data_inside,"sampling on "+cas.sampling_on+" with "+str(dict["n_collocations"])+" points")

        else:
            plt.figure(figsize=(10,5))

            plt.subplot(1,2,1)
            data_inside = sampler.sampling(dict["n_collocations"])[0].detach().numpy()
            plot_sampling(bornes,data_inside,"sampling on "+cas.sampling_on)

            plt.subplot(1,2,2)
            data_boundary = sampler.bc_sampling(dict["n_bc_collocation"])[0].detach().numpy()
            plot_sampling(bornes,data_boundary,"border sampling")

        plt.savefig(dir_name / "models" / ("sampling_"+str(num_config)+".png"))

    ###
    # Model
    ###
    
    name = "model_"+str(num_config)

    file_path = dir_name / "models" / (name+".pth")
    if new_training:
        file_path.unlink(missing_ok=True)
    
    tlayers = dict["layers"]
    network = pinn_x.PINNx(
        net=mlp.GenericMLP, pde=pde, layer_sizes=tlayers, activation_type=dict["activation"]
    )

    ###
    # Trainer
    ###

    if impose_exact_bc:
        dict["n_bc_collocation"] = 0
        dict["w_bc"] = 0.

    trainer = training_x.TrainerPINNSpace(
        pde=pde,
        network=network,
        sampler=sampler,
        file_name=file_path,
        bc_loss_bool=not impose_exact_bc,
        decay=dict["decay"],
        batch_size=dict["n_collocations"],
        w_data=dict["w_data"],
        w_res=dict["w_res"],
        w_bc=dict["w_bc"],
    )

    if new_training or trainer.to_be_trained:
        # si lr est une liste on entrainer n_epoch/len(lr) fois avec chaque lr
        if isinstance(dict["lr"],list):
            # on récupère le nombre d'époques (pas forcément divisible par len(lr))
            n_epochs = dict["n_epochs"]//len(dict["lr"])
            tab_n_epochs = [n_epochs]*len(dict["lr"])
            tab_n_epochs[-1] += dict["n_epochs"]%len(dict["lr"])
            for (i,lr) in enumerate(dict["lr"]): 
                print("## Train ", tab_n_epochs[i], " epochs with lr = ", lr)
                trainer.learning_rate = lr
                trainer.train(epochs=tab_n_epochs[i], n_collocation=dict["n_collocations"], n_bc_collocation=dict["n_bc_collocation"], n_data=dict["n_data"])
        else:
            trainer.learning_rate = dict["lr"]
            trainer.train(epochs=dict["n_epochs"], n_collocation=dict["n_collocations"], n_bc_collocation=dict["n_bc_collocation"], n_data=dict["n_data"])
    
    ###
    # Plot solutions
    ###

    # plot sur O_cal
    fig_path = dir_name / "solutions" / (name+"_O_cal.png")
    plot_solution(cas,"O_cal",trainer,fig_path)
    
    # plot sur Omega
    fig_path = dir_name / "solutions" / (name+"_Omega.png")
    plot_solution(cas,"Omega",trainer,fig_path)

    return trainer

