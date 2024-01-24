###########
# Imports #
###########

import matplotlib.pyplot as plt
from pathlib import Path

from scimba.equations import domain, pde_2d_laplacian
from scimba.pinns import pinn_x, training_x
from scimba.sampling import sampling_parameters, sampling_pde, uniform_sampling

from modules.problem.Poisson2D import *
import torch
torch.cuda.set_per_process_memory_fraction(0.8)

###############
# Define case #
###############

current = Path(__file__).parent.parent

def plot_sampling(bornes,data,name):
    plt.plot(data[:,0],data[:,1],"+")
    plt.xlim(bornes[0][0],bornes[0][1])
    plt.ylim(bornes[1][0],bornes[1][1])
    plt.title(name)

def plot_solution(cas,trainer,fig_path):
    pde = cas.pde
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(sampler=uniform_sampling.UniformSampling, model=pde)
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)
    
    trainer.plot(50000,filename=fig_path,sampler=sampler)

def test_laplacian_2d(cas, num_config, dict, save_sampling = False, new_training = False):
    ###
    # Sampler
    ###

    # xdomain = cas.xdomain
    pde = cas.pde

    dir_name = current / cas.dir_name

    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    if save_sampling:
        bornes = pde.space_domain.surrounding_domain.bound
        
        data_inside = sampler.sampling(dict["n_collocations"])[0].cpu().detach().numpy()
        plt.figure(figsize=(5,5))
        plot_sampling(bornes,data_inside,str(dict["n_collocations"])+" points")

        plt.savefig(dir_name / "models" / ("sampling_"+str(num_config)+".png"))

    save_phi = True
    if save_phi:
        data_inside = sampler.sampling(10000)[0]
        phi_inside = cas.sd_function.sdf(data_inside)[:,0].detach().cpu().numpy()
        data_inside = data_inside.detach().cpu().numpy()

        print("phi_inside.shape = ",phi_inside.shape)
        print("data_inside.shape = ",data_inside.shape)

        plt.figure(figsize=(15,5))
        plt.tricontourf(data_inside[:,0],data_inside[:,1],phi_inside,"o",levels=100)#,cmap="hot")
        plt.title("phi")
        plt.colorbar()

        plt.savefig(dir_name / "models" / ("phi_"+str(num_config)+".png"))

    ###
    # Model
    ###
    
    name = "model_"+str(num_config)

    file_path = dir_name / "models" / (name+".pth")
    if new_training:
        file_path.unlink(missing_ok=True)
    
    tlayers = dict["layers"]
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers, activation_type=dict["activation"])
    pinn = pinn_x.PINNx(network, pde)
    # network = pinn_x.PINNx(
    #     net=mlp.GenericMLP, pde=pde, layer_sizes=tlayers, activation_type=dict["activation"]
    # )

    ###
    # Trainer
    ###

    trainer = training_x.TrainerPINNSpace(
        pde=pde,
        network=pinn,
        sampler=sampler,
        file_name=file_path,
        bc_loss_bool=False,
        decay=dict["decay"],
        batch_size=dict["n_collocations"],
        w_data=dict["w_data"],
        w_res=dict["w_res"]
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
                trainer.train(epochs=tab_n_epochs[i], n_collocation=dict["n_collocations"], n_data=dict["n_data"])
        else:
            trainer.learning_rate = dict["lr"]
            trainer.train(epochs=dict["n_epochs"], n_collocation=dict["n_collocations"], n_data=dict["n_data"])
    
    ###
    # Plot solutions
    ###

    fig_path = dir_name / "solutions" / (name+".png")
    plot_solution(cas,trainer,fig_path)

    return trainer

