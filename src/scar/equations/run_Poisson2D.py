###########
# Imports #
###########

import matplotlib.pyplot as plt
from pathlib import Path

from scimba.pinns import pinn_x, training_x
from scimba.sampling import sampling_parameters, sampling_pde, uniform_sampling
import scimba.nets.training_tools as training_tools
import scimba.pinns.pinn_losses as pinn_losses

from scar.equations.Poisson2D import *

###########
# Globals #
###########

current = Path(__file__).parent.parent.parent.parent

def plot_sampling(bornes,data,name):
    plt.plot(data[:,0],data[:,1],"+")
    plt.xlim(bornes[0][0],bornes[0][1])
    plt.ylim(bornes[1][0],bornes[1][1])
    plt.title(name)

def plot_solution(trainer,fig_path):
    trainer.plot(50000,filename=fig_path) #,sampler=sampler)

def plot_derivatives(trainer,derivees_path,derivees2_path,phi_path):

    trainer.plot_first_derivative_x(filename=derivees_path,n_visu=5000)
    trainer.plot_second_derivative_x(filename=derivees2_path,n_visu=5000)

    trainer.plot_mul_derivatives_x(filename=phi_path,n_visu=5000)

def run_Poisson2D(cas, num_config, dict, save_sampling = False, save_phi=False, new_training = False):
    pde = cas.pde
    dir_name = current / cas.dir_name

    ###
    # Sampling
    ###

    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    ###
    # Plot sampling and phi
    ###

    if save_sampling:
        bornes = pde.space_domain.large_domain.surrounding_domain.bound
        
        data_inside = sampler.sampling(dict["n_collocations"])[0].x.cpu().detach().numpy()
        plt.figure(figsize=(5,5))
        plot_sampling(bornes,data_inside,str(dict["n_collocations"])+" points")

        plt.savefig(dir_name / "models" / ("sampling_"+str(num_config)+"_diapo.png"),dpi=500)

    if save_phi:
        data_inside = sampler.sampling(50000)[0]#.x
        phi_inside = cas.sd_function.sdf(data_inside)[:,0].detach().cpu().numpy()
        data_inside = data_inside.x.detach().cpu().numpy()

        plt.figure(figsize=(10,10))
        plt.scatter(data_inside[:,0],data_inside[:,1],c=phi_inside)
        # plt.tricontourf(data_inside[:,0],data_inside[:,1],phi_inside,"o",levels=100)#,cmap="hot")
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

    ###
    # Trainer
    ###

    losses = pinn_losses.PinnLossesData(w_res=dict["w_res"])
    optimizers = training_tools.OptimizerData(learning_rate=dict["lr"], decay=dict["decay"])
    trainer = training_x.TrainerPINNSpace(
        pde=pde,
        network=pinn,
        sampler=sampler,
        losses=losses,
        optimizers=optimizers,
        file_name=file_path,
        batch_size=dict["n_collocations"],
    )

    ###
    # Training
    ###

    if new_training or trainer.to_be_trained:
        # si lr est une liste on entrainer n_epoch/len(lr) fois avec chaque lr
        # if isinstance(dict["lr"],list):
        #     # on récupère le nombre d'époques (pas forcément divisible par len(lr))
        #     n_epochs = dict["n_epochs"]//len(dict["lr"])
        #     tab_n_epochs = [n_epochs]*len(dict["lr"])
        #     tab_n_epochs[-1] += dict["n_epochs"]%len(dict["lr"])
        #     for (i,lr) in enumerate(dict["lr"]): 
        #         print("## Train ", tab_n_epochs[i], " epochs with lr = ", lr)
        #         trainer.learning_rate = lr
        #         trainer.train(epochs=tab_n_epochs[i], n_collocation=dict["n_collocations"], n_data=dict["n_data"])
        # else:
        #     trainer.learning_rate = dict["lr"]
        trainer.train(epochs=dict["n_epochs"], n_collocation=dict["n_collocations"], n_data=dict["n_data"])

    ###
    # Plot solutions and derivatives
    ###

    fig_path = dir_name / "solutions" / (name+".png")
    # derivees_path = dir_name / "solutions" / (name+"_first.png")
    # derivees2_path = dir_name / "solutions" / (name+"_second.png")
    # phi_path = dir_name / "solutions" / (name+"_phi.png")
    
    plot_solution(trainer,fig_path)
    
    # plot_derivatives(trainer,derivees_path,derivees2_path,phi_path)

    return trainer

