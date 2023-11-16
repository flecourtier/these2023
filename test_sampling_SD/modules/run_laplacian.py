import matplotlib.pyplot as plt

new_training = False
align_levelset = ""
# align_levelset = "_align_levelset" 

import torch
torch.seed()

from pathlib import Path

from scimba.nets import mlp
from scimba.pinns import pinn_x, training_x
from scimba.sampling import sampling_pde, sampling_parameters, uniform_sampling
from modules.Poisson2D import *

current = Path(__file__).parent.parent

def plot_sampling(domain,data,name):
    plt.plot(data[:,0],data[:,1],"+")
    # bornes = domain.surrounding_domain.bound
    bornes = domain.bound
    plt.xlim(bornes[0][0],bornes[0][1])
    plt.ylim(bornes[1][0],bornes[1][1])
    plt.title(name)

def test_laplacian_2d(class_problem, class_pde, num_config, dict, use_levelset,save_sampling = False,save_phi = False):
    problem = class_problem()
    pde = class_pde(problem, use_levelset)
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    dir_name = current / "networks" / class_problem.__name__ / class_pde.__name__

    if save_sampling:
        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        data_inside = sampler.sampling(dict["n_collocations"])[0].detach().numpy()
        print(data_inside.shape)
        plot_sampling(pde.space_domain,data_inside,"inside Omega")

        plt.subplot(1,2,2)
        data_boundary = sampler.bc_sampling(dict["n_bc_collocation"])[0].detach().numpy()
        plot_sampling(pde.space_domain,data_boundary,"on the border")

        plt.savefig(dir_name / "results" / ("sampling_"+str(num_config)+align_levelset+".png"))

    # if save_phi:
    #     data_inside = sampler.sampling(10000)[0]
    #     phi_inside = problem.levelset(data_inside).detach().cpu().numpy()
    #     data_inside = data_inside.detach().numpy()

    #     plt.figure(figsize=(15,5))

    #     partial_S_plus = problem.polygon.detach().numpy()
    #     nb_pts = problem.nb_pts
    #     partial_S_plus = np.concatenate([partial_S_plus,[partial_S_plus[0]]])

    #     plt.tricontourf(data_inside[:,0],data_inside[:,1],phi_inside,"o",levels=100)#,cmap="hot")
    #     for i in range(nb_pts):
    #         pt1 = partial_S_plus[i]
    #         pt2 = partial_S_plus[i+1]
    #         plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],"black",linewidth=3)
    #     plt.title("phi")
    #     plt.colorbar()

    #     plt.savefig(dir_name / "results" / ("phi_"+str(num_config)+align_levelset+".png"))

    name = "model_"+str(num_config)
    if use_levelset:
        name += "_exact_bc"
    name += align_levelset

    file_path = dir_name / "models" / (name+".pth")
    fig_path = dir_name / "results" / (name+".png")
    if new_training:
        file_path.unlink(missing_ok=True)
    
    tlayers = dict["layers"]
    network = pinn_x.PINNx(
        net=mlp.GenericMLP, pde=pde, layer_sizes=tlayers, activation_type=dict["activation"]
    )

    trainer = training_x.TrainerPINNSpace(
        pde=pde,
        network=network,
        sampler=sampler,
        file_name=file_path,
        bc_loss_bool=not use_levelset,
        # learning_rate=dict["lr"],
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

    # plot sans mask
    trainer.plot(50000,filename=fig_path)
    
    # plot with mask
    pde = Poisson2D_fixed2(problem, use_levelset)
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)
    trainer.plot_with_mask(sampler,50000,filename=fig_path)

    return trainer

