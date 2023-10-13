from modules.data import *
import matplotlib.pyplot as plt

x0,y0,r,domain_O = domain_parameters()
new_training = False
save_sampling = False

from pathlib import Path

from scimba.nets import mlp
from scimba.pinns import domain, pinn_x, training_x
from scimba.sampling import sampling_pde, sampling_parameters, uniform_sampling
from modules.Poisson2D import *

current = Path(__file__).parent.parent

def levelset(X):
    return call_phi(torch, X.T)

def plot_sampling(domain,data,name):
    plt.plot(data[:,0],data[:,1],"+")
    bornes = domain.surrounding_domain.bound
    plt.xlim(bornes[0][0],bornes[0][1])
    plt.ylim(bornes[1][0],bornes[1][1])
    plt.title(name)

def test_laplacian_2d(num_config,dict,use_levelset):
    xdomain = domain.SignedDistanceBasedDomain(2, domain_O, levelset)
    pde = Poisson2D(xdomain, use_levelset)
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    if save_sampling:
        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        data_inside = sampler.sampling(10000)[0].detach().numpy()
        print(data_inside.shape)
        plot_sampling(xdomain,data_inside,"inside Omega")

        plt.subplot(1,2,2)
        data_boundary = sampler.bc_sampling(1000)[0].detach().numpy()
        plot_sampling(xdomain,data_boundary,"on the border")

        plt.savefig(current / "networks" / "results" / ("sampling_"+str(num_config)+".png"))

    name = "model_"+str(num_config)
    if use_levelset:
        name += "_exact_bc"

    file_path = current / "networks" / "models" / (name+".pth")
    fig_path = current / "networks" / "results" / (name+".png")
    if new_training:
        file_path.unlink(missing_ok=True)
    
    tlayers = dict["layers"]
    network = pinn_x.PINNx(
        net=mlp.GenericMLP, pde=pde, layer_sizes=tlayers, activation_type=dict["activation"]
    )

    print(file_path)
    trainer = training_x.TrainerPINNSpace(
        pde=pde,
        network=network,
        sampler=sampler,
        file_name=file_path,
        bc_loss_bool=not use_levelset,
        learning_rate=dict["lr"],
        decay=dict["decay"],
        batch_size=dict["n_collocations"],
        w_data=dict["w_data"],
        w_res=dict["w_res"],
        w_bc=dict["w_bc"],
    )

    if new_training or trainer.to_be_trained:
        trainer.train(epochs=dict["n_epochs"], n_collocation=dict["n_collocations"], n_bc_collocation=dict["n_bc_collocation"], n_data=dict["n_data"])
    trainer.plot(50000,filename=fig_path)

    assert True