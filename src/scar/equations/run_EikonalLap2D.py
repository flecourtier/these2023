###########
# Imports #
###########

from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import scimba.pinns.pinn_x as pinn_x

from scimba.shape.eikonal_x import EikonalPINNx
from scimba.shape.training_x import TrainerEikonal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

torch.set_default_dtype(torch.double)
torch.set_default_device(device)

###########
# Globals #
###########

current = Path(__file__).parent.parent.parent.parent

def from_xyz_normals(path):
    f = open(path, "r")
    s = f.readline()
    L = []
    n = []
    while s:
        t = s.split()
        L.append(np.array([float(t[0]), float(t[1])]))
        n.append(np.array([float(t[2]), float(t[3])]))
        s = f.readline()
    f.close()
    return np.array(L), np.array(n)

def run_EikonalLap2D(form,num_config,dict,new_training = False,createxyzfile = False):
    bound = [[form.bord_a,form.bord_b],[form.bord_a2,form.bord_b2]]
    class_name = form.__class__.__name__
    dir_name = current / "networks" / "EikonalLap2D" / class_name

    n_bc_points = dict["n_bc_collocations"]
    surface_filename = "../xyzfiles/"+class_name+"_"+str(n_bc_points)+".xy"
    if not Path(surface_filename).exists():
        if createxyzfile:
            t = np.linspace(0,1,n_bc_points)
            c_t = form.c(t)
            grad_c_t = form.c_prime_rot(t,theta=-np.pi/2)
            grad_c_t_norm = np.linalg.norm(grad_c_t,axis=0)
            normals = grad_c_t/grad_c_t_norm

            fig, ax = plt.subplots()

            form.plot_curve(color="black")
            ax.quiver(c_t[0,::10],c_t[1,::10],normals[0,::10],normals[1,::10],color="red")

            plt.xlim(form.bord_a,form.bord_b)
            plt.ylim(form.bord_a2,form.bord_b2)

            plt.show()

            assert np.allclose(np.linalg.norm(normals,axis=0),1)

            x,y = c_t
            normals = normals.T
            with open(surface_filename,"w") as f:
                for i in range(len(x)):
                    f.write(f"{x[i]} {y[i]} {normals[i,0]} {normals[i,1]}\n")
                f.close()
        else:
            assert False, f"File {surface_filename} does not exist"

    bc_points, bc_normals = from_xyz_normals(surface_filename)
    print(f"bc_points.shape = {bc_points.shape}, bc_normals.shape = {bc_normals.shape}")
    bc_points = torch.tensor(bc_points, dtype=torch.double, device=device, requires_grad=True)
    bc_normals = torch.tensor(bc_normals, dtype=torch.double, device=device)

    ###
    # Model
    ###

    name = "model_"+str(num_config)
    file_path = dir_name / "models" / (name+".pth")
    if new_training:
        file_path.unlink(missing_ok=True)

    net = pinn_x.MLP_x
    tlayers = dict["layers"]
    eik = EikonalPINNx(2, bound, bc_points, bc_normals, net, layer_sizes=tlayers, activation_type=dict["activation"],lap=True)

    ##
    # Trainer
    ### 

    trainer = TrainerEikonal(
        eik=eik,
        file_name=file_path,
        learning_rate=dict["lr"],
        decay=dict["decay"],
        batch_size=dict["n_collocations"],
        w_eik=dict["w_eik"],
        w_bc=dict["w_bc"],
        w_tv=dict["w_lap"],
        tv_loss_f=torch.nn.MSELoss()

    )

    if new_training or trainer.to_be_trained:
        trainer.train(epochs=dict["n_epochs"], n_collocation=dict["n_collocations"])

    ###
    # Plot solutions
    ###

    fig_path = dir_name / "solutions" / (name+".png")
    trainer.plot(20000,filename=fig_path)
    fig_path = dir_name / "solutions" / (name+"_mask.png")
    trainer.plot_with_mask(20000,filename=fig_path)

    return eik, trainer