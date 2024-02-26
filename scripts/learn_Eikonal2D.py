################
# Commentaires #
################

# Attention, ici il n'y a pas de recherche qui permet de savoir si cette configuration existe déjà
# On remarquera également que si n_layers et units sont renseignés en plus de layeres, layers n'est pas pris en compte
# Pas de garantit non plus que le fichier de config associé au modèle 1 soit le bon (si on suuprime 1 et pas l'autre par exemple)

###########
# Imports #
###########

import sys
import argparse

from scar.problem.Case import *
from scar.utils import *
from scar.equations.run_Eikonal2D import *

#############
# Arguments #
#############

# Récupération des arguments lors de l'exécution du script python
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", help="Index of configuration file.", type=int)
    parser.add_argument("--form", help="Form considered.", type=str, default="Circle")

    # Model arguments
    parser.add_argument("--n_layers", help="Number of layers in the model (required units).", required="--units" in sys.argv, type=int, default=6)
    parser.add_argument("--units", help="Number of units in each model layer (required n_layers).", required="--n_layers" in sys.argv, type=int, default=64)
    parser.add_argument("--layers", help="Number of units in each model layer.", nargs='+', type=int, default=[64, 64, 64, 64, 64, 64])
    parser.add_argument("--activation", help="Type of the activation function.", type=str, default="sine")

    # Trainer arguments
    parser.add_argument("--lr","--learning_rate", help="Learning rate of the trainer.", nargs="+", type=float, default=1e-2)
    parser.add_argument("--decay", help="Multiplicative factor of learning rate decay.", type=float, default=0.99)

    parser.add_argument("--w_eik", help="Weight in the Eikonal loss.", type=float, default=100.0)
    parser.add_argument("--w_bc", help="Weight in the Boundary loss.", type=float, default=100.0)
    parser.add_argument("--w_tv", help="Weight in the TV loss.", type=float, default=0.0)
    
    # Training arguments
    parser.add_argument("--n_epochs", help="Number of epochs during training.", type=int, default=10000)
    parser.add_argument("--n_collocations", help="Number of collocation points inside the domain during training.", type=int, default=4000)

    parser.add_argument("--n_bc_collocations", help="Number of collocation points on boundary during training.", type=int, default=2000)

    args = parser.parse_args()

    return args, parser

#############
# Run model #
#############

args, parser = get_args()

geom_class_name = args.form
geom_class = get_class(geom_class_name,Geometry)
form = geom_class()

dir_name = "../networks/Eikonal2D/"+geom_class_name+"/"
if not Path(dir_name).exists():
    Path(dir_name).mkdir(parents=True)
# create models and solutions dirs
if not Path(dir_name+"models").exists():
    Path(dir_name+"models").mkdir(parents=True)
if not Path(dir_name+"solutions").exists():
    Path(dir_name+"solutions").mkdir(parents=True)

condition = (len(sys.argv)!=3 and not "--form" in sys.argv) or len(sys.argv)>5
print("len(sys.argv) : ", len(sys.argv))
print("cond1 : ", (len(sys.argv)!=3 and not "--form" in sys.argv))
print("cond2 : ", len(sys.argv)>5)
config, args, config_filename, model_filename = get_config_filename(args,parser,dir_name,condition=condition)
print("### Config file : ",config_filename)
print("### Model file : ",model_filename)

dict_config = read_config(config_filename)

run_Eikonal2D(form,config,dict_config,new_training = False,createxyzfile=False)


# ######


# from pathlib import Path
# import torch
# import numpy as np

# import scimba.pinns.pinn_x as pinn_x

# from scimba.shape.eikonal_x import EikonalPINNx
# from scimba.shape.training_x import TrainerEikonal

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"torch loaded; device is {device}")

# torch.set_default_dtype(torch.double)
# torch.set_default_device(device)

# def from_xyz_normals(path):
#     f = open(path, "r")
#     s = f.readline()
#     L = []
#     n = []
#     while s:
#         t = s.split()
#         L.append(np.array([float(t[0]), float(t[1])]))
#         n.append(np.array([float(t[2]), float(t[3])]))
#         s = f.readline()
#     f.close()
#     return np.array(L), np.array(n)

# def run_Eikonal2D(form):
#     bound = [[form.bord_a,form.bord_b],[form.bord_a2,form.bord_b2]]
#     class_name = form.__class__.__name__
#     surface_filename = "../xyzfiles/"+class_name+".xyz"

#     bc_points, bc_normals = from_xyz_normals(surface_filename)
#     bc_points = torch.tensor(bc_points, dtype=torch.double, device=device, requires_grad=True)
#     bc_normals = torch.tensor(bc_normals, dtype=torch.double, device=device)

#     net = pinn_x.MLP_x
#     eik = EikonalPINNx(2, bound, bc_points, bc_normals, net)

#     file_name = "test.pth"
#     new_training = False

#     if new_training:
#         (
#             Path.cwd()
#             / Path(TrainerEikonal.FOLDER_FOR_SAVED_NETWORKS)
#             / file_name
#         ).unlink(missing_ok=True)

#     trainer = TrainerEikonal(
#         eik=eik,
#         file_name=file_name,
#         learning_rate=1.e-2,
#         decay=0.99,
#         batch_size=5000,
#         w_tv = 0.0,
#         # w_data=0,
#         # w_res=0.1,
#         # w_bc=10,
#     )

#     if new_training:
#         trainer.train(epochs=20000, n_collocation=2000)

#     trainer.plot(20000)

#     return eik, trainer


# if __name__ == "__main__":
#     network, trainer = run_Eikonal2D()
