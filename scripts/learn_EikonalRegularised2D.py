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
    parser.add_argument("--eps", help="Regularisation parameter.", type=float)

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
geom_class = get_class(geom_class_name,Geometry2D)
form = geom_class()

dir_name = "../networks/EikonalRegularised2D/"+geom_class_name+"/"
if not Path(dir_name).exists():
    Path(dir_name).mkdir(parents=True)
# create models and solutions dirs
if not Path(dir_name+"models").exists():
    Path(dir_name+"models").mkdir(parents=True)
if not Path(dir_name+"solutions").exists():
    Path(dir_name+"solutions").mkdir(parents=True)

default = len(sys.argv)==1 or (len(sys.argv)==3 and "--form" in sys.argv)
from_config = (len(sys.argv)!=3 and not "--form" in sys.argv) or len(sys.argv)>5
config, args, config_filename, model_filename = get_config_filename(args,parser,dir_name,default,from_config)
print("### Config file : ",config_filename)
print("### Model file : ",model_filename)

dict_config = read_config(config_filename)

run_Eikonal2D(form,config,dict_config,new_training = False,createxyzfile=True,regularised=True)