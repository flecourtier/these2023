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
from scar.equations.run_Poisson2D import *

#############
# Arguments #
#############

# Récupération des arguments lors de l'exécution du script python
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", help="Index of configuration file.", type=int)
    parser.add_argument("--casefile", help="Path to the case file.", type=str, default="case.json")

    # Model arguments
    parser.add_argument("--n_layers", help="Number of layers in the model (required units).", required="--units" in sys.argv, type=int, default=4)
    parser.add_argument("--units", help="Number of units in each model layer (required n_layers).", required="--n_layers" in sys.argv, type=int, default=20)
    parser.add_argument("--layers", help="Number of units in each model layer.", nargs='+', type=int, default=[20, 20, 20, 20])
    parser.add_argument("--activation", help="Type of the activation function.", type=str, default="sine")

    # Trainer arguments
    parser.add_argument("--lr","--learning_rate", help="Learning rate of the trainer.", nargs="+", type=float, default=1e-2)
    parser.add_argument("--decay", help="Multiplicative factor of learning rate decay.", type=float, default=0.99)

    parser.add_argument("--w_data", help="Weight of data in the loss.", type=float, default=0.0)
    parser.add_argument("--w_res", help="Weight of residue in the loss.", type=float, default=1.0)
    
    # Training arguments
    parser.add_argument("--n_epochs", help="Number of epochs during training.", type=int, default=10000)
    parser.add_argument("--n_collocations", help="Number of collocation points inside the domain during training.", type=int, default=2000)
    parser.add_argument("--n_data", help="Number of data during training.", type=int, default=0)

    args = parser.parse_args()

    return args, parser

####################
# Define Test case #
####################

args, parser = get_args()

casefile = args.casefile
cas = Case("../testcases/"+casefile)

dir_name = "../"+cas.dir_name

# Création des dossiers si ils n'existent pas
create_tree(dir_name)
for subdir in [dir_name+"models", dir_name+"solutions"]:
    create_tree(subdir)

#############
# Run model #
#############

condition = len(sys.argv)!=3 and not (len(sys.argv)==5 and "--casefile" in sys.argv)
config, args, config_filename, model_filename = get_config_filename(args,parser,dir_name,condition=condition)
print("### Config file : ",config_filename)
print("### Model file : ",model_filename)

dict = read_config(config_filename)

run_Poisson2D(cas,config,dict,save_sampling=True,save_phi=True,new_training=False)