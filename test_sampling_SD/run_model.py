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

from modules.Case import *
# from modules.Poisson2D import *
from modules.utils import *
from modules.run_laplacian import *

from create_xlsx_file import *

###############
# Define case #
###############

cas = Case("case.json")

impose_exact_bc = cas.impose_exact_bc
problem_considered = cas.Problem
pde_considered = cas.PDE
dir_name = cas.dir_name
create_tree(dir_name)
for subdir in [dir_name+"models", dir_name+"solutions"]: # dir_name+"corrections", dir_name+"derivatives", 
    create_tree(subdir)

save_sampling = True
new_training = False

############
# A REVOIR #
############

# écrire un fichier de configuration au format json (A mettre dans utils)
def write_config(opts, filename):
    config_file = open(filename, "w")
    config_file.write("{\n")
    for i,arg in enumerate(vars(opts)):
        value = getattr(opts, arg)
        if type(value) == str:
            config_file.write("\t\"" + arg + "\":\"" + str(value) + "\"")
        elif arg != "config" and value!=None:
            config_file.write("\t\""+arg+"\":"+str(value))
        
        if arg!="config" and value!=None:
            if i!=len(vars(opts))-1:
                config_file.write(",\n")
            else:
                config_file.write("\n")
    config_file.write("}")
    config_file.close()

#############
# Arguments #
#############

# Récupération des arguments lors de l'exécution du script python
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", help="Index of configuration file.", type=int)

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
    if not impose_exact_bc:
        parser.add_argument("--w_bc", help="Weight of boundary conditions in the loss.", type=float, default=10.0)

    # Training arguments
    parser.add_argument("--n_epochs", help="Number of epochs during training.", type=int, default=10000)
    parser.add_argument("--n_collocations", help="Number of collocation points inside the domain during training.", type=int, default=2000)
    if not impose_exact_bc:
        parser.add_argument("--n_bc_collocation", help="Number of collocation points on the boundary during training.", type=int, default=500)
    parser.add_argument("--n_data", help="Number of data during training.", type=int, default=0)

    args = parser.parse_args()

    return args, parser

##############################################
# Création/Récupération du fichier de config #
##############################################

def get_config_filename(args,parser):
    # si l'utilisateur a rentré n_layers et units, on remplace layers par [units, units, ...]
    if not arg_is_default(args, parser, "n_layers") or not arg_is_default(args, parser, "units"):
        args.layers = [args.units]*args.n_layers
    vars(args)["n_layers"] = None
    vars(args)["units"] = None

    ###
    # Création/Récupération du fichier de configuration
    ###

    # cas par défaut (modèle 0)
    if len(sys.argv)==1: #or args.config==0:
        print("### Default case")
        config=0
    else:
        print("### Not default case")
        # si il n'y a pas de fichier de configuration fournit ("--config" n'est pas dans les arguments)
        if args.config==None:
            print("## No config file provided")
            print("# New model created")
            config = get_empty_num_config(dir_name+"models")

        # si il y a un fichier de configuration fournit ("--config" est dans les arguments)
        else:
            print("## Config file provided")
            config = args.config       
            config_filename = dir_name+"models/config_"+str(config)+".json"
            model_filename = dir_name+"models/model_"+str(config)+".pth"
            
            # on lit le fichier de configuration et on remplace les arguments par défaut par ceux du fichier
            args_config = argparse.Namespace(**vars(args))
            dict = read_config(config_filename)
            for arg,value in dict.items():
                vars(args_config)[arg] = value      

            if len(sys.argv)!=3:
                print("# New model created from config file")
                # si l'utilisateur rajoute des args, on modifie les valeurs du fichier de config
                # (c'est alors un nouveau modèle)
                for arg in vars(args):
                    if arg!="config" and not arg_is_default(args, parser, arg):
                        value = getattr(args, arg)
                        vars(args_config)[arg] = value

                config = get_empty_num_config(dir_name+"models")
            else:
                print("# Load model from config file")

            args = args_config

    config_filename = dir_name+"models/config_"+str(config)+".json"
    model_filename = dir_name+"models/model_"+str(config)+".pth"
    write_config(args, config_filename)

    return config, args, config_filename, model_filename


#############
# Run model #
#############

args, parser = get_args()
config, args, config_filename, model_filename = get_config_filename(args,parser)
print("### Config file : ",config_filename)
print("### Model file : ",model_filename)

dict = read_config(config_filename)

test_laplacian_2d(cas,config,dict,save_sampling=save_sampling,new_training=new_training)
# create_xlsx_file(problem_considered,pde_considered)