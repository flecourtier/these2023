import argparse
import sys
import json
import os

from modules.run_laplacian import *
from create_xlsx_file import *
from modules.Poisson2D import *


# Attention, ici il n'y a pas de recherche qui permet de savoir si cette configuration existe déjà
# on remarquera également que si n_layers et units sont renseignés en plus de layeres, layers n'est pas pris en compte
# Pas de garantit non plus que le fichier de config associé au modèle 1 soit le bon (si on suuprime 1 et pas l'autre par exemple)


problem_considered = Circle
pde_considered = Poisson2D_fixed

save_sampling = False
save_phi = False


# create directories if they don't exist
name_problem_considered = problem_considered.__name__+"/"
if not os.path.isdir("networks/"+name_problem_considered):
    os.mkdir("networks/"+name_problem_considered)
name_pde_considered = pde_considered.__name__+"/"

dir_name = "networks/"+name_problem_considered+name_pde_considered
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
    os.mkdir(dir_name+"configs")
    os.mkdir(dir_name+"models")
    os.mkdir(dir_name+"results")


# on récupère un indice de configuration non utilisée
def get_empty_num_config():
    config = 1
    while True:
        config_filename = dir_name+"configs/config_"+str(config)+".json"
        # si il n'existe pas, break
        if not os.path.isfile(config_filename):
            break
        config += 1
    return config



# écrire un fichier de configuration au format json
def write_config(opts, filename):
    config_file = open(filename, "w")
    config_file.write("{\n")
    for i,arg in enumerate(vars(opts)):
        value = getattr(opts, arg)
        if type(value) == str:
            config_file.write("\t\""+arg+"\":\""+str(getattr(opts, arg))+"\"")
        elif arg != "config" and value!=None and arg!="bc":
            config_file.write("\t\""+arg+"\":"+str(getattr(opts, arg)))
        
        if arg!="config" and value!=None and arg!="bc":
            if i!=len(vars(opts))-1:
                config_file.write(",\n")
            else:
                config_file.write("\n")
    config_file.write("}")
    config_file.close()

# lire un fichier de configuration
def read_config(filename):
    with open(filename) as f:
        raw_config = f.read()
        dict = json.loads(raw_config)
    return  dict



# retourne True si la valeur de l'agument est la valeur par défaut (ne signifie pas que l'utilisateur n'a pas rentré l'argument)
def is_default(opts, parser, arg):
  if getattr(opts, arg) == parser.get_default(arg):
    return True
  return False

###
# Récupération des arguments lors de l'exécution du script python
###

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", help="Index of configuration file.", type=int)
    parser.add_argument("--bc", "--boundary_conditions", help="Don't use the signed distance function to impose exact boundary conditions.", action='store_false')

    # Model arguments
    parser.add_argument("--n_layers", help="Number of layers in the model (required units).", required="--units" in sys.argv, type=int, default=4)
    parser.add_argument("--units", help="Number of units in each model layer (required n_layers).", required="--n_layers" in sys.argv, type=int, default=20)
    parser.add_argument("--layers", help="Number of units in each model layer.", nargs='+', type=int, default=[20, 20, 20, 20])
    parser.add_argument("--activation", help="Type of the activation function.", type=str, default="sine")

    # Trainer arguments
    parser.add_argument("--lr","--learning_rate", help="Learning rate of the trainer.", nargs="+", type=float, default=1e-2)
    parser.add_argument("--decay", help="Multiplicative factor of learning rate decay.", type=float, default=0.99)

    parser.add_argument("--w_data", help="Weight of data in the loss.", type=float, default=0.0)
    parser.add_argument("--w_res", help="Weight of residue in the loss.", type=float, default=0.01)
    parser.add_argument("--w_bc", help="Weight of boundary conditions in the loss.", type=float, default=10.0)

    # Training arguments
    parser.add_argument("--n_epochs", help="Number of epochs during training.", type=int, default=1000)
    parser.add_argument("--n_collocations", help="Number of collocation points inside the domain during training.", type=int, default=500)
    parser.add_argument("--n_bc_collocation", help="Number of collocation points on the boundary during training.", type=int, default=500)
    parser.add_argument("--n_data", help="Number of data during training.", type=int, default=0)

    args = parser.parse_args()

    return args, parser

def get_config_filename(args,parser):
    # imposition des conditions exactes au bord ?
    if args.bc:
        end = "_exact_bc"
    else:
        end = ""

    # si l'utilisateur a rentré n_layers et units, on remplace layers par [units, units, ...]
    if not is_default(args, parser, "n_layers") or not is_default(args, parser, "units"):
        args.layers = [args.units]*args.n_layers
    vars(args)["n_layers"] = None
    vars(args)["units"] = None

    ###
    # Création/Récupération du fichier de configuration
    ###

    # cas par défaut (modèle 0)
    if len(sys.argv)==1 or (len(sys.argv)==2 and sys.argv[1]=="--bc") or (len(sys.argv)==3 and sys.argv[1]=="--pb"): #or args.config==0:
        print("### Default case")
        config=0
    else:
        print("### Not default case")
        # si il n'y a pas de fichier de configuration fournit ("--config" n'est pas dans les arguments)
        if args.config==None:
            print("## No config file provided")
            print("# New model created")
            config = get_empty_num_config()

        # si il y a un fichier de configuration fournit ("--config" est dans les arguments)
        else:
            print("## Config file provided")
            config = args.config       
            config_filename = dir_name+"configs/config_"+str(config)+".json"
            model_filename = dir_name+"models/model_"+str(config)+end+".pth"
            
            # on lit le fichier de configuration et on remplace les arguments par défaut par ceux du fichier
            args_config = argparse.Namespace(**vars(args))
            dict = read_config(config_filename)
            for arg,value in dict.items():
                vars(args_config)[arg] = value      

            if len(sys.argv)!=3 and not (len(sys.argv)==4 and "--bc" in sys.argv):
                print("# New model created from config file")
                # si l'utilisateur rajoute des args, on modifie les valeurs du fichier de config
                # (c'est alors un nouveau modèle)
                for arg in vars(args):
                    if arg!="config" and not is_default(args, parser, arg):
                        value = getattr(args, arg)
                        vars(args_config)[arg] = value

                config = get_empty_num_config()
            else:
                print("# Load model from config file")

            args = args_config

    config_filename = dir_name+"configs/config_"+str(config)+".json"
    model_filename = dir_name+"models/model_"+str(config)+end+".pth"
    write_config(args, config_filename)

    return config, args, config_filename, model_filename


args, parser = get_args()
config, args, config_filename, model_filename = get_config_filename(args,parser)
print("### Config file : ",config_filename)
print("### Model file : ",model_filename)

dict = read_config(config_filename)
print("### Config : ")
for arg,value in dict.items():
    print(arg,":",value)


test_laplacian_2d(problem_considered,pde_considered,config,dict,args.bc,save_sampling,save_phi)
create_xlsx_file(problem_considered,pde_considered)