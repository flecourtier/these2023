import os,sys,argparse,inspect,json

# Récupérer un indice de configuration non utilisé
def get_empty_num_config(dir_name):
    config = 1
    while True:
        config_filename = dir_name+"config_"+str(config)+".json"
        # si il n'existe pas, break
        if not os.path.isfile(config_filename):
            # print("LA",config_filename)
            break
        config += 1
    return config

# retourne True si la valeur de l'agument est la valeur par défaut (ne signifie pas que l'utilisateur n'a pas rentré l'argument)
def arg_is_default(opts, parser, arg):
  if getattr(opts, arg) == parser.get_default(arg):
    return True
  return False

# lire un fichier de configuration
def read_config(filename):
    with open(filename) as f:
        raw_config = f.read()
        dict = json.loads(raw_config)
    return  dict

# écrire un fichier de configuration au format json
def write_config(opts, filename):
    config_file = open(filename, "w")
    config_file.write("{\n")
    for i,arg in enumerate(vars(opts)):
        value = getattr(opts, arg)
        if arg!="casefile":
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

# Créer tous les sous dossiers si ils n'existent pas
def create_tree(path):
    path_split = path.split("/")
    if path[0]=="/":
        path_split = path_split[1:]
        start = "/"
    else:
        start = ""
    for i in range(1,len(path_split)+1):
        subdir = "/".join(path_split[:i])
        if not os.path.isdir(start+subdir):
            os.mkdir(start+subdir)

# Création/Récupération du fichier de config
def get_config_filename(args,parser,dir_name,default,from_config):
    # si l'utilisateur a rentré n_layers et units, on remplace layers par [units, units, ...]
    if not arg_is_default(args, parser, "n_layers") or not arg_is_default(args, parser, "units"):
        args.layers = [args.units]*args.n_layers
    vars(args)["n_layers"] = None
    vars(args)["units"] = None

    # cas par défaut (modèle 0)
    if default: #or args.config==0:
        print("### Default case")
        config=0
    else:
        print("### Not default case")
        # si il n'y a pas de fichier de configuration fournit ("--config" n'est pas dans les arguments)
        if args.config==None:
            print("## No config file provided")
            print("# New model created")
            print(dir_name+"models")
            config = get_empty_num_config(dir_name+"models/")

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

            if from_config:
                print("# New model created from config file")
                # si l'utilisateur rajoute des args, on modifie les valeurs du fichier de config
                # (c'est alors un nouveau modèle)
                for arg in vars(args):
                    if arg!="config" and not arg_is_default(args, parser, arg):
                        value = getattr(args, arg)
                        vars(args_config)[arg] = value

                config = get_empty_num_config(dir_name+"models/")
            else:
                print("# Load model from config file")

            args = args_config

    config_filename = dir_name+"models/config_"+str(config)+".json"
    model_filename = dir_name+"models/model_"+str(config)+".pth"
    write_config(args, config_filename)

    return config, args, config_filename, model_filename

# get all the class name in the module (not abstract class)
def get_class_name(module):
    class_name = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and not inspect.isabstract(obj):
            class_name.append(name)
    return class_name
    
# get the class by its name
def get_class(name,module):
    try:
        # Récupérer la classe par son nom
        class_ = getattr(module, name)
        return class_
    except AttributeError:
        # Gestion de l'erreur si la classe n'est pas trouvée
        print(f"Erreur : Classe {name} non trouvée dans le module {module.__name__}.")
    except Exception as e:
        # Gestion d'autres exceptions
        print(f"Une erreur s'est produite : {e}")