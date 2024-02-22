import json
import os 

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
    for i in range(1,len(path_split)+1):
        subdir = "/".join(path_split[:i])
        if not os.path.isdir(subdir):
            os.mkdir(subdir)