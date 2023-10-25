import argparse
import sys
import os

from modules.Poisson2D import *
from create_xlsx_file import *

parser = argparse.ArgumentParser()
parser.add_argument("config", help="Index of configuration file to remove.", type=int)
parser.add_argument("--bc", help="0 - exact_bc : 1 - no exact bc ; 2 - both.", type=int, default=0)
args = parser.parse_args()

problem_considered = None #Circle
pde_considered = None #Poisson2D_f
name_problem_considered = problem_considered.__name__+"/"
name_pde_considered = pde_considered.__name__+"/"
dir_name = "networks/"+name_problem_considered+name_pde_considered

config = args.config

# bc = 0 or bc = 2
if args.bc != 1:
    end = "_exact_bc"

    # remove model and result with exact_bc
    model_filename = dir_name+"models/model_"+str(config)+end+".pth"
    if os.path.exists(model_filename):
        os.remove(model_filename)
    result_filename = dir_name+"results/model_"+str(config)+end+".png"
    if os.path.exists(result_filename):
        os.remove(result_filename)

# bc = 1 or bc = 2
if args.bc !=0:
    end = ""

    # remove model and result with NO exact_bc
    model_filename = dir_name+"models/model_"+str(config)+end+".pth"
    if os.path.exists(model_filename):
        os.remove(model_filename)
    result_filename = dir_name+"results/model_"+str(config)+end+".png"
    if os.path.exists(result_filename):
        os.remove(result_filename)

# remove config file if there is no file with exact_bc and no file with NO exact_bc
end = "_exact_bc"
model_filename_exact = dir_name+"models/model_"+str(config)+"_exact_bc"+".pth"
model_filename_no_exact = dir_name+"models/model_"+str(config)+".pth"
if not os.path.exists(model_filename_exact) and not os.path.exists(model_filename_no_exact):
    config_filename = dir_name+"configs/config_"+str(config)+".json"
    if os.path.exists(config_filename):
        os.remove(config_filename)

# update xlsx file
create_xlsx_file(problem_considered, pde_considered)