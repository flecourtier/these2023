import numpy as np
import os
import shutil
from pathlib import Path

from utils import *
from ReadLatex import *
from InitAntora import *
from WriteAntora import *

"""
But : Convertir un fichier latex en une documentation antora complète
"""

# def rm_all():
#     shutil.rmtree(page_dir)
#     os.mkdir(page_dir)
#     shutil.rmtree(attachments_dir)
#     os.mkdir(attachments_dir)
#     shutil.rmtree(images_dir)
#     os.mkdir(images_dir)
#     os.remove(result_dir + "nav.adoc")

current_dir = Path(__file__).parent
root_dir = str(current_dir.parent.parent) + "/"

rapport_dir = root_dir + "results/"

# si il y a déjà un dossier antora, on le supprime et on copie le dossier antora_base
# rm_all()

# for dir in os.listdir(rapport_dir):
#     print(dir)

# Read the latex file
section_files,section_names = get_sections(rapport_dir)
sections = get_subsections(section_files,section_names,rapport_dir)

print("section_files :",section_files)
print("section_names :",section_names)
print("sections :",sections)

# InitAntora

create_nav_file(section_files,sections)
create_nav(section_files,sections)
create_main_page_file(section_files,sections)

presentation_name = cp_assets(section_files,rapport_dir)
print("presentation_name :",presentation_name)
cp_all_sections(section_files,section_names,sections,rapport_dir)
create_presentation_file(presentation_name)

# label_sections = get_label_sections(section_files)
# print("label_sections :",label_sections)
