from pathlib import Path

from utils import *
from ReadLatex import *
from InitAntora import *
from WriteAntora import *

"""
But : Créer la documentation antora complète
"""

current_dir = Path(__file__).parent
root_dir = str(current_dir.parent.parent) + "/"

nav_file = create_nav_file()

#############
# Resultats #
#############

####
# Create Results
####

nav_file.write("\n.Résultats\n")

# ReadLatex

rapport_dir = root_dir + "results/"
rapport_file = rapport_dir + "results.tex"

section_files,section_names = get_sections(rapport_file)
sections = get_subsections(section_files,section_names,rapport_dir)

# InitAntora

new_section_files = []
for i in range(len(section_files)):
    section_file_name = section_files[i].split("/")[1]
    section_file_name = "results/" + section_file_name
    new_section_files.append(section_file_name)
write_in_nav_file(nav_file,new_section_files,sections,level=1)
create_nav(section_files,sections,"results/")
create_main_page_file(section_files,sections)

cp_assets(section_files,rapport_dir)
cp_all_sections(section_files,section_names,sections,rapport_dir,"results/")

# label_sections = get_label_sections(section_files)
# print("label_sections :",label_sections)

############################
# Contenus supplémentaires #
############################

####
# Create Slides
####

nav_file.write("\n.Contenus supplémetaires\n")
presentation_name = cp_slides()
create_presentation_file(presentation_name)
nav_file.write("* xref:slides.adoc[Mes slides]\n")

####
# Create Abstracts
####

abstract_dir = root_dir + "abstracts/"
abstract_file = abstract_dir + "abstracts.tex"

section_files,section_names = get_sections(abstract_file)
sections = get_subsections(section_files,section_names,abstract_dir)

cp_all_sections(section_files,section_names,sections,abstract_dir,"abstracts/")
section_files,section_names,sections = group_by_months(section_files,section_names,sections,"abstracts/")

nav_file.write("* xref:abstracts.adoc[Résumés hebdomadaires]\n")
write_in_nav_file(nav_file,section_files,sections,level=2)