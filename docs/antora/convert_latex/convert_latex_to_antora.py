from pathlib import Path

from utils import *
from ReadLatex import *
from InitAntora import *
from WriteAntora import *

"""
But : Créer la documentation antora complète
"""

# current_dir = Path(__file__).parent
# root_dir = str(current_dir.parent.parent) + "/"

root_dir, result_dir, page_dir, images_dir, attachments_dir = get_dir()

nav_file = create_nav_file()

#############
# Report #
#############

####
# Create Results
####

rapport_dir = root_dir + "report/"

subdir_list = []
for subdir in os.listdir(rapport_dir):
    if os.path.isdir(rapport_dir + subdir) and subdir[-1] != "_":
        subdir_list.append(subdir)

subdir_list = sorted(subdir_list)
for subdir in subdir_list:
    chapter_dir = rapport_dir + subdir + "/"
    chapter_name = subdir[2].upper() + subdir[3:]
    print("## "+chapter_name)

    rapport_file = chapter_dir + "report.tex"
    nav_file.write("\n."+chapter_name+"\n")

    # ReadLatex

    section_files,section_names = get_sections(rapport_file)
    sections,section_names = get_subsections(section_files,section_names,chapter_dir)

    print("section_files :",section_files)
    print("section_names :",section_names)
    print("sections :",sections)

    # InitAntora

    new_section_files = []
    for i in range(len(section_files)):
        section_file_name = section_files[i].split("/")[1]
        section_file_name = subdir + "/" + section_file_name
        new_section_files.append(section_file_name)
    write_in_nav_file(nav_file,new_section_files,sections,level=1)
    create_nav(section_files,sections,subdir+"/")

    cp_assets(new_section_files,chapter_dir)
    cp_all_sections(section_files,section_names,sections,chapter_dir,subdir+"/")

    # # label_sections = get_label_sections(section_files)
    # # print("label_sections :",label_sections)

shutil.copyfile(rapport_dir + "report.pdf",attachments_dir + "report.pdf")

############################
# Contenus supplémentaires #
############################

####
# Create Slides
####

nav_file.write("\n.Contenus supplémentaires\n")
presentation_name = cp_slides()
create_presentation_file(presentation_name)
nav_file.write("* xref:slides.adoc[Mes slides]\n")
poster_name = cp_posters()
create_poster_file(poster_name)
nav_file.write("* xref:posters.adoc[Mes posters]\n")

####
# Create Abstracts
####

abstract_dir = root_dir + "abstracts/"
abstract_file = abstract_dir + "abstracts.tex"

section_files,section_names = get_sections(abstract_file)
sections,section_names = get_subsections(section_files,section_names,abstract_dir)

cp_all_sections(section_files,section_names,sections,abstract_dir,"abstracts/")
section_files,section_names,sections = group_by_months(section_files,section_names,sections,"abstracts/")
create_abstract_file(section_files,section_names)

nav_file.write("* xref:abstracts.adoc[Résumés hebdomadaires]\n")
write_in_nav_file(nav_file,section_files,sections,level=2)

#####
# Finish
#####

nav_file.close()
nav_filename = result_dir + "nav.adoc"
create_main_page_file(nav_filename)