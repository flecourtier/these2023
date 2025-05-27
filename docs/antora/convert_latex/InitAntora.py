import numpy as np
import os
import shutil
from utils import *

root_dir, result_dir, page_dir, images_dir, attachments_dir = get_dir()

# create all directories and files of the documentation
def create_nav(section_files,sections,write_dir):
    # remove all the files in the page directory
    if os.path.exists(page_dir + write_dir):
        shutil.rmtree(page_dir + write_dir)
    os.mkdir(page_dir + write_dir)

    for i,(section,subsections) in enumerate(sections.items()):        
        if section_files[i]!="":
            section_file_name = section_files[i].split("/")[1]
            section_file = open(page_dir + write_dir + section_file_name + ".adoc", 'w')
            section_file.write("= " + section + "\n")
            if subsections!=None:
                os.mkdir(page_dir + write_dir + section_file_name)
                for j,(subsection,subsubsections) in enumerate(subsections.items()):
                    subsection_file = open(page_dir + write_dir + section_file_name + "/subsec_" + str(j) + ".adoc", 'w')
                    subsection_file.write("= " + subsection + "\n")
                    subsection_file.close()
                    if len(subsubsections)!=0:
                        for k,subsubsection in enumerate(subsubsections):
                            subsubsection_file = open(page_dir + write_dir + section_file_name + "/subsec_" + str(j) + "_subsubsec_" + str(k) + ".adoc", 'w')
                            subsubsection_file.write("= " + subsubsection + "\n")
                            subsubsection_file.close()
                    
            section_file.close()
        else:
            section_file_name = "section_" + str(i)
            section_file = open(page_dir + write_dir + section_file_name + ".adoc", 'w')
            section_file.write("= " + section + "\n")
            section_file.close()   

## Cp presentation
def cp_slides():
    presentation_dir = root_dir + "presentation/"
    pres_attachments_dir = attachments_dir + "presentation/"
    if os.path.exists(pres_attachments_dir):
        shutil.rmtree(pres_attachments_dir)
    os.mkdir(pres_attachments_dir)

    presentation_name = {}
    for file in os.listdir(presentation_dir):
        if file!="georgia_files":
            pdf_to_copy = presentation_dir + file + "/presentation.pdf"
            shutil.copyfile(pdf_to_copy,pres_attachments_dir + file + ".pdf")
            
            latex_file = presentation_dir + file + "/presentation.tex"
            file_read = open(latex_file, 'r')
            while line := file_read.readline():
                if search_word_in_line("\\title[PhiFEM]", line):
                    name = line.split("{")[1].split("}")[0]
                    presentation_name[file] = name

    return presentation_name

## Cp posters
def cp_posters():
    poster_dir = root_dir + "poster/"
    pres_attachments_dir = attachments_dir + "poster/"
    if os.path.exists(pres_attachments_dir):
        shutil.rmtree(pres_attachments_dir)
    os.mkdir(pres_attachments_dir)

    poster_name = {}
    for file in os.listdir(poster_dir):
        if file!="georgia_files":
            pdf_to_copy = poster_dir + file + "/poster.pdf"
            shutil.copyfile(pdf_to_copy,pres_attachments_dir + file + ".pdf")
            
            latex_file = poster_dir + file + "/poster.tex"
            file_read = open(latex_file, 'r')
            while line := file_read.readline():
                if search_word_in_line("% Titre : ", line):
                    name = line.split(": ")[1]
                    poster_name[file] = name
                else:
                    ValueError("Poster file " + latex_file + " does not contain the title line '% Titre : '")

    return poster_name

# copy all the images of the tex report in the antora documentation
def cp_assets(section_files,rapport_dir):
    for i,section_file in enumerate(section_files):
        section = section_file.split("/")[1]
        dir_name = section_file.split("/")[0]+"/"
        if os.path.exists(images_dir+dir_name+section):
            shutil.rmtree(images_dir+dir_name+section)
        # Cp images
        if os.path.exists(rapport_dir + "images/" + section):
            shutil.copytree(rapport_dir + "images/" + section, images_dir+dir_name+section)

    # Cp attachments
    if os.path.exists(attachments_dir):
        # shutil.rmtree(attachments_dir)
        for file in os.listdir(attachments_dir):
            # check if not dir
            if os.path.isfile(attachments_dir + file):
                os.remove(attachments_dir + file)
    # os.mkdir(attachments_dir)
    shutil.copyfile(root_dir + "abstracts/abstracts.pdf",attachments_dir + "abstracts.pdf")
    

# create the nav.adoc file
def create_nav_file():
    # create the nav.adoc file
    nav_file = result_dir + "nav.adoc"
    file_write = open(nav_file, 'w')
    file_write.write(":stem: latexmath\n")
    
    file_write.write("\n* xref:main_page.adoc[Sommaire]\n")
    
    # file_write.close()
    return file_write

# complete the nav.adoc file with the sections and subsections
def write_in_nav_file(file_write,section_files,sections,level=1):
    for i,(section,subsections) in enumerate(sections.items()):    
        if section_files[i]!="":
            section_file_name = section_files[i]
            # section_file_name = section_files[i].split("/")[1]
            file_write.write(level*"*"+" xref:" + section_file_name + ".adoc[" + section + "]\n")
            if subsections!=None:
                for j,(subsection,subsubsections) in enumerate(subsections.items()):
                    file_write.write(level*"*"+"* xref:" + section_file_name + "/subsec_" + str(j) + ".adoc[" + subsection + "]\n")
                    if len(subsubsections)!=0:
                        for k,subsubsection in enumerate(subsubsections):
                            file_write.write(level*"*"+"** xref:" + section_file_name + "/subsec_" + str(j) + "_subsubsec_" + str(k) + ".adoc[" + subsubsection + "]\n")
        else:
            section_file_name = "section_" + str(i)
            file_write.write(level*"*"+" xref:" + section_file_name + ".adoc[" + section + "]\n")

    # file_write.close()

# create the main page of the documentation
def create_main_page_file(nav_filename):
    nav_file = open(nav_filename, 'r')
    # create the main_page.adoc file
    main_page_file = page_dir + "main_page.adoc"
    file_write = open(main_page_file, 'w')

    file_write.write("# Sommaire\n\n")
    intro = "Après un stage dans l'équipe INRIA MIMESIS dans le cadre de mon master (Master CSMI à l'Université de Strasbourg), j'ai rejoint l'équipe en tant que doctorante en octobre 2023 sous la direction d'Emmanuel Franck, Michel Duprez et Vanessa Lleras. L'objectif de ce projet est le *\"Développement de méthodes hybrides éléments finis/réseaux neuronaux pour aider à la création de jumeaux chirurgicaux numériques\"*.\n\n"
    file_write.write(intro)

    # Résultats
    file_write.write("== Rapport\n\n")
    file_write.write("Vous trouverez un rapport complété au fur et à mesure des avancements et des résultats obtenus (au format xref:attachment$report.pdf[PDF]) :\n\n")
    
    write_results = False
    while line := nav_file.readline():
        if search_word_in_line(".Contenus supplémentaires", line):
            break
        if line[0]==".":
            file_write.write("=== " + line[1:])
            write_results = True
        elif write_results:
            file_write.write(line)
        

    # Contenus supplémentaires
    file_write.write("\n== Contenus supplémentaires\n\n")
    # file_write.write(attach)
    file_write.write("Vous pouvez trouver les contenus supplémentaires suivants:\n\n")
    file_write.write("* différentes xref:slides.adoc[présentations]\n\n")
    file_write.write("* des xref:abstracts.adoc[résumés hebdomadaires] (au format xref:attachment$abstracts.pdf[PDF])\n\n")
    
    file_write.write("Vous trouverez également:\n\n")
    file_write.write("* une https://drive.google.com/file/d/1mA1_JrBOlv6OsjKCtzuZGMHcKeHAZ4s9/view?usp=drive_link[ToDo List] des travaux à effectuer chaque semaine\n\n")
    file_write.write("* une documentation du code (à rajouter)\n\n")

    file_write.close()  

# create the presentation page
def create_presentation_file(presentation_name):
    # create the nav.adoc file
    presentation_file = page_dir + "slides.adoc"
    
    date,name = [],[]
    for key,value in presentation_name.items():
        date.append(key)
        name.append(value)
    date,name = np.array(date), np.array(name)
    index = np.argsort(date)[::-1]
    date,name = date[index],name[index]

    date_fr = []
    for d in date:
        date_fr.append(d[8:10] + "/" + d[5:7] + "/" + d[:4])
        
    file_write = open(presentation_file, 'w')
    file_write.write("# Slides\n\n")

    for i in range(len(date)):
        file_write.write("* "+date_fr[i]+" : xref:attachment$presentation/" + date[i] + ".pdf[" + name[i] + "]\n")
    
    file_write.close() 
    
# create the poster page
def create_poster_file(poster_name):
    # create the nav.adoc file
    poster_file = page_dir + "posters.adoc"
    
    date,name = [],[]
    for key,value in poster_name.items():
        date.append(key)
        name.append(value)
    date,name = np.array(date), np.array(name)
    index = np.argsort(date)[::-1]
    date,name = date[index],name[index]

    date_fr = []
    for d in date:
        date_fr.append(d[8:10] + "/" + d[5:7] + "/" + d[:4])
        
    file_write = open(poster_file, 'w')
    file_write.write("# Posters\n\n")

    for i in range(len(date)):
        file_write.write("* "+date_fr[i]+" : xref:attachment$poster/" + date[i] + ".pdf[" + name[i] + "]\n")
    
    file_write.close() 

# create the presentation page
def create_abstract_file(section_files,section_names):
    # create the nav.adoc file
    abstract_file = page_dir + "abstracts.adoc"

    file_write = open(abstract_file, 'w')
    file_write.write("# Résumés hebdomadaires\n\n")
    file_write.write("Vous trouverez ici des résumés des travaux effectués chaque semaine :\n\n")

    for i in range(len(section_files)):
        section_file_name = section_files[i]
        file_write.write("* xref:" + section_file_name + ".adoc[" + section_names[i] + "]\n")

    file_write.close() 