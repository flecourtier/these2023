import numpy as np
import os
import shutil
from utils import *

root_dir, result_dir, page_dir, images_dir, attachments_dir = get_dir()

# create all directories and files of the documentation
def create_nav(section_files,sections):
    # remove all the files in the page directory
    if os.path.exists(page_dir):
        shutil.rmtree(page_dir)
    os.mkdir(page_dir)

    for i,(section,subsections) in enumerate(sections.items()):        
        if section_files[i]!="":
            section_file_name = section_files[i].split("/")[1]
            section_file = open(page_dir + section_file_name + ".adoc", 'w')
            section_file.write("= " + section + "\n")
            if subsections!=None:
                os.mkdir(page_dir + section_file_name)
                for j,(subsection,subsubsections) in enumerate(subsections.items()):
                    subsection_file = open(page_dir + section_file_name + "/subsec_" + str(j) + ".adoc", 'w')
                    subsection_file.write("= " + subsection + "\n")
                    subsection_file.close()
                    if len(subsubsections)!=0:
                        for k,subsubsection in enumerate(subsubsections):
                            subsubsection_file = open(page_dir + section_file_name + "/subsec_" + str(j) + "_subsubsec_" + str(k) + ".adoc", 'w')
                            subsubsection_file.write("= " + subsubsection + "\n")
                            subsubsection_file.close()
                    
            section_file.close()
        else:
            section_file_name = "section_" + str(i)
            section_file = open(page_dir + section_file_name + ".adoc", 'w')
            section_file.write("= " + section + "\n")
            section_file.close()   

# copy all the images of the tex report in the antora documentation
def cp_assets(section_files,rapport_dir):
    for i,section_file in enumerate(section_files):
        # Cp images
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
        dir_name = section_file.split("/")[0]
        section = section_file.split("/")[1]
        shutil.copytree(rapport_dir + dir_name + "/images/" + section, images_dir+section)

    # Cp attachments
    if os.path.exists(attachments_dir):
        shutil.rmtree(attachments_dir)
    os.mkdir(attachments_dir)
    shutil.copyfile(root_dir + "abstracts/abstracts.pdf",attachments_dir + "abstracts.pdf")
    shutil.copyfile(root_dir + "meetings/meetings.pdf",attachments_dir + "meetings.pdf")
    shutil.copyfile(rapport_dir + "results.pdf",attachments_dir + "results.pdf")

    ## Cp presentation
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

# create the nav.adoc file
def create_nav_file(section_files,sections):
    # create the nav.adoc file
    nav_file = result_dir + "nav.adoc"
    file_write = open(nav_file, 'w')
    file_write.write(":stem: latexmath\n")
    
    file_write.write("* xref:main_page.adoc[Résultats]\n")

    for i,(section,subsections) in enumerate(sections.items()):    
        print(section,subsections)    
        if section_files[i]!="":
            section_file_name = section_files[i].split("/")[1]
            file_write.write("** xref:" + section_file_name + ".adoc[" + section + "]\n")
            if subsections!=None:
                for j,(subsection,subsubsections) in enumerate(subsections.items()):
                    file_write.write("*** xref:" + section_file_name + "/subsec_" + str(j) + ".adoc[" + subsection + "]\n")
                    if len(subsubsections)!=0:
                        for k,subsubsection in enumerate(subsubsections):
                            file_write.write("**** xref:" + section_file_name + "/subsec_" + str(j) + "_subsubsec_" + str(k) + ".adoc[" + subsubsection + "]\n")
        else:
            section_file_name = "section_" + str(i)
            file_write.write("** xref:" + section_file_name + ".adoc[" + section + "]\n")

    file_write.write("* xref:presentation.adoc[Slides]\n")
    file_write.close()

# create the main page of the documentation
def create_main_page_file(section_files,sections):
    # create the nav.adoc file
    main_page_file = page_dir + "main_page.adoc"
    file_write = open(main_page_file, 'w')

    file_write.write("# Phi-FEM Project\n\n")

    intro = "Après un stage dans l'équipe INRIA MIMESIS dans le cadre de mon master (Master CSMI à l'Université de Strasbourg), j'ai rejoint l'équipe en tant que doctorante en octobre 2023 sous la direction d'Emmanuel Franck, Michel Duprez et Vanessa Lleras. L'objectif de ce projet est le *\"Développement de méthodes hybrides éléments finis/réseaux neuronaux pour aider à la création de jumeaux chirurgicaux numériques\"*.\n\n"

    attach = "Vous pouvez trouver les contenus supplémentaires suivants:\n\n* un xref:attachment$abstracts.pdf[résumé hebdomadaire]\n* des xref:attachment$meetings.pdf[préparations aux meetings] (avec les résultats à présenter)\n* une https://drive.google.com/file/d/1mA1_JrBOlv6OsjKCtzuZGMHcKeHAZ4s9/view?usp=drive_link[ToDo List] des travaux à effectuer chaque semaine\n\n"

    file_write.write(intro)

    file_write.write("== Contenus supplémentaires\n\n")

    file_write.write(attach)

    file_write.write("== Résultats\n\n")

    file_write.write("Vous trouverez xref:attachment$results.pdf[ICI] les résultats obtenus sous le format PDF.\n\n")

    for i,(section,subsections) in enumerate(sections.items()):        
        if section_files[i]!="":
            section_file_name = section_files[i].split("/")[1]
            file_write.write("* xref:" + section_file_name + ".adoc[" + section + "]\n")
        else:
            section_file_name = "section_" + str(i)
            file_write.write("* xref:" + section_file_name + ".adoc[" + section + "]\n")

    file_write.close()  

# create the presentation page
def create_presentation_file(presentation_name):
    # create the nav.adoc file
    presentation_file = page_dir + "presentation.adoc"
    
    date,name = [],[]
    for key,value in presentation_name.items():
        date.append(key)
        name.append(value)
    date,name = np.array(date), np.array(name)
    index = np.argsort(date)
    date,name = date[index],name[index]

    date_fr = []
    for d in date:
        date_fr.append(d[-2:] + "/" + d[5:7] + "/" + d[:4])

    file_write = open(presentation_file, 'w')
    file_write.write("# Slides\n\n")

    for i in range(len(date)):
        file_write.write("* "+date_fr[i]+" : xref:attachment$presentation/" + date[i] + ".pdf[" + name[i] + "]\n")
    
    file_write.close() 