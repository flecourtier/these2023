import numpy as np
import os
import shutil

check_while = False

"""
But : Convertir un fichier latex en une documentation antora complète
"""

root_dir = "../../docs/"

source_dir = "results/"
rapport_dir = root_dir + source_dir
# suivi_dir = root_dir + "suivi/"
rapport_file = rapport_dir + "results.tex"

result_dir = root_dir + "antora/modules/ROOT/"
page_dir = result_dir + "pages/"
images_dir = result_dir + "assets/images/"
attachments_dir = result_dir + "assets/attachments/"

# return true if word is in line
def search_word_in_line(word, line):
    return word in line

# test if the title contains a latex formula
def test_latex_title(title):
    if search_word_in_line("$", title):
        title_split = title.split("$")
        title = ""
        start_stem = 0
        for i,part in enumerate(title_split):
            title += part
            if start_stem==0 and i!=len(title_split)-1:
                title += "stem:["
                start_stem = 1
            elif start_stem==1 and i!=len(title_split)-1:
                title += "]"
                start_stem = 0

    return title

# read "rapport.tex" and return the list of the files of the sections (e.g. "sections/section_1")
# for section which are no input, we juste create an empty section
def get_sections():
    file_read = open(rapport_file, 'r')

    # we start by complete sections
    section_files = []
    sections_name = []
    while line := file_read.readline():
        if search_word_in_line("\section", line):
            section_name = line.split("{")[1].split("}")[0]
            section_name = test_latex_title(section_name)
            sections_name.append(section_name)
        if search_word_in_line("\input", line):
            key = line.split("{")[1].split("}")[0]
            section_files.append(key)

    return section_files,sections_name

# return the list of the sections and the list of the subsections
# if there is paragraph, we creates file for subsubsection 
# {"section1":[subsection1,subsection2],"section2":[subsection1,subsection2]}
def get_subsections(section_files,sections_name):
    def test_paragraph(section):
        file_read = open(root_dir + source_dir + section + ".tex", 'r')
        while line := file_read.readline():
            if search_word_in_line("\paragraph", line):
                return True
        return False

    sections = {}
    for (i,section_file) in enumerate(section_files):
        section_name = sections_name[i]
        # if section_file!="":
        file_read = open(root_dir + source_dir + section_file + ".tex", 'r')
        subsections = {}
        add_subsubsections = test_paragraph(section_file)
        while line := file_read.readline():
            # if search_word_in_line("\section", line):
            #     pass
            if search_word_in_line("\subsection", line):
                subsection = line.split("{")[1].split("}")[0]
                subsection = test_latex_title(subsection)
                subsections[subsection] = []
            if search_word_in_line("\subsubsection", line):
                subsubsection = line.split("{")[1].split("}")[0]
                subsubsection = test_latex_title(subsubsection)
                if add_subsubsections:    
                    subsections[subsection].append(subsubsection)
                
        sections[section_name] = subsections
        # else:
        #     key = empty_sections.pop(0)
        #     key = test_latex_title(key)
        #     sections[key] = None

    return sections

# create the nav.adoc file
def create_nav_file(section_files,sections):
    # create the nav.adoc file
    nav_file = result_dir + "nav.adoc"
    file_write = open(nav_file, 'w')

    file_write.write(":stem: latexmath\n")
    file_write.write("* xref:main_page.adoc[PhiFEM PhD]\n")

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

    file_write.close()

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

# create the main page of the documentation
def create_main_page_file(section_files,sections):
    # create the nav.adoc file
    main_page_file = page_dir + "main_page.adoc"
    file_write = open(main_page_file, 'w')

    file_write.write("# Phi-FEM Project\n\n")

    intro = "Après un stage dans l'équipe INRIA MIMESIS dans le cadre de mon master (Master CSMI à l'Université de Strasbourg), j'ai rejoint l'équipe en tant que doctorante en octobre 2023 sous la direction d'Emmanuel Franck, Michel Duprez et Vanessa Lleras. L'objectif de ce projet est le *\"Développement de méthodes hybrides éléments finis/réseaux neuronaux pour aider à la création de jumeaux chirurgicaux numériques\"*.\n\n"

    attach = "Vous pouvez trouver les contenus supplémentaires suivants:\n\n* un xref:attachment$abstracts.pdf[résumé hebdomadaire]\n* des xref:attachment$meetings.pdf[préparations aux meetings] (avec les résultats à présenter)\n* une xref:attachment$to_do_list.pdf[ToDo List] des travaux à effectuer chaque semaine\n\n"

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

    # file_write.write("\nYou can find the internship report in PDF just xref:attachment$rapport.pdf[HERE] as well as a weekly tracking of the internship xref:attachment$suivi.pdf[HERE] (in french).\n")


    file_write.close()   

# copy all the images of the tex report in the antora documentation
def cp_assets(section_files):       
    for i,section_file in enumerate(section_files):
        # Cp images
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
        dir_name = section_file.split("/")[0]
        section = section_file.split("/")[1]
        shutil.copytree(root_dir + source_dir + dir_name + "/images/" + section, images_dir+section)

        # Cp attachments
        if os.path.exists(attachments_dir):
            shutil.rmtree(attachments_dir)
        os.mkdir(attachments_dir)
        shutil.copyfile(root_dir + "abstracts/abstracts.pdf",attachments_dir + "abstracts.pdf")
        shutil.copyfile(root_dir + "meetings/meetings.pdf",attachments_dir + "meetings.pdf")
        shutil.copyfile(root_dir + "to_do_list/to_do_list.pdf",attachments_dir + "to_do_list.pdf")
        shutil.copyfile(rapport_dir + "results.pdf",attachments_dir + "results.pdf")

# Test if there is a refernce to a figure in the line (many configurations possibles)
def test_fig(line):
    possible_ref = ["Figure \\ref","Figure~\\ref","Fig \\ref","Fig~\\ref","Fig.~\\ref","Fig.\\ref"]
    for ref in possible_ref:
        if search_word_in_line(ref,line):
            return ref,True
    return None,False

# Test if there is a refernce to a section in the line (many configurations possibles)
def test_section(line):
    possible_ref = ["Section \\ref","Section~\\ref","Sec \\ref","Sec~\\ref","Sec.~\\ref","Sec.\\ref"]
    for ref in possible_ref:
        if search_word_in_line(ref,line):
            return ref,True
    return None,False

# create a dict which contains all the label of sections, subsections and subsubsections
def get_label_sections(section_files):
    label_sections = {}
    for (s,(section,subsections)) in enumerate(sections.items()):
        section_file = section_files[s]
        if section_file!="":
            section_file_name = section_file.split("/")[1]
            file_read = open(root_dir + source_dir + section_file + ".tex", 'r')
            
            num_subsection = -1
            num_subsubsection = -1
            while line := file_read.readline():
                if search_word_in_line("\section", line):
                    if search_word_in_line("\label", line):
                        section_name = line.split("\label{")[1].split("}")[0]
                        section_name = test_latex_title(section_name)
                        label_sections[section_name]={"xref":[section_file_name,section]}
                
                if search_word_in_line("\subsection", line):
                    num_subsection += 1
                    num_subsubsection = -1
                    subsection_name = line.split("{")[1].split("}")[0]
                    subsection_name = test_latex_title(subsection_name)
                    if search_word_in_line("\label", line):
                        label_sections[line.split("\label{")[1].split("}")[0]]={"xref":[section_file_name+"/subsec_"+str(num_subsection),subsection_name]}

                if search_word_in_line("\subsubsection", line):
                    num_subsubsection += 1
                    if search_word_in_line("\label", line):
                        subsubsection_name = line.split("{")[1].split("}")[0]
                        subsubsection_name = test_latex_title(subsubsection_name)
                        subsubsection_name_ = subsubsection_name.replace(" ","_")
                        subsubsection_name_ = "_"+subsubsection_name_.lower()
                        label_sections[line.split("\label{")[1].split("}")[0]]={"":subsubsection_name_,"xref":[section_file_name+"/subsec_"+str(num_subsection)+"_subsubsec_"+str(num_subsubsection),subsubsection_name]}

                if search_word_in_line("\paragraph", line):
                    if search_word_in_line("\label", line):
                        paragraph_name = line.split("{")[1].split("}")[0]
                        paragraph_name = test_latex_title(paragraph_name)
                        paragraph_name_ = paragraph_name.replace(" ","_")
                        paragraph_name_ = "_"+paragraph_name_.lower()
                        label_sections[line.split("\label{")[1].split("}")[0]]={"":paragraph_name_}

    return label_sections

def cp_section(section_file,section_name,subsections,label_sections):
    file_read = open(root_dir + source_dir + section_file + ".tex", 'r')
    
    name_section_file = section_file.split("/")[1]
    file_write = open(page_dir + name_section_file + ".adoc", 'w')
    
    num_subsection = -1
    subsection_file = None
    first_minipage=True
    minipage_width = 1.
    bf_and_it = False

    section = section_name
    file_write.write(":stem: latexmath\n")
    file_write.write(":xrefstyle: short\n")
    file_write.write("= " + section + "\n")

    while line := file_read.readline():
        # if search_word_in_line("\section", line):
        #     section = line.split("{")[1].split("}")[0]
        #     section = test_latex_title(section)
        #     subsections = sections[section]
        #     file_write.write(":stem: latexmath\n")
        #     file_write.write(":xrefstyle: short\n")
        #     file_write.write("= " + section + "\n")
        #     line = ""

        if search_word_in_line("\subsection", line):
            num_subsection += 1
            num_subsubsection = -1

            if num_subsection==0 and subsections!=None:
                file_write.write("\n---\n")
                file_write.write("The features include\n\n")
                section_file_name = section_file.split("/")[1]
                for i,subsection_ in enumerate(subsections):
                    file_write.write("** xref:" + section_file_name + "/subsec_"+ str(i) + ".adoc[" + subsection_ + "]\n\n")

            subsection_file = "subsec_" + str(num_subsection) + ".adoc"
            file_write = open(page_dir + name_section_file + "/" + subsection_file, 'w')

            subsection = line.split("{")[1].split("}")[0]
            subsection = test_latex_title(subsection)
            subsubsections = subsections[subsection]
            file_write.write(":stem: latexmath\n")
            file_write.write(":xrefstyle: short\n")
            file_write.write("= " + subsection + "\n")
            line=""
        
        if search_word_in_line("\subsubsection", line):
            num_subsubsection += 1
            name_subsubsection = line.split("{")[1].split("}")[0]
            name_subsubsection = test_latex_title(name_subsubsection)
            if subsubsections!=[]:
                if num_subsubsection==0:
                    file_write.write("\n---\n")
                    file_write.write("The features include\n\n")
                    section_file_name = section_file.split("/")[1]
                    for i,subsubsection_ in enumerate(subsubsections):
                        file_write.write("** xref:" + section_file_name + "/subsec_"+ str(num_subsection) + "_subsubsec_" + str(i) + ".adoc[" + subsubsection_ + "]\n\n")

                subsubsection_file = "subsec_" + str(num_subsection) + "_subsubsec_" + str(num_subsubsection) + ".adoc"
                file_write = open(page_dir + name_section_file + "/" + subsubsection_file, 'w')
                file_write.write(":stem: latexmath\n")
                file_write.write(":xrefstyle: short\n")
                line = "= " + name_subsubsection + "\n"
            else:
                line = "== " + name_subsubsection + "\n"

        if search_word_in_line("\paragraph", line):
            name_paragraph = line.split("{")[1].split("}")[0]
            name_paragraph = test_latex_title(name_paragraph)
            line = "== " + name_paragraph + "\n"

        if search_word_in_line("\graphicspath", line):
            line = ":imagesdir: \{moduledir\}/assets/" + line.split("{")[2].split("}")[0] + "\n"

        if search_word_in_line("\modif", line):
            sentence = line.split("\modif")[1].split("{")[1].split("}")[0]
            to_replace = "\modif{" + sentence + "}"
            line = line.replace(to_replace, "#" + sentence + "#")

        if search_word_in_line("$",line):
            tab_line = line.split("$")
            start_stem = 0
            line_modif = ""
            for i,part in enumerate(tab_line):
                line_modif += part
                if start_stem==0 and i!=len(tab_line)-1:
                    line_modif += "stem:["
                    start_stem = 1
                elif start_stem==1 and i!=len(tab_line)-1:
                    line_modif += "]"
                    start_stem = 0
            line = line_modif

        if search_word_in_line("\\begin{figure}", line):
            while not search_word_in_line("\end{figure}", line):
                line = file_read.readline()
                if search_word_in_line("\includegraphics", line):
                    image_name = line.split("{")[1].split("}")[0]
                    image_name = image_name[1:-1]   

                    linewidth = line.split("width=")[1].split("\\linewidth")[0]
                    if linewidth=="":
                        linewidth="1.0"
                    linewidth = float(linewidth)

                    width = minipage_width * linewidth * 600
                    height = minipage_width * linewidth * 480

                if search_word_in_line("\caption", line):
                    count = line.count("{")
                    if count>=3:
                        while line[0]=="\t":
                            line = line[1:]
                        caption = line.replace("\captionof{figure}{","")[:-2]+"\n"
                    else:
                        caption = line.split("{")[1].split("}")[0]
                    caption = test_latex_title(caption)
                    
                if search_word_in_line("\label", line):
                    label = line.split("{")[1].split("}")[0]
                    file_write.write("[["+label+"]]\n")
            minipage_width = 1.
            line = "."+caption+"\nimage::" + name_section_file + "/" + image_name + "[width="+str(width)+",height="+str(height)+"]\n"

        if search_word_in_line("\\begin{minipage}", line):
            minipage_width = 0.5
            if first_minipage:
                line = "[cols=\"a,a\"]\n|===\n|"
                first_minipage=False
            else:
                line = "|"
                first_minipage=True
        
        if search_word_in_line("\end{minipage}", line):
            if first_minipage:
                line="\n|===\n"
            else:
                line="" 

        if search_word_in_line("\\begin{equation", line):
            line = "[stem]\n++++\n"
        
        if search_word_in_line("\end{equation", line):
            line = "++++\n"

        if search_word_in_line("\\begin{align*}", line) or search_word_in_line("\\begin{align}", line):
            line = "[stem]\n++++\n\\begin{aligned}\n"

        if search_word_in_line("\end{align*}", line) or search_word_in_line("\end{align}", line):
            line = "\\end{aligned}\n++++\n"

        if search_word_in_line("\\begin{enumerate", line) or search_word_in_line("\end{enumerate", line):
            line = "\n"

        if search_word_in_line("\\begin{itemize", line) or search_word_in_line("\end{itemize", line):
            line = "\n"
        
        if search_word_in_line("\item",line):
            line = line.replace("\item", "* ")

        if search_word_in_line("\\begin{Rem",line):
            line = "\n[NOTE]\n====\n"

        if search_word_in_line("\\end{Rem",line):
            line = "====\n"

        ref,test= test_fig(line)
        while test:
            if check_while:
                print("while 1")
            name_label_fig = line.split(ref+"{")[1].split("}")[0]
            line = line.replace(ref+"{"+name_label_fig+"}","<<"+name_label_fig+">>")
            ref,test = test_fig(line)

        ref,test = test_section(line)
        while test:
            if check_while:
                print("while 2")
            name_label_sec = line.split(ref+"{")[1].split("}")[0]
            label = label_sections[name_label_sec]
            print(label)
            if "xref" in label and "" in label:
                if subsubsections!=[]:
                    # print("xref")
                    line = line.replace(ref+"{"+name_label_sec+"}","xref:"+label["xref"][0]+".adoc"+"[Section \""+label["xref"][1]+"\"]")
                else:
                    # print("")
                    line = line.replace(ref+"{"+name_label_sec+"}","<<"+label[""]+">>")
            elif "xref" in label:
                line = line.replace(ref+"{"+name_label_sec+"}","xref:"+label["xref"][0]+".adoc"+"[Section \""+label["xref"][1]+"\"]")
            else:
                line = line.replace(ref+"{"+name_label_sec+"}","<<"+label[""]+">>")
            
            ref,test = test_section(line)

        if search_word_in_line("\\newpage",line):
            line=""

        if search_word_in_line("\_",line):
            line = line.replace("\_","_")

        while search_word_in_line("\href",line):
            if check_while:
                print("while 3")
            url = line.split("{")[1].split("}")[0]
            text = line.split("{")[2].split("}")[0]
            line = line.replace("\href{"+url+"}{"+text+"}",url+"["+text+"]")

        while search_word_in_line("\\textbf{\\textit",line):
            bf_and_it = True
            sentence = line.split("\\textit")[1].split("{")[1].split("}")[0]
            line = line.replace("\\textbf{\\textit{"+sentence+"}}","*_"+sentence+"_*")

        while search_word_in_line("\\textit{\\textbf",line):
            bf_and_it = True
            sentence = line.split("\\textbf")[1].split("{")[1].split("}")[0]
            line = line.replace("\\textit{\\textbf{"+sentence+"}}","*_"+sentence+"_*")

        if bf_and_it:
            bf_and_it=False
        else:
            while search_word_in_line("\\textbf",line):
                if check_while:
                    print("while 4")
                sentence = line.split("\\textbf")[1].split("{")[1].split("}")[0]
                line = line.replace("\\textbf{"+sentence+"}","*"+sentence+"*")

            while search_word_in_line("\\textit",line):
                if check_while:
                    print("while 5")
                sentence = line.split("\\textit")[1].split("{")[1].split("}")[0]
                line = line.replace("\\textit{"+sentence+"}","_"+sentence+"_")

        if search_word_in_line("\\begin{Prop}",line):
            if search_word_in_line("[",line):
                prop_title = line.split("[")[1].split("]")[0]
                line = "\n[]\n====\n*Propositon ("+prop_title+").*\n"
            else:
                line = "\n[]\n====\n*Propositon.*\n"

        if search_word_in_line("\\end{Prop}",line):
            line = "====\n"

        if search_word_in_line("\\begin{Def}",line):
            if search_word_in_line("[",line):
                def_title = line.split("[")[1].split("]")[0]
                line = "\n[]\n====\n*Definition ("+def_title+").*\n"
            else:
                line = "\n[]\n====\n*Definition.*\n"

        if search_word_in_line("\\end{Def}",line):
            line = "====\n"

        if search_word_in_line("\\begin{Example}",line):
                line = "\n---\n*Example.*\n"

        if search_word_in_line("\\end{Example}",line):
            line = "\n---\n"

        if line!="":
            while line[0]=="\t":
                line = line[1:]

            if line[0]!="%":
                file_write.write(line)
    
def cp_all_sections(section_files,section_names,sections):
    label_sections = get_label_sections(section_files)
    for i in range(len(section_files)):
        subsections = sections[section_names[i]]        
        # if section_files[i]!="":
        print(section_files[i])
        print(sections)
        print(label_sections)
        cp_section(section_files[i],section_names[i],subsections,label_sections)
        # else:
        #     section_file_name = "section_" + str(i)
        #     file_write = open(page_dir + section_file_name + ".adoc", 'w')
        #     file_write.write("= " + section + "\n\n")
        #     file_write.write("#TO COMPLETE !#\n")
        #     file_write.close()

def rm_all():
    shutil.rmtree(page_dir)
    os.mkdir(page_dir)
    shutil.rmtree(attachments_dir)
    os.mkdir(attachments_dir)
    shutil.rmtree(images_dir)
    os.mkdir(images_dir)
    os.remove(result_dir + "nav.adoc")

# si il y a déjà un dossier antora, on le supprime et on copie le dossier antora_base
# rm_all()
section_files,section_names = get_sections()
print(section_files)
sections = get_subsections(section_files,section_names)
print(sections)
create_nav_file(section_files,sections)
create_nav(section_files,sections)
create_main_page_file(section_files,sections)
cp_assets(section_files)
cp_all_sections(section_files,section_names,sections)
label_sections = get_label_sections(section_files)
print(label_sections)
