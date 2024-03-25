from utils import *
from ReadLatex import get_label_sections

check_while = False

root_dir, result_dir, page_dir, images_dir, attachments_dir = get_dir()

def cp_section(section_file,section_name,subsections,label_sections,rapport_dir):
    file_read = open(rapport_dir + section_file + ".tex", 'r')
    
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
    
def cp_all_sections(section_files,section_names,sections,rapport_dir):
    label_sections = get_label_sections(section_files,sections,rapport_dir)
    for i in range(len(section_files)):
        subsections = sections[section_names[i]]        
        # if section_files[i]!="":
        print(section_files[i])
        print(sections)
        print(label_sections)
        cp_section(section_files[i],section_names[i],subsections,label_sections,rapport_dir)
        # else:
        #     section_file_name = "section_" + str(i)
        #     file_write = open(page_dir + section_file_name + ".adoc", 'w')
        #     file_write.write("= " + section + "\n\n")
        #     file_write.write("#TO COMPLETE !#\n")
        #     file_write.close()