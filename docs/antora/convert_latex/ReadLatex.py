from utils import *

# read "rapport.tex" and return the list of the files of the sections (e.g. "sections/section_1")
# for section which are no input, we juste create an empty section
def get_sections(rapport_file):
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
            sections_name.append("")

    return section_files,sections_name

# return the list of the sections and the list of the subsections
# if there is paragraph, we creates file for subsubsection 
# {"section1":[subsection1,subsection2],"section2":[subsection1,subsection2]}
def get_subsections(section_files,sections_name,rapport_dir):
    def test_paragraph(section):
        file_read = open(rapport_dir + section + ".tex", 'r')
        while line := file_read.readline():
            if search_word_in_line("\paragraph", line):
                return True
        return False

    sections = {}
    for (i,section_file) in enumerate(section_files):
        section_name = sections_name[i]

        file_read = open(rapport_dir + section_file + ".tex", 'r')
        subsections = {}
        add_subsubsections = test_paragraph(section_file)
        while line := file_read.readline():
            if search_word_in_line("\section", line):
                assert section_name == ""
                section_name = line.split("{")[1].split("}")[0]
                section_name = test_latex_title(section_name)
                sections_name[i] = section_name
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

    return sections,sections_name

# create a dict which contains all the label of sections, subsections and subsubsections
def get_label_sections(section_files,sections,rapport_dir):
    chapter_dir = rapport_dir.split("/")[-2] + "/"
    label_sections = {}
    for (s,(section,subsections)) in enumerate(sections.items()):
        section_file = section_files[s]
        if section_file!="":
            section_file_name = chapter_dir+section_file.split("/")[1]
            file_read = open(rapport_dir + section_file + ".tex", 'r')
            
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