from datetime import date,timedelta
import numpy as np
import os
import shutil

import subprocess

def write_entete(file_write, title, tableofcontents):
    file_write.write('\n\\begin{document}\n')
    name = "LECOURTIER Frédérique \hfill \\today\n"
    entete = '\t'+name+'\t\\begin{center}\n\t\t\Large\\textbf{{'+title+'}}\n\t\end{center}\n'
    if tableofcontents:
        entete += '\t\\tableofcontents\n'
    file_write.write(entete)

def get_values():
    monday = date(2023, 10, 2)
    friday = monday + timedelta(days=4)
    week = timedelta(days=7) # 7 days 

    today = date.today()
    delta = today - monday
    current_week_num = int(np.ceil((delta.days+1) / 7))

    return monday, friday, week, current_week_num

def run_abstracts():
    monday, friday, week, current_week_num = get_values()

    print("### Abstracts ###")
    current_dir = "abstracts/"
    results_file = current_dir+"abstracts.tex"
    if os.path.exists(results_file):
        os.remove(results_file)

    to_include = current_dir+"include.txt"
    shutil.copyfile(to_include, results_file)
    file_write = open(results_file,"a")
    title = "Asbtracts : Week 1 - Week "+str(current_week_num)
    write_entete(file_write, title, False)

    abstracts_repo = "weeks/"
    liste = []
    for i in range(1,current_week_num+1):
        asbtract_filename = abstracts_repo+"week_"+str(i)

        monday_str = monday.strftime("%d/%m/%Y")
        friday_str = friday.strftime("%d/%m/%Y")

        week_title = "Week "+str(i)+" : "+monday_str+" - "+friday_str
        file_write.write('\n\t\\section{'+week_title+'}\n')

        if os.path.exists(current_dir+asbtract_filename+".tex"):
            liste.append(i)
            file_write.write('\t\input{'+asbtract_filename+'}\n')
            
        monday+=week
        friday+=week
    print(liste)

    file_write.write('\end{document}')

def run_results():
    monday, friday, week, current_week_num = get_values()

    print("### Results ###")
    current_dir = "results/"
    results_file = current_dir+"results.tex"
    if os.path.exists(results_file):
        os.remove(results_file)

    to_include = current_dir+"include.txt"
    shutil.copyfile(to_include, results_file)
    file_write = open(results_file,"a")
    title = "Results : Week 1 - Week "+str(current_week_num)
    write_entete(file_write, title, True)

    results_repo = "weeks/"
    liste = []
    for i in range(1,current_week_num+1):
        results_filename = results_repo+"week_"+str(i)

        monday_str = monday.strftime("%d/%m/%Y")
        friday_str = friday.strftime("%d/%m/%Y")

        if os.path.exists(current_dir+results_filename+".tex"):
            liste.append(i)
            file_write.write('\t\\newpage\n')
            week_title = "Week "+str(i)+" : "+monday_str+" - "+friday_str
            file_write.write('\n\t\\section{'+week_title+'}\n')
            file_write.write('\t\input{'+results_filename+'}\n')

        monday+=week
        friday+=week
    print(liste)

    file_write.write('\end{document}')

def run_report():
    monday, friday, week, current_week_num = get_values()

    print("### Report ###")
    current_dir = "report/"
    results_file = current_dir+"report.tex"
    if os.path.exists(results_file):
        os.remove(results_file)

    to_include = current_dir+"include.txt"
    shutil.copyfile(to_include, results_file)
    file_write = open(results_file,"a")
    title = "Report : Week 1 - Week "+str(current_week_num)
    write_entete(file_write, title, True)

    for subdir in os.listdir(current_dir):
        if os.path.isdir(current_dir+subdir) and subdir[-1] != "_":
            chapter_name = subdir[0].upper()+subdir[1:]
            file_write.write('\n\t\\chapter{'+chapter_name+'}\n')
            results_repo = subdir+"/"+"sections/"

            for section in os.listdir(current_dir+results_repo):
                section_name = section.replace(".tex","")
                file_write.write('\t\\newpage\n')
                file_write.write('\t\input{'+results_repo+section_name+'}\n')

            # print(liste)

    file_write.write('\end{document}')

def run_chapter_report():
    monday, friday, week, current_week_num = get_values()

    print("### Chapter ###")
    current_dir = "report/"
    to_include = current_dir+"include_chapter.txt"
    
    for subdir in os.listdir(current_dir):
        if os.path.isdir(current_dir+subdir) and subdir[-1] != "_":
            title = subdir[0].upper()+subdir[1:]
            print("## "+title)
            chapter_dir = current_dir+subdir+"/"
            results_file = chapter_dir+"report.tex"
            if os.path.exists(results_file):
                os.remove(results_file)

            shutil.copyfile(to_include, results_file)
            file_write = open(results_file,"a")
            write_entete(file_write, title, True)

            results_repo = "sections/"
            for section in os.listdir(chapter_dir+results_repo+"/"):
                section_name = section.replace(".tex","")
                file_write.write('\t\\newpage\n')
                file_write.write('\t\input{'+results_repo+section_name+'}\n')

            file_write.write('\end{document}')

run_abstracts()
run_report()
run_chapter_report()