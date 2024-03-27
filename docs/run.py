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

def run_todolist():
    monday, friday, week, current_week_num = get_values()

    print("### TODOList ###")
    current_dir = "to_do_list/"
    results_file = current_dir+"to_do_list.tex"
    if os.path.exists(results_file):
        os.remove(results_file)

    to_include = current_dir+"include.txt"
    shutil.copyfile(to_include, results_file)
    file_write = open(results_file,"a")
    title = "ToDoList : Week 1 - Week "+str(current_week_num)
    write_entete(file_write, title, True)

    todo_repo = "weeks/"
    liste = []
    for i in range(1,current_week_num+1):
        todo_filename = todo_repo+"week_"+str(i)+".tex"

        monday_str = monday.strftime("%d/%m/%Y")
        friday_str = friday.strftime("%d/%m/%Y")

        # si to_do_list existe 
        if os.path.exists(current_dir+todo_filename):
            # lire première ligne de week_i.tex
            with open(current_dir+todo_filename, 'r') as file:
                first_line = file.readline()
                status=False
                if "COMPLET" in first_line:
                    status=True  
            liste.append(i)
            file_write.write('\n\t\\newpage\n')
            week_title = "Week "+str(i)+" : "+monday_str+" - "+friday_str
            if status:
                week_title += " - COMPLET"
            file_write.write('\n\t\\section*{'+week_title+'}\n')
            file_write.write('\t\\addcontentsline{toc}{section}{'+week_title+'}\n')
            file_write.write('\t\input{'+todo_filename+'}\n')
            
        monday+=week
        friday+=week
    print(liste)
    
    file_write.write('\end{document}')

def run_meetings():
    monday = date(2023, 10, 2)
    day = timedelta(days=1)

    today = date.today()+timedelta(days=7)
    delta = (today - monday).days+1

    print("### Meetings ###")
    current_dir = "meetings/"

    ######################
    # create one file for all meetings
    ######################

    to_include = current_dir+"include.txt"

    meeting_file = current_dir+"meetings.tex"
    if os.path.exists(meeting_file):
        os.remove(meeting_file)

    shutil.copyfile(to_include, meeting_file)
    file_write = open(meeting_file,"a")
    file_write.write('\n\\title{Meeting\'s results}\n')
    file_write.write('\n\\author{LECOURTIER Frédérique}\n')
    file_write.write('\n\\date{\today}\n')
    file_write.write('\n\\begin{document}\n')
    file_write.write('\n\t\\maketitle\n')
    file_write.write('\n\t\\tableofcontents\n')

    results_repo = "days/"
    current_day = monday
    for i in range(delta):
        meeting_filedir = results_repo+current_day.strftime("%m_%d_%Y")+"/"
        meeting_filename = meeting_filedir+"day.tex"
        
        current_day_str = current_day.strftime("%A %d %B %Y")
        if os.path.exists(current_dir+meeting_filename):
            print(meeting_filename)
        
            file_write.write('\t\\newpage\n')
            week_title = "Meeting - "+current_day_str
            file_write.write('\n\t\\chapter{'+week_title+'}\n')
            file_write.write('\t\\graphicspath{{'+meeting_filedir+'images/}}\n')
            file_write.write('\t\\newpage\n')
            file_write.write('\t\input{'+meeting_filename+'}\n')

        current_day+=day

    file_write.write('\end{document}')

    ######################
    # create one file for each meetings
    ######################

    results_repo = "days/"
    
    to_include = current_dir+results_repo+"include.txt"

    current_day = monday
    for i in range(delta):
        meeting_filedir = results_repo+current_day.strftime("%m_%d_%Y")+"/"
        meeting_filename = meeting_filedir+"day.tex"
        meeting_file = meeting_filedir+"meeting.tex"
        current_day_str = current_day.strftime("%A %d %B %Y")

        if os.path.exists(current_dir+meeting_filename):
            shutil.copyfile(to_include, current_dir+meeting_file)
            file_write = open(current_dir+meeting_file,"a")
            # title = "Meeting's results - "+current_day_str
            # write_entete(file_write, title, False)

            file_write.write('\n\\begin{document}\n')
            file_write.write('\n\t\\begin{titlepage}\n')
            file_write.write('\n\t\t\\vspace*{\\stretch{1}}\n')
            file_write.write('\n\t\t\\begin{center}\n')
            file_write.write('\n\t\t\t{\\Huge\\bfseries Meeting\'s results} \\\\[1ex]\n')
            file_write.write('\n\t\t\t{\\Large\\bfseries '+current_day_str+'} \\\\[1ex]\n')
            file_write.write('\n\t\t\tLECOURTIER Frédérique\n')
            file_write.write('\n\t\t\\end{center}\n')
            file_write.write('\n\t\t\\vspace{\\stretch{2}}\n')
            file_write.write('\n\t\\end{titlepage}\n')

            file_write.write('\t\\graphicspath{{images/}}\n')
            file_write.write('\t\input{day.tex}\n')
            file_write.write('\end{document}')

        current_day+=day

run_abstracts()
run_results()
run_todolist()
run_meetings()