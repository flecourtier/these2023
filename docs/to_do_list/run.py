from datetime import date,timedelta
import numpy as np
import os
import shutil

monday = date(2023, 10, 2)
friday = monday + timedelta(days=4)
week = timedelta(days=7) # 7 days 

today = date.today()
delta = today - monday
current_week_num = int(np.ceil((delta.days+1) / 7))
print("Current week number: "+str(current_week_num))



to_include = "include.txt"
results_file = "to_do_list.tex"
if os.path.exists(results_file):
    os.remove(results_file)

shutil.copyfile(to_include, results_file)
file_write = open(results_file,"a")
file_write.write('\n\\begin{document}\n')
name = "LECOURTIER Frédérique \hfill \\today\n"
title = "ToDoList : Week 1 - Week "+str(current_week_num)
entete = '\t'+name+'\t\\begin{center}\n\t\t\Large\\textbf{{'+title+'}}\n\t\end{center}\n\t\\tableofcontents\n'
file_write.write(entete)

todo_repo = "weeks/"


for i in range(1,current_week_num+1):
    todo_filename = todo_repo+"week_"+str(i)+".tex"

    monday_str = monday.strftime("%d/%m/%Y")
    friday_str = friday.strftime("%d/%m/%Y")

    # si to_do_list existe 
    if os.path.exists(todo_filename):
        file_write.write('\n\t\\newpage\n')
        week_title = "Week "+str(i)+" : "+monday_str+" - "+friday_str
        file_write.write('\n\t\\section*{'+week_title+'}\n')
        file_write.write('\n\t\\addcontentsline{toc}{section}{'+week_title+'}\n')
        file_write.write('\t\t\input{'+todo_filename+'}\n')
        
        # file_read = open(todo_filename, 'r')
        # lines_to_write = file_read.readlines()
        # for line_to_write in lines_to_write:
        #     file_write.write("\t"+line_to_write)
        # file_write.write('\n')
        
    monday+=week
    friday+=week

file_write.write('\end{document}')