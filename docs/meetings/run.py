from datetime import date,timedelta
import numpy as np
import os
import shutil

monday = date(2023, 10, 2)
# friday = monday + timedelta(days=4)
day = timedelta(days=1) # 7 days 

today = date.today()+timedelta(days=7)
delta = (today - monday).days+1


to_include = "include.txt"

# create one file for all meetings
meeting_file = "meeting.tex"
if os.path.exists(meeting_file):
    os.remove(meeting_file)

shutil.copyfile(to_include, meeting_file)
file_write = open(meeting_file,"a")
file_write.write('\n\\begin{document}\n')
name = "LECOURTIER Frédérique \hfill \\today\n"
title = "Meeting's results"
entete = '\t'+name+'\t\\begin{center}\n\t\t\Large\\textbf{{'+title+'}}\n\t\end{center}\n\t\\tableofcontents\n'
file_write.write(entete)

results_repo = "days/"

current_day = monday
for i in range(delta):
    meeting_filedir = results_repo+current_day.strftime("%m_%d_%Y")+"/"
    meeting_filename = meeting_filedir+"day.tex"
    
    if os.path.exists(meeting_filename):
        print(meeting_filename)
    current_day_str = current_day.strftime("%A %d %B %Y")

    if os.path.exists(meeting_filename):
        file_write.write('\t\\newpage\n')
        week_title = "Meeting - "+current_day_str
        file_write.write('\n\t\\section{'+week_title+'}\n')
        file_write.write('\t\\graphicspath{{'+meeting_filedir+'images/}}\n')
        file_write.write('\t\input{'+meeting_filename+'}\n')

    current_day+=day

file_write.write('\end{document}')

# create one file for each meetings
results_repo = "days/"
current_day = monday
for i in range(delta):
    meeting_filedir = results_repo+current_day.strftime("%m_%d_%Y")+"/"
    meeting_filename = meeting_filedir+"day.tex"
    meeting_file = meeting_filedir+"meeting.tex"
    current_day_str = current_day.strftime("%A %d %B %Y")

    if os.path.exists(meeting_filename):
        shutil.copyfile(to_include, meeting_file)
        file_write = open(meeting_file,"a")
        file_write.write('\n\\begin{document}\n')
        name = "LECOURTIER Frédérique \hfill \\today\n"
        title = "Meeting's results - "+current_day_str
        entete = '\t'+name+'\t\\begin{center}\n\t\t\Large\\textbf{{'+title+'}}\n\t\end{center}\n'
        file_write.write(entete)

        # file_write.write('\t\\newpage\n')
        # week_title = "Meeting - "+current_day_str
        # file_write.write('\n\t\\section{'+week_title+'}\n')
        file_write.write('\t\\graphicspath{{images/}}\n')
        file_write.write('\t\input{day.tex}\n')

        file_write.write('\end{document}')

    current_day+=day

    