
# import xlsxwriter module
import xlsxwriter
import os
import openpyxl as xl

from modules.utils import read_config
from modules.Case import *

# define cells formats
def create_formats(workbook):
    title_format = workbook.add_format(
        {
            "bold": 1,
        }
    )

    training_format = workbook.add_format(
        {
            "bold": 1,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "fg_color": "yellow",
        }
    )

    results_format = workbook.add_format(
        {
            "bold": 1,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "fg_color": "green",
        }
    ) 

    format_white = workbook.add_format(
        {
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        }
    )

    format_grey = workbook.add_format(
        {
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "fg_color": "C0C0C0",
        }
    )

    return {"title_format":title_format,"training_format":training_format,"results_format":results_format,"format_white":format_white,"format_grey":format_grey}

# define title lines
def training_titles(worksheet,cell,formats):
    row,col = cell

    worksheet.merge_range(row, col, row+1, col, "Configuration", formats["training_format"])

    col+=1
    worksheet.merge_range(row, col, row, col+1, "Model parameters", formats["training_format"])
    worksheet.write(row+1, col, 'Layers',formats["training_format"])
    col+=1
    worksheet.write(row+1, col, 'Activation Function',formats["training_format"])

    col+=1
    worksheet.merge_range(row, col, row, col+4, "Trainer parameters", formats["training_format"])
    worksheet.write(row+1, col, 'Learning rate',formats["training_format"])
    col+=1
    worksheet.write(row+1, col, 'Decay',formats["training_format"])
    col+=1
    worksheet.write(row+1, col, 'w_data',formats["training_format"])
    col+=1
    worksheet.write(row+1, col, 'w_res',formats["training_format"])
    col+=1
    worksheet.write(row+1, col, 'w_bc',formats["training_format"])

    col+=1
    worksheet.merge_range(row, col, row, col+3, "Training parameters", formats["training_format"])
    worksheet.write(row+1, col, 'n_epochs',formats["training_format"])
    col+=1
    worksheet.write(row+1, col, 'n_collocation',formats["training_format"])
    col+=1
    worksheet.write(row+1, col, 'n_bc_collocation',formats["training_format"])
    col+=1
    worksheet.write(row+1, col, 'n_data',formats["training_format"])

    # loss columns

    col+=1
    worksheet.merge_range(row, col, row, col+1, 'Training results',formats["results_format"])
    worksheet.write(row+1, col, "Omega", formats["results_format"])
    worksheet.set_column(col, col, 20)
    col+=1
    worksheet.write(row+1, col, "O_cal", formats["results_format"])
    worksheet.set_column(col, col, 20)

    row+=2

    return row,col

def training(worksheet,results_dir_sampling,cell,formats,num=0):
    row,col = cell

    column_keys = ["layers","activation","lr","decay","w_data","w_res","w_bc","n_epochs","n_collocations","n_bc_collocations","n_data"]

    models_dir = results_dir_sampling+"models/"

    config_filename = models_dir+"config_"+str(num)+".json"
    previous_dict = None

    format = formats["format_white"]
    while os.path.isfile(config_filename):

        dict = read_config(config_filename)

        worksheet.set_row(row, 60)

        col = 0
        worksheet.write(row, col, str(num),format)
        for key in column_keys:
            if key not in dict:
                dict[key] = 0
            value = dict[key]

            if previous_dict != None:
                if previous_dict[key] != value:
                    format = formats["format_grey"]
                else:
                    format = formats["format_white"]
            col+=1
            worksheet.write(row, col, str(value), format)

        col += 1
        result_filename = results_dir_sampling+"solutions/model_"+str(num)+"_Omega.png"
        worksheet.write(row, col, "", format)
        if os.path.isfile(result_filename):
            worksheet.insert_image(row, col, result_filename, {"x_scale": 0.1, "y_scale": 0.1})

        col += 1
        result_filename = results_dir_sampling+"solutions/model_"+str(num)+"_O_cal.png"
        worksheet.write(row, col, "", format)
        if os.path.isfile(result_filename):
            worksheet.insert_image(row, col, result_filename, {"x_scale": 0.1, "y_scale": 0.1})

        num+=1
        config_filename = models_dir+"config_"+str(num)+".json"
        row+=1

        previous_dict = dict.copy()

    if num==0:
        training(worksheet,results_dir_sampling,cell,formats,num=1)

    return row,col

def results_titles(worksheet,cell,formats):
    row,col = cell

    worksheet.set_column(col, col+11, 18)

    worksheet.merge_range(row, col, row, col+11, "Additive correction", formats["results_format"])
    row+=1

    worksheet.merge_range(row, col, row+3, col, "Configuration", formats["results_format"])

    # FEM
    col+=1
    worksheet.merge_range(row, col, row, col+4, "Correction with FEM", formats["results_format"])

    worksheet.merge_range(row+1, col, row+3, col, "Correction", formats["results_format"])
    worksheet.merge_range(row+1, col+1, row+1, col+4, "Derivatives", formats["results_format"])
    worksheet.merge_range(row+2, col+1, row+2, col+2, "First", formats["results_format"])
    worksheet.write(row+3, col+1, "x", formats["results_format"])
    worksheet.write(row+3, col+2, "y", formats["results_format"])
    worksheet.merge_range(row+2, col+3, row+2, col+4, "Second", formats["results_format"])
    worksheet.write(row+3, col+3, "x", formats["results_format"])
    worksheet.write(row+3, col+4, "y", formats["results_format"])

    # PhiFEM
    col+=5
    worksheet.merge_range(row, col, row, col+5, "Correction with PhiFEM", formats["results_format"])

    worksheet.merge_range(row+1, col, row+1, col+1, "Correction", formats["results_format"])
    worksheet.merge_range(row+2, col, row+3, col, "Omega_h", formats["results_format"])
    worksheet.merge_range(row+2, col+1, row+3, col+1, "Omega", formats["results_format"])
    worksheet.merge_range(row+1, col+2, row+1, col+5, "Derivatives", formats["results_format"])
    worksheet.merge_range(row+2, col+2, row+2, col+3, "First", formats["results_format"])
    worksheet.write(row+3, col+2, "x", formats["results_format"])
    worksheet.write(row+3, col+3, "y", formats["results_format"])
    worksheet.merge_range(row+2, col+4, row+2, col+5, "Second", formats["results_format"])
    worksheet.write(row+3, col+4, "x", formats["results_format"])
    worksheet.write(row+3, col+5, "y", formats["results_format"])

    row+=4
    col+=11

    return row,col

def results(worksheet,results_dir_sampling,cell,formats,num=0):
    row,col = cell

    models_dir = results_dir_sampling+"models/"

    corr_dir = results_dir_sampling+"corrections/add/"
    derivees_dir = results_dir_sampling+"derivees/"

    config_filename = models_dir+"config_"+str(num)+".json"

    format = formats["format_white"]
    while os.path.isfile(config_filename):
        dict = read_config(config_filename)

        worksheet.set_row(row, 60)

        col = 0
        worksheet.write(row, col, str(num),format)

        corr_type="FEM"

        col += 1
        result_filename = corr_dir+corr_type+"/corr_"+corr_type+"_"+str(num)+".png"
        worksheet.write(row, col, "", format)
        if os.path.isfile(result_filename):
            worksheet.insert_image(row, col, result_filename, {"x_scale": 0.05, "y_scale": 0.05})

        tab_derivees=["x","y","xx","yy"]
        for derivee in tab_derivees:
            config_dir = derivees_dir+"config_"+str(num)+"/"

            col += 1
            result_filename = config_dir+"derivees_Omega_"+derivee+".png"
            worksheet.write(row, col, "", format)
            if os.path.isdir(config_dir) and os.path.isfile(result_filename):
                worksheet.insert_image(row, col, result_filename, {"x_scale": 0.08, "y_scale": 0.08})

        corr_type="PhiFEM"

        col += 1
        result_filename = corr_dir+corr_type+"/corr_"+corr_type+"_"+str(num)+".png"
        worksheet.write(row, col, "", format)
        if os.path.isfile(result_filename):
            worksheet.insert_image(row, col, result_filename, {"x_scale": 0.05, "y_scale": 0.05})

        col += 1
        result_filename = corr_dir+corr_type+"/corr_"+corr_type+"_"+str(num)+"_Omega.png"
        worksheet.write(row, col, "", format)
        if os.path.isfile(result_filename):
            worksheet.insert_image(row, col, result_filename, {"x_scale": 0.08, "y_scale": 0.08})

        tab_derivees=["x","y","xx","yy"]
        for derivee in tab_derivees:
            config_dir = derivees_dir+"config_"+str(num)+"/"

            col += 1
            result_filename = config_dir+"derivees_Omega_h_"+derivee+".png"
            worksheet.write(row, col, "", format)
            if os.path.isdir(config_dir) and os.path.isfile(result_filename):
                worksheet.insert_image(row, col, result_filename, {"x_scale": 0.08, "y_scale": 0.08})

        # col += 1
        # result_filename = dir_name+"results/corr/corr_phifem_"+str(num)+"_exact_bc.png"
        # worksheet.write(row, col, "", format)
        # if os.path.isfile(result_filename):
        #     worksheet.insert_image(row, col, result_filename, {"x_scale": 0.05, "y_scale": 0.05})

        # col += 1
        # result_filename = dir_name+"results/corr/corr_phifem_"+str(num)+".png"
        # worksheet.write(row, col, "", format)
        # if os.path.isfile(result_filename):
        #     worksheet.insert_image(row, col, result_filename, {"x_scale": 0.05, "y_scale": 0.05})


        num+=1
        config_filename = models_dir+"config_"+str(num)+".json"
        row+=1

        previous_dict = dict.copy()

    if num==0:
        results(worksheet,results_dir_sampling,cell,formats,num=1)

    return row,col

def create_xlsx_file(cas):
    impose_exact_bc = cas.impose_exact_bc
    class_Problem = cas.class_Problem
    name_class_Problem = class_Problem.__name__
    class_PDE = cas.class_PDE
    name_class_PDE = class_PDE.__name__

    root_dir = "networks/"
    if impose_exact_bc:
        root_dir += "exact_bc/"
    else:
        root_dir += "approach_bc/"

    # to search results
    results_dir = root_dir+name_class_PDE+"/"+name_class_Problem+"/"
    print("results_dir :",results_dir)

    # to create excel file
    excel_filename = root_dir+name_class_PDE+"_"+name_class_Problem
    if os.path.isfile(excel_filename+".xlsx"):
        os.remove(excel_filename+".xlsx")

    workbook = xlsxwriter.Workbook(excel_filename+".xlsx")
    formats = create_formats(workbook)


    def worksheet1():
        worksheet = workbook.add_worksheet("Training")

        def training_on(sampling_on,cell):
            row,col = cell

            worksheet.write(row, col, "Training on "+sampling_on+" :",formats["title_format"])
            row+=1
            results_dir_sampling = results_dir+sampling_on+"_training/"
            row,col = training_titles(worksheet,[row,col],formats)   
            row,col = training(worksheet,results_dir_sampling,[row,col],formats)
            
            return row,col
        
        row,col = 0,0
        row,col = training_on("Omega",[row,col])

        row+=1
        col=0
        row,col = training_on("O_cal",[row,col])

        worksheet.autofit()

        return row,col
    
    row,col = worksheet1()

    def worksheet2():
        worksheet = workbook.add_worksheet("Results")

        def results_on(sampling_on,cell):
            row,col = cell

            worksheet.write(row, col, "Results on "+sampling_on+" :",formats["title_format"])
            row+=1
            results_dir_sampling = results_dir+sampling_on+"_training/"
            row,col = results_titles(worksheet,[row,col],formats)   
            row,col = results(worksheet,results_dir_sampling,[row,col],formats)
            
            return row,col
        
        row,col = 0,0
        row,col = results_on("Omega",[row,col])

        row+=1
        col=0
        row,col = results_on("O_cal",[row,col])

        # col=0
        # row,col = results_titles(worksheet,[row,col],formats)

        worksheet.autofit()

        return row,col
    
    row,col = worksheet2()

    workbook.close()

cas = Case("case.json")
create_xlsx_file(cas)