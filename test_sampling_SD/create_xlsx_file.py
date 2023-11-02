
# import xlsxwriter module
import xlsxwriter
import os
import json
import openpyxl as xl


def read_config(filename):
    with open(filename) as f:
        raw_config = f.read()
        dict = json.loads(raw_config)
    return  dict

# define title lines
def create_title_lines(workbook,worksheet):
    title_format = workbook.add_format(
        {
            "bold": 1,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "fg_color": "yellow",
        }
    )

    image_format = workbook.add_format(
        {
            "bold": 1,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "fg_color": "green",
        }
    ) 

    row=0
    col=0

    worksheet.merge_range(row, col, row+2, col, "Configuration", title_format)

    col+=1
    worksheet.merge_range(row, col, row+1, col+1, "Model parameters", title_format)
    worksheet.write(row+2, col, 'Layers',title_format)
    col+=1
    worksheet.write(row+2, col, 'Activation Function',title_format)

    col+=1
    worksheet.merge_range(row, col, row+1, col+4, "Trainer parameters", title_format)
    worksheet.write(row+2, col, 'Learning rate',title_format)
    col+=1
    worksheet.write(row+2, col, 'Decay',title_format)
    col+=1
    worksheet.write(row+2, col, 'w_data',title_format)
    col+=1
    worksheet.write(row+2, col, 'w_res',title_format)
    col+=1
    worksheet.write(row+2, col, 'w_bc',title_format)

    col+=1
    worksheet.merge_range(row, col, row+1, col+3, "Training parameters", title_format)
    worksheet.write(row+2, col, 'n_epochs',title_format)
    col+=1
    worksheet.write(row+2, col, 'n_collocation',title_format)
    col+=1
    worksheet.write(row+2, col, 'n_bc_collocation',title_format)
    col+=1
    worksheet.write(row+2, col, 'n_data',title_format)

    # loss columns

    col+=1
    worksheet.merge_range(row, col, row+1, col+1, 'Training loss',image_format)
    worksheet.write(row+2, col, "Exact BC.", image_format)
    worksheet.set_column(col, col, 20)
    col+=1
    worksheet.write(row+2, col, "No exact BC.", image_format)
    worksheet.set_column(col, col, 20)

    # corr columns

    col+=1
    worksheet.merge_range(row, col, row, col+3, 'Correction',image_format)
    worksheet.merge_range(row+1, col, row+1, col+1, "FEM", image_format)
    worksheet.write(row+2, col, "Exact BC.", image_format)
    worksheet.set_column(col, col, 14)
    col+=1
    worksheet.write(row+2, col, "No exact BC.", image_format)
    worksheet.set_column(col, col, 14)
    col+=1
    worksheet.merge_range(row+1, col, row+1, col+1, "PhiFEM", image_format)
    worksheet.write(row+2, col, "Exact BC.", image_format)
    worksheet.set_column(col, col, 14)
    col+=1
    worksheet.write(row+2, col, "No exact BC.", image_format)
    worksheet.set_column(col, col, 14)

    row+=3

    return row,col

def create_xlsx_file(problem_considered, pde_considered):
    excel_filename = "networks/"+problem_considered.__name__+"/models_config_"+pde_considered.__name__
    if os.path.isfile(excel_filename+".xlsx"):
        os.remove(excel_filename+".xlsx")

    workbook = xlsxwriter.Workbook(excel_filename+".xlsx")
    worksheet = workbook.add_worksheet()

    row,col = create_title_lines(workbook,worksheet)   

    # Start config
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

    dir_name = "networks/"+problem_considered.__name__+"/"+pde_considered.__name__+"/"

    num = 0
    config_filename = dir_name+"configs/config_"+str(num)+".json"
    print(config_filename)
    previous_dict = None

    format = format_white
    while os.path.isfile(config_filename):
        dict = read_config(config_filename)

        worksheet.set_row(row, 60)

        col = 0
        worksheet.write(row, col, str(num),format)
        for key,value in dict.items():
            if previous_dict != None:
                if previous_dict[key] != value:
                    format = format_grey
                else:
                    format = format_white
            col+=1
            worksheet.write(row, col, str(value), format)

        col += 1
        result_filename = dir_name+"results/model_"+str(num)+"_exact_bc.png"
        worksheet.write(row, col, "", format)
        if os.path.isfile(result_filename):
            worksheet.insert_image(row, col, result_filename, {"x_scale": 0.1, "y_scale": 0.1})

        col += 1
        result_filename = dir_name+"results/model_"+str(num)+".png"
        worksheet.write(row, col, "", format)
        if os.path.isfile(result_filename):
            worksheet.insert_image(row, col, result_filename, {"x_scale": 0.1, "y_scale": 0.1})

        col += 1
        result_filename = dir_name+"results/corr/corr_fem_"+str(num)+"_exact_bc.png"
        worksheet.write(row, col, "", format)
        if os.path.isfile(result_filename):
            worksheet.insert_image(row, col, result_filename, {"x_scale": 0.05, "y_scale": 0.05})

        col += 1
        result_filename = dir_name+"results/corr/corr_fem_"+str(num)+".png"
        worksheet.write(row, col, "", format)
        if os.path.isfile(result_filename):
            worksheet.insert_image(row, col, result_filename, {"x_scale": 0.05, "y_scale": 0.05})

        col += 1
        result_filename = dir_name+"results/corr/corr_phifem_"+str(num)+"_exact_bc.png"
        worksheet.write(row, col, "", format)
        if os.path.isfile(result_filename):
            worksheet.insert_image(row, col, result_filename, {"x_scale": 0.05, "y_scale": 0.05})

        col += 1
        result_filename = dir_name+"results/corr/corr_phifem_"+str(num)+".png"
        worksheet.write(row, col, "", format)
        if os.path.isfile(result_filename):
            worksheet.insert_image(row, col, result_filename, {"x_scale": 0.05, "y_scale": 0.05})


        num+=1
        config_filename = dir_name+"configs/config_"+str(num)+".json"
        row+=1


        previous_dict = dict.copy()

    worksheet.autofit()

    workbook.close()
