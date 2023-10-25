
# import xlsxwriter module
import xlsxwriter
import os
import json

def read_config(filename):
    with open(filename) as f:
        raw_config = f.read()
        dict = json.loads(raw_config)
    return  dict

def create_xlsx_file(problem_considered, pde_considered):
    excel_filename = "networks/"+problem_considered.__name__+"/models_config_"+pde_considered.__name__
    if os.path.isfile(excel_filename+".xlsx"):
        os.remove(excel_filename+".xlsx")

    workbook = xlsxwriter.Workbook(excel_filename+".xlsx")
    worksheet = workbook.add_worksheet()

    # define title lines
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

    worksheet.merge_range(row, col, row+1, col, "Configuration", title_format)

    col+=1
    worksheet.merge_range(row, col, row, col+1, "Model parameters", title_format)
    worksheet.write(row+1, col, 'Layers',title_format)
    col+=1
    worksheet.write(row+1, col, 'Activation Function',title_format)

    col+=1
    worksheet.merge_range(row, col, row, col+4, "Trainer parameters", title_format)
    worksheet.write(row+1, col, 'Learning rate',title_format)
    col+=1
    worksheet.write(row+1, col, 'Decay',title_format)
    col+=1
    worksheet.write(row+1, col, 'w_data',title_format)
    col+=1
    worksheet.write(row+1, col, 'w_res',title_format)
    col+=1
    worksheet.write(row+1, col, 'w_bc',title_format)

    col+=1
    worksheet.merge_range(row, col, row, col+3, "Training parameters", title_format)
    worksheet.write(row+1, col, 'n_epochs',title_format)
    col+=1
    worksheet.write(row+1, col, 'n_collocation',title_format)
    col+=1
    worksheet.write(row+1, col, 'n_bc_collocation',title_format)
    col+=1
    worksheet.write(row+1, col, 'n_data',title_format)

    # insert images

    col+=1
    worksheet.merge_range(row, col, row+1, col, "Exact BC.", image_format)
    worksheet.set_column(col, col, 20)
    col+=1
    worksheet.merge_range(row, col, row+1, col, "No exact BC.", image_format)
    worksheet.set_column(col, col, 20)
    

    # Start config
    format = workbook.add_format(
        {
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        }
    )

    row+=2

    dir_name = "networks/"+problem_considered.__name__+"/"+pde_considered.__name__+"/"

    num = 0
    config_filename = dir_name+"configs/config_"+str(num)+".json"
    while os.path.isfile(config_filename):
        dict = read_config(config_filename)

        worksheet.set_row(row, 60)

        col = 0
        worksheet.write(row, col, str(num),format)
        for key,value in dict.items():
            if key=="layers":
                col+=1
                worksheet.write(row, col, str(str(value)), format)
            elif key!="pb":
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

        num+=1
        config_filename = dir_name+"configs/config_"+str(num)+".json"
        row+=1

    worksheet.autofit()

    workbook.close()
