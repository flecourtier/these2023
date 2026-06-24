import csv
precision = 54

filename = f"output{precision}.csv"

tPINN = 707.84
#1e-3
if precision == 13:
    tfem = 76.51430344195187
    tfem_mesh = 40.18

    tadd = 0.05198588045
    tadd_mesh = 1.14e-2
#5e-4
if precision == 54:
    tfem = 226.1171859646457
    tfem_mesh = 98.26

    tadd = 0.08519871752982358
    tadd_mesh = 3.22e-2
#2e-4
if precision == 24:
    tfem = 947.1547919826207
    tfem_mesh = 390.02

    tadd = 0.3365887224544619
    tadd_mesh = 1.36e-1
#1e-4
elif precision == 14:
    tfem = 2799.058040154791
    tfem_mesh = 1105.45

    tadd = 0.9255318879
    tadd_mesh = 0.463

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["np", "tfem", "tadd"])
    for np in range(51):
        tfem_p = tfem_mesh + np * (tfem - tfem_mesh)
        tadd_p = tPINN + tadd_mesh + np * (tadd - tadd_mesh)
        writer.writerow([np, tfem_p, tadd_p])

print(f"CSV file '{filename}' created successfully.")