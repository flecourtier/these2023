input_file = "../explications.tex"
output_file = "explications.tex"

with open(input_file, 'r') as file_in, open(output_file, 'w') as file_out:
    ignore_section = False
    for line in file_in:
        # si line contient a
        if "images/" in line:
            line = line.replace("images/", "../images/")
        if line.strip() == "\\begin{preuve}":
            file_out.write(line)
            file_out.write("...\n")
            ignore_section = True
        elif line.strip() == "\\end{preuve}":
            file_out.write(line)
            ignore_section = False
        elif not ignore_section:
            file_out.write(line)
