from pathlib import Path

def get_dir():
    current_dir = Path(__file__).parent
    root_dir = str(current_dir.parent.parent) + "/"

    result_dir = root_dir + "antora/modules/ROOT/"
    page_dir = result_dir + "pages/"
    images_dir = result_dir + "assets/images/"
    attachments_dir = result_dir + "assets/attachments/"

    return root_dir, result_dir, page_dir, images_dir, attachments_dir

# return true if word is in line
def search_word_in_line(word, line):
    return word in line

# test if the title contains a latex formula
def test_latex_title(title):
    if search_word_in_line("$", title):
        title_split = title.split("$")
        title = ""
        start_stem = 0
        for i,part in enumerate(title_split):
            title += part
            if start_stem==0 and i!=len(title_split)-1:
                title += "stem:["
                start_stem = 1
            elif start_stem==1 and i!=len(title_split)-1:
                title += "]"
                start_stem = 0
    return title

# Test if there is a refernce to a figure in the line (many configurations possibles)
def test_fig(line):
    possible_ref = ["Figure \\ref","Figure~\\ref","Fig \\ref","Fig~\\ref","Fig.~\\ref","Fig.\\ref"]
    for ref in possible_ref:
        if search_word_in_line(ref,line):
            return ref,True
    return None,False

# Test if there is a refernce to a section in the line (many configurations possibles)
def test_section(line):
    possible_ref = ["Section \\ref","Section~\\ref","Sec \\ref","Sec~\\ref","Sec.~\\ref","Sec.\\ref"]
    for ref in possible_ref:
        if search_word_in_line(ref,line):
            return ref,True
    return None,False