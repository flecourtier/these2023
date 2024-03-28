#!/bin/bash

# run one pdflatex command
function run_pdflatex {
    if pdflatex -synctex=1 -interaction=nonstopmode -output-directory=$1 $2 >/dev/null 2>&1; then
        echo "pdflatex a été exécuté avec succès"
        # exit 0
    else
        echo "Erreur lors de l'exécution de pdflatex"        
        pdflatex -synctex=1 -interaction=nonstopmode -output-directory=$1 $2 #>/dev/null
        pwd_dir=$(pwd)
        filename="${2%.*}"
        echo "Voir $pwd_dir/$filename.log pour plus d'informations"
        exit 1
    fi
}

function run_abstracts {
    echo "### Abstracts"

    # run pdflatex for the main file which contain all the meetings
    dir="abstracts/"
    latexfilename="abstracts.tex"
    cd $dir
    echo $dir$latexfilename
    run_pdflatex "./" $latexfilename
    run_pdflatex "./" $latexfilename
    cd "../"
}

function run_report {
    echo "### Report"

    # run pdflatex for each chapter of the report
    latexfilename="report.tex"
    dir="report/"
    cd $dir
    for subdir in */
    do
        ## passer si le répertoire finit par "_" (dossier temporaire)
        if [[ $subdir == *"_/" ]]; then
            continue
        fi
        cd $subdir
        echo $subdir$latexfilename
        run_pdflatex "./" $latexfilename
        run_pdflatex "./" $latexfilename
        cd "../"
    done
    cd "../"

    # run pdflatex for the main file which contain all the chapter
    dir="report/"
    latexfilename="report.tex"
    cd $dir
    pwd
    echo $dir$latexfilename
    # compile twice for table of contents
    run_pdflatex "./" $latexfilename
    # https://stackoverflow.com/questions/64809221/why-does-one-need-to-compile-two-times-to-have-a-table-of-contents-in-the-pdf
    run_pdflatex "./" $latexfilename
    cd "../"
}

function run_all {
    echo "###### ALL"

    run_abstracts
    run_report
}

function show_help {
    echo "Usage: pdflatex.sh [OPTION]"
    echo "Options:"
    echo "  ALL     Compile all LaTeX files."
    echo "  a       Compile the abstracts."
    echo "  r       Compile the report."
}

# -z vérifie si la chaine passé en argument est vide
if [ ! -z "$1" ]; then
    echo "L'argument passé est : $1"
    case $1 in
        "ALL")
            run_all
            ;;
        "a")
            run_abstracts
            ;;
        "r")
            run_report
            ;;
        *)
            echo "Argument invalide"
            ;;
    esac
else
    show_help
fi
