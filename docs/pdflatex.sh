#!/bin/bash

# run one pdflatex command
function run_pdflatex {
    if pdflatex -synctex=1 -interaction=nonstopmode -output-directory=$1 $2 >/dev/null 2>&1; then
        echo "pdflatex a été exécuté avec succès"
    else
        echo "Erreur lors de l'exécution de pdflatex"        
        pdflatex -synctex=1 -interaction=nonstopmode -output-directory=$1 $2 #>/dev/null
        pwd_dir=$(pwd)
        filename="${2%.*}"
        echo "Voir $pwd_dir/$filename.log pour plus d'informations"
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

function run_meetings {
    echo "### Meetings"

    # run pdflatex for each meeting
    latexfilename="meeting.tex"
    dir="meetings/days/"
    cd $dir
    for subdir in */
    do
        cd $subdir
        echo $subdir$latexfilename
        run_pdflatex "./" $latexfilename
        run_pdflatex "./" $latexfilename
        cd "../"
    done
    cd "../../"

    # run pdflatex for the main file which contain all the meetings
    dir="meetings/"
    latexfilename="meetings.tex"
    cd $dir
    echo $dir$latexfilename
    # compile twice for table of contents
    run_pdflatex "./" $latexfilename
    # https://stackoverflow.com/questions/64809221/why-does-one-need-to-compile-two-times-to-have-a-table-of-contents-in-the-pdf
    run_pdflatex "./" $latexfilename
    cd "../"
}

function run_results {
    echo "### Results"

    # run pdflatex for the main file which contain all the meetings
    dir="results/"
    latexfilename="results.tex"
    cd $dir
    echo $dir$latexfilename
    run_pdflatex "./" $latexfilename
    run_pdflatex "./" $latexfilename
    cd "../"
}

function run_to_do_list {
    echo "### ToDo List"

    # run pdflatex for the main file which contain all the meetings
    dir="to_do_list/"
    cd $dir
    latexfilename="to_do_list.tex"
    echo $dir$latexfilename
    run_pdflatex "./" $latexfilename
    run_pdflatex "./" $latexfilename
    cd "../"
}

function run_all {
    echo "###### ALL"

    run_abstracts
    run_meetings
    run_results
    run_to_do_list
}

function show_help {
    echo "Usage: pdflatex.sh [OPTION]"
    echo "Options:"
    echo "  ALL     Compile all LaTeX files."
    echo "  a       Compile the abstracts."
    echo "  m       Compile the meetings."
    echo "  r       Compile the results."
    echo "  t       Compile the to-do list."
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
        "m")
            run_meetings
            ;;
        "r")
            run_results
            ;;
        "t")
            run_to_do_list
            ;;
        *)
            echo "Argument invalide"
            ;;
    esac
else
    show_help
fi
