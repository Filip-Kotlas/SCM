#!/bin/bash
# task 2 subtask 2
# Filip Kotlas 08/04/2024

sed -i 's/Funktion[ ]\{1\}/function /g' ./Script/latex/p_7.tex

sed '/^%.*$/d' ./Script/latex/p_7.tex > t7.tex

echo 'Number of characters in the files is:'
wc -c  ./Script/latex/p_7.tex t7.tex
echo

echo 'Number of lines in the files is: '
wc -l ./Script/latex/p_7.tex t7.tex
echo

echo 'Difference between file ./Script/latex/p_7.tex and t7.tex: '
diff ./Script/latex/p_7.tex t7.tex
echo
