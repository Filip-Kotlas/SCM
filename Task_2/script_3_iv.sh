#!/bin/bash
# task 3 subtask 4
# Filip Kotlas 13/04/2024

if [ ! -e "Codes.zip" ]; then
    wget http://imsc.uni-graz.at/haasegu/Lectures/SciComp/SS20/Codes.zip
    unzip Codes.zip
fi

cd Codes
IFS=$'\n' read -d '' -r -a DIRS <<< "$(ls -d */)"
cd ${DIRS[0]}
IFS=$'\n' read -d '' -r -a FILES <<< "$(ls | grep -E '\.cpp$|\.h$')"
cd ..

for ((i = 0; i < ${#DIRS[@]} -1; i++))
do
    for FILE in ${FILES[@]}
    do
        MESSAGE=`diff  ${DIRS[i]}/$FILE ${DIRS[i+1]}/$FILE`
        if [ -n "$MESSAGE" ]; then
            echo "Difference between " ${DIRS[i]}$FILE " and " ${DIRS[i+1]}$FILE is:
            echo $MESSAGE
        else
            echo "There is no difference between " ${DIRS[i]}$FILE " and " ${DIRS[i+1]}$FILE
        fi
        echo
    done
done
echo

for ((i = 0; i < ${#DIRS[@]}; i++))
do
    count=0
    for FILE in ${FILES[@]}
    do
        add=`grep -e "goto" -e "continue" -e "break" -c ${DIRS[i]}$FILE`
        (( count += add ))
    done
    echo "In directory " ${DIRS[i]} " there is " $count " cases of forbidden key words."
    if [ $count -eq 0 ]; then
        echo "Code passed the test."
    else
        echo "Code did not passed the test."
    fi
done

cd ..
tar -u -v -v Codes.tar Codes
gzip Codes.tar
