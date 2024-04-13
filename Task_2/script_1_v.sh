#!/bin/bash
# task 1 subtask 5
# Filip Kotlas 07/04/2024

http://imsc.uni-graz.at/haasegu/Lectures/SciComp/SS20/kurs.tar.gz
if [ ! -e "kurs.tar.gz" ]; then
    wget http://imsc.uni-graz.at/haasegu/Lectures/SciComp/SS20/kurs.tar.gz
fi

ARCHIVE=kurs.tar.gz

tar -t -f $ARCHIVE > list_of_content.txt
LAYER_1_FOLDERS=`tar -t -f $ARCHIVE | grep -E '^[^/]*/$' | tr '\n' ' '`
tar -x -z -f $ARCHIVE

echo 'Size of the new directory(ies):'
du -sh $LAYER_1_FOLDERS
echo

echo 'Size of the directories and subdirectories:'
du -h $LAYER_1_FOLDERS

files_to_delete=`find . -name "*.log" -o -name "*.o" -o -name "main.GCC_" -o -type d -name "*html*" | tr '\n' ' '`
rm -r $files_to_delete