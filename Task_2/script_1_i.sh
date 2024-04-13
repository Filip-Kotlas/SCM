#!/bin/bash
# task 1 subtask 1
# Filip Kotlas 07/04/2024

path_to_working_directory=`pwd`
main_file=Task_1_directory

if [ -e $main_file ]
then
    echo 'There already is a folder with the name' $main_file
    echo 'Exiting script'
    exit
fi

mkdir $main_file
cd $main_file
mkdir Project_1
mkdir Project_2
cd Project_1

#Just example files
touch main.cpp
touch mylib.cpp
touch mylib.h

cd ../Project_2
touch main.cpp
touch factorial.cpp
touch factorial.h
cd ..

echo 'Content of folder Project_2 is:'
ls Project_2/*.cpp
cd $path_to_working_directory
chmod go-r -R $main_file

