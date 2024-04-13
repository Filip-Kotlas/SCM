#!/bin/bash
# task 3 subtask 2
# Filip Kotlas 08/04/2024

file=output_file.txt
ls -R > $file

echo 'Files with extension .cpp and .h are: '
grep -E '.*[.]h$|.*[.]cpp$' $file


