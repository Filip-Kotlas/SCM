#!/bin/bash
# task 2 subtask 1
# Filip Kotlas 08.04.2024

files=`find . -name "*.tex"`

echo "Files containing word 'Alternative':"
fgrep -l "Alternative" $files
echo

echo "Files containing words 'lauf' or 'Lauf':"
grep -E -n "[Ll]auf" $files

