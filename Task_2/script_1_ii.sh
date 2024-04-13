#!/bin/bash
# task 1 subtask 2
# Filip Kotlas 07/04/2024


find . > directory_tree.txt
echo Files with suffix .h
find . -name '*.h'
find . -name '*.cpp' -exec touch {} \;
