#!/bin/bash
# task 1 subtask 6
# Filip Kotlas 07.04.2024

# I was not able to find the file, so I tested it on one of my own.
g++ Skript/Beispiele/Ex433.cpp &> out_first.txt
g++ Ex433.cpp 2>&1 | tee out.txt
