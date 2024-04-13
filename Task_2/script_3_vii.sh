#!/bin/bash
# task 3 subtask 7
# Filip Kotlas 08/04/2024

if [ ! -e Goldbach.out ]; then
   echo "Program Goldbach.out does not exist. Attempting to compile."
   if [ ! -e Project_1/main.cpp ] || [ ! -e Project_1/mylib.cpp ]; then
	echo "Cannot find the source files. Exiting script."
	exit
   else
	g++ Project_1/main.cpp Project_1/mylib.cpp -o Goldbach.out
   fi
fi


iterations=( "1000" "5000" "10000" "50000" "100000" "200000" )
times=( "0" "0" "0" "0" "0" "0" )

echo > results.txt
for i in ${!iterations[@]}; do
    start=$(date +%s%N)
    ./Goldbach.out ${iterations[$i]}
    end=$(date +%s%N) 
    times[$i]=$(($(($end-$start))/1000000))
done

for i in ${!iterations[@]}; do
    echo ${iterations[$i]} ${times[$i]} >> results.txt
done

gnuplot -persist <<-EOFMarker
    set title "Goldbach" font ",14" textcolor rgbcolor "royalblue"
    set timefmt "%y/%m/%d"
    set xlabel "Iterations"
    set ylabel "Time"
    set pointsize 1
    plot "results.txt" using 1:2 with linespoints
EOFMarker