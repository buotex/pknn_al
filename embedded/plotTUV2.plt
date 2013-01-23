#!/bin/zsh
gnuplot -persist <<EOF 
set term epslatex color
set output 'plotTUV.tex'
#set title "$ TUV(x|\\\alpha)$ with $\\\alpha_3 = \\\alpha_4 = \\\alpha_5 = 1/5$"
set size square
set xlabel "$\\\alpha_1$"
set ylabel "$\\\alpha_2$"
#set xtics 5
#set ytics 5
#set zlabel "TUV"
set cblabel "TUV"
set noztics
set pm3d map
set zrange [0:]
splot 'plotTUV.txt' notitle
EOF
