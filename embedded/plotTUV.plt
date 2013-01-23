#!/bin/zsh
gnuplot -persist <<EOF 

set term wxt
set xlabel "alpha1"
set ylabel "alpha2"
set zlabel "R(E)"
set pm3d; set palette
splot 'plotTUV_Uncertainty.txt'
EOF

gnuplot -persist <<EOF 
set term wxt
set xlabel "alpha1"
set ylabel "alpha2"
set zlabel "TUV"
splot 'plotTUV.txt'
EOF

gnuplot -persist <<EOF 
set term wxt
set xlabel "alpha1"
set ylabel "alpha2"
set zlabel "-E(R)"
splot 'plotTUV_Exploration.txt'
EOF
