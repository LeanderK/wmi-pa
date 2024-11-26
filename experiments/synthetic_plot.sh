#!/bin/bash

SYN_DIR=synthetic_exp

mkdir -p $SYN_DIR/plots

python3 plot.py $SYN_DIR/results/r3_b3_d[2-6]* -o $SYN_DIR/plots/ -f r3_b3_d2-6 --timeout 3600 --legend-pos 4
python3 plot.py $SYN_DIR/results/r3_b3_d[2-6]* -o $SYN_DIR/plots/ -f r3_b3_d2-6_cactus --cactus --timeout 3600 --legend-pos 2
