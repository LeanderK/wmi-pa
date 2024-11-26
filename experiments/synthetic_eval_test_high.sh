#!/bin/bash

SYN_DIR=synthetic_exp
DATA_DIR=$SYN_DIR/data

for dir in $(ls -d $DATA_DIR/pa_r4_b0_d2_m10_e6*)
do
	res_dir=$(sed "s+data+results+g" <<< $dir)
	mkdir -p $res_dir
	echo Evaluating $dir

	for mode in "SAE4WMI torch" "SAPA torch"
	do
		echo Mode $mode
		python3 evaluateModels.py $dir -o $res_dir -m $mode --monomials_use_float64 --sum_seperately --with_sorting
	done
done
