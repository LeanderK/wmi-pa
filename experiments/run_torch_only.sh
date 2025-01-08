#!/bin/bash

SYN_DIR=synthetic_exp
DATA_DIR=$SYN_DIR/data

for dir in $(ls -d $DATA_DIR/*)
do
	res_dir=$(sed "s+data+results+g" <<< $dir)
	mkdir -p $res_dir
	echo Evaluating $dir

	echo Mode SAE4WMI torch
	python3 evaluateModels.py $dir -o $res_dir -m SAE4WMI torch --monomials_use_float64 --sum_seperately --with_sorting

done
