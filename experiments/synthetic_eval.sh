#!/bin/bash

SYN_DIR=synthetic_exp
DATA_DIR=$SYN_DIR/data

for dir in $(ls -d $DATA_DIR/*)
do
	res_dir=$(sed "s+data+results+g" <<< $dir)
	mkdir -p $res_dir
	echo Evaluating $dir

	for mode in "SAE4WMI latte"
	do
		echo Mode $mode
		python3 evaluateModels.py $dir -o $res_dir -m $mode 
	done

	for mode in "SAE4WMI torch"
	do
		echo Mode $mode
		python3 evaluateModels.py $dir -o $res_dir -m $mode --monomials_use_float64 --sum_seperately --with_sorting
	done

  error=0.1
	for N in 100 1000 10000
	do
    echo "Mode SAE4WMI volesti, N $N, 5 seeds"
    python3 evaluateModels.py "$dir" -o "$res_dir" -m SAE4WMI volesti -e $error -N $N --seed 666 --n-seeds 5
  done
done
