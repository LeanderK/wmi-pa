#!/bin/bash

MLC_DIR=mlc

for dir in $(ls -d mlc/data/uci-det-m:100-M:200-N:5-Q:0.0-S:666/small/*)
do
	res_dir=$(sed "s+data+results+g" <<< $dir)
  mkdir -p $res_dir
	echo Evaluating $dir
	# for mode in XSDD XADD FXSDD "PA latte" "SAPA latte" "SAE4WMI latte"
        for mode in "SAE4WMI latte" "SAE4WMI torch"
        do
                echo Mode $mode
                python3 evaluateModels.py $dir -o $res_dir --timeout 1200 -m $mode
        done
done
