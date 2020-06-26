#! /bin/bash

for i in $(seq 1 999)
do
    echo ""
    echo ======round$i======
for file in ../../metaflow_sysml19/non_sq  ../../metaflow_sysml19/opt_sq ../code/opt_time_sq ../code/opt_energy_sq
	    do
	    #echo $file
	    ./run.sh $file 2>/dev/null
	    done
done
