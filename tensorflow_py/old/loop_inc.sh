#! /bin/bash


list=(../../metaflow_sysml19/non_inc  ../../metaflow_sysml19/opt_inc ../code/opt_time_inc ../code/opt_energy_inc)

for i in $(seq 1 99999)
do
    #idx=$((RANDOM % 4))
    num=`od -A n -t d -N 1 /dev/urandom |tr -d ' '`
    idx=$((num % 4))
     ./run.sh ${list[idx]} 2>/dev/null
done
