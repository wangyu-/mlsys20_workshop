#! /bin/bash


list=(../../metaflow_sysml19/non_sq16  ../../metaflow_sysml19/opt_sq16 ../code/opt_time_sq16 ../code/opt_energy_sq16)

for i in $(seq 1 99999)
do
    #idx=$((RANDOM % 4))
    num=`od -A n -t d -N 1 /dev/urandom |tr -d ' '`
    idx=$((num % 4))
    #ls -l ${list[idx]}
    #echo ${list[idx]}
     ./run.sh ${list[idx]} 2>/dev/null
done
