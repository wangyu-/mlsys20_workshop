#! /bin/bash


list=( "$@" )
len=${#list[@]}

#echo $list, $len

#list=(../../metaflow_sysml19/non_sq  ../../metaflow_sysml19/opt_sq ../code/opt_time_sq ../code/opt_energy_sq)

for i in $(seq 1 99999)
do
    #idx=$((RANDOM % 4))
    num=`od -A n -t d -N 1 /dev/urandom |tr -d ' '`
    idx=$((num % $len))
     ./run.sh ${list[idx]} 2>/dev/null
done
