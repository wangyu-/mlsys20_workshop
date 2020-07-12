#! /bin/bash


list=(../code/test_inc0 ../code/test_inc1 ../code/test_inc2)

for i in $(seq 1 99999)
do
    #idx=$((RANDOM % 4))
    num=`od -A n -t d -N 1 /dev/urandom |tr -d ' '`
    idx=$((num % 3))
     ./run.sh ${list[idx]} 2>/dev/null
done
