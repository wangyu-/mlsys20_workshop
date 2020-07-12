#! /bin/bash

killall sh

#echo $1 $2
name=$(basename $1)

sh manual_measure.sh >$name.power &
pid=$!

time0=`python tf_executor.py --graph_file $1 2>/dev/null |grep -o "0\.[0-9]*"`
time=`echo $time0|awk '{ printf "%0.5f\n" ,$1*1000}'`
kill $pid 
power=`python avg_power.py $name.power|grep -o "[0-9\.]*"|awk '{ printf "%0.5f\n" ,$1}'`
energy=`echo $time $power|awk '{ printf "%0.5f\n" ,$1*$2}'`
energy2=`echo $time $power|awk '{ printf "%0.5f\n" ,$1*($2-23)}'`
echo "RESULT:  name= "$name"	time= "$time"	power= "$power"	energy= "$energy"	energy2= "$energy2





#echo $name

