#! /bin/bash

file=$1
list0=`cat $1 |awk '{print $3}'|sort -u|xargs`
list=( "$list0" )
#echo $list
for i in $list
	 do
echo -ne $i"	"
res=`cat $file |grep $i|awk 'BEGIN{sum1=0;sum2=0;sum3=0;sum4=0;cnt=0;}{sum1+=$5;sum2+=$7;sum3+=$9;sum4+=$11;cnt++} END{print "time=",sum1/cnt,"power=",sum2/cnt,"energy="sum3/cnt,"energy2="sum4/cnt}'`
echo $res
done
