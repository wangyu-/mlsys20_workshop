

file=$1
for i in non_inc opt_inc opt_time_inc opt_energy_inc
	 do
res=`cat $file |grep $i|awk 'BEGIN{sum1=0;sum2=0;sum3=0;sum4=0;cnt=0;}{sum1+=$5;sum2+=$7;sum3+=$9;sum4+=$11;cnt++} END{print "time=",sum1/cnt,"power=",sum2/cnt,"energy="sum3/cnt,"energy2="sum4/cnt}'`
echo $res
done
