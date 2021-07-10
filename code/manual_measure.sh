#! /bin/sh

while [ 1 ]
do
sleep 0.2
./measure_power.sh |tee power_result.tmp
mv power_result.tmp power_result
done
