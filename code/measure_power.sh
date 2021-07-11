#! /bin/sh
nvidia-smi -q -d POWER -i 0|grep "Power Draw"|grep -o "[0-9]\+\.[0-9]\+"
#nvidia-smi | grep -o  '[0-9]*[0-9]W' |head -1|grep -o '[0-9]*'
