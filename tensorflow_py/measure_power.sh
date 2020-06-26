#! /bin/sh
nvidia-smi | grep -o  '[0-9]*[0-9]W' |head -1|grep -o '[0-9]*'
