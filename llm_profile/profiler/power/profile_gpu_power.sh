#!/bin/bash

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw --format=csv,noheader,nounits -lms 1000 -i 0 > gpu_power_log.txt