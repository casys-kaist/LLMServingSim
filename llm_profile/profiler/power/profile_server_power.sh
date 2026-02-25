#!/bin/bash

while true; do
    echo -n "$(date +"%F %T")," >> server_power_log.csv
    ipmitool dcmi power reading \
      | grep "Instantaneous power reading" \
      | awk '{print $(NF-1)}' >> server_power_log.csv
    sleep 1
done