#!/bin/bash
nvidia-smi --query-gpu=name,power.limit,driver_version --format=csv,noheader > host_metadata.txt
