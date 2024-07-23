#!/bin/bash

# Start time of the script
echo "Start Evaluation"
# Function to format time in hours, minutes, and seconds
format_time() {
  local total_seconds=$1
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  echo "${hours}h ${minutes}m ${seconds}s"
}

# Start time of the script
start_script=$SECONDS

# evaluation 1
start=$SECONDS
./evaluation1.sh
end=$SECONDS
elapsed=$((end - start))
echo "Evaluation 1 took $(format_time $elapsed)."

# evaluation 2
start=$SECONDS
./evaluation2.sh
end=$SECONDS
elapsed=$((end - start))
echo "Evaluation 2 took $(format_time $elapsed)."

# evaluation 3
start=$SECONDS
./evaluation3.sh
end=$SECONDS
elapsed=$((end - start))
echo "Evaluation 3 took $(format_time $elapsed)."

# evaluation 4
start=$SECONDS
./evaluation4.sh
end=$SECONDS
elapsed=$((end - start))
echo "Evaluation 4 took $(format_time $elapsed)."

# evaluation 5
start=$SECONDS
./evaluation5.sh
end=$SECONDS
elapsed=$((end - start))
echo "Evaluation 5 took $(format_time $elapsed)."

# End time of the script
end_script=$SECONDS
total_elapsed=$((end_script - start_script))
echo "Total time taken: $(format_time $total_elapsed)."