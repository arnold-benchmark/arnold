#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "No checkpoint file provided. Usage: $0 <checkpoint_file>"
    exit 1
fi

checkpoint_file=$1

# Define an array of tasks
tasks=('pickup_object' 'reorient_object' 'open_drawer' 'close_drawer' 'open_cabinet' 'close_cabinet' 'pour_water' 'transfer_water') 

# Iterate three times for each set of tasks
for i in {1..3}; do
    output_root="/root/arnold/output/output_iteration_${i}"
    echo "Iteration: $i, Output root: $output_root"

    # Iterate over the tasks
    for task in "${tasks[@]}"; do
        echo "Running task: $task with checkpoint_file: $checkpoint_file and output_root: $output_root"
        /isaac-sim/python.sh eval.py task="$task" model=peract lang_encoder=clip mode=eval use_gt=[0,0] visualize=0 record=True checkpoint_file="$checkpoint_file" output_root="$output_root"
    done
done