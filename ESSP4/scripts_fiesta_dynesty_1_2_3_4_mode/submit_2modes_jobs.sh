#!/bin/bash

echo "Submitting 2modes FIESTA jobs (mode1, mode2)..."
echo "==============================================="

job_count=0
for script in run_*_2modes.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 2modes FIESTA jobs"
