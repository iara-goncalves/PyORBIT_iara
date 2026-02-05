#!/bin/bash

echo "Submitting ALL FIESTA jobs (2modes and 3modes)..."
echo "=================================================="

job_count=0
for script in run_*_2modes.sh run_*_3modes.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo ""
echo "Submitted $job_count FIESTA jobs total"