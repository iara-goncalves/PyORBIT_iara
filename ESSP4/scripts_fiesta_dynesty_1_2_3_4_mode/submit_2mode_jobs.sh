#!/bin/bash

echo "Submitting 2mode FIESTA jobs (mode2)..."
echo "========================================"

job_count=0
for script in run_*_2mode.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 2mode FIESTA jobs"
