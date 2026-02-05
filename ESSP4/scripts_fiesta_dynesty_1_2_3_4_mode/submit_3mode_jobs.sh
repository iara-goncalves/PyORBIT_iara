#!/bin/bash

echo "Submitting 3mode FIESTA jobs (mode3)..."
echo "========================================"

job_count=0
for script in run_*_3mode.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 3mode FIESTA jobs"
