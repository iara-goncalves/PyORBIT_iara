#!/bin/bash

echo "Submitting white_noise jobs (RV only)..."
echo "========================================"

job_count=0
for script in run_*_white_noise.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count white_noise jobs"
