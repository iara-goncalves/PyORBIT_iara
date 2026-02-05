#!/bin/bash

echo "Submitting 1mode FIESTA jobs (mode1)..."
echo "========================================"

job_count=0
for script in run_*_1mode.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 1mode FIESTA jobs"
