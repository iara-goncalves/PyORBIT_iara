#!/bin/bash

echo "Submitting 4mode FIESTA jobs (mode4)..."
echo "========================================"

job_count=0
for script in run_*_4mode.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 4mode FIESTA jobs"
