#!/bin/bash

echo "Submitting FIESTA jobs (2 modes only)..."
echo "========================================="

job_count=0
for script in run_*_fiesta.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count FIESTA jobs (2 modes)"
