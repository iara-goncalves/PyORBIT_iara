#!/bin/bash

echo "Submitting 3MODES jobs (3 modes only)..."
echo "========================================="

job_count=0
for script in run_*_3modes.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 3MODES jobs (3 modes)"
