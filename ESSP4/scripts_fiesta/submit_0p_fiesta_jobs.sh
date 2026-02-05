#!/bin/bash

echo "Submitting 0p FIESTA jobs (0 planets with GP)..."
echo "================================================"

job_count=0
for script in run_*_0p_2modes.sh run_*_0p_3modes.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 0p FIESTA jobs"
