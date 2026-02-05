#!/bin/bash

echo "Submitting 1p FIESTA jobs (1 planet + GP)..."
echo "============================================"

job_count=0
for script in run_*_1p_2modes.sh run_*_1p_3modes.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 1p FIESTA jobs"
