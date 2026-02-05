#!/bin/bash

echo "Submitting 3p FIESTA jobs (3 planets + GP)..."
echo "============================================="

job_count=0
for script in run_*_3p_2modes.sh run_*_3p_3modes.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 3p FIESTA jobs"
