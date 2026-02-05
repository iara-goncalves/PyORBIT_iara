#!/bin/bash

echo "Submitting 0p jobs (single files, 0 planets with GP)..."
echo "======================================================"

job_count=0
for script in run_*_0p_*_single.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 0p single file jobs"
