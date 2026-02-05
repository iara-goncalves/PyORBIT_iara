#!/bin/bash

echo "Submitting ccfs jobs (single files, FWHM, Contrast)..."
echo "====================================================="

job_count=0
for script in run_*_ccfs_single.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count ccfs single file jobs"
