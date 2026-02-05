#!/bin/bash

echo "Submitting 4_activity_indi jobs (single files, BIS, FWHM, Contrast, Halpha)..."
echo "=============================================================================="

job_count=0
for script in run_*_4_activity_indi_single.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 4_activity_indi single file jobs"
