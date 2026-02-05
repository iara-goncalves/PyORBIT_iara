#!/bin/bash

echo "Submitting 5_activity_indi jobs (BIS, FWHM, Contrast, Halpha, CaII)..."
echo "======================================================================="

job_count=0
for script in run_*_5_activity_indi.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 5_activity_indi jobs"
