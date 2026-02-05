#!/bin/bash

echo "Submitting HD102365 0p UltraNest jobs..."
echo "======================================================"

job_count=0
for script in run_HD102365_*_0p_ultranest.sh; do
  if [ -f "$script" ]; then
    echo "Submitting: $script"
    bsub < "$script"
    ((job_count++))
    sleep 0.5
  fi
done

echo "Submitted $job_count 0p jobs."
