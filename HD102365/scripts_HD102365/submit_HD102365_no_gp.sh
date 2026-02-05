#!/bin/bash

echo "Submitting HD102365 no-GP jobs..."
echo "================================="

job_count=0
for script in run_HD102365_*_no_gp_*_*.sh; do
  if [ -f "$script" ]; then
    echo "Submitting: $script"
    bsub < "$script"
    ((job_count++))
    sleep 0.5
  fi
done

echo "Submitted $job_count no-GP jobs."
