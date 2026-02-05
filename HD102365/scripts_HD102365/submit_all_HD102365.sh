#!/bin/bash

echo "Submitting ALL HD102365 PyORBIT jobs..."
echo "======================================="

job_count=0
for script in run_HD102365_*.sh; do
  if [ -f "$script" ]; then
    echo "Submitting: $script"
    bsub < "$script"
    ((job_count++))
    sleep 0.5
  fi
done

echo "Submitted $job_count jobs."
