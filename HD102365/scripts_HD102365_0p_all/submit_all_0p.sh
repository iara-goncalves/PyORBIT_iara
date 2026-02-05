#!/bin/bash
job_count=0
for script in run_*.sh; do
  [ -f "$script" ] || continue
  echo "Submitting $script"
  bsub < "$script"
  ((job_count++))
  sleep 0.5
done
echo "Submitted $job_count jobs."
