#!/bin/bash
echo "Submitting UCLES-only HD102365 GP jobs..."
job_count=0
for script in run_HD102365_UCLES_only_gp_*.sh; do
  [ -f "$script" ] || continue
  echo "Submitting: $script"
  bsub < "$script"
  ((job_count++))
  sleep 0.5
done
echo "Submitted $job_count GP jobs."
