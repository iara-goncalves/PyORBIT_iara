#!/bin/bash

echo "Submitting HD102365 all_instr_espresso_gp UltraNest jobs..."
echo "======================================================"

job_count=0
for script in run_HD102365_all_instr_espresso_gp_*_ultranest.sh; do
  if [ -f "$script" ]; then
    echo "Submitting: $script"
    bsub < "$script"
    ((job_count++))
    sleep 0.5
  fi
done

echo "Submitted $job_count all_instr_espresso_gp jobs."
