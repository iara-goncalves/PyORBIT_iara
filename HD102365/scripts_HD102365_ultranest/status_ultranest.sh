#!/bin/bash

echo "HD102365 UltraNest Job Status Summary"
echo "======================================"
echo ""

results_dir="/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365_ultranest"

for config in all_instr all_instr_espresso_gp espresso_only no_espresso ucles_only; do
  echo "Configuration: ${config}"
  for planets in 0p 1p 2p 3p; do
    job_dir="${results_dir}/${config}/${planets}/HD102365_${config}_${planets}_ultranest"
    if [ -d "${job_dir}" ]; then
      if [ -f "${job_dir}/ultranest_results.pkl" ]; then
        echo "  ${planets}: COMPLETED"
      elif [ -f "${job_dir}/configuration_file_ultranest_run_HD102365_${config}_${planets}_ultranest.log" ]; then
        echo "  ${planets}: RUNNING"
      else
        echo "  ${planets}: NOT STARTED"
      fi
    else
      echo "  ${planets}: NOT CREATED"
    fi
  done
  echo ""
done
