#!/bin/bash

echo "HD102365 UltraNest Job Monitor"
echo "==============================="
bjobs | grep "HD102365_.*_ultranest" || echo "No UltraNest jobs found."
echo ""
echo "Job counts by configuration:"
for config in all_instr all_instr_espresso_gp espresso_only no_espresso ucles_only; do
  count=$(bjobs | grep "HD102365_${config}_.*_ultranest" | wc -l)
  echo "  ${config}: ${count}"
done
echo ""
echo "Job counts by planet configuration:"
for planets in 0p 1p 2p 3p; do
  count=$(bjobs | grep "HD102365_.*_${planets}_ultranest" | wc -l)
  echo "  ${planets}: ${count}"
done
