#!/bin/bash
echo "Canceling all UCLES-only HD102365 jobs..."
job_ids=$(bjobs | grep "HD102365_UCLES_only_" | awk '{print $1}')
if [ -z "$job_ids" ]; then
  echo "No UCLES-only HD102365 jobs to cancel."
  exit 0
fi
echo "Jobs to cancel:"
echo "$job_ids"
for id in $job_ids; do
  echo "Canceling job $id"
  bkill "$id"
done
echo "Done."
