#!/bin/bash

echo "Canceling all HD102365 jobs..."
echo "=============================="

job_ids=$(bjobs | grep "HD102365_" | awk '{print $1}')
if [ -z "$job_ids" ]; then
  echo "No HD102365 jobs to cancel."
  exit 0
fi

echo "Jobs to cancel:"
echo "$job_ids"
echo ""
read -p "Proceed? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
  for id in $job_ids; do
    echo "Canceling job $id"
    bkill "$id"
  done
  echo "Done."
else
  echo "Aborted."
fi
