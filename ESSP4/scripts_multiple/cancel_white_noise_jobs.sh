#!/bin/bash

echo "Canceling all jobs containing 'white_noise'..."
echo "=============================================="

job_ids=$(bjobs -w | grep white_noise | awk '{print $1}')

if [ -z "$job_ids" ]; then
    echo "No white_noise jobs found to cancel."
    exit 0
fi

echo "Found white_noise jobs to cancel:"
echo "$job_ids"
echo ""

read -p "Are you sure you want to cancel all these jobs? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    for job_id in $job_ids; do
        echo "Canceling job: $job_id"
        bkill $job_id
    done
    echo "All white_noise jobs canceled."
else
    echo "Operation canceled."
fi
