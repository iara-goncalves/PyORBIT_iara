#!/bin/bash

echo "Canceling all PyORBIT FIESTA jobs..."
echo "===================================="

job_ids=$(bjobs | grep -E "(DS[1-9]_[0-3]p_(2modes|3modes))" | awk '{print $1}')

if [ -z "$job_ids" ]; then
    echo "No FIESTA jobs found to cancel."
    exit 0
fi

echo "Found FIESTA jobs to cancel:"
echo "$job_ids"
echo ""

read -p "Are you sure you want to cancel all these jobs? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    for job_id in $job_ids; do
        echo "Canceling job: $job_id"
        bkill $job_id
    done
    echo "All FIESTA jobs canceled."
else
    echo "Operation canceled."
fi
