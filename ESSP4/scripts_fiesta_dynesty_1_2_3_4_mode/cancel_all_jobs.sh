#!/bin/bash
# cancel_all_jobs.sh - Cancel all LSF jobs for current user

USER=$(whoami)

echo "=========================================="
echo "  LSF Job Cancellation Script"
echo "=========================================="
echo "User: $USER"
echo ""

# Get list of all jobs
JOBS=$(bjobs -u $USER -o "jobid" -noheader 2>/dev/null)

if [ -z "$JOBS" ]; then
    echo "✓ No jobs found to cancel."
    exit 0
fi

# Count jobs
NUM_JOBS=$(echo "$JOBS" | wc -l)
echo "Found $NUM_JOBS job(s) to cancel:"
echo ""

# Show job details before canceling
bjobs -u $USER -o "jobid stat queue job_name submit_time" -noheader

echo ""
read -p "Cancel all these jobs? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Canceling jobs..."
    
    # Cancel all jobs
    for jobid in $JOBS; do
        bkill $jobid 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "  ✓ Canceled job $jobid"
        else
            echo "  ✗ Failed to cancel job $jobid"
        fi
    done
    
    echo ""
    echo "=========================================="
    echo "  Cancellation complete!"
    echo "=========================================="
    
    # Wait a moment and show remaining jobs
    sleep 2
    echo ""
    echo "Remaining jobs:"
    bjobs -u $USER 2>/dev/null || echo "  ✓ No jobs remaining"
else
    echo "Cancellation aborted."
fi
