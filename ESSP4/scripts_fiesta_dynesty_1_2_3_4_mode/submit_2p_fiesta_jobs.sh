#!/bin/bash

echo "Submitting 2p FIESTA jobs (2 planets + GP)..."
echo "============================================="

job_count=0
for script in run_*_2p_1mode.sh run_*_2p_2mode.sh run_*_2p_3mode.sh run_*_2p_4mode.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 2p FIESTA jobs"
