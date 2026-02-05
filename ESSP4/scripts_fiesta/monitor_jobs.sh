#!/bin/bash

echo "PyORBIT FIESTA Job Monitor (2 and 3 modes)"
echo "============================================"

echo "All your jobs:"
bjobs

echo ""
echo "PyORBIT FIESTA jobs:"
bjobs | grep -E "(DS[1-9]_[1-3]p_(fiesta|3modes))"

echo ""
echo "Job summary:"
total_jobs=$(bjobs | grep -c -E "(DS[1-9]_[1-3]p_(fiesta|3modes))")
running_jobs=$(bjobs | grep RUN | grep -c -E "(DS[1-9]_[1-3]p_(fiesta|3modes))")
pending_jobs=$(bjobs | grep PEND | grep -c -E "(DS[1-9]_[1-3]p_(fiesta|3modes))")
fiesta_jobs=$(bjobs | grep -c -E "(DS[1-9]_[1-3]p_fiesta)")
modes3_jobs=$(bjobs | grep -c -E "(DS[1-9]_[1-3]p_3modes)")

echo "Total jobs: $total_jobs"
echo "Running: $running_jobs"
echo "Pending: $pending_jobs"
echo "FIESTA (2 modes): $fiesta_jobs"
echo "3MODES (3 modes): $modes3_jobs"

echo ""
echo "Jobs by dataset:"
for ds in DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8 DS9; do
    ds_count=$(bjobs | grep -c -E "(${ds}_[1-3]p_(fiesta|3modes))")
    if [ $ds_count -gt 0 ]; then
        echo "  $ds: $ds_count jobs"
    fi
done

echo ""
echo "Jobs by planet configuration:"
for planet in 1p 2p 3p; do
    planet_count=$(bjobs | grep -c -E "(DS[1-9]_${planet}_(fiesta|3modes))")
    if [ $planet_count -gt 0 ]; then
        echo "  $planet: $planet_count jobs"
    fi
done

echo ""
echo "Refresh with: ./monitor_jobs.sh"
echo "Detailed job info: bjobs -l JOB_ID"
