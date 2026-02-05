#!/bin/bash

echo "PyORBIT FIESTA Job Monitor"
echo "=========================="

echo "All your jobs:"
bjobs

echo ""
echo "FIESTA jobs by configuration:"
echo ""
echo "2modes jobs:"
bjobs | grep -E "(DS[1-9]_[0-3]p_2modes)"

echo ""
echo "3modes jobs:"
bjobs | grep -E "(DS[1-9]_[0-3]p_3modes)"

echo ""
echo "Job summary:"
total_jobs=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_(2modes|3modes))")
running_jobs=$(bjobs | grep RUN | grep -c -E "(DS[1-9]_[0-3]p_(2modes|3modes))")
pending_jobs=$(bjobs | grep PEND | grep -c -E "(DS[1-9]_[0-3]p_(2modes|3modes))")

echo "Total FIESTA jobs: $total_jobs"
echo "Running: $running_jobs"
echo "Pending: $pending_jobs"

echo ""
echo "Jobs by FIESTA configuration:"
modes2_count=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_2modes)")
modes3_count=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_3modes)")
echo "  2modes: $modes2_count jobs"
echo "  3modes: $modes3_count jobs"

echo ""
echo "Jobs by planet configuration:"
for planet in 0p 1p 2p 3p; do
    planet_count=$(bjobs | grep -c -E "(DS[1-9]_${planet}_(2modes|3modes))")
    if [ $planet_count -gt 0 ]; then
        echo "  $planet: $planet_count jobs"
    fi
done

echo ""
echo "Jobs by dataset:"
for ds in DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8 DS9; do
    ds_count=$(bjobs | grep -c -E "(${ds}_[0-3]p_(2modes|3modes))")
    if [ $ds_count -gt 0 ]; then
        echo "  $ds: $ds_count jobs"
    fi
done

echo ""
echo "Refresh with: ./monitor_fiesta_jobs.sh"
echo "Detailed job info: bjobs -l JOB_ID"
