#!/bin/bash

echo "PyORBIT FIESTA Job Monitor"
echo "=========================="

echo "All your jobs:"
bjobs

echo ""
echo "FIESTA jobs by configuration:"
echo ""
echo "1mode jobs:"
bjobs | grep -E "(DS[1-9]_[0-3]p_1mode)"

echo ""
echo "2mode jobs:"
bjobs | grep -E "(DS[1-9]_[0-3]p_2mode)"

echo ""
echo "3mode jobs:"
bjobs | grep -E "(DS[1-9]_[0-3]p_3mode)"

echo ""
echo "4mode jobs:"
bjobs | grep -E "(DS[1-9]_[0-3]p_4mode)"

echo ""
echo "Job summary:"
total_jobs=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_(1mode|2mode|3mode|4mode))")
running_jobs=$(bjobs | grep RUN | grep -c -E "(DS[1-9]_[0-3]p_(1mode|2mode|3mode|4mode))")
pending_jobs=$(bjobs | grep PEND | grep -c -E "(DS[1-9]_[0-3]p_(1mode|2mode|3mode|4mode))")

echo "Total FIESTA jobs: $total_jobs"
echo "Running: $running_jobs"
echo "Pending: $pending_jobs"

echo ""
echo "Jobs by FIESTA configuration:"
mode1_count=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_1mode)")
mode2_count=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_2mode)")
mode3_count=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_3mode)")
mode4_count=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_4mode)")
echo "  1mode: $mode1_count jobs"
echo "  2mode: $mode2_count jobs"
echo "  3mode: $mode3_count jobs"
echo "  4mode: $mode4_count jobs"

echo ""
echo "Jobs by planet configuration:"
for planet in 0p 1p 2p 3p; do
    planet_count=$(bjobs | grep -c -E "(DS[1-9]_${planet}_(1mode|2mode|3mode|4mode))")
    if [ $planet_count -gt 0 ]; then
        echo "  $planet: $planet_count jobs"
    fi
done

echo ""
echo "Jobs by dataset:"
for ds in DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8 DS9; do
    ds_count=$(bjobs | grep -c -E "(${ds}_[0-3]p_(1mode|2mode|3mode|4mode))")
    if [ $ds_count -gt 0 ]; then
        echo "  $ds: $ds_count jobs"
    fi
done

echo ""
echo "Refresh with: ./monitor_fiesta_jobs.sh"
echo "Detailed job info: bjobs -l JOB_ID"
