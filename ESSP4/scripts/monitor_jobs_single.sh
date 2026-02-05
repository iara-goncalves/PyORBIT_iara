#!/bin/bash

echo "PyORBIT Job Monitor (Single Files, Multi-Configuration)"
echo "======================================================="

echo "All your jobs:"
bjobs

echo ""
echo "PyORBIT single file jobs by configuration:"
echo "0p (0 planets with GP):"
bjobs | grep -E "(DS[1-9]_0p_(2_activity_indi|4_activity_indi|5_activity_indi|ccfs)_single)"

echo ""
echo "2_activity_indi (BIS, FWHM):"
bjobs | grep -E "(DS[1-9]_[0-3]p_2_activity_indi_single)"

echo ""
echo "4_activity_indi (BIS, FWHM, Contrast, Halpha):"
bjobs | grep -E "(DS[1-9]_[0-3]p_4_activity_indi_single)"

echo ""
echo "5_activity_indi (BIS, FWHM, Contrast, Halpha, CaII):"
bjobs | grep -E "(DS[1-9]_[0-3]p_5_activity_indi_single)"

echo ""
echo "ccfs (FWHM, Contrast):"
bjobs | grep -E "(DS[1-9]_[0-3]p_ccfs_single)"

echo ""
echo "Job summary:"
total_jobs=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_(2_activity_indi|4_activity_indi|5_activity_indi|ccfs)_single)")
running_jobs=$(bjobs | grep RUN | grep -c -E "(DS[1-9]_[0-3]p_(2_activity_indi|4_activity_indi|5_activity_indi|ccfs)_single)")
pending_jobs=$(bjobs | grep PEND | grep -c -E "(DS[1-9]_[0-3]p_(2_activity_indi|4_activity_indi|5_activity_indi|ccfs)_single)")

echo "Total PyORBIT single file jobs: $total_jobs"
echo "Running: $running_jobs"
echo "Pending: $pending_jobs"

echo ""
echo "Jobs by configuration:"
for config in 2_activity_indi 4_activity_indi 5_activity_indi ccfs; do
    config_count=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_${config}_single)")
    if [ $config_count -gt 0 ]; then
        echo "  $config: $config_count jobs"
    fi
done

echo ""
echo "Jobs by planet configuration:"
for planet in 0p 1p 2p 3p; do
    planet_count=$(bjobs | grep -c -E "(DS[1-9]_${planet}_(2_activity_indi|4_activity_indi|5_activity_indi|ccfs)_single)")
    if [ $planet_count -gt 0 ]; then
        echo "  $planet: $planet_count jobs"
    fi
done

echo ""
echo "Jobs by dataset:"
for ds in DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8 DS9; do
    ds_count=$(bjobs | grep -c -E "(${ds}_[0-3]p_(2_activity_indi|4_activity_indi|5_activity_indi|ccfs)_single)")
    if [ $ds_count -gt 0 ]; then
        echo "  $ds: $ds_count jobs"
    fi
done

echo ""
echo "Refresh with: ./monitor_jobs_single.sh"
echo "Detailed job info: bjobs -l JOB_ID"
