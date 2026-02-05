#!/bin/bash
echo "UCLES-only HD102365 Job Monitor"
echo "==============================="
bjobs | grep "HD102365_UCLES_only_" || echo "No UCLES-only HD102365 jobs found."
