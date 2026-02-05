#!/bin/bash

echo "HD102365 Job Monitor"
echo "===================="
bjobs | grep "HD102365_" || echo "No HD102365 jobs found."
