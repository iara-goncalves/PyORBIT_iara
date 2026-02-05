#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J DS9_0p_4_activity_indi
### -- ask for number of cores (default: 1) -- 
#BSUB -n 16
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 1GB of memory per core/slot -- 
#BSUB -R "rusage[mem=1GB]"
### -- specify that we want the job to get killed if it exceeds 2GB per core/slot -- 
#BSUB -M 2GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00
### -- set the email address -- 
#BSUB -u icogo@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o /work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/out_multiple/Output_DS9_0p_4_activity_indi.out

# Change to configuration directory
cd /work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/results_multiple/DS9/DS9_0p/DS9_0p_4_activity_indi

# Clean up previous runs
rm -f configuration_file_emcee_run_DS9_0p_4_activity_indi.log

# Activate PyORBIT environment
source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Run PyORBIT analysis
pyorbit_run emcee DS9_0p_4_activity_indi.yaml > configuration_file_emcee_run_DS9_0p_4_activity_indi.log
pyorbit_results emcee DS9_0p_4_activity_indi.yaml -all >> configuration_file_emcee_run_DS9_0p_4_activity_indi.log

# Create results directory and copy files
mkdir -p ./DS9_0p_4_activity_indi
cp DS9_0p_4_activity_indi.yaml ./DS9_0p_4_activity_indi/
cp configuration_file_emcee_run_DS9_0p_4_activity_indi.log ./DS9_0p_4_activity_indi/

# Deactivate environment
conda deactivate

echo "Job DS9_0p_4_activity_indi completed at: $(date)"
