#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J DS8_3p_5_activity_indi_single
### -- ask for number of cores (default: 1) -- 
#BSUB -n 16
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 1GB of memory per core/slot -- 
#BSUB -R "rusage[mem=1GB]"
### -- specify that we want the job to get killed if it exceeds 2GB per core/slot -- 
#BSUB -M 2GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 48:00
### -- set the email address -- 
#BSUB -u jzhao@space.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o /work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/out_single/Output_DS8_3p_5_activity_indi_single.out

# Change to configuration directory
cd /work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/results_single/DS8/DS8_3p/DS8_3p_5_activity_indi

# Clean up previous runs
rm -f configuration_file_emcee_run_DS8_3p_5_activity_indi_single.log

# Activate PyORBIT environment
source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Run PyORBIT analysis
pyorbit_run emcee DS8_3p_5_activity_indi.yaml > configuration_file_emcee_run_DS8_3p_5_activity_indi_single.log
pyorbit_results emcee DS8_3p_5_activity_indi.yaml -all >> configuration_file_emcee_run_DS8_3p_5_activity_indi_single.log

# Create results directory and copy files
# mkdir -p ./DS8_3p_5_activity_indi_single
# cp DS8_3p_5_activity_indi.yaml ./DS8_3p_5_activity_indi_single/
# cp configuration_file_emcee_run_DS8_3p_5_activity_indi_single.log ./DS8_3p_5_activity_indi_single/

# Deactivate environment
conda deactivate

echo "Job DS8_3p_5_activity_indi_single completed at: $(date)"
