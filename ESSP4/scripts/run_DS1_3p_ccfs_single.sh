#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J DS1_3p_ccfs_single
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
#BSUB -o /work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/out_single/Output_DS1_3p_ccfs_single.out

# Change to configuration directory
cd /work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/results_single/DS1/DS1_3p/DS1_3p_ccfs

# Clean up previous runs
rm -f configuration_file_emcee_run_DS1_3p_ccfs_single.log

# Activate PyORBIT environment
source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Run PyORBIT analysis
pyorbit_run emcee DS1_3p_ccfs.yaml > configuration_file_emcee_run_DS1_3p_ccfs_single.log
pyorbit_results emcee DS1_3p_ccfs.yaml -all >> configuration_file_emcee_run_DS1_3p_ccfs_single.log

# Create results directory and copy files
# mkdir -p ./DS1_3p_ccfs_single
# cp DS1_3p_ccfs.yaml ./DS1_3p_ccfs_single/
# cp configuration_file_emcee_run_DS1_3p_ccfs_single.log ./DS1_3p_ccfs_single/

# Deactivate environment
conda deactivate

echo "Job DS1_3p_ccfs_single completed at: $(date)"
