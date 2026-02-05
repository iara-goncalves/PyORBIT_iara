#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J HD189567_2p_4activity
### -- ask for number of cores (default: 1) -- 
#BSUB -n 16
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5GB per core/slot -- 
#BSUB -M 5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o Output_HD189567_2p_4activity.out 

# Change to GitHub HD189567 directory
cd /work2/lbuc/iara/GitHub/PyORBIT_examples/HD189567/

# Clean up previous runs
rm -f configuration_file_emcee_run_HD189567_2p_4activity.log Output_HD189567_2p_4activity.out

# Activate PyORBIT environment
source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Run PyORBIT analysis (YAML is in data_HD189567 folder)
pyorbit_run emcee ./data_HD189567/2p_HD189567.yaml > configuration_file_emcee_run_HD189567_2p_4activity.log
pyorbit_results emcee ./data_HD189567/2p_HD189567.yaml -all >> configuration_file_emcee_run_HD189567_2p_4activity.log

# Create the 2p_HD189567 folder inside data_HD189567 and organize files
mkdir -p ./data_HD189567/2p_HD189567
mv 2p_HD189567_emcee/* ./data_HD189567/2p_HD189567/
rmdir 2p_HD189567_emcee
mv configuration_file_emcee_run_HD189567_2p_4activity.log ./data_HD189567/2p_HD189567/

# The YAML file is already in the right location, so no need to move it

# Deactivate environment
conda deactivate

echo "Job HD189567_2p_4activity completed at: $(date)"