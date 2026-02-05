#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J DS3_1p_2modes
### -- ask for number of cores (default: 1) -- 
#BSUB -n 16
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
### -- specify that we want the job to get killed if it exceeds 3GB per core/slot -- 
#BSUB -M 3GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 48:00
### -- set the email address -- 
#BSUB -u jzhao@space.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o /work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/out_fiesta/Output_DS3_1p_2modes.out

# Change to configuration directory
cd /work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/results_fiesta_emcee_kostas/DS3/DS3_1p_2modes

# Clean up previous runs
rm -f configuration_file_emcee_run_DS3_1p_2modes.log



# Set up environment paths for jinzhao's installations
export CONDA_BASE="/work2/lbuc/iara/anaconda3"
export PATH="$CONDA_BASE/bin:$PATH"

# Initialize conda for any user
source $CONDA_BASE/etc/profile.d/conda.sh

# Activate PyORBIT environment using CORRECT FULL PATH
conda activate /work2/lbuc/iara/conda_envs/pyorbit


# # Activate PyORBIT environment
# source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
# conda activate pyorbit

# Run PyORBIT analysis
pyorbit_run emcee DS3_1p_2modes.yaml > configuration_file_emcee_run_DS3_1p_2modes.log
pyorbit_results emcee DS3_1p_2modes.yaml -all >> configuration_file_emcee_run_DS3_1p_2modes.log

# Create results directory and copy files
# mkdir -p ./DS3_1p_2modes
# cp DS3_1p_2modes.yaml ./DS3_1p_2modes/
# cp configuration_file_emcee_run_DS3_1p_2modes.log ./DS3_1p_2modes/

# Deactivate environment
conda deactivate

echo "Job DS3_1p_2modes completed at: $(date)"
