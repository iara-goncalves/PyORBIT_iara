#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J DS6_2p_white_noise
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
#BSUB -o /work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/out_multiple/Output_DS6_2p_white_noise.out

# Change to configuration directory
cd /work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/results_multiple/DS6/DS6_2p/DS6_2p_white_noise

# Clean up previous runs
rm -f configuration_file_emcee_run_DS6_2p_white_noise.log

# Activate PyORBIT environment
source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Run PyORBIT analysis
pyorbit_run emcee DS6_2p_white_noise.yaml > configuration_file_emcee_run_DS6_2p_white_noise.log
pyorbit_results emcee DS6_2p_white_noise.yaml -all >> configuration_file_emcee_run_DS6_2p_white_noise.log

# Create results directory and copy files
mkdir -p ./DS6_2p_white_noise
cp DS6_2p_white_noise.yaml ./DS6_2p_white_noise/
cp configuration_file_emcee_run_DS6_2p_white_noise.log ./DS6_2p_white_noise/

# Deactivate environment
conda deactivate

echo "Job DS6_2p_white_noise completed at: $(date)"
