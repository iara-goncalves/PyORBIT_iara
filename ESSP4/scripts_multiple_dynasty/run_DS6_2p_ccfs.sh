#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J DS6_2p_ccfs
### -- ask for number of cores (default: 1) -- 
#BSUB -n 16
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
### -- specify that we want the job to get killed if it exceeds 3GB per core/slot -- 
#BSUB -M 3GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 72:00
### -- set the email address -- 
#BSUB -u jzhao@space.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o /work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/out_multiple_dynasty/Output_DS6_2p_ccfs.out

# Change to configuration directory
cd /work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/results_multiple_dynasty/DS6/DS6_2p/DS6_2p_ccfs

# Clean up previous runs
rm -f configuration_file_emcee_run_DS6_2p_ccfs.log

# Activate PyORBIT environment
source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Run PyORBIT analysis
pyorbit_run dynesty DS6_2p_ccfs.yaml > configuration_file_emcee_run_DS6_2p_ccfs.log
pyorbit_results dynesty DS6_2p_ccfs.yaml -all >> configuration_file_emcee_run_DS6_2p_ccfs.log

# Create results directory and copy files
# mkdir -p ./DS6_2p_ccfs
# cp DS6_2p_ccfs.yaml ./DS6_2p_ccfs/
# cp configuration_file_emcee_run_DS6_2p_ccfs.log ./DS6_2p_ccfs/

# Deactivate environment
conda deactivate

echo "Job DS6_2p_ccfs completed at: $(date)"
