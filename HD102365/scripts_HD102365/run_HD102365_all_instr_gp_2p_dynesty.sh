#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J HD102365_all_instr_gp_2p_dynesty
### -- ask for number of cores (default: 1) -- 
#BSUB -n 32
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 8GB of memory per core/slot -- 
#BSUB -R "rusage[mem=8GB]"
### -- specify that we want the job to get killed if it exceeds 9GB per core/slot -- 
#BSUB -M 9GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 72:00
### -- set the email address -- 
#BSUB -u icogo@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/out_HD102365/Output_HD102365_all_instr_gp_2p_dynesty.out

# Change to configuration directory
cd /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/all_instr/gp/2p/HD102365_all_instr_gp_2p_dynesty

# Clean up previous runs
rm -f configuration_file_dynesty_run_HD102365_all_instr_gp_2p_dynesty.log

# Activate PyORBIT environment
source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Run PyORBIT analysis
pyorbit_run dynesty HD102365_all_instr_gp_2p_dynesty.yaml > configuration_file_dynesty_run_HD102365_all_instr_gp_2p_dynesty.log
pyorbit_results dynesty HD102365_all_instr_gp_2p_dynesty.yaml -all >> configuration_file_dynesty_run_HD102365_all_instr_gp_2p_dynesty.log

# Create results directory and copy files
mkdir -p /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/all_instr/gp/2p/HD102365_all_instr_gp_2p_dynesty/HD102365_all_instr_gp_2p_dynesty
cp HD102365_all_instr_gp_2p_dynesty.yaml /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/all_instr/gp/2p/HD102365_all_instr_gp_2p_dynesty/HD102365_all_instr_gp_2p_dynesty/
cp configuration_file_dynesty_run_HD102365_all_instr_gp_2p_dynesty.log /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/all_instr/gp/2p/HD102365_all_instr_gp_2p_dynesty/HD102365_all_instr_gp_2p_dynesty/

# Deactivate environment
conda deactivate

echo "Job HD102365_all_instr_gp_2p_dynesty completed at: $(date)"
