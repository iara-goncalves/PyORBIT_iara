#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J HD102365_all_instr_no_gp_1p_dynesty
### -- ask for number of cores (default: 1) -- 
#BSUB -n 16
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
### -- specify that we want the job to get killed if it exceeds 2GB per core/slot -- 
#BSUB -M 2GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 48:00
### -- set the email address -- 
#BSUB -u icogo@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/out_HD102365/Output_HD102365_all_instr_no_gp_1p_dynesty.out

# Change to configuration directory
cd /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/all_instr/no_gp/1p/HD102365_all_instr_no_gp_1p_dynesty

# Clean up previous runs
rm -f configuration_file_dynesty_run_HD102365_all_instr_no_gp_1p_dynesty.log

# Activate PyORBIT environment
source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Run PyORBIT analysis
pyorbit_run dynesty HD102365_all_instr_no_gp_1p_dynesty.yaml > configuration_file_dynesty_run_HD102365_all_instr_no_gp_1p_dynesty.log
pyorbit_results dynesty HD102365_all_instr_no_gp_1p_dynesty.yaml -all >> configuration_file_dynesty_run_HD102365_all_instr_no_gp_1p_dynesty.log

# Create results directory and copy files
mkdir -p /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/all_instr/no_gp/1p/HD102365_all_instr_no_gp_1p_dynesty/HD102365_all_instr_no_gp_1p_dynesty
cp HD102365_all_instr_no_gp_1p_dynesty.yaml /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/all_instr/no_gp/1p/HD102365_all_instr_no_gp_1p_dynesty/HD102365_all_instr_no_gp_1p_dynesty/
cp configuration_file_dynesty_run_HD102365_all_instr_no_gp_1p_dynesty.log /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/all_instr/no_gp/1p/HD102365_all_instr_no_gp_1p_dynesty/HD102365_all_instr_no_gp_1p_dynesty/

# Deactivate environment
conda deactivate

echo "Job HD102365_all_instr_no_gp_1p_dynesty completed at: $(date)"
