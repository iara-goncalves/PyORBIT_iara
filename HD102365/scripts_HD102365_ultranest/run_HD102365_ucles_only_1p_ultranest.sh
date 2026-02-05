#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J HD102365_ucles_only_1p_ultranest
### -- ask for number of cores (default: 1) -- 
#BSUB -n 32
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5GB per core/slot -- 
#BSUB -M 5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 72:00
### -- set the email address -- 
#BSUB -u jzhao@space.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/out_HD102365_ultranest/Output_HD102365_ucles_only_1p_ultranest.out

# Change to configuration directory
cd /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365_ultranest/ucles_only/1p/HD102365_ucles_only_1p_ultranest

# Clean up previous runs
rm -f configuration_file_ultranest_run_HD102365_ucles_only_1p_ultranest.log

# Activate PyORBIT environment
## source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Set environment variables for parallel processing
export OMP_NUM_THREADS=1

# Run PyORBIT analysis with UltraNest
mpiexec -np 32 pyorbit_run ultranest HD102365_ucles_only_1p_ultranest.yaml > configuration_file_ultranest_run_HD102365_ucles_only_1p_ultranest.log
pyorbit_results ultranest HD102365_ucles_only_1p_ultranest.yaml -all >> configuration_file_ultranest_run_HD102365_ucles_only_1p_ultranest.log

# Create results directory and copy files
mkdir -p /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365_ultranest/ucles_only/1p/HD102365_ucles_only_1p_ultranest/HD102365_ucles_only_1p_ultranest
cp HD102365_ucles_only_1p_ultranest.yaml /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365_ultranest/ucles_only/1p/HD102365_ucles_only_1p_ultranest/HD102365_ucles_only_1p_ultranest/
cp configuration_file_ultranest_run_HD102365_ucles_only_1p_ultranest.log /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365_ultranest/ucles_only/1p/HD102365_ucles_only_1p_ultranest/HD102365_ucles_only_1p_ultranest/

# Deactivate environment
conda deactivate

echo "Job HD102365_ucles_only_1p_ultranest completed at: $(date)"
