#!/bin/sh
#BSUB -q hpc
#BSUB -J HD102365_espresso_only_gp_0p_emcee
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 9GB
#BSUB -W 72:00
#BSUB -u icogo@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/out_HD102365_0p_all/Output_HD102365_espresso_only_gp_0p_emcee.out

cd /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365_0p_all/HD102365_espresso_only_gp_0p_emcee
rm -f configuration_file_emcee_run_HD102365_espresso_only_gp_0p_emcee.log

source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

pyorbit_run emcee HD102365_espresso_only_gp_0p_emcee.yaml > configuration_file_emcee_run_HD102365_espresso_only_gp_0p_emcee.log
pyorbit_results emcee HD102365_espresso_only_gp_0p_emcee.yaml -all >> configuration_file_emcee_run_HD102365_espresso_only_gp_0p_emcee.log

mkdir -p /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365_0p_all/HD102365_espresso_only_gp_0p_emcee/HD102365_espresso_only_gp_0p_emcee
cp HD102365_espresso_only_gp_0p_emcee.yaml /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365_0p_all/HD102365_espresso_only_gp_0p_emcee/HD102365_espresso_only_gp_0p_emcee/
cp configuration_file_emcee_run_HD102365_espresso_only_gp_0p_emcee.log /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365_0p_all/HD102365_espresso_only_gp_0p_emcee/HD102365_espresso_only_gp_0p_emcee/

conda deactivate
echo "Job HD102365_espresso_only_gp_0p_emcee completed at: $(date)"
