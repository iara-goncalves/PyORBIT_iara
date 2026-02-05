#!/bin/sh
#BSUB -q hpc
#BSUB -J HD102365_UCLES_only_no_gp_3p_emcee
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -M 2GB
#BSUB -W 24:00
#BSUB -u icogo@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/out_UCLES_HD102365/Output_HD102365_UCLES_only_no_gp_3p_emcee.out

cd /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_UCLES_HD102365/no_gp/3p/HD102365_UCLES_only_no_gp_3p_emcee
rm -f configuration_file_emcee_run_HD102365_UCLES_only_no_gp_3p_emcee.log

source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

pyorbit_run emcee HD102365_UCLES_only_no_gp_3p_emcee.yaml > configuration_file_emcee_run_HD102365_UCLES_only_no_gp_3p_emcee.log
pyorbit_results emcee HD102365_UCLES_only_no_gp_3p_emcee.yaml -all >> configuration_file_emcee_run_HD102365_UCLES_only_no_gp_3p_emcee.log

mkdir -p /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_UCLES_HD102365/no_gp/3p/HD102365_UCLES_only_no_gp_3p_emcee/HD102365_UCLES_only_no_gp_3p_emcee
cp HD102365_UCLES_only_no_gp_3p_emcee.yaml /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_UCLES_HD102365/no_gp/3p/HD102365_UCLES_only_no_gp_3p_emcee/HD102365_UCLES_only_no_gp_3p_emcee/
cp configuration_file_emcee_run_HD102365_UCLES_only_no_gp_3p_emcee.log /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_UCLES_HD102365/no_gp/3p/HD102365_UCLES_only_no_gp_3p_emcee/HD102365_UCLES_only_no_gp_3p_emcee/

conda deactivate
echo "Job HD102365_UCLES_only_no_gp_3p_emcee completed at: $(date)"
