#!/bin/sh
#BSUB -q hpc
#BSUB -J HD102365_UCLES_only_no_gp_2p_dynesty
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -M 2GB
#BSUB -W 48:00
#BSUB -u icogo@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/out_UCLES_HD102365/Output_HD102365_UCLES_only_no_gp_2p_dynesty.out

cd /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_UCLES_HD102365/no_gp/2p/HD102365_UCLES_only_no_gp_2p_dynesty
rm -f configuration_file_dynesty_run_HD102365_UCLES_only_no_gp_2p_dynesty.log

source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

pyorbit_run dynesty HD102365_UCLES_only_no_gp_2p_dynesty.yaml > configuration_file_dynesty_run_HD102365_UCLES_only_no_gp_2p_dynesty.log
pyorbit_results dynesty HD102365_UCLES_only_no_gp_2p_dynesty.yaml -all >> configuration_file_dynesty_run_HD102365_UCLES_only_no_gp_2p_dynesty.log

mkdir -p /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_UCLES_HD102365/no_gp/2p/HD102365_UCLES_only_no_gp_2p_dynesty/HD102365_UCLES_only_no_gp_2p_dynesty
cp HD102365_UCLES_only_no_gp_2p_dynesty.yaml /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_UCLES_HD102365/no_gp/2p/HD102365_UCLES_only_no_gp_2p_dynesty/HD102365_UCLES_only_no_gp_2p_dynesty/
cp configuration_file_dynesty_run_HD102365_UCLES_only_no_gp_2p_dynesty.log /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_UCLES_HD102365/no_gp/2p/HD102365_UCLES_only_no_gp_2p_dynesty/HD102365_UCLES_only_no_gp_2p_dynesty/

conda deactivate
echo "Job HD102365_UCLES_only_no_gp_2p_dynesty completed at: $(date)"
