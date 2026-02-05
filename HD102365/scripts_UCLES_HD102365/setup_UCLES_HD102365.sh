#!/bin/bash

# HD102365 UCLES-only PyORBIT Setup Generator
# Creates directory structure, YAML configs, and LSF job scripts for:
#   - UCLES_only
#   - GP vs no-GP
#   - 1p, 2p, 3p
#   - emcee & dynesty
#
# Job names:
#   HD102365_UCLES_only_{gp|no_gp}_{1p|2p|3p}_{emcee|dynesty}

base_dir="/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365"
data_dir="${base_dir}/processed_data"
results_dir="${base_dir}/results_UCLES_HD102365"
out_dir="${base_dir}/out_UCLES_HD102365"
scripts_dir="${base_dir}/scripts_UCLES_HD102365"

mkdir -p "${results_dir}" "${out_dir}" "${scripts_dir}"
cd "${scripts_dir}" || exit 1

# LSF configuration
queue="hpc"
email="icogo@dtu.dk"

# Resources for no-GP jobs
cores_emcee_no_gp=16
cores_dynesty_no_gp=16
mem_per_core_no_gp="2GB"
mem_limit_no_gp="2GB"
walltime_emcee_no_gp="24:00"
walltime_dynesty_no_gp="48:00"

# Resources for GP jobs
cores_emcee_gp=32
cores_dynesty_gp=32
mem_per_core_gp="8GB"
mem_limit_gp="9GB"
walltime_emcee_gp="72:00"
walltime_dynesty_gp="72:00"

gp_flags=("no_gp" "gp")
planets=("1p" "2p" "3p")
samplers=("emcee" "dynesty")

yaml_count=0
script_count=0

echo "HD102365 UCLES-only PyORBIT Setup"
echo "================================"
echo "Base dir:      ${base_dir}"
echo "Data dir:      ${data_dir}"
echo "Results dir:   ${results_dir}"
echo "Output dir:    ${out_dir}"
echo "Scripts dir:   ${scripts_dir}"
echo ""

# ------------------ YAML GENERATOR ------------------ #
generate_yaml_ucles() {
  local yaml_file="$1"      # full path
  local gp_flag="$2"        # gp / no_gp
  local planet_conf="$3"    # 1p / 2p / 3p

  local num_planets
  case "${planet_conf}" in
    "1p") num_planets=1 ;;
    "2p") num_planets=2 ;;
    "3p") num_planets=3 ;;
    *) echo "Unknown planet_conf ${planet_conf}" ; exit 1 ;;
  esac

  local rv_name="HD102365_UCLES_RV"

  cat > "${yaml_file}" << EOF
inputs:
  ${rv_name}:
    file: ${data_dir}/${rv_name}.dat
    kind: RV
    models:
      - radial_velocities
EOF

  if [ "${gp_flag}" == "gp" ]; then
    cat >> "${yaml_file}" << EOF
      - gp_multidimensional

  HD102365_UCLES_EWHa:
    file: ${data_dir}/HD102365_UCLES_EWHa.dat
    kind: EWHa
    models:
      - gp_multidimensional
EOF
  fi

  cat >> "${yaml_file}" << EOF

common:
  planets:
EOF

  local planet_letters=("b" "c" "d")
  for ((p=0; p<num_planets; p++)); do
    local pl="${planet_letters[$p]}"
    cat >> "${yaml_file}" << EOF
    ${pl}:
      orbit: keplerian
      parametrization: Eastman2013
      boundaries:
        P: [1.1, 8000.0]
        K: [0.1, 10.0]
        e: [0.00, 0.40]
EOF
  done

  if [ "${gp_flag}" == "gp" ]; then
    # UCLES-only GP priors/bounds (as in your 0p UCLES-only setup)
    cat >> "${yaml_file}" << EOF
  activity:
    boundaries:
      Prot: [3000.0, 4000.0]
      Pdec: [1000.0, 10000.0]
      Oamp: [0.01, 1.0]
    priors:
      Prot: ['Gaussian', 3500.00, 500.0]
EOF
  fi

  cat >> "${yaml_file}" << EOF
  star:
    star_parameters:
      priors:
        mass: ['Gaussian', 0.85, 0.03]
        radius: ['Gaussian', 0.99, 0.02]
        density: ['Gaussian', 0.8760186293091098, 0.04]

models:
  radial_velocities:
    planets:
EOF

  for ((p=0; p<num_planets; p++)); do
    local pl="${planet_letters[$p]}"
    echo "      - ${pl}" >> "${yaml_file}"
  done

  if [ "${gp_flag}" == "gp" ]; then
    cat >> "${yaml_file}" << EOF
  gp_multidimensional:
    model: spleaf_multidimensional_esp
    common: activity
    n_harmonics: 4
    hyperparameters_condition: True
    rotation_decay_condition: True

    ${rv_name}:
      boundaries:
        rot_amp: [0.0, 10.0]
        con_amp: [-30.0, 20.0]
      derivative: True

    HD102365_UCLES_EWHa:
      boundaries:
        rot_amp: [-1.0, 1.0]
        con_amp: [-10.0, 10.0]
      derivative: True
EOF
  fi

  local cpu_threads
  if [ "${gp_flag}" == "gp" ]; then
    cpu_threads=32
  else
    cpu_threads=16
  fi

  cat >> "${yaml_file}" << EOF

parameters:
  Tref: 2450830.213590
  low_ram_plot: True
  plot_split_threshold: 1000
  cpu_threads: ${cpu_threads}

solver:
  pyde:
    ngen: 50000
    npop_mult: 6
  emcee:
    npop_mult: 6
    nsteps: 50000
    nburn: 15000
    nsave: 35000
    thin: 15
  nested_sampling:
    nlive: 1000
  recenter_bounds: True
EOF
}

# ------------------ JOB SCRIPT GENERATOR ------------------ #
generate_job_script() {
  local script_file="$1"
  local job_name="$2"
  local config_dir="$3"
  local yaml_name="$4"
  local sampler="$5"   # emcee / dynesty
  local gp_flag="$6"   # gp / no_gp

  local cores mem_per_core mem_limit walltime

  if [ "${gp_flag}" == "gp" ]; then
    if [ "${sampler}" == "emcee" ]; then
      cores=${cores_emcee_gp}
      mem_per_core=${mem_per_core_gp}
      mem_limit=${mem_limit_gp}
      walltime=${walltime_emcee_gp}
    else
      cores=${cores_dynesty_gp}
      mem_per_core=${mem_per_core_gp}
      mem_limit=${mem_limit_gp}
      walltime=${walltime_dynesty_gp}
    fi
  else
    if [ "${sampler}" == "emcee" ]; then
      cores=${cores_emcee_no_gp}
      mem_per_core=${mem_per_core_no_gp}
      mem_limit=${mem_limit_no_gp}
      walltime=${walltime_emcee_no_gp}
    else
      cores=${cores_dynesty_no_gp}
      mem_per_core=${mem_per_core_no_gp}
      mem_limit=${mem_limit_no_gp}
      walltime=${walltime_dynesty_no_gp}
    fi
  fi

  cat > "${script_file}" << EOF
#!/bin/sh
#BSUB -q ${queue}
#BSUB -J ${job_name}
#BSUB -n ${cores}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${mem_per_core}]"
#BSUB -M ${mem_limit}
#BSUB -W ${walltime}
#BSUB -u ${email}
#BSUB -B
#BSUB -N
#BSUB -o ${out_dir}/Output_${job_name}.out

cd ${config_dir}
rm -f configuration_file_${sampler}_run_${job_name}.log

source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

pyorbit_run ${sampler} ${yaml_name} > configuration_file_${sampler}_run_${job_name}.log
pyorbit_results ${sampler} ${yaml_name} -all >> configuration_file_${sampler}_run_${job_name}.log

mkdir -p ${config_dir}/${job_name}
cp ${yaml_name} ${config_dir}/${job_name}/
cp configuration_file_${sampler}_run_${job_name}.log ${config_dir}/${job_name}/

conda deactivate
echo "Job ${job_name} completed at: \$(date)"
EOF

  chmod +x "${script_file}"
}

# ------------------ MAIN LOOP ------------------ #
for gp_flag in "${gp_flags[@]}"; do
  for planet_conf in "${planets[@]}"; do
    for sampler in "${samplers[@]}"; do

      job_name="HD102365_UCLES_only_${gp_flag}_${planet_conf}_${sampler}"
      config_dir="${results_dir}/${gp_flag}/${planet_conf}/${job_name}"
      mkdir -p "${config_dir}"

      yaml_name="${job_name}.yaml"
      yaml_path="${config_dir}/${yaml_name}"

      generate_yaml_ucles "${yaml_path}" "${gp_flag}" "${planet_conf}"
      ((yaml_count++))

      script_path="${scripts_dir}/run_${job_name}.sh"
      generate_job_script "${script_path}" "${job_name}" "${config_dir}" "${yaml_name}" "${sampler}" "${gp_flag}"
      ((script_count++))

      echo "Created: ${yaml_path}"
      echo "Created: ${script_path}"
      echo ""
    done
  done
done

# ------------------ SUBMIT / MONITOR / CANCEL HELPERS ------------------ #
cat > "${scripts_dir}/submit_all_UCLES_HD102365.sh" << 'EOF'
#!/bin/bash
echo "Submitting ALL UCLES-only HD102365 jobs..."
job_count=0
for script in run_HD102365_UCLES_only_*.sh; do
  [ -f "$script" ] || continue
  echo "Submitting: $script"
  bsub < "$script"
  ((job_count++))
  sleep 0.5
done
echo "Submitted $job_count jobs."
EOF
chmod +x "${scripts_dir}/submit_all_UCLES_HD102365.sh"

cat > "${scripts_dir}/submit_UCLES_HD102365_gp.sh" << 'EOF'
#!/bin/bash
echo "Submitting UCLES-only HD102365 GP jobs..."
job_count=0
for script in run_HD102365_UCLES_only_gp_*.sh; do
  [ -f "$script" ] || continue
  echo "Submitting: $script"
  bsub < "$script"
  ((job_count++))
  sleep 0.5
done
echo "Submitted $job_count GP jobs."
EOF
chmod +x "${scripts_dir}/submit_UCLES_HD102365_gp.sh"

cat > "${scripts_dir}/submit_UCLES_HD102365_no_gp.sh" << 'EOF'
#!/bin/bash
echo "Submitting UCLES-only HD102365 no-GP jobs..."
job_count=0
for script in run_HD102365_UCLES_only_no_gp_*.sh; do
  [ -f "$script" ] || continue
  echo "Submitting: $script"
  bsub < "$script"
  ((job_count++))
  sleep 0.5
done
echo "Submitted $job_count no-GP jobs."
EOF
chmod +x "${scripts_dir}/submit_UCLES_HD102365_no_gp.sh"

cat > "${scripts_dir}/monitor_UCLES_HD102365.sh" << 'EOF'
#!/bin/bash
echo "UCLES-only HD102365 Job Monitor"
echo "==============================="
bjobs | grep "HD102365_UCLES_only_" || echo "No UCLES-only HD102365 jobs found."
EOF
chmod +x "${scripts_dir}/monitor_UCLES_HD102365.sh"

cat > "${scripts_dir}/cancel_UCLES_HD102365.sh" << 'EOF'
#!/bin/bash
echo "Canceling all UCLES-only HD102365 jobs..."
job_ids=$(bjobs | grep "HD102365_UCLES_only_" | awk '{print $1}')
if [ -z "$job_ids" ]; then
  echo "No UCLES-only HD102365 jobs to cancel."
  exit 0
fi
echo "Jobs to cancel:"
echo "$job_ids"
for id in $job_ids; do
  echo "Canceling job $id"
  bkill "$id"
done
echo "Done."
EOF
chmod +x "${scripts_dir}/cancel_UCLES_HD102365.sh"

echo "Setup complete."
echo "YAML files created:   ${yaml_count}"
echo "Job scripts created:  ${script_count}"
echo ""
echo "Use:"
echo "  cd ${scripts_dir}"
echo "  ./submit_all_UCLES_HD102365.sh"
echo "  ./submit_UCLES_HD102365_gp.sh"
echo "  ./submit_UCLES_HD102365_no_gp.sh"
echo "  ./monitor_UCLES_HD102365.sh"
echo "  ./cancel_UCLES_HD102365.sh"
