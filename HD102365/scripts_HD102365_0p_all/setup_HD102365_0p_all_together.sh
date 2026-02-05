#!/bin/bash
# HD102365 0p Baselines (ALL together, GP-only)
# Creates YAMLs + LSF run scripts for:
#  (1) all_instr / no_espresso / espresso_only x (gp) x (emcee,dynesty)
#  (2) UCLES_only                               x (gp) x (emcee,dynesty)
#  (3) all_instr, GP only on ESPRESSO           x (emcee,dynesty)
#
# 0p+GP: RV + activity indicators
# Note: 0p baseline uses: models: radial_velocities: planets: []

base_dir="/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365"
data_dir="${base_dir}/processed_data"
results_dir="${base_dir}/results_HD102365_0p_all"
out_dir="${base_dir}/out_HD102365_0p_all"
scripts_dir="${base_dir}/scripts_HD102365_0p_all"

mkdir -p "${results_dir}" "${out_dir}" "${scripts_dir}"
cd "${scripts_dir}" || exit 1

# LSF configuration
queue="hpc"
email="icogo@dtu.dk"

# Resources for GP jobs
cores_emcee_gp=32
cores_dynesty_gp=32
mem_per_core_gp="8GB"
mem_limit_gp="9GB"
walltime_emcee_gp="72:00"
walltime_dynesty_gp="72:00"

samplers=("emcee" "dynesty")
gp_flags=("gp")   # GP-only
data_sets=("all_instr" "no_espresso" "espresso_only")

rv_inputs_classical=(
  "HD102365_HARPS-Post_RV"
  "HD102365_HARPS-Pre_RV"
  "HD102365_HIRES-Post_RV"
  "HD102365_PFS-Post_RV"
  "HD102365_PFS-Pre_RV"
  "HD102365_UCLES_RV"
)

yaml_count=0
script_count=0

# ------------------ JOB SCRIPT GENERATOR ------------------ #
generate_job_script() {
  local script_file="$1"
  local job_name="$2"
  local config_dir="$3"
  local yaml_name="$4"
  local sampler="$5"   # emcee / dynesty

  local cores mem_per_core mem_limit walltime
  if [ "${sampler}" == "emcee" ]; then
    cores=${cores_emcee_gp}; mem_per_core=${mem_per_core_gp}; mem_limit=${mem_limit_gp}; walltime=${walltime_emcee_gp}
  else
    cores=${cores_dynesty_gp}; mem_per_core=${mem_per_core_gp}; mem_limit=${mem_limit_gp}; walltime=${walltime_dynesty_gp}
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

# ------------------ YAML: Set 1 (instrument subsets, GP) ------------------ #
generate_yaml_set1() {
  local yaml_file="$1"
  local data_set="$2"   # all_instr / no_espresso / espresso_only

  local include_espresso=false
  local include_classical=true
  case "${data_set}" in
    "all_instr") include_espresso=true; include_classical=true ;;
    "no_espresso") include_espresso=false; include_classical=true ;;
    "espresso_only") include_espresso=true; include_classical=false ;;
    *) echo "Unknown data_set ${data_set}"; exit 1 ;;
  esac

  cat > "${yaml_file}" << EOF
inputs:
EOF

  write_rv_models_gp() {
    cat >> "${yaml_file}" << EOF
      - radial_velocities
      - gp_multidimensional
EOF
  }

  if [ "${include_espresso}" == true ]; then
    cat >> "${yaml_file}" << EOF
  HD102365_ESPRESSO_RV:
    file: ${data_dir}/HD102365_ESPRESSO_RV.dat
    kind: RV
    models:
EOF
    write_rv_models_gp
  fi

  if [ "${include_classical}" == true ]; then
    for rv_name in "${rv_inputs_classical[@]}"; do
      cat >> "${yaml_file}" << EOF
  ${rv_name}:
    file: ${data_dir}/${rv_name}.dat
    kind: RV
    models:
EOF
      write_rv_models_gp
    done
  fi

  # Activity indicators (GP)
  if [ "${include_espresso}" == true ]; then
    cat >> "${yaml_file}" << EOF
  HD102365_ESPRESSO_BIS:
    file: ${data_dir}/HD102365_ESPRESSO_BIS.dat
    kind: BIS
    models:
      - gp_multidimensional
EOF
  fi

  if [ "${include_classical}" == true ]; then
    cat >> "${yaml_file}" << EOF
  HD102365_HARPS-Post_SHK:
    file: ${data_dir}/HD102365_HARPS-Post_SHK.dat
    kind: SHK
    models:
      - gp_multidimensional
  HD102365_HARPS-Pre_SHK:
    file: ${data_dir}/HD102365_HARPS-Pre_SHK.dat
    kind: SHK
    models:
      - gp_multidimensional
  HD102365_HIRES-Post_SHK:
    file: ${data_dir}/HD102365_HIRES-Post_SHK.dat
    kind: SHK
    models:
      - gp_multidimensional
  HD102365_PFS-Post_SHK:
    file: ${data_dir}/HD102365_PFS-Post_SHK.dat
    kind: SHK
    models:
      - gp_multidimensional
  HD102365_PFS-Pre_SHK:
    file: ${data_dir}/HD102365_PFS-Pre_SHK.dat
    kind: SHK
    models:
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
  activity:
    boundaries:
      Prot: [20.0, 50.0]
      Pdec: [10.0, 1000.0]
      Oamp: [0.01, 1.0]
    priors:
      Prot: ['Gaussian', 36.00, 10.0]

  star:
    star_parameters:
      priors:
        mass: ['Gaussian', 0.85, 0.03]
        radius: ['Gaussian', 0.99, 0.02]
        density: ['Gaussian', 0.8760186293091098, 0.04]

models:
  radial_velocities:
    planets: []

  gp_multidimensional:
    model: spleaf_multidimensional_esp
    common: activity
    n_harmonics: 4
    hyperparameters_condition: True
    rotation_decay_condition: True
EOF

  if [ "${include_espresso}" == true ]; then
    cat >> "${yaml_file}" << EOF
    HD102365_ESPRESSO_RV:
      boundaries:
        rot_amp: [0.0, 10.0]
        con_amp: [-20.0, 20.0]
      derivative: True
EOF
  fi

  if [ "${include_classical}" == true ]; then
    for rv_name in "${rv_inputs_classical[@]}"; do
      local con_amp
      if [[ "${rv_name}" == "HD102365_UCLES_RV" ]]; then
        con_amp="-30.0, 20.0"
      else
        con_amp="-20.0, 20.0"
      fi
      cat >> "${yaml_file}" << EOF
    ${rv_name}:
      boundaries:
        rot_amp: [0.0, 10.0]
        con_amp: [${con_amp}]
      derivative: True
EOF
    done
  fi

  if [ "${include_espresso}" == true ]; then
    cat >> "${yaml_file}" << EOF
    HD102365_ESPRESSO_BIS:
      boundaries:
        rot_amp: [-80.0, 80.0]
        con_amp: [-10.0, 10.0]
      derivative: True
EOF
  fi

  if [ "${include_classical}" == true ]; then
    cat >> "${yaml_file}" << EOF
    HD102365_HARPS-Post_SHK:
      boundaries:
        rot_amp: [-1.0, 1.0]
        con_amp: [-10.0, 10.0]
      derivative: True
    HD102365_HARPS-Pre_SHK:
      boundaries:
        rot_amp: [-1.0, 1.0]
        con_amp: [-10.0, 10.0]
      derivative: True
    HD102365_HIRES-Post_SHK:
      boundaries:
        rot_amp: [-1.0, 1.0]
        con_amp: [-10.0, 10.0]
      derivative: True
    HD102365_PFS-Post_SHK:
      boundaries:
        rot_amp: [-1.0, 1.0]
        con_amp: [-10.0, 10.0]
      derivative: True
    HD102365_PFS-Pre_SHK:
      boundaries:
        rot_amp: [-1.0, 1.0]
        con_amp: [-10.0, 10.0]
      derivative: True
    HD102365_UCLES_EWHa:
      boundaries:
        rot_amp: [-1.0, 1.0]
        con_amp: [-10.0, 10.0]
      derivative: True
EOF
  fi

  cat >> "${yaml_file}" << EOF

parameters:
  Tref: 2450830.213590
  low_ram_plot: True
  plot_split_threshold: 1000
  cpu_threads: 32

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

# ------------------ YAML: Set 2 (UCLES-only, GP) ------------------ #
generate_yaml_ucles_only() {
  local yaml_file="$1"
  local rv_name="HD102365_UCLES_RV"

  cat > "${yaml_file}" << EOF
inputs:
  ${rv_name}:
    file: ${data_dir}/${rv_name}.dat
    kind: RV
    models:
      - radial_velocities
      - gp_multidimensional

  HD102365_UCLES_EWHa:
    file: ${data_dir}/HD102365_UCLES_EWHa.dat
    kind: EWHa
    models:
      - gp_multidimensional

common:
  activity:
    boundaries:
      Prot: [3000.0, 4000.0]
      Pdec: [1000.0, 10000.0]
      Oamp: [0.01, 1.0]
    priors:
      Prot: ['Gaussian', 3500.00, 500.0]

  star:
    star_parameters:
      priors:
        mass: ['Gaussian', 0.85, 0.03]
        radius: ['Gaussian', 0.99, 0.02]
        density: ['Gaussian', 0.8760186293091098, 0.04]

models:
  radial_velocities:
    planets: []

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

parameters:
  Tref: 2450830.213590
  low_ram_plot: True
  plot_split_threshold: 1000
  cpu_threads: 32

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

# ------------------ YAML: Set 3 (GP only on ESPRESSO; others no GP) ------------------ #
generate_yaml_espresso_gp_only() {
  local yaml_file="$1"

  cat > "${yaml_file}" << EOF
inputs:
  HD102365_ESPRESSO_RV:
    file: ${data_dir}/HD102365_ESPRESSO_RV.dat
    kind: RV
    models:
      - radial_velocities
      - gp_multidimensional
EOF

  for rv_name in "${rv_inputs_classical[@]}"; do
    cat >> "${yaml_file}" << EOF
  ${rv_name}:
    file: ${data_dir}/${rv_name}.dat
    kind: RV
    models:
      - radial_velocities
EOF
  done

  cat >> "${yaml_file}" << EOF
  HD102365_ESPRESSO_BIS:
    file: ${data_dir}/HD102365_ESPRESSO_BIS.dat
    kind: BIS
    models:
      - gp_multidimensional

common:
  activity:
    boundaries:
      Prot: [20.0, 50.0]
      Pdec: [10.0, 1000.0]
      Oamp: [0.01, 1.0]
    priors:
      Prot: ['Gaussian', 36.00, 10.0]

  star:
    star_parameters:
      priors:
        mass: ['Gaussian', 0.85, 0.03]
        radius: ['Gaussian', 0.99, 0.02]
        density: ['Gaussian', 0.8760186293091098, 0.04]

models:
  radial_velocities:
    planets: []

  gp_multidimensional:
    model: spleaf_multidimensional_esp
    common: activity
    n_harmonics: 4
    hyperparameters_condition: True
    rotation_decay_condition: True

    HD102365_ESPRESSO_RV:
      boundaries:
        rot_amp: [0.0, 10.0]
        con_amp: [-20.0, 20.0]
      derivative: True

    HD102365_ESPRESSO_BIS:
      boundaries:
        rot_amp: [-80.0, 80.0]
        con_amp: [-10.0, 10.0]
      derivative: True

parameters:
  Tref: 2450830.213590
  low_ram_plot: True
  plot_split_threshold: 1000
  cpu_threads: 32

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

# ============================================================
# MAIN GENERATION
# ============================================================

# --- Set 1 (GP only)
for data_set in "${data_sets[@]}"; do
  for sampler in "${samplers[@]}"; do
    job_name="HD102365_${data_set}_gp_0p_${sampler}"
    config_dir="${results_dir}/${job_name}"
    mkdir -p "${config_dir}"

    yaml_name="${job_name}.yaml"
    generate_yaml_set1 "${config_dir}/${yaml_name}" "${data_set}"
    ((yaml_count++))

    generate_job_script "${scripts_dir}/run_${job_name}.sh" "${job_name}" "${config_dir}" "${yaml_name}" "${sampler}"
    ((script_count++))
  done
done

# --- Set 2 (UCLES only, GP only)
for sampler in "${samplers[@]}"; do
  job_name="HD102365_UCLES_only_gp_0p_${sampler}"
  config_dir="${results_dir}/${job_name}"
  mkdir -p "${config_dir}"

  yaml_name="${job_name}.yaml"
  generate_yaml_ucles_only "${config_dir}/${yaml_name}"
  ((yaml_count++))

  generate_job_script "${scripts_dir}/run_${job_name}.sh" "${job_name}" "${config_dir}" "${yaml_name}" "${sampler}"
  ((script_count++))
done

# --- Set 3 (GP only on ESPRESSO; others no GP)  [RENAMED]
for sampler in "${samplers[@]}"; do
  job_name="HD102365_all_instr_ESPRESSO_GP_0p_${sampler}"
  config_dir="${results_dir}/${job_name}"
  mkdir -p "${config_dir}"

  yaml_name="${job_name}.yaml"
  generate_yaml_espresso_gp_only "${config_dir}/${yaml_name}"
  ((yaml_count++))

  generate_job_script "${scripts_dir}/run_${job_name}.sh" "${job_name}" "${config_dir}" "${yaml_name}" "${sampler}"
  ((script_count++))
done

# ============================================================
# SUBMIT SCRIPT (everything)
# ============================================================
cat > "${scripts_dir}/submit_all_0p.sh" << 'EOF'
#!/bin/bash
job_count=0
for script in run_*.sh; do
  [ -f "$script" ] || continue
  echo "Submitting $script"
  bsub < "$script"
  ((job_count++))
  sleep 0.5
done
echo "Submitted $job_count jobs."
EOF
chmod +x "${scripts_dir}/submit_all_0p.sh"

echo ""
echo "Setup complete."
echo "YAML files created:  ${yaml_count}"
echo "Job scripts created: ${script_count}"
echo ""
echo "Run:"
echo "  cd ${scripts_dir}"
echo "  ./submit_all_0p.sh"
