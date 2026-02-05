#!/bin/bash

# HD102365 PyORBIT Setup Generator
# Creates directory structure, YAML configs, and job scripts
# for:
#   - GP vs no-GP
#   - all instruments vs no ESPRESSO vs ESPRESSO only
#   - 1p, 2p, 3p
#   - emcee & dynesty

base_dir="/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365"
data_dir="${base_dir}/processed_data"
results_dir="${base_dir}/results_HD102365"
out_dir="${base_dir}/out_HD102365"
scripts_dir="${base_dir}/scripts_HD102365"

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

# Configuration axes
data_sets=("all_instr" "no_espresso" "espresso_only")
gp_flags=("no_gp" "gp")
planets=("1p" "2p" "3p")
samplers=("emcee" "dynesty")

yaml_count=0
script_count=0

echo "HD102365 PyORBIT Setup"
echo "======================"
echo "Base dir:      ${base_dir}"
echo "Results dir:   ${results_dir}"
echo "Output dir:    ${out_dir}"
echo "Scripts dir:   ${scripts_dir}"
echo ""

# ------------------ YAML GENERATOR ------------------ #
generate_yaml() {
    local yaml_file="$1"
    local data_set="$2"    # all_instr / no_espresso / espresso_only
    local gp_flag="$3"     # gp / no_gp
    local planet_conf="$4" # 1p / 2p / 3p
    local sampler="$5"     # emcee / dynesty

    # Determine number of planets
    local num_planets
    case "${planet_conf}" in
        "1p") num_planets=1 ;;
        "2p") num_planets=2 ;;
        "3p") num_planets=3 ;;
        *) echo "Unknown planet_conf ${planet_conf}" ; exit 1 ;;
    esac

    # Classical RV inputs
    local rv_inputs_classical=(
        "HD102365_HARPS-Post_RV"
        "HD102365_HARPS-Pre_RV"
        "HD102365_HIRES-Post_RV"
        "HD102365_PFS-Post_RV"
        "HD102365_PFS-Pre_RV"
        "HD102365_UCLES_RV"
    )

    # Which instruments to include
    local include_espresso=false
    local include_classical=true

    case "${data_set}" in
        "all_instr")
            include_espresso=true
            include_classical=true
            ;;
        "no_espresso")
            include_espresso=false
            include_classical=true
            ;;
        "espresso_only")
            include_espresso=true
            include_classical=false
            ;;
        *)
            echo "Unknown data_set ${data_set}"
            exit 1
            ;;
    esac

    # Start YAML
    cat > "${yaml_file}" << EOF
inputs:
EOF

    # ESPRESSO RV
    if [ "${include_espresso}" == true ]; then
        cat >> "${yaml_file}" << EOF
  HD102365_ESPRESSO_RV:
    file: ${data_dir}/HD102365_ESPRESSO_RV.dat
    kind: RV
    models:
EOF
        echo "      - radial_velocities" >> "${yaml_file}"
        if [ "${gp_flag}" == "gp" ]; then
            echo "      - gp_multidimensional" >> "${yaml_file}"
        fi
    fi

    # Classical RV (only if this data_set includes them)
    if [ "${include_classical}" == true ]; then
        for rv_name in "${rv_inputs_classical[@]}"; do
            cat >> "${yaml_file}" << EOF
  ${rv_name}:
    file: ${data_dir}/${rv_name}.dat
    kind: RV
    models:
EOF
            echo "      - radial_velocities" >> "${yaml_file}"
            if [ "${gp_flag}" == "gp" ]; then
                echo "      - gp_multidimensional" >> "${yaml_file}"
            fi
        done
    fi

    # Activity indicators only if GP
    if [ "${gp_flag}" == "gp" ]; then
        # ESPRESSO BIS (no FWHM)
        if [ "${include_espresso}" == true ]; then
            cat >> "${yaml_file}" << EOF
  HD102365_ESPRESSO_BIS:
    file: ${data_dir}/HD102365_ESPRESSO_BIS.dat
    kind: BIS
    models:
      - gp_multidimensional
EOF
        fi

        # SHK / EWHa — only if classical data are included
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
    fi

    # ---------- COMMON SECTION ---------- #
    cat >> "${yaml_file}" << EOF

common:
  planets:
EOF

    planet_letters=("b" "c" "d")
    for ((p=0; p<num_planets; p++)); do
        pl=${planet_letters[$p]}
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
        cat >> "${yaml_file}" << EOF
  activity:
    boundaries:
      Prot: [20.0, 50.0]
      Pdec: [10.0, 1000.0]
      Oamp: [0.01, 1.0]
    priors:
      Prot: ['Gaussian', 36.00, 10.0]
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
        pl=${planet_letters[$p]}
        echo "      - ${pl}" >> "${yaml_file}"
    done

    # ---------- GP MODEL BLOCK ---------- #
    if [ "${gp_flag}" == "gp" ]; then
        cat >> "${yaml_file}" << EOF
  gp_multidimensional:
    model: spleaf_multidimensional_esp
    common: activity
    n_harmonics: 4
    hyperparameters_condition: True
    rotation_decay_condition: True
EOF

        # RV contributions
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

        # ESPRESSO BIS
        if [ "${include_espresso}" == true ]; then
            cat >> "${yaml_file}" << EOF
    HD102365_ESPRESSO_BIS:
      boundaries:
        rot_amp: [-80.0, 80.0]
        con_amp: [-10.0, 10.0]
      derivative: True
EOF
        fi

        # SHK/EWHa contributions (only if classical)
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
    fi

    # Parameters & solver: cpu_threads depends on gp_flag
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
    local sampler="$5"  # emcee / dynesty
    local gp_flag="$6"  # gp / no_gp

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
### General options 
### -- specify queue -- 
#BSUB -q ${queue}
### -- set the job Name -- 
#BSUB -J ${job_name}
### -- ask for number of cores (default: 1) -- 
#BSUB -n ${cores}
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need ${mem_per_core} of memory per core/slot -- 
#BSUB -R "rusage[mem=${mem_per_core}]"
### -- specify that we want the job to get killed if it exceeds ${mem_limit} per core/slot -- 
#BSUB -M ${mem_limit}
### -- set walltime limit: hh:mm -- 
#BSUB -W ${walltime}
### -- set the email address -- 
#BSUB -u ${email}
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o ${out_dir}/Output_${job_name}.out

# Change to configuration directory
cd ${config_dir}

# Clean up previous runs
rm -f configuration_file_${sampler}_run_${job_name}.log

# Activate PyORBIT environment
source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Run PyORBIT analysis
pyorbit_run ${sampler} ${yaml_name} > configuration_file_${sampler}_run_${job_name}.log
pyorbit_results ${sampler} ${yaml_name} -all >> configuration_file_${sampler}_run_${job_name}.log

# Create results directory and copy files
mkdir -p ${config_dir}/${job_name}
cp ${yaml_name} ${config_dir}/${job_name}/
cp configuration_file_${sampler}_run_${job_name}.log ${config_dir}/${job_name}/

# Deactivate environment
conda deactivate

echo "Job ${job_name} completed at: \$(date)"
EOF

    chmod +x "${script_file}"
}

# ------------------ MAIN LOOP ------------------ #
for data_set in "${data_sets[@]}"; do
  for gp_flag in "${gp_flags[@]}"; do
    for planet_conf in "${planets[@]}"; do
      for sampler in "${samplers[@]}"; do
        job_name="HD102365_${data_set}_${gp_flag}_${planet_conf}_${sampler}"
        config_dir="${results_dir}/${data_set}/${gp_flag}/${planet_conf}/${job_name}"
        mkdir -p "${config_dir}"

        yaml_name="${job_name}.yaml"
        yaml_path="${config_dir}/${yaml_name}"

        generate_yaml "${yaml_path}" "${data_set}" "${gp_flag}" "${planet_conf}" "${sampler}"
        ((yaml_count++))

        script_name="run_${job_name}.sh"
        script_path="${scripts_dir}/${script_name}"
        generate_job_script "${script_path}" "${job_name}" "${config_dir}" "${yaml_name}" "${sampler}" "${gp_flag}"
        ((script_count++))

        echo "Created: ${yaml_path}"
        echo "Created: ${script_path}"
      done
    done
  done
done

# ------------------ MANAGEMENT SCRIPTS ------------------ #
cat > submit_all_HD102365.sh << 'EOF'
#!/bin/bash

echo "Submitting ALL HD102365 PyORBIT jobs..."
echo "======================================="

job_count=0
for script in run_HD102365_*.sh; do
  if [ -f "$script" ]; then
    echo "Submitting: $script"
    bsub < "$script"
    ((job_count++))
    sleep 0.5
  fi
done

echo "Submitted $job_count jobs."
EOF
chmod +x submit_all_HD102365.sh

cat > submit_HD102365_no_gp.sh << 'EOF'
#!/bin/bash

echo "Submitting HD102365 no-GP jobs..."
echo "================================="

job_count=0
for script in run_HD102365_*_no_gp_*_*.sh; do
  if [ -f "$script" ]; then
    echo "Submitting: $script"
    bsub < "$script"
    ((job_count++))
    sleep 0.5
  fi
done

echo "Submitted $job_count no-GP jobs."
EOF
chmod +x submit_HD102365_no_gp.sh

cat > submit_HD102365_gp.sh << 'EOF'
#!/bin/bash

echo "Submitting HD102365 GP jobs..."
echo "=============================="

job_count=0
for script in run_HD102365_*_gp_*_*.sh; do
  if [ -f "$script" ]; then
    echo "Submitting: $script"
    bsub < "$script"
    ((job_count++))
    sleep 0.5
  fi
done

echo "Submitted $job_count GP jobs."
EOF
chmod +x submit_HD102365_gp.sh

cat > monitor_HD102365.sh << 'EOF'
#!/bin/bash

echo "HD102365 Job Monitor"
echo "===================="
bjobs | grep "HD102365_" || echo "No HD102365 jobs found."
EOF
chmod +x monitor_HD102365.sh

cat > cancel_HD102365.sh << 'EOF'
#!/bin/bash

echo "Canceling all HD102365 jobs..."
echo "=============================="

job_ids=$(bjobs | grep "HD102365_" | awk '{print $1}')
if [ -z "$job_ids" ]; then
  echo "No HD102365 jobs to cancel."
  exit 0
fi

echo "Jobs to cancel:"
echo "$job_ids"
echo ""
read -p "Proceed? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
  for id in $job_ids; do
    echo "Canceling job $id"
    bkill "$id"
  done
  echo "Done."
else
  echo "Aborted."
fi
EOF
chmod +x cancel_HD102365.sh

echo ""
echo "Setup complete."
echo "YAML files created:   ${yaml_count}"
echo "Job scripts created:  ${script_count}"
echo "Use:"
echo "  cd ${scripts_dir}"
echo "  ./submit_all_HD102365.sh"
echo "  ./submit_HD102365_no_gp.sh"
echo "  ./submit_HD102365_gp.sh"
echo "  ./monitor_HD102365.sh"
echo "  ./cancel_HD102365.sh"
