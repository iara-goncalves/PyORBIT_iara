#!/bin/bash

# PyORBIT Complete Setup Generator - FIESTA Mode Version
# Creates directory structure, YAML configs, and job scripts for FIESTA analysis
# Structure: ESSP4/scripts/ (this script) and ESSP4/results_fiesta/ (results)
# Data files: ESSP4/data/ (instrument-specific .dat files)
# RV data: expres, harps, neid (3 instruments)
# FIESTA data: expres, harps, harpsn, neid (4 instruments)
# Four configurations: 1mode (mode1), 2mode (mode2), 3mode (mode3), 4mode (mode4)

# Define arrays
datasets=(DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8 DS9)
planets=(0p 1p 2p 3p)  # 0p for 0 planets
rv_instruments=(expres harps neid)  # RV data instruments
fiesta_instruments=(expres harps harpsn neid)  # FIESTA data instruments
fiesta_configs=(1mode 2mode 3mode 4mode)  # FIESTA mode configurations

# Base directories
data_dir="../data"
results_dir="../results_fiesta_dynesty_1_2_3_4_mode"
out_dir="../out_fiesta"

# LSF configuration
queue="hpc"
cores=16
mem_per_core="2GB"
mem_limit="3GB"
walltime="48:00"
email="jzhao@space.dtu.dk"

echo "PyORBIT Complete Setup Generator - FIESTA Mode Version"
echo "======================================================="
echo "Datasets: ${#datasets[@]} (DS1-DS9)"
echo "Planets: ${#planets[@]} (0p, 1p, 2p, 3p)"
echo "RV Instruments: 3 (expres, harps, neid)"
echo "FIESTA Instruments: 4 (expres, harps, harpsn, neid)"
echo "FIESTA Configurations:"
echo "  - 1mode: mode1"
echo "  - 2mode: mode2"
echo "  - 3mode: mode3"
echo "  - 4mode: mode4"
echo "Data directory: $data_dir"
echo "Results directory: $results_dir"
echo "Output directory: $out_dir"
echo ""

# Create base directories
mkdir -p "$results_dir"
mkdir -p "$out_dir"

# Counter for generated files
script_count=0
yaml_count=0
total_combinations=$((${#datasets[@]} * ${#planets[@]} * ${#fiesta_configs[@]}))

echo "Creating complete directory structure and files..."
echo "Total combinations: $total_combinations"
echo ""

# Function to generate YAML configuration
generate_yaml() {
    local yaml_file="$1"
    local dataset="$2"
    local planet="$3"
    local fiesta_config="$4"
    
    # Determine number of planets for configuration
    case $planet in
        "0p") num_planets=0 ;;
        "1p") num_planets=1 ;;
        "2p") num_planets=2 ;;
        "3p") num_planets=3 ;;
    esac
    
    # Determine which FIESTA mode to use
    case $fiesta_config in
        "1mode") mode_num=1 ;;
        "2mode") mode_num=2 ;;
        "3mode") mode_num=3 ;;
        "4mode") mode_num=4 ;;
    esac
    
    # Generate YAML configuration
    cat > "$yaml_file" << EOF
inputs:
EOF

    # Add RV data for RV instruments (expres, harps, neid)
    for instrument in "${rv_instruments[@]}"; do
        cat >> "$yaml_file" << EOF
  RVdata_${instrument}:
    file: ../../../data/${dataset}_${instrument}_RV.dat
    kind: RV
    models:
EOF
        
        # Add radial_velocities model only if we have planets
        if [ $num_planets -gt 0 ]; then
            cat >> "$yaml_file" << EOF
      - radial_velocities
EOF
        fi
        
        # Always add GP model for RV
        cat >> "$yaml_file" << EOF
      - gp_multidimensional
EOF
    done

    # Add FIESTA mode for FIESTA instruments (expres, harps, harpsn, neid)
    for instrument in "${fiesta_instruments[@]}"; do
        cat >> "$yaml_file" << EOF
  FIESTAdata_${instrument}_mode${mode_num}:
    file: ../../../data/${dataset}_${instrument}_fiesta_mode${mode_num}.dat
    kind: FIESTA_mode${mode_num}
    models:
      - gp_multidimensional
EOF
    done

    # Add common section
    cat >> "$yaml_file" << EOF

common:
EOF

    # Add planet configurations only if we have planets
    if [ $num_planets -gt 0 ]; then
        cat >> "$yaml_file" << EOF
  planets:
EOF
        
        planet_letters=("b" "c" "d")
        for ((p=0; p<num_planets; p++)); do
            planet_letter=${planet_letters[$p]}
            cat >> "$yaml_file" << EOF
    ${planet_letter}:
      orbit: keplerian
      parametrization: Eastman2013
      boundaries:
        P: [1.1, 200.0]
        K: [0.001, 10.0]
        e: [0.00, 0.70]
EOF
        done
    fi

    # Always add activity section for GP
    cat >> "$yaml_file" << EOF
  activity:
    boundaries:
      Prot: [22.0, 32.0]
      Pdec: [10.0, 300.0]
      Oamp: [0.01, 1.0]
    priors:
      Prot: ['Gaussian', 27.00, 2.0]
      Oamp: ['Gaussian', 0.35, 0.035]
EOF

    # Add star section
    cat >> "$yaml_file" << EOF
  star:
    star_parameters:
      priors:
        mass: ['Gaussian', 1, 0.000001]
        radius: ['Gaussian', 1, 0.000001]
        density: ['Gaussian', 1, 0.000001]

models:
EOF

    # Add radial_velocities model only if we have planets
    if [ $num_planets -gt 0 ]; then
        cat >> "$yaml_file" << EOF
  radial_velocities:
    planets:
EOF
        
        # Add planet list for radial_velocities model
        planet_letters=("b" "c" "d")
        for ((p=0; p<num_planets; p++)); do
            planet_letter=${planet_letters[$p]}
            cat >> "$yaml_file" << EOF
      - ${planet_letter}
EOF
        done
    fi

    # Always add GP multidimensional model
    cat >> "$yaml_file" << EOF
  gp_multidimensional:
    model: spleaf_multidimensional_esp
    common: activity
    n_harmonics: 4
    hyperparameters_condition: True
    rotation_decay_condition: True
EOF

    # Add RV data configurations for GP model (3 RV instruments)
    for instrument in "${rv_instruments[@]}"; do
        cat >> "$yaml_file" << EOF
    RVdata_${instrument}:
      boundaries:
        rot_amp: [0.0, 20.0] #at least one must be positive definite
        con_amp: [-20.0, 20.0]
      derivative: True
EOF
    done

    # Add FIESTA mode configuration for GP model (4 FIESTA instruments)
    for instrument in "${fiesta_instruments[@]}"; do
        cat >> "$yaml_file" << EOF
    FIESTAdata_${instrument}_mode${mode_num}:
      boundaries:
        rot_amp: [-50.0, 50.0]
        con_amp: [-20.0, 20.0]
      derivative: True
EOF
    done

    # Add parameters and solver sections
    cat >> "$yaml_file" << EOF

parameters:
  Tref: 59334.700184
  low_ram_plot: True
  plot_split_threshold: 1000
  cpu_threads: 16

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
    #use_threading_pool: False
  nested_sampling:
    nlive: 2000
  recenter_bounds: True
EOF
}

# Function to generate job script
generate_job_script() {
    local script_name="$1"
    local job_name="$2"
    local config_dir="$3"
    local yaml_name="$4"
    
    cat > "$script_name" << EOF
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
rm -f configuration_file_emcee_run_${job_name}.log



# # Set up environment paths for jinzhao's installations
# export CONDA_BASE="/work2/lbuc/iara/anaconda3"
# export PATH="\$CONDA_BASE/bin:\$PATH"

# # Initialize conda for any user
# source \$CONDA_BASE/etc/profile.d/conda.sh

# # Activate PyORBIT environment using CORRECT FULL PATH
# conda activate /work2/lbuc/iara/conda_envs/pyorbit


# Activate PyORBIT environment
source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Run PyORBIT analysis
pyorbit_run dynesty ${yaml_name} > configuration_file_emcee_run_${job_name}.log
pyorbit_results dynesty ${yaml_name} -all >> configuration_file_emcee_run_${job_name}.log

# Create results directory and copy files
# mkdir -p ./${job_name}
# cp ${yaml_name} ./${job_name}/
# cp configuration_file_emcee_run_${job_name}.log ./${job_name}/

# Deactivate environment
conda deactivate

echo "Job ${job_name} completed at: \$(date)"
EOF

    chmod +x "$script_name"
}

# Generate complete structure
for dataset in "${datasets[@]}"; do
    # Create main dataset directory in results_fiesta
    dataset_dir="${results_dir}/${dataset}"
    mkdir -p "$dataset_dir"
    echo "Created main directory: $dataset_dir"
    
    for planet in "${planets[@]}"; do
        for fiesta_config in "${fiesta_configs[@]}"; do
            # Create configuration directory inside dataset
            config_dir="${dataset_dir}/${dataset}_${planet}_${fiesta_config}"
            mkdir -p "$config_dir"
            echo "  Created config directory: $config_dir"
            
            # Generate YAML configuration file
            yaml_file="${config_dir}/${dataset}_${planet}_${fiesta_config}.yaml"
            generate_yaml "$yaml_file" "$dataset" "$planet" "$fiesta_config"
            ((yaml_count++))
            echo "    Created YAML: $yaml_file"
            
            # Generate LSF job script
            job_name="${dataset}_${planet}_${fiesta_config}"
            script_name="run_${job_name}.sh"
            generate_job_script "$script_name" "$job_name" "$config_dir" "${dataset}_${planet}_${fiesta_config}.yaml"
            ((script_count++))
            echo "    Created job script: $script_name"
        done
        echo ""
    done
    echo ""
done

# Generate management scripts
echo "Creating job management scripts..."

# Submit all jobs
cat > "submit_all_fiesta_jobs.sh" << 'EOF'
#!/bin/bash

echo "Submitting ALL PyORBIT FIESTA jobs..."
echo "====================================="

job_count=0
submitted_jobs=()
failed_jobs=()

for script in run_*_1mode.sh run_*_2mode.sh run_*_3mode.sh run_*_4mode.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        output=$(bsub < "$script" 2>&1)
        job_id=$(echo "$output" | grep -oE '[0-9]+' | head -1)
        
        if [ $? -eq 0 ] && [ -n "$job_id" ]; then
            submitted_jobs+=("$job_id")
            ((job_count++))
            echo "   ✓ Job ID: $job_id"
        else
            failed_jobs+=("$script")
            echo "   ✗ Failed to submit: $script"
            echo "     Error: $output"
        fi
        sleep 1
    fi
done

echo ""
echo "=========================================="
echo "Submission Summary"
echo "=========================================="
echo "Successfully submitted: $job_count jobs"
echo "Failed submissions: ${#failed_jobs[@]}"

if [ ${#failed_jobs[@]} -gt 0 ]; then
    echo ""
    echo "Failed jobs:"
    for failed in "${failed_jobs[@]}"; do
        echo "  - $failed"
    done
fi

if [ $job_count -gt 0 ]; then
    echo ""
    echo "Submitted Job IDs: ${submitted_jobs[*]}"
    echo ""
    echo "Monitor with: ./monitor_fiesta_jobs.sh"
    echo "Cancel all with: ./cancel_all_fiesta_jobs.sh"
fi
EOF

# Submit 1mode jobs only
cat > "submit_1mode_jobs.sh" << 'EOF'
#!/bin/bash

echo "Submitting 1mode FIESTA jobs (mode1)..."
echo "========================================"

job_count=0
for script in run_*_1mode.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 1mode FIESTA jobs"
EOF

# Submit 2mode jobs only
cat > "submit_2mode_jobs.sh" << 'EOF'
#!/bin/bash

echo "Submitting 2mode FIESTA jobs (mode2)..."
echo "========================================"

job_count=0
for script in run_*_2mode.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 2mode FIESTA jobs"
EOF

# Submit 3mode jobs only
cat > "submit_3mode_jobs.sh" << 'EOF'
#!/bin/bash

echo "Submitting 3mode FIESTA jobs (mode3)..."
echo "========================================"

job_count=0
for script in run_*_3mode.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 3mode FIESTA jobs"
EOF

# Submit 4mode jobs only
cat > "submit_4mode_jobs.sh" << 'EOF'
#!/bin/bash

echo "Submitting 4mode FIESTA jobs (mode4)..."
echo "========================================"

job_count=0
for script in run_*_4mode.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 4mode FIESTA jobs"
EOF

# Submit 0p jobs only
cat > "submit_0p_fiesta_jobs.sh" << 'EOF'
#!/bin/bash

echo "Submitting 0p FIESTA jobs (0 planets with GP)..."
echo "================================================"

job_count=0
for script in run_*_0p_1mode.sh run_*_0p_2mode.sh run_*_0p_3mode.sh run_*_0p_4mode.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 0p FIESTA jobs"
EOF

# Submit 1p jobs only
cat > "submit_1p_fiesta_jobs.sh" << 'EOF'
#!/bin/bash

echo "Submitting 1p FIESTA jobs (1 planet + GP)..."
echo "============================================"

job_count=0
for script in run_*_1p_1mode.sh run_*_1p_2mode.sh run_*_1p_3mode.sh run_*_1p_4mode.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 1p FIESTA jobs"
EOF

# Submit 2p jobs only
cat > "submit_2p_fiesta_jobs.sh" << 'EOF'
#!/bin/bash

echo "Submitting 2p FIESTA jobs (2 planets + GP)..."
echo "============================================="

job_count=0
for script in run_*_2p_1mode.sh run_*_2p_2mode.sh run_*_2p_3mode.sh run_*_2p_4mode.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 2p FIESTA jobs"
EOF

# Submit 3p jobs only
cat > "submit_3p_fiesta_jobs.sh" << 'EOF'
#!/bin/bash

echo "Submitting 3p FIESTA jobs (3 planets + GP)..."
echo "============================================="

job_count=0
for script in run_*_3p_1mode.sh run_*_3p_2mode.sh run_*_3p_3mode.sh run_*_3p_4mode.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 3p FIESTA jobs"
EOF

# Monitor jobs
cat > "monitor_fiesta_jobs.sh" << 'EOF'
#!/bin/bash

echo "PyORBIT FIESTA Job Monitor"
echo "=========================="

echo "All your jobs:"
bjobs

echo ""
echo "FIESTA jobs by configuration:"
echo ""
echo "1mode jobs:"
bjobs | grep -E "(DS[1-9]_[0-3]p_1mode)"

echo ""
echo "2mode jobs:"
bjobs | grep -E "(DS[1-9]_[0-3]p_2mode)"

echo ""
echo "3mode jobs:"
bjobs | grep -E "(DS[1-9]_[0-3]p_3mode)"

echo ""
echo "4mode jobs:"
bjobs | grep -E "(DS[1-9]_[0-3]p_4mode)"

echo ""
echo "Job summary:"
total_jobs=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_(1mode|2mode|3mode|4mode))")
running_jobs=$(bjobs | grep RUN | grep -c -E "(DS[1-9]_[0-3]p_(1mode|2mode|3mode|4mode))")
pending_jobs=$(bjobs | grep PEND | grep -c -E "(DS[1-9]_[0-3]p_(1mode|2mode|3mode|4mode))")

echo "Total FIESTA jobs: $total_jobs"
echo "Running: $running_jobs"
echo "Pending: $pending_jobs"

echo ""
echo "Jobs by FIESTA configuration:"
mode1_count=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_1mode)")
mode2_count=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_2mode)")
mode3_count=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_3mode)")
mode4_count=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_4mode)")
echo "  1mode: $mode1_count jobs"
echo "  2mode: $mode2_count jobs"
echo "  3mode: $mode3_count jobs"
echo "  4mode: $mode4_count jobs"

echo ""
echo "Jobs by planet configuration:"
for planet in 0p 1p 2p 3p; do
    planet_count=$(bjobs | grep -c -E "(DS[1-9]_${planet}_(1mode|2mode|3mode|4mode))")
    if [ $planet_count -gt 0 ]; then
        echo "  $planet: $planet_count jobs"
    fi
done

echo ""
echo "Jobs by dataset:"
for ds in DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8 DS9; do
    ds_count=$(bjobs | grep -c -E "(${ds}_[0-3]p_(1mode|2mode|3mode|4mode))")
    if [ $ds_count -gt 0 ]; then
        echo "  $ds: $ds_count jobs"
    fi
done

echo ""
echo "Refresh with: ./monitor_fiesta_jobs.sh"
echo "Detailed job info: bjobs -l JOB_ID"
EOF

# Cancel all jobs
cat > "cancel_all_fiesta_jobs.sh" << 'EOF'
#!/bin/bash

echo "Canceling all PyORBIT FIESTA jobs..."
echo "===================================="

job_ids=$(bjobs | grep -E "(DS[1-9]_[0-3]p_(1mode|2mode|3mode|4mode))" | awk '{print $1}')

if [ -z "$job_ids" ]; then
    echo "No FIESTA jobs found to cancel."
    exit 0
fi

echo "Found FIESTA jobs to cancel:"
echo "$job_ids"
echo ""

read -p "Are you sure you want to cancel all these jobs? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    for job_id in $job_ids; do
        echo "Canceling job: $job_id"
        bkill $job_id
    done
    echo "All FIESTA jobs canceled."
else
    echo "Operation canceled."
fi
EOF

chmod +x submit_all_fiesta_jobs.sh
chmod +x submit_1mode_jobs.sh
chmod +x submit_2mode_jobs.sh
chmod +x submit_3mode_jobs.sh
chmod +x submit_4mode_jobs.sh
chmod +x submit_0p_fiesta_jobs.sh
chmod +x submit_1p_fiesta_jobs.sh
chmod +x submit_2p_fiesta_jobs.sh
chmod +x submit_3p_fiesta_jobs.sh
chmod +x monitor_fiesta_jobs.sh
chmod +x cancel_all_fiesta_jobs.sh

echo "Setup Complete!"
echo "==============="
echo "Created $yaml_count YAML configuration files"
echo "Created $script_count job scripts"
echo "Created job management scripts"
echo ""
echo "Directory structure created in: $results_dir"
echo "Output files will be saved in: $out_dir"
echo ""
echo "Structure example:"
echo "  ${results_dir}/DS1/"
echo "    ├── DS1_0p_1mode/     (0 planets + mode1)"
echo "    ├── DS1_0p_2mode/     (0 planets + mode2)"
echo "    ├── DS1_0p_3mode/     (0 planets + mode3)"
echo "    ├── DS1_0p_4mode/     (0 planets + mode4)"
echo "    ├── DS1_1p_1mode/     (1 planet + mode1)"
echo "    ├── DS1_1p_2mode/     (1 planet + mode2)"
echo "    ├── DS1_1p_3mode/     (1 planet + mode3)"
echo "    ├── DS1_1p_4mode/     (1 planet + mode4)"
echo "    ├── DS1_2p_1mode/     (2 planets + mode1)"
echo "    ├── DS1_2p_2mode/     (2 planets + mode2)"
echo "    ├── DS1_2p_3mode/     (2 planets + mode3)"
echo "    ├── DS1_2p_4mode/     (2 planets + mode4)"
echo "    ├── DS1_3p_1mode/     (3 planets + mode1)"
echo "    ├── DS1_3p_2mode/     (3 planets + mode2)"
echo "    ├── DS1_3p_3mode/     (3 planets + mode3)"
echo "    └── DS1_3p_4mode/     (3 planets + mode4)"
echo ""
echo "Configuration details:"
echo "- RV data: 3 instruments (expres, harps, neid)"
echo "- FIESTA data: 4 instruments (expres, harps, harpsn, neid)"
echo "- Each configuration uses a single FIESTA mode:"
echo "  * 1mode: mode1 only (4 FIESTA files)"
echo "  * 2mode: mode2 only (4 FIESTA files)"
echo "  * 3mode: mode3 only (4 FIESTA files)"
echo "  * 4mode: mode4 only (4 FIESTA files)"
echo "- Total data files per analysis: 7 (3 RV + 4 FIESTA)"
echo "- GP multidimensional model with spleaf_multidimensional_esp"
echo "- MCMC: 50k steps, 15k burn-in, 35k save, thin=15"
echo ""
echo "Submission scripts:"
echo "- ./submit_all_fiesta_jobs.sh    - Submit ALL FIESTA configurations"
echo "- ./submit_1mode_jobs.sh         - Submit 1mode configurations only"
echo "- ./submit_2mode_jobs.sh         - Submit 2mode configurations only"
echo "- ./submit_3mode_jobs.sh         - Submit 3mode configurations only"
echo "- ./submit_4mode_jobs.sh         - Submit 4mode configurations only"
echo "- ./submit_0p_fiesta_jobs.sh     - Submit 0 planets (all 4 modes)"
echo "- ./submit_1p_fiesta_jobs.sh     - Submit 1 planet (all 4 modes)"
echo "- ./submit_2p_fiesta_jobs.sh     - Submit 2 planets (all 4 modes)"
echo "- ./submit_3p_fiesta_jobs.sh     - Submit 3 planets (all 4 modes)"
echo "- ./monitor_fiesta_jobs.sh       - Monitor all jobs"
echo "- ./cancel_all_fiesta_jobs.sh    - Cancel all jobs"
echo ""
echo "Total jobs: 144 (9 datasets × 4 planet configs × 4 FIESTA modes)"
echo "  - 1mode jobs: 36 (9 datasets × 4 planet configs)"
echo "  - 2mode jobs: 36 (9 datasets × 4 planet configs)"
echo "  - 3mode jobs: 36 (9 datasets × 4 planet configs)"
echo "  - 4mode jobs: 36 (9 datasets × 4 planet configs)"
echo "  Breakdown by planets:"
echo "    * 0p: 36 jobs (9 datasets × 4 FIESTA modes)"
echo "    * 1p: 36 jobs (9 datasets × 4 FIESTA modes)"
echo "    * 2p: 36 jobs (9 datasets × 4 FIESTA modes)"
echo "    * 3p: 36 jobs (9 datasets × 4 FIESTA modes)"
