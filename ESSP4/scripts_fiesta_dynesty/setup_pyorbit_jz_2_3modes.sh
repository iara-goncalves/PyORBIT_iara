#!/bin/bash

# PyORBIT Complete Setup Generator - FIESTA Mode Version
# Creates directory structure, YAML configs, and job scripts for FIESTA analysis
# Structure: ESSP4/scripts/ (this script) and ESSP4/results_fiesta/ (results)
# Data files: ESSP4/data/ (instrument-specific .dat files)
# RV data: expres, harps, neid (3 instruments)
# FIESTA data: expres, harps, harpsn, neid (4 instruments)
# Two configurations: 2modes (mode1, mode2) and 3modes (mode1, mode2, mode3)

# Define arrays
datasets=(DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8 DS9)
planets=(0p 1p 2p 3p)  # 0p for 0 planets
rv_instruments=(expres harps neid)  # RV data instruments
fiesta_instruments=(expres harps harpsn neid)  # FIESTA data instruments
fiesta_configs=(2modes 3modes)  # FIESTA mode configurations

# Base directories
data_dir="../data"
results_dir="../results_fiesta_dynesty"
out_dir="../out_fiesta_dynesty"

# LSF configuration
queue="hpc"
cores=16
mem_per_core="2GB"
mem_limit="3GB"
walltime="72:00"
email="jzhao@space.dtu.dk"

echo "PyORBIT Complete Setup Generator - FIESTA Mode Version"
echo "======================================================="
echo "Datasets: ${#datasets[@]} (DS1-DS9)"
echo "Planets: ${#planets[@]} (0p, 1p, 2p, 3p)"
echo "RV Instruments: 3 (expres, harps, neid)"
echo "FIESTA Instruments: 4 (expres, harps, harpsn, neid)"
echo "FIESTA Configurations:"
echo "  - 2modes: mode1, mode2"
echo "  - 3modes: mode1, mode2, mode3"
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
    
    # Determine number of FIESTA modes
    case $fiesta_config in
        "2modes") num_modes=2 ;;
        "3modes") num_modes=3 ;;
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

    # Add FIESTA modes for FIESTA instruments (expres, harps, harpsn, neid)
    for instrument in "${fiesta_instruments[@]}"; do
        for ((mode=1; mode<=num_modes; mode++)); do
            cat >> "$yaml_file" << EOF
  FIESTAdata_${instrument}_mode${mode}:
    file: ../../../data/${dataset}_${instrument}_fiesta_mode${mode}.dat
    kind: FIESTA_mode${mode}
    models:
      - gp_multidimensional
EOF
        done
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
        P: [1.1, 100.0]
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

    # Add FIESTA mode configurations for GP model (4 FIESTA instruments)
    for instrument in "${fiesta_instruments[@]}"; do
        for ((mode=1; mode<=num_modes; mode++)); do
            cat >> "$yaml_file" << EOF
    FIESTAdata_${instrument}_mode${mode}:
      boundaries:
        rot_amp: [-50.0, 50.0]
        con_amp: [-20.0, 20.0]
      derivative: True
EOF
        done
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
    sampling_efficiency: 0.10  # ← Lower = more thorough
    bound: 'multi'  # ← Already set (good!)
    sample: 'rwalk'  # ← Try random walk instead of auto
    walks: 50  # ← More MCMC steps per iteration
    maxiter: null  # ← Let it run until convergence
    maxcall: null  # ← No call limit
    dlogz: 0.01  # ← Already set (good!)
    # Add these:
    enlarge: 1.5  # ← Expand bounding ellipsoids (helps with multimodality)
    bootstrap: 0  # ← Disable bootstrap (faster, more stable)
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



# Set up environment paths for jinzhao's installations
export CONDA_BASE="/work2/lbuc/iara/anaconda3"
export PATH="\$CONDA_BASE/bin:\$PATH"

# Initialize conda for any user
source \$CONDA_BASE/etc/profile.d/conda.sh

# Activate PyORBIT environment using CORRECT FULL PATH
conda activate /work2/lbuc/iara/conda_envs/pyorbit


# # Activate PyORBIT environment
# source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
# conda activate pyorbit

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

for script in run_*_2modes.sh run_*_3modes.sh; do
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

# Submit 2modes jobs only
cat > "submit_2modes_jobs.sh" << 'EOF'
#!/bin/bash

echo "Submitting 2modes FIESTA jobs (mode1, mode2)..."
echo "==============================================="

job_count=0
for script in run_*_2modes.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 2modes FIESTA jobs"
EOF

# Submit 3modes jobs only
cat > "submit_3modes_jobs.sh" << 'EOF'
#!/bin/bash

echo "Submitting 3modes FIESTA jobs (mode1, mode2, mode3)..."
echo "======================================================"

job_count=0
for script in run_*_3modes.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 3modes FIESTA jobs"
EOF

# Submit 0p jobs only
cat > "submit_0p_fiesta_jobs.sh" << 'EOF'
#!/bin/bash

echo "Submitting 0p FIESTA jobs (0 planets with GP)..."
echo "================================================"

job_count=0
for script in run_*_0p_2modes.sh run_*_0p_3modes.sh; do
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
for script in run_*_1p_2modes.sh run_*_1p_3modes.sh; do
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
for script in run_*_2p_2modes.sh run_*_2p_3modes.sh; do
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
for script in run_*_3p_2modes.sh run_*_3p_3modes.sh; do
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
echo "2modes jobs:"
bjobs | grep -E "(DS[1-9]_[0-3]p_2modes)"

echo ""
echo "3modes jobs:"
bjobs | grep -E "(DS[1-9]_[0-3]p_3modes)"

echo ""
echo "Job summary:"
total_jobs=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_(2modes|3modes))")
running_jobs=$(bjobs | grep RUN | grep -c -E "(DS[1-9]_[0-3]p_(2modes|3modes))")
pending_jobs=$(bjobs | grep PEND | grep -c -E "(DS[1-9]_[0-3]p_(2modes|3modes))")

echo "Total FIESTA jobs: $total_jobs"
echo "Running: $running_jobs"
echo "Pending: $pending_jobs"

echo ""
echo "Jobs by FIESTA configuration:"
modes2_count=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_2modes)")
modes3_count=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_3modes)")
echo "  2modes: $modes2_count jobs"
echo "  3modes: $modes3_count jobs"

echo ""
echo "Jobs by planet configuration:"
for planet in 0p 1p 2p 3p; do
    planet_count=$(bjobs | grep -c -E "(DS[1-9]_${planet}_(2modes|3modes))")
    if [ $planet_count -gt 0 ]; then
        echo "  $planet: $planet_count jobs"
    fi
done

echo ""
echo "Jobs by dataset:"
for ds in DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8 DS9; do
    ds_count=$(bjobs | grep -c -E "(${ds}_[0-3]p_(2modes|3modes))")
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

job_ids=$(bjobs | grep -E "(DS[1-9]_[0-3]p_(2modes|3modes))" | awk '{print $1}')

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
chmod +x submit_2modes_jobs.sh
chmod +x submit_3modes_jobs.sh
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
echo "    ├── DS1_0p_2modes/     (0 planets + 2 FIESTA modes)"
echo "    ├── DS1_0p_3modes/     (0 planets + 3 FIESTA modes)"
echo "    ├── DS1_1p_2modes/     (1 planet + 2 FIESTA modes)"
echo "    ├── DS1_1p_3modes/     (1 planet + 3 FIESTA modes)"
echo "    ├── DS1_2p_2modes/     (2 planets + 2 FIESTA modes)"
echo "    ├── DS1_2p_3modes/     (2 planets + 3 FIESTA modes)"
echo "    ├── DS1_3p_2modes/     (3 planets + 2 FIESTA modes)"
echo "    └── DS1_3p_3modes/     (3 planets + 3 FIESTA modes)"
echo ""
echo "Configuration details:"
echo "- RV data: 3 instruments (expres, harps, neid)"
echo "- FIESTA data: 4 instruments (expres, harps, harpsn, neid)"
echo "- 2modes: mode1, mode2 (8 FIESTA files)"
echo "- 3modes: mode1, mode2, mode3 (12 FIESTA files)"
echo "- Total data files per analysis:"
echo "  * 2modes: 11 files (3 RV + 8 FIESTA)"
echo "  * 3modes: 15 files (3 RV + 12 FIESTA)"
echo "- GP multidimensional model with spleaf_multidimensional_esp"
echo "- MCMC: 10k steps, 3k burn-in, 7k save, thin=10"
echo ""
echo "Submission scripts:"
echo "- ./submit_all_fiesta_jobs.sh    - Submit ALL FIESTA configurations"
echo "- ./submit_2modes_jobs.sh        - Submit 2modes configurations only"
echo "- ./submit_3modes_jobs.sh        - Submit 3modes configurations only"
echo "- ./submit_0p_fiesta_jobs.sh     - Submit 0 planets (both 2modes & 3modes)"
echo "- ./submit_1p_fiesta_jobs.sh     - Submit 1 planet (both 2modes & 3modes)"
echo "- ./submit_2p_fiesta_jobs.sh     - Submit 2 planets (both 2modes & 3modes)"
echo "- ./submit_3p_fiesta_jobs.sh     - Submit 3 planets (both 2modes & 3modes)"
echo "- ./monitor_fiesta_jobs.sh       - Monitor all jobs"
echo "- ./cancel_all_fiesta_jobs.sh    - Cancel all jobs"
echo ""
echo "Total jobs: 72 (9 datasets × 4 planet configs × 2 FIESTA configs)"
echo "  - 2modes jobs: 36 (9 datasets × 4 planet configs)"
echo "  - 3modes jobs: 36 (9 datasets × 4 planet configs)"
echo "  Breakdown by planets:"
echo "    * 0p: 18 jobs (9 datasets × 2 FIESTA configs)"
echo "    * 1p: 18 jobs (9 datasets × 2 FIESTA configs)"
echo "    * 2p: 18 jobs (9 datasets × 2 FIESTA configs)"
echo "    * 3p: 18 jobs (9 datasets × 2 FIESTA configs)"
