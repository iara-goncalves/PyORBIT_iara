#!/bin/bash

# PyORBIT Complete Setup Generator - FIESTA Mode Version
# Creates directory structure, YAML configs, and job scripts for FIESTA analysis
# Structure: ESSP4/scripts/ (this script) and ESSP4/results_fiesta/ (results)
# Data files: ESSP4/data/ (instrument-specific .dat files)
# RV data: expres, harps, neid (3 instruments)
# FIESTA data: expres, harps, harpsn, neid (4 instruments)

# Define arrays
datasets=(DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8 DS9)
planets=(0p 1p 2p 3p)  # 0p for 0 planets
rv_instruments=(expres harps neid)  # RV data instruments
fiesta_instruments=(expres harps harpsn neid)  # FIESTA data instruments

# Base directories
data_dir="/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/data"
results_dir="/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/results_fiesta_dynesty"
out_dir="/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/out_fiesta_dynesty"

# LSF configuration
queue="hpc"
cores=16
mem_per_core="2GB"
mem_limit="2GB"
walltime="72:00"
# email="icogo@dtu.dk"
email="jzhao@space.dtu.dk"

echo "PyORBIT Complete Setup Generator - FIESTA Mode Version"
echo "======================================================="
echo "Datasets: ${#datasets[@]} (DS1-DS9)"
echo "Planets: ${#planets[@]} (0p, 1p, 2p, 3p)"
echo "RV Instruments: 3 (expres, harps, neid)"
echo "FIESTA Instruments: 4 (expres, harps, harpsn, neid)"
echo "FIESTA modes: mode1 and mode2 for each instrument"
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
total_combinations=$((${#datasets[@]} * ${#planets[@]}))

echo "Creating complete directory structure and files..."
echo "Total combinations: $total_combinations"
echo ""

# Function to generate YAML configuration
generate_yaml() {
    local yaml_file="$1"
    local dataset="$2"
    local planet="$3"
    
    # Determine number of planets for configuration
    case $planet in
        "0p") num_planets=0 ;;
        "1p") num_planets=1 ;;
        "2p") num_planets=2 ;;
        "3p") num_planets=3 ;;
    esac
    
    # Generate YAML configuration
    cat > "$yaml_file" << EOF
inputs:
EOF

    # Add RV data for RV instruments (expres, harps, neid)
    for instrument in "${rv_instruments[@]}"; do
        cat >> "$yaml_file" << EOF
  RVdata_${instrument}:
    file: ${data_dir}/${dataset}_${instrument}_RV.dat
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

    # Add FIESTA mode1 and mode2 for FIESTA instruments (expres, harps, harpsn, neid)
    for instrument in "${fiesta_instruments[@]}"; do
        cat >> "$yaml_file" << EOF
  FIESTAdata_${instrument}_mode1:
    file: ${data_dir}/${dataset}_${instrument}_fiesta_mode1.dat
    kind: FIESTA_mode1
    models:
      - gp_multidimensional
  FIESTAdata_${instrument}_mode2:
    file: ${data_dir}/${dataset}_${instrument}_fiesta_mode2.dat
    kind: FIESTA_mode2
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
        P: [1.3, 100.0]
        K: [0.001, 10.0]
        e: [0.00, 0.70]
      priors:
        e: ['Gaussian', 0.00, 0.098]
EOF
        done
    fi

    # Always add activity section for GP
    cat >> "$yaml_file" << EOF
  activity:
    boundaries:
      Prot: [20.0, 35.0]
      Pdec: [5.0, 1000.0]
      Oamp: [0.01, 1.0]
    priors:
      Prot: ['Gaussian', 28.00, 0.50]
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
        rot_amp: [0.0, 10.0] #at least one must be positive definite
        con_amp: [-20.0, 20.0]
      derivative: True
EOF
    done

    # Add FIESTA mode configurations for GP model (4 FIESTA instruments)
    for instrument in "${fiesta_instruments[@]}"; do
        cat >> "$yaml_file" << EOF
    FIESTAdata_${instrument}_mode1:
      boundaries:
        rot_amp: [-10.0, 10.0]
        con_amp: [-20.0, 20.0]
      derivative: True
    FIESTAdata_${instrument}_mode2:
      boundaries:
        rot_amp: [-10.0, 10.0]
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
    ngen: 5000
    npop_mult: 4
  emcee:
    npop_mult: 4
    nsteps: 10000
    nburn: 3000
    nsave: 7000
    thin: 50
    #use_threading_pool: False
  nested_sampling:
    nlive: 1000
    sampling_efficiency: 0.30
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

# Activate PyORBIT environment
source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Run PyORBIT analysis
pyorbit_run emcee ${yaml_name} > configuration_file_emcee_run_${job_name}.log
pyorbit_results emcee ${yaml_name} -all >> configuration_file_emcee_run_${job_name}.log

# Create results directory and copy files
mkdir -p ./${job_name}
cp ${yaml_name} ./${job_name}/
cp configuration_file_emcee_run_${job_name}.log ./${job_name}/

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
        # Create planet directory inside dataset
        planet_dir="${dataset_dir}/${dataset}_${planet}_fiesta"
        mkdir -p "$planet_dir"
        echo "  Created planet directory: $planet_dir"
        
        # Generate YAML configuration file
        yaml_file="${planet_dir}/${dataset}_${planet}_fiesta.yaml"
        generate_yaml "$yaml_file" "$dataset" "$planet"
        ((yaml_count++))
        echo "    Created YAML: $yaml_file"
        
        # Generate LSF job script
        job_name="${dataset}_${planet}_fiesta"
        script_name="run_${job_name}.sh"
        generate_job_script "$script_name" "$job_name" "$planet_dir" "${dataset}_${planet}_fiesta.yaml"
        ((script_count++))
        echo "    Created job script: $script_name"
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

for script in run_*_fiesta.sh; do
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

# Submit 0p jobs only
cat > "submit_0p_fiesta_jobs.sh" << 'EOF'
#!/bin/bash

echo "Submitting 0p FIESTA jobs (0 planets with GP)..."
echo "================================================"

job_count=0
for script in run_*_0p_fiesta.sh; do
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
for script in run_*_1p_fiesta.sh; do
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
for script in run_*_2p_fiesta.sh; do
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
for script in run_*_3p_fiesta.sh; do
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
echo "FIESTA jobs by planet configuration:"
for planet in 0p 1p 2p 3p; do
    echo ""
    echo "${planet} jobs:"
    bjobs | grep -E "(DS[1-9]_${planet}_fiesta)"
done

echo ""
echo "Job summary:"
total_jobs=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_fiesta)")
running_jobs=$(bjobs | grep RUN | grep -c -E "(DS[1-9]_[0-3]p_fiesta)")
pending_jobs=$(bjobs | grep PEND | grep -c -E "(DS[1-9]_[0-3]p_fiesta)")

echo "Total FIESTA jobs: $total_jobs"
echo "Running: $running_jobs"
echo "Pending: $pending_jobs"

echo ""
echo "Jobs by planet configuration:"
for planet in 0p 1p 2p 3p; do
    planet_count=$(bjobs | grep -c -E "(DS[1-9]_${planet}_fiesta)")
    if [ $planet_count -gt 0 ]; then
        echo "  $planet: $planet_count jobs"
    fi
done

echo ""
echo "Jobs by dataset:"
for ds in DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8 DS9; do
    ds_count=$(bjobs | grep -c -E "(${ds}_[0-3]p_fiesta)")
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

job_ids=$(bjobs | grep -E "(DS[1-9]_[0-3]p_fiesta)" | awk '{print $1}')

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
echo "    ├── DS1_0p_fiesta/     (0 planets + FIESTA modes)"
echo "    ├── DS1_1p_fiesta/     (1 planet + FIESTA modes)"
echo "    ├── DS1_2p_fiesta/     (2 planets + FIESTA modes)"
echo "    └── DS1_3p_fiesta/     (3 planets + FIESTA modes)"
echo ""
echo "Configuration details:"
echo "- RV data: 3 instruments (expres, harps, neid)"
echo "- FIESTA data: 4 instruments (expres, harps, harpsn, neid)"
echo "- Each FIESTA instrument has: mode1 + mode2"
echo "- Total data files per analysis: 11 (3 RV + 8 FIESTA)"
echo "- GP multidimensional model with spleaf_multidimensional_esp"
echo "- MCMC: 10k steps, 3k burn-in, 7k save, thin=50"
echo ""
echo "Submission scripts:"
echo "- ./submit_all_fiesta_jobs.sh    - Submit ALL FIESTA configurations"
echo "- ./submit_0p_fiesta_jobs.sh     - Submit 0 planets only"
echo "- ./submit_1p_fiesta_jobs.sh     - Submit 1 planet only"
echo "- ./submit_2p_fiesta_jobs.sh     - Submit 2 planets only"
echo "- ./submit_3p_fiesta_jobs.sh     - Submit 3 planets only"
echo "- ./monitor_fiesta_jobs.sh       - Monitor all jobs"
echo "- ./cancel_all_fiesta_jobs.sh    - Cancel all jobs"
echo ""
echo "Total jobs: 36 (9 datasets × 4 planet configurations)"
echo "  - 0p jobs: 9"
echo "  - 1p jobs: 9"
echo "  - 2p jobs: 9"
echo "  - 3p jobs: 9"
