#!/bin/bash

# PyORBIT Complete Setup Generator - Multi-Configuration Version
# Creates directory structure, YAML configs, and job scripts
# Structure: ESSP4/scripts/ (this script) and ESSP4/results_multiple/ (results)
# Data files: ESSP4/data/ (instrument-specific .dat files)
# Each analysis uses ALL three instruments (expres, harps, neid) simultaneously

# Define arrays
datasets=(DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8 DS9)
planets=(0p 1p 2p 3p)  # Added 0p for 0 planets
instruments=(expres harps neid)

# Configuration definitions
declare -A config_indicators
config_indicators["2_activity_indi"]="BIS FWHM"
config_indicators["4_activity_indi"]="BIS FWHM Contrast Halpha"
config_indicators["5_activity_indi"]="BIS FWHM Contrast Halpha CaII"
config_indicators["ccfs"]="FWHM Contrast"
# config_indicators["white_noise"]=""  # Only RV, no activity indicators REMOVED

# Activity indicator ranges for con_amp
declare -A indicator_ranges
indicator_ranges["Contrast"]="-500.0, 500.0"
indicator_ranges["FWHM"]="-20.0, 20.0"
indicator_ranges["BIS"]="-10.0, 10.0"
indicator_ranges["Halpha"]="-1.0, 1.0"
indicator_ranges["CaII"]="-1.0, 1.0"

# Base directories
data_dir="/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/data"
results_dir="/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/results_multiple_dynasty"
out_dir="/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/out_multiple_dynasty"

# LSF configuration
queue="hpc"
cores=16
mem_per_core="2GB"
mem_limit="3GB"
walltime="72:00"
# email="icogo@dtu.dk"
email="jzhao@space.dtu.dk"

echo "PyORBIT Complete Setup Generator - Multi-Configuration Version"
echo "=============================================================="
echo "Datasets: ${#datasets[@]} (DS1-DS9)"
echo "Planets: ${#planets[@]} (0p, 1p, 2p, 3p)"
echo "Instruments: ALL 3 instruments per analysis (expres, harps, neid)"
echo "Configurations:"
echo "  - 2_activity_indi: BIS, FWHM"
echo "  - 4_activity_indi: BIS, FWHM, Contrast, Halpha"
echo "  - 5_activity_indi: BIS, FWHM, Contrast, Halpha, CaII"
echo "  - ccfs: FWHM, Contrast"
# echo "  - white_noise: RV only"    # REMOVED
echo "Activity indicator con_amp ranges:"
echo "  - Contrast: [-500, 500]"
echo "  - FWHM: [-20, 20]"
echo "  - BIS: [-10, 10]"
echo "  - Halpha: [-1, 1]"
echo "  - CaII: [-1, 1]"
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
# Remove white_noise from count:
total_combinations=$((${#datasets[@]} * ${#planets[@]} * ${#config_indicators[@]}))

echo "Creating complete directory structure and files..."
echo "Total combinations: $total_combinations"
echo ""

# Function to generate YAML configuration
generate_yaml() {
    local yaml_file="$1"
    local dataset="$2"
    local planet="$3"
    local config_name="$4"
    local indicators_string="$5"
    
    # Convert indicators string to array
    read -ra indicators <<< "$indicators_string"
    
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

    # Add RV data for each instrument
    for instrument in "${instruments[@]}"; do
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
        
        # Add GP model only if we have activity indicators
        if [ ${#indicators[@]} -gt 0 ]; then
            cat >> "$yaml_file" << EOF
      - gp_multidimensional
EOF
        fi
    done

    # Add activity indicator inputs for each instrument (if any)
    if [ ${#indicators[@]} -gt 0 ]; then
        for indicator in "${indicators[@]}"; do
            for instrument in "${instruments[@]}"; do
                cat >> "$yaml_file" << EOF
  ${indicator}data_${instrument}:
    file: ${data_dir}/${dataset}_${instrument}_${indicator}.dat
    kind: ${indicator}
    models:
      - gp_multidimensional
EOF
            done
        done
    fi

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
EOF
        done
    fi

    # Add activity section only if we have indicators
    if [ ${#indicators[@]} -gt 0 ]; then
        cat >> "$yaml_file" << EOF
  activity:
    boundaries:
      Prot: [20.0, 35.0]
      Pdec: [10.0, 1000.0]
      Oamp: [0.01, 1.0]
    priors:
      Prot: ['Gaussian', 28.00, 0.50]
      Oamp: ['Gaussian', 0.35, 0.035]
EOF
    fi

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

    # Add GP multidimensional model only if we have activity indicators
    if [ ${#indicators[@]} -gt 0 ]; then
        cat >> "$yaml_file" << EOF
  gp_multidimensional:
    model: spleaf_multidimensional_esp
    common: activity
    n_harmonics: 4
    hyperparameters_condition: True
    rotation_decay_condition: True
EOF

        # Add RV data configurations for GP model (all instruments)
        for instrument in "${instruments[@]}"; do
            cat >> "$yaml_file" << EOF
    RVdata_${instrument}:
      boundaries:
        rot_amp: [0.0, 10.0] #at least one must be positive definite
        con_amp: [-20.0, 20.0]
      derivative: True
EOF
        done

        # Add activity indicator configurations for GP model with specific ranges
        for indicator in "${indicators[@]}"; do
            # Get the specific range for this indicator
            con_amp_range="${indicator_ranges[$indicator]}"
            
            for instrument in "${instruments[@]}"; do
                cat >> "$yaml_file" << EOF
    ${indicator}data_${instrument}:
      boundaries:
        rot_amp: [${con_amp_range}]
        con_amp: [-10.0, 10.0]
      derivative: True
EOF
            done
        done
    fi

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
    npop_mult: 4
  emcee:
    npop_mult: 6
    nsteps: 50000
    nburn: 15000
    nsave: 15000
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
    # Create main dataset directory in results_multiple
    dataset_dir="${results_dir}/${dataset}"
    mkdir -p "$dataset_dir"
    echo "Created main directory: $dataset_dir"
    
    for planet in "${planets[@]}"; do
        # Create planet directory inside dataset
        planet_dir="${dataset_dir}/${dataset}_${planet}"
        mkdir -p "$planet_dir"
        echo "  Created planet directory: $planet_dir"
        
        # Create each configuration
        for config_name in "${!config_indicators[@]}"; do
            indicators_string="${config_indicators[$config_name]}"
            
            # Remove all white_noise handling
            #if [ "$planet" == "0p" ] && [ "$config_name" == "white_noise" ]; then
            #    echo "    Skipping 0p + white_noise (no data to analyze)"
            #    continue
            #fi
            
            # Create configuration subdirectory
            config_dir="${planet_dir}/${dataset}_${planet}_${config_name}"
            mkdir -p "$config_dir"
            echo "    Created config directory: $config_dir"
            
            # Generate YAML configuration file
            yaml_file="${config_dir}/${dataset}_${planet}_${config_name}.yaml"
            generate_yaml "$yaml_file" "$dataset" "$planet" "$config_name" "$indicators_string"
            ((yaml_count++))
            echo "      Created YAML: $yaml_file"
            
            # Generate LSF job script
            job_name="${dataset}_${planet}_${config_name}"
            script_name="run_${job_name}.sh"
            generate_job_script "$script_name" "$job_name" "$config_dir" "${dataset}_${planet}_${config_name}.yaml"
            ((script_count++))
            echo "      Created job script: $script_name"
        done
        echo ""
    done
    echo ""
done

# [Rest of the script remains the same - all the management scripts generation]
# Generate management scripts
echo "Creating job management scripts..."

# Submit all jobs
cat > "submit_all_jobs.sh" << 'EOF'
#!/bin/bash

echo "Submitting ALL PyORBIT jobs (all configurations)..."
echo "===================================================="

job_count=0
submitted_jobs=()
failed_jobs=()

for script in run_*.sh; do
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
    echo "Monitor with: ./monitor_jobs.sh"
    echo "Cancel all with: ./cancel_all_jobs.sh"
fi
EOF

# Submit 0p jobs only
cat > "submit_0p_jobs.sh" << 'EOF'
#!/bin/bash

echo "Submitting 0p jobs (0 planets with GP)..."
echo "========================================="

job_count=0
for script in run_*_0p_*.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 0p jobs"
EOF

# Submit 2_activity_indi jobs only
cat > "submit_2_activity_indi.sh" << 'EOF'
#!/bin/bash

echo "Submitting 2_activity_indi jobs (BIS, FWHM)..."
echo "==============================================="

job_count=0
for script in run_*_2_activity_indi.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 2_activity_indi jobs"
EOF

# Submit 4_activity_indi jobs only
cat > "submit_4_activity_indi.sh" << 'EOF'
#!/bin/bash

echo "Submitting 4_activity_indi jobs (BIS, FWHM, Contrast, Halpha)..."
echo "================================================================="

job_count=0
for script in run_*_4_activity_indi.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 4_activity_indi jobs"
EOF

# Submit 5_activity_indi jobs only
cat > "submit_5_activity_indi.sh" << 'EOF'
#!/bin/bash

echo "Submitting 5_activity_indi jobs (BIS, FWHM, Contrast, Halpha, CaII)..."
echo "======================================================================="

job_count=0
for script in run_*_5_activity_indi.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count 5_activity_indi jobs"
EOF

# Submit ccfs jobs only
cat > "submit_ccfs.sh" << 'EOF'
#!/bin/bash

echo "Submitting ccfs jobs (FWHM, Contrast)..."
echo "========================================"

job_count=0
for script in run_*_ccfs.sh; do
    if [ -f "$script" ]; then
        echo "Submitting: $script"
        bsub < "$script"
        ((job_count++))
        sleep 1
    fi
done

echo "Submitted $job_count ccfs jobs"
EOF

# Remove white noise submission script entirely
# Submit white_noise jobs only -- REMOVED

# Monitor jobs
cat > "monitor_jobs.sh" << 'EOF'
#!/bin/bash

echo "PyORBIT Job Monitor (Multi-Configuration)"
echo "=========================================="

echo "All your jobs:"
bjobs

echo ""
echo "PyORBIT jobs by configuration:"
echo "0p (0 planets with GP):"
bjobs | grep -E "(DS[1-9]_0p_(2_activity_indi|4_activity_indi|5_activity_indi|ccfs))"

echo ""
echo "2_activity_indi (BIS, FWHM):"
bjobs | grep -E "(DS[1-9]_[0-3]p_2_activity_indi)"

echo ""
echo "4_activity_indi (BIS, FWHM, Contrast, Halpha):"
bjobs | grep -E "(DS[1-9]_[0-3]p_4_activity_indi)"

echo ""
echo "5_activity_indi (BIS, FWHM, Contrast, Halpha, CaII):"
bjobs | grep -E "(DS[1-9]_[0-3]p_5_activity_indi)"

echo ""
echo "ccfs (FWHM, Contrast):"
bjobs | grep -E "(DS[1-9]_[0-3]p_ccfs)"

# Removed white_noise monitoring
# echo ""
# echo "white_noise (RV only):"
# bjobs | grep -E "(DS[1-9]_[1-3]p_white_noise)"

echo ""
echo "Job summary:"
total_jobs=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_(2_activity_indi|4_activity_indi|5_activity_indi|ccfs))")
running_jobs=$(bjobs | grep RUN | grep -c -E "(DS[1-9]_[0-3]p_(2_activity_indi|4_activity_indi|5_activity_indi|ccfs))")
pending_jobs=$(bjobs | grep PEND | grep -c -E "(DS[1-9]_[0-3]p_(2_activity_indi|4_activity_indi|5_activity_indi|ccfs))")

echo "Total PyORBIT jobs: $total_jobs"
echo "Running: $running_jobs"
echo "Pending: $pending_jobs"

echo ""
echo "Jobs by configuration:"
for config in 2_activity_indi 4_activity_indi 5_activity_indi ccfs; do
    config_count=$(bjobs | grep -c -E "(DS[1-9]_[0-3]p_${config})")
    if [ $config_count -gt 0 ]; then
        echo "  $config: $config_count jobs"
    fi
done

echo ""
echo "Jobs by planet configuration:"
for planet in 0p 1p 2p 3p; do
    planet_count=$(bjobs | grep -c -E "(DS[1-9]_${planet}_(2_activity_indi|4_activity_indi|5_activity_indi|ccfs))")
    if [ $planet_count -gt 0 ]; then
        echo "  $planet: $planet_count jobs"
    fi
done

echo ""
echo "Jobs by dataset:"
for ds in DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8 DS9; do
    ds_count=$(bjobs | grep -c -E "(${ds}_[0-3]p_(2_activity_indi|4_activity_indi|5_activity_indi|ccfs))")
    if [ $ds_count -gt 0 ]; then
        echo "  $ds: $ds_count jobs"
    fi
done

echo ""
echo "Refresh with: ./monitor_jobs.sh"
echo "Detailed job info: bjobs -l JOB_ID"
EOF

# Cancel all jobs
cat > "cancel_all_jobs.sh" << 'EOF'
#!/bin/bash

echo "Canceling all PyORBIT jobs (all configurations)..."
echo "=================================================="

job_ids=$(bjobs | grep -E "(DS[1-9]_[0-3]p_(2_activity_indi|4_activity_indi|5_activity_indi|ccfs))" | awk '{print $1}')

if [ -z "$job_ids" ]; then
    echo "No PyORBIT jobs found to cancel."
    exit 0
fi

echo "Found PyORBIT jobs to cancel:"
echo "$job_ids"
echo ""

read -p "Are you sure you want to cancel all these jobs? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    for job_id in $job_ids; do
        echo "Canceling job: $job_id"
        bkill $job_id
    done
    echo "All PyORBIT jobs canceled."
else
    echo "Operation canceled."
fi
EOF

chmod +x submit_all_jobs.sh
chmod +x submit_0p_jobs.sh
chmod +x submit_2_activity_indi.sh
chmod +x submit_4_activity_indi.sh
chmod +x submit_5_activity_indi.sh
chmod +x submit_ccfs.sh
#chmod +x submit_white_noise.sh # removed
chmod +x monitor_jobs.sh
chmod +x cancel_all_jobs.sh

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
echo "  ${results_dir}/DS5/"
echo "    ├── DS5_0p/"
echo "    │   ├── DS5_0p_2_activity_indi/     (0 planets + BIS, FWHM)"
echo "    │   ├── DS5_0p_4_activity_indi/     (0 planets + BIS, FWHM, Contrast, Halpha)"
echo "    │   ├── DS5_0p_5_activity_indi/     (0 planets + BIS, FWHM, Contrast, Halpha, CaII)"
echo "    │   └── DS5_0p_ccfs/                (0 planets + FWHM, Contrast)"
echo "    ├── DS5_1p/"
echo "    │   ├── DS5_1p_2_activity_indi/     (1 planet + BIS, FWHM)"
echo "    │   ├── DS5_1p_4_activity_indi/     (1 planet + BIS, FWHM, Contrast, Halpha)"
echo "    │   ├── DS5_1p_5_activity_indi/     (1 planet + BIS, FWHM, Contrast, Halpha, CaII)"
echo "    │   ├── DS5_1p_ccfs/                (1 planet + FWHM, Contrast)"
#echo "    │   └── DS5_1p_white_noise/         (1 planet + RV only)" # removed white_noise dir
echo "    ├── DS5_2p/ (same 4 configurations)"
echo "    └── DS5_3p/ (same 4 configurations)"
echo ""
echo "Configuration details:"
echo "- 0p: 0 planets with GP activity model only (4 configurations)"
echo "- 1p-3p: 1-3 planets with optional GP activity model (4 configurations each)"
echo "- 2_activity_indi: BIS, FWHM (6 files per analysis)"
echo "- 4_activity_indi: BIS, FWHM, Contrast, Halpha (15 files per analysis)"
echo "- 5_activity_indi: BIS, FWHM, Contrast, Halpha, CaII (18 files per analysis)"
echo "- ccfs: FWHM, Contrast (9 files per analysis)"
#echo "- white_noise: RV only (3 files per analysis, not available for 0p)" # removed
echo "- All use 3 instruments: expres, harps, neid"
echo "- GP multidimensional model"
echo "- MCMC: 500k steps, 150k burn-in, 150k save, thin=100"
echo ""
echo "Activity indicator con_amp ranges:"
echo "- Contrast: [-500, 500]"
echo "- FWHM: [-20, 20]"
echo "- BIS: [-10, 10]"
echo "- Halpha: [-1, 1]"
echo "- CaII: [-1, 1]"
echo ""
echo "Submission scripts:"
echo "- ./submit_all_jobs.sh           - Submit ALL configurations"
echo "- ./submit_0p_jobs.sh            - Submit 0 planets configurations only"
echo "- ./submit_2_activity_indi.sh    - Submit 2 activity indicators only"
echo "- ./submit_4_activity_indi.sh    - Submit 4 activity indicators only"
echo "- ./submit_5_activity_indi.sh    - Submit 5 activity indicators only"
echo "- ./submit_ccfs.sh               - Submit ccfs only"
#echo "- ./submit_white_noise.sh        - Submit white noise only" # removed
echo "- ./monitor_jobs.sh              - Monitor all jobs"
echo "- ./cancel_all_jobs.sh           - Cancel all jobs"
echo ""
echo "Total jobs: $yaml_count"
