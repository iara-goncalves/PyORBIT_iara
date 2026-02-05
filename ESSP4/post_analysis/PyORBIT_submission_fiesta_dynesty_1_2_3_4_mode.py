"""
PyORBIT CSV Export for ESSP Submission - FIESTA Dynesty Configurations
Only process best models from the selection CSV
"""

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
import os
import glob
import re

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

def parse_log_file(log_file_path):
    """Parse the log file to extract hyperparameters from the LAST median parameter section"""
    
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
        
        # Find ALL occurrences of "Statistics on the model parameters obtained from the posteriors samples"
        model_params_marker = "Statistics on the model parameters obtained from the posteriors samples"
        
        # Find all positions of this marker
        positions = []
        start = 0
        while True:
            pos = content.find(model_params_marker, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        if not positions:
            return {}
        
        # Use the LAST occurrence
        last_position = positions[-1]
        
        # Get content starting from the last occurrence
        model_section = content[last_position:]
        
        # Find the end of this section
        end_markers = [
            "Statistics on the derived parameters",
            "====================================================================================================\n\n\n",
            "Inclination fixed to 90 deg!"
        ]
        
        end_pos = len(model_section)
        for marker in end_markers:
            pos = model_section.find(marker, 200)
            if pos != -1 and pos < end_pos:
                end_pos = pos
        
        model_section = model_section[:end_pos]
        
        # Parse the hyperparameters
        hyperparams = {}
        
        lines = model_section.split('\n')
        current_dataset = None
        current_model = None
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines and section markers
            if not line_stripped or '=====' in line_stripped or 'Statistics on' in line_stripped:
                continue
            
            # Check for dataset headers with model
            if line_stripped.startswith('----- dataset:') and 'model:' in line_stripped:
                parts = line_stripped.split('-----')
                for part in parts:
                    if 'dataset:' in part:
                        current_dataset = part.replace('dataset:', '').strip()
                    elif 'model:' in part:
                        current_model = part.replace('model:', '').strip()
                        
            elif line_stripped.startswith('----- dataset:'):
                current_dataset = line_stripped.replace('----- dataset:', '').strip()
                current_model = None
                
            elif line_stripped.startswith('----- common model:'):
                current_dataset = line_stripped.replace('----- common model:', '').strip()
                current_model = 'common'
            
            # Extract parameter values
            elif current_dataset:
                parts = line_stripped.split()
                if len(parts) >= 2:
                    try:
                        param_name = parts[0]
                        param_value = float(parts[1])
                        
                        # Create unique key for parameter
                        if current_model == 'common':
                            key = f"{current_dataset}_{param_name}"
                        elif current_model:
                            key = f"{current_dataset}_{current_model}_{param_name}"
                        else:
                            key = f"{current_dataset}_{param_name}"
                        
                        hyperparams[key] = param_value
                        
                    except (ValueError, IndexError):
                        continue
        
        return hyperparams
        
    except Exception as e:
        print(f"        Error parsing log file: {e}")
        return {}

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

# Base directories
base_results_dir = '/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/results_fiesta_dynesty_1_2_3_4_mode'
output_base_dir = '/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/submission_csv_files/results_fiesta_dynesty_1_2_3_4_mode'
best_models_csv = '/work2/lbuc/jzhao/PyORBIT_ESSP/post_analysis/results_fiesta_dynesty_1_2_3_4_mode/results_fiesta_dynesty_1_2_3_4_mode.csv'

# Common settings
datasets_list = ['RVdata_expres', 'RVdata_harps', 'RVdata_neid']
instruments = ['expres', 'harps', 'harpsn', 'neid']
activity_model = 'gp_multidimensional'
reference_planet = 'b'
group_name = "DTU-Padova-PSU"
method_name = "dynesty_multiple"

# Create output base directory
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

print("="*80)
print("PROCESSING BEST FIESTA DYNESTY MODELS")
print("="*80)

# Read the best models CSV
try:
    best_models_df = pd.read_csv(best_models_csv)
    print(f"\nLoaded best models CSV: {best_models_csv}")
    print(f"Found {len(best_models_df)} best model configurations\n")
except Exception as e:
    print(f"ERROR: Could not read best models CSV: {e}")
    exit(1)

# Process each best model
for idx, row in best_models_df.iterrows():
    # Extract information from the CSV
    # The CSV contains full paths, so we need to extract the configuration from the path
    
    # Try to get the path from the CSV
    config_path = None
    for col in best_models_df.columns:
        if isinstance(row[col], str) and 'DS' in str(row[col]) and '/' in str(row[col]):
            config_path = row[col]
            break
    
    if not config_path:
        print(f"  ✗ Could not determine configuration path from row {idx}")
        continue
    
    # Extract the configuration name from the path
    # Example: /work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/results_fiesta_dynesty_1026/DS1/DS1_1p_2modes
    # We want: DS1_1p_2modes
    config_name = os.path.basename(config_path)
    
    # Extract dataset name (e.g., DS1, DS2, DS3) from the config_name
    dataset_match = re.match(r'(DS\d+)_', config_name)
    if not dataset_match:
        print(f"  ✗ Could not extract dataset name from: {config_name}")
        continue
    
    dataset_name = dataset_match.group(1)
    
    print(f"\nProcessing: {config_name}")
    
    # Create output directory for this dataset
    dataset_output_dir = output_base_dir
    if not os.path.exists(dataset_output_dir):
        os.makedirs(dataset_output_dir)
    
    # Extract number of modes from config_name
    if '1mode' in config_name:
        n_modes = 1
        modes_str = '1mode'
    elif '2mode' in config_name:
        n_modes = 2
        modes_str = '2mode'
    elif '3mode' in config_name:
        n_modes = 3
        modes_str = '3mode'
    elif '4mode' in config_name:
        n_modes = 4
        modes_str = '4mode'
    else:
        print(f"  ✗ Unknown mode configuration in {config_name}")
        continue
    
    # Extract planet configuration (but won't use in filename)
    if '0p' in config_name:
        is_0p_config = True
    elif '1p' in config_name or '2p' in config_name or '3p' in config_name:
        is_0p_config = False
    else:
        print(f"  ✗ Unknown planet configuration in {config_name}")
        continue
    
    # Build paths - the structure is DS2/DS2_3p_3modes/DS2_3p_3modes/dynesty_plot/model_files
    config_dir = os.path.join(base_results_dir, dataset_name, config_name)
    model_files_dir = os.path.join(config_dir, config_name, 'dynesty_plot', 'model_files')
    
    if not os.path.exists(model_files_dir):
        print(f"  ✗ Model files directory not found: {model_files_dir}")
        continue
    
    # ============================================================================
    # PROCESS RV DATA
    # ============================================================================
    
    try:
        # Collect all data by timestamp
        data_by_time = {}
        
        for dataset in datasets_list:
            instrument = dataset.split('_')[1]
            
            # For 0p configurations, we only have activity data
            if is_0p_config:
                activity_file = os.path.join(model_files_dir, f'{dataset}_{activity_model}.dat')
                if not os.path.exists(activity_file):
                    continue
                
                activity_mod = np.genfromtxt(activity_file, skip_header=1)
                
                time_emjd = activity_mod[:, 0]
                RV_activity = activity_mod[:, 6]
                eRV_activity = np.sqrt(activity_mod[:, 9]**2 + activity_mod[:, 12]**2)
                
                # Store RV data by timestamp
                for i in range(len(time_emjd)):
                    t = time_emjd[i]
                    if t not in data_by_time:
                        data_by_time[t] = {
                            'Time [eMJD]': t,
                            'RV_C': np.nan,
                            'eRV_C': np.nan,
                            'RV_A': RV_activity[i],
                            'eRV_A': eRV_activity[i]
                        }
            
            else:
                # For non-0p configurations
                RV_file = os.path.join(model_files_dir, f'{dataset}_radial_velocities_{reference_planet}.dat')
                if not os.path.exists(RV_file):
                    continue
                
                RV_mod = np.genfromtxt(RV_file, skip_header=1)
                
                activity_file = os.path.join(model_files_dir, f'{dataset}_{activity_model}.dat')
                if not os.path.exists(activity_file):
                    continue
                
                activity_mod = np.genfromtxt(activity_file, skip_header=1)
                
                time_emjd = RV_mod[:, 0]
                RV_full_model = RV_mod[:, 3]
                RV_offset = RV_mod[:, 5]
                RV_CA = RV_full_model - RV_offset
                RV_activity = activity_mod[:, 6]
                RV_clean = RV_CA - RV_activity
                
                eRV_clean = np.sqrt(RV_mod[:, 9]**2 + RV_mod[:, 12]**2)
                eRV_activity = np.sqrt(activity_mod[:, 9]**2 + activity_mod[:, 12]**2)
                
                # Store RV data by timestamp
                for i in range(len(time_emjd)):
                    t = time_emjd[i]
                    if t not in data_by_time:
                        data_by_time[t] = {
                            'Time [eMJD]': t,
                            'RV_C': RV_clean[i],
                            'eRV_C': eRV_clean[i],
                            'RV_A': RV_activity[i],
                            'eRV_A': eRV_activity[i]
                        }
        
        # Now load FIESTA modes as activity indicators - COMBINED BY MODE
        for mode_num in range(1, n_modes + 1):
            mode_column = f"FIESTAdata_mode{mode_num}"
            error_column = f"eFIESTAdata_mode{mode_num}"
            
            # Try each instrument for this mode
            for inst in instruments:
                fiesta_name = f"FIESTAdata_{inst}_mode{mode_num}"
                try:
                    fiesta_file = os.path.join(model_files_dir, f"{fiesta_name}_{activity_model}.dat")
                    if os.path.exists(fiesta_file):
                        fiesta_mod = np.genfromtxt(fiesta_file, skip_header=1)
                        
                        fiesta_times = fiesta_mod[:, 0]
                        fiesta_values = fiesta_mod[:, 6]  # Activity indicator model values
                        fiesta_errors = np.sqrt(fiesta_mod[:, 9]**2 + fiesta_mod[:, 12]**2)
                        
                        # Add FIESTA data to matching timestamps (using combined column name)
                        for i in range(len(fiesta_times)):
                            t = fiesta_times[i]
                            if t in data_by_time:
                                # Add as activity indicator columns (without instrument name)
                                data_by_time[t][mode_column] = fiesta_values[i]
                                data_by_time[t][error_column] = fiesta_errors[i]
                            else:
                                # Create new entry if timestamp doesn't exist yet
                                data_by_time[t] = {
                                    'Time [eMJD]': t,
                                    'RV_C': np.nan,
                                    'eRV_C': np.nan,
                                    'RV_A': np.nan,
                                    'eRV_A': np.nan,
                                    mode_column: fiesta_values[i],
                                    error_column: fiesta_errors[i]
                                }
                except Exception as e:
                    # Silently continue if file doesn't exist for this instrument
                    continue
        
        if not data_by_time:
            print(f"  ✗ No data found")
            continue
        
        # Convert to DataFrame
        all_data = list(data_by_time.values())
        df = pd.DataFrame(all_data)
        df = df.sort_values('Time [eMJD]').reset_index(drop=True)
        
        # Organize columns: Time, RV_C, eRV_C, RV_A, eRV_A, then FIESTA modes
        required_columns = ['Time [eMJD]', 'RV_C', 'eRV_C', 'RV_A', 'eRV_A']
        
        # Add FIESTA mode columns in order
        fiesta_columns = []
        for mode_num in range(1, n_modes + 1):
            mode_col = f"FIESTAdata_mode{mode_num}"
            error_col = f"eFIESTAdata_mode{mode_num}"
            if mode_col in df.columns:
                fiesta_columns.append(mode_col)
            if error_col in df.columns:
                fiesta_columns.append(error_col)
        
        final_columns = required_columns + fiesta_columns
        
        # Only include columns that exist in the dataframe
        final_columns = [col for col in final_columns if col in df.columns]
        df = df[final_columns]
        
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        df[numerical_columns] = df[numerical_columns].round(6)
        
        # CSV NAMING FORMAT (WITHOUT planet config): DS1_DTU-Padova-PSU_dynesty_2modes_results.csv
        csv_filename = f"{dataset_name}_{group_name}_{method_name}_{modes_str}_results.csv"
        csv_filepath = os.path.join(dataset_output_dir, csv_filename)
        df.to_csv(csv_filepath, index=False)
        
        print(f"  ✓ Results CSV saved: {csv_filename} ({len(df)} observations)")
        print(f"     Columns: {', '.join(df.columns.tolist())}")
        
    except Exception as e:
        print(f"  ✗ Error processing RV data: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # ============================================================================
    # PROCESS HYPERPARAMETERS
    # ============================================================================
    
    # Log file path
    log_file = os.path.join(config_dir, f'configuration_file_emcee_run_{config_name}.log')
    
    if os.path.exists(log_file):
        hyperparams = parse_log_file(log_file)
        
        if hyperparams:
            config_hyperparams = []
            
            # Extract common activity parameters
            common_prot = hyperparams.get('activity_Prot', np.nan)
            common_pdec = hyperparams.get('activity_Pdec', np.nan)
            common_oamp = hyperparams.get('activity_Oamp', np.nan)
            
            # Export hyperparameters only for the single mode used by this configuration
            target_mode = n_modes  # e.g., 4 for "4mode"

            # Optional: detect which instruments contributed this mode in the results df
            present_instruments = []
            mode_col = f"FIESTAdata_mode{target_mode}"
            if mode_col in df.columns:
                # We can’t see instrument-specific columns in df (they’re combined),
                # so we’ll include all instruments but filter out rows with missing amps.
                present_instruments = instruments
            else:
                present_instruments = instruments  # fallback

            for inst in present_instruments:
                fiesta_name = f"FIESTAdata_{inst}_mode{target_mode}"

                param_row = {
                    'name': fiesta_name,
                    'Prot': common_prot,
                    'Pdec': common_pdec,
                    'Oamp': common_oamp,
                    'rot_amp': hyperparams.get(f"{fiesta_name}_{activity_model}_rot_amp", np.nan),
                    'con_amp': hyperparams.get(f"{fiesta_name}_{activity_model}_con_amp", np.nan),
                }

                # Optional: only keep rows that have at least one GP amp present
                if not (np.isnan(param_row['rot_amp']) and np.isnan(param_row['con_amp'])):
                    config_hyperparams.append(param_row)

            # Save hyperparameters CSV
            if config_hyperparams:
                config_hyperparams_df = pd.DataFrame(config_hyperparams)
                
                numerical_cols = ['Prot', 'Pdec', 'Oamp', 'rot_amp', 'con_amp']
                for col in numerical_cols:
                    if col in config_hyperparams_df.columns:
                        config_hyperparams_df[col] = config_hyperparams_df[col].round(6)
                
                # CSV NAMING FORMAT (WITHOUT planet config): DS1_DTU-Padova-PSU_dynesty_2modes_hyperparameters.csv
                hyperparams_filename = f"{dataset_name}_{group_name}_{method_name}_{modes_str}_hyperparameters.csv"
                hyperparams_filepath = os.path.join(dataset_output_dir, hyperparams_filename)
                config_hyperparams_df.to_csv(hyperparams_filepath, index=False)
                
                print(f"  ✓ Hyperparameters CSV saved: {hyperparams_filename}")
        else:
            print(f"  ✗ No hyperparameters found in log file")
    else:
        print(f"  ✗ Log file not found: {log_file}")

print("\n" + "="*80)
print("ALL BEST FIESTA DYNESTY MODELS PROCESSED!")
print("="*80)
print(f"Results saved in: {output_base_dir}")
