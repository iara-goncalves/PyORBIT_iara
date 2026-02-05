"""
PyORBIT CSV Export for ESSP Submission - Single Instrument Configurations
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

def get_activity_config(config_type):
    """Return activity indicators based on configuration type"""
    
    configs = {
        '2_activity_indi': ['BISdata', 'FWHMdata'],
        '4_activity_indi': ['BISdata', 'FWHMdata', 'Contrastdata', 'Halphadata'],
        '5_activity_indi': ['BISdata', 'FWHMdata', 'Contrastdata', 'Halphadata', 'CaIIdata'],
        'ccfs': ['FWHMdata', 'Contrastdata'],
        'white_noise': []
    }
    
    if config_type not in configs:
        raise ValueError(f"Unknown configuration: {config_type}")
    
    return configs[config_type]

def parse_log_file(log_file_path):
    """Parse the log file to extract hyperparameters from the LAST median parameter section"""
    
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
        
        model_params_marker = "Statistics on the model parameters obtained from the posteriors samples"
        
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
        
        last_position = positions[-1]
        model_section = content[last_position:]
        
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
        
        hyperparams = {}
        lines = model_section.split('\n')
        current_dataset = None
        current_model = None
        
        for line in lines:
            line_stripped = line.strip()
            
            if not line_stripped or '=====' in line_stripped or 'Statistics on' in line_stripped:
                continue
            
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
            
            elif current_dataset:
                parts = line_stripped.split()
                if len(parts) == 2:
                    try:
                        param_name = parts[0]
                        param_value = float(parts[1])
                        
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

def extract_csv_name_components(subdir_name):
    """
    Extract components from subdirectory name for CSV naming (WITHOUT planet config)
    Example: DS1_3p_2_activity_indi_single -> DS1, 2_activity_indi
    """
    # Remove '_single' suffix if present
    base_name = subdir_name.replace('_single', '')
    
    # Split by underscore
    parts = base_name.split('_')
    
    # First part is dataset (DS1, DS2, etc.)
    dataset = parts[0]
    
    # Skip the planet config (0p, 1p, 2p, 3p) - it's parts[1]
    # Rest is the activity config
    activity_config = '_'.join(parts[2:])
    
    return dataset, activity_config

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

base_results_dir = '/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/results_single'
output_base_dir = '/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/submission_csv_files/results_single'
best_models_csv = '/work2/lbuc/jzhao/PyORBIT_ESSP/post_analysis/results_single.csv'

activity_model = 'gp_multidimensional'
reference_planet = 'b'
group_name = "DTU-Padova-PSU"
method_name = "emcee_single"

if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

print("="*80)
print("PROCESSING BEST SINGLE INSTRUMENT MODELS")
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
    # Extract configuration path from the CSV
    config_path = None
    for col in best_models_df.columns:
        if isinstance(row[col], str) and 'DS' in str(row[col]) and '/' in str(row[col]):
            config_path = row[col]
            break
    
    if not config_path:
        print(f"  ✗ Could not determine configuration path from row {idx}")
        continue
    
    # Extract the subdirectory name from the path
    # Example: /work2/.../results_single/DS1/DS1_3p/DS1_3p_2_activity_indi_single
    subdir_name = os.path.basename(config_path)
    
    # Extract dataset name from the subdir_name
    dataset_match = re.match(r'(DS\d+)_', subdir_name)
    if not dataset_match:
        print(f"  ✗ Could not extract dataset name from: {subdir_name}")
        continue
    
    dataset_name = dataset_match.group(1)
    
    print(f"\nProcessing: {subdir_name}")
    
    # Create output directory for this dataset
    dataset_output_dir = output_base_dir
    if not os.path.exists(dataset_output_dir):
        os.makedirs(dataset_output_dir)
    
    # Extract configuration type from subdirectory name
    if 'white_noise' in subdir_name:
        config_type = 'white_noise'
    elif '5_activity_indi' in subdir_name:
        config_type = '5_activity_indi'
    elif '4_activity_indi' in subdir_name:
        config_type = '4_activity_indi'
    elif '2_activity_indi' in subdir_name:
        config_type = '2_activity_indi'
    elif 'ccfs' in subdir_name:
        config_type = 'ccfs'
    else:
        print(f"  ✗ Unknown configuration type in {subdir_name}")
        continue
    
    # Get activity indicators
    activity_indicators = get_activity_config(config_type)
    
    # Check if this is a 0p configuration
    is_0p_config = '0p' in subdir_name
    
    # Build paths - use the actual path from CSV
    model_files_dir = os.path.join(config_path, subdir_name, 'emcee_plot', 'model_files')
    
    if not os.path.exists(model_files_dir):
        print(f"  ✗ Model files directory not found: {model_files_dir}")
        continue
    
    # ============================================================================
    # EXTRACT CSV NAME COMPONENTS (WITHOUT planet config)
    # ============================================================================
    
    ds_name, activity_cfg = extract_csv_name_components(subdir_name)
    
    # ============================================================================
    # PROCESS RV DATA
    # ============================================================================
    
    try:
        all_data = []
        
        # For single instrument, use the generic names without instrument suffix
        if is_0p_config:
            # Only activity data for 0p configurations
            activity_file = os.path.join(model_files_dir, f'RVdata_{activity_model}.dat')
            if not os.path.exists(activity_file):
                print(f"  ✗ Activity file not found: {activity_file}")
                continue
            
            activity_mod = np.genfromtxt(activity_file, skip_header=1)
            
            time_emjd = activity_mod[:, 0]
            RV_activity = activity_mod[:, 6]
            eRV_activity = np.sqrt(activity_mod[:, 9]**2 + activity_mod[:, 12]**2)
            
            # Load activity indicators
            activity_data = {}
            for indicator in activity_indicators:
                try:
                    activity_ind_file = os.path.join(model_files_dir, f"{indicator}_{activity_model}.dat")
                    if os.path.exists(activity_ind_file):
                        activity_ind_mod = np.genfromtxt(activity_ind_file, skip_header=1)
                        
                        activity_data[indicator] = {
                            'values': activity_ind_mod[:, 6],
                            'errors': np.sqrt(activity_ind_mod[:, 9]**2 + activity_ind_mod[:, 12]**2)
                        }
                except:
                    continue
            
            # Create rows with NaN for RV_C and eRV_C
            for i in range(len(time_emjd)):
                row_data = {
                    'Time [eMJD]': time_emjd[i],
                    'RV_C': np.nan,
                    'eRV_C': np.nan,
                    'RV_A': RV_activity[i],
                    'eRV_A': eRV_activity[i]
                }
                
                for indicator in activity_indicators:
                    if indicator in activity_data:
                        if 'BIS' in indicator:
                            unit = ' [m/s]'
                        elif 'FWHM' in indicator:
                            unit = ' [m/s]'
                        else:
                            unit = ''
                        
                        row_data[f'{indicator}{unit}'] = activity_data[indicator]['values'][i]
                        row_data[f'e{indicator}{unit}'] = activity_data[indicator]['errors'][i]
                
                all_data.append(row_data)
        
        else:
            # For non-0p configurations
            RV_file = os.path.join(model_files_dir, f'RVdata_radial_velocities_{reference_planet}.dat')
            if not os.path.exists(RV_file):
                print(f"  ✗ RV file not found: {RV_file}")
                continue
            
            RV_mod = np.genfromtxt(RV_file, skip_header=1)
            
            activity_file = os.path.join(model_files_dir, f'RVdata_{activity_model}.dat')
            if not os.path.exists(activity_file):
                print(f"  ✗ Activity file not found: {activity_file}")
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
            
            activity_data = {}
            for indicator in activity_indicators:
                try:
                    activity_ind_file = os.path.join(model_files_dir, f"{indicator}_{activity_model}.dat")
                    if os.path.exists(activity_ind_file):
                        activity_ind_mod = np.genfromtxt(activity_ind_file, skip_header=1)
                        
                        activity_data[indicator] = {
                            'values': activity_ind_mod[:, 6],
                            'errors': np.sqrt(activity_ind_mod[:, 9]**2 + activity_ind_mod[:, 12]**2)
                        }
                except:
                    continue
            
            for i in range(len(time_emjd)):
                row_data = {
                    'Time [eMJD]': time_emjd[i],
                    'RV_C': RV_clean[i],
                    'eRV_C': eRV_clean[i],
                    'RV_A': RV_activity[i],
                    'eRV_A': eRV_activity[i]
                }
                
                for indicator in activity_indicators:
                    if indicator in activity_data:
                        if 'BIS' in indicator:
                            unit = ' [m/s]'
                        elif 'FWHM' in indicator:
                            unit = ' [m/s]'
                        else:
                            unit = ''
                        
                        row_data[f'{indicator}{unit}'] = activity_data[indicator]['values'][i]
                        row_data[f'e{indicator}{unit}'] = activity_data[indicator]['errors'][i]
                
                all_data.append(row_data)
        
        if not all_data:
            print(f"  ✗ No data found")
            continue
        
        df = pd.DataFrame(all_data)
        df = df.sort_values('Time [eMJD]').reset_index(drop=True)
        
        required_columns = ['Time [eMJD]', 'RV_C', 'eRV_C', 'RV_A', 'eRV_A']
        indicator_columns = [col for col in df.columns if col not in required_columns]
        
        activity_pairs = []
        for indicator in sorted(activity_indicators):
            data_cols = [col for col in indicator_columns if col.startswith(f'{indicator}') and not col.startswith(f'e{indicator}')]
            error_cols = [col for col in indicator_columns if col.startswith(f'e{indicator}')]
            
            for data_col in data_cols:
                activity_pairs.append(data_col)
                # Match the error column
                if 'BIS' in indicator or 'FWHM' in indicator:
                    error_col = data_col.replace(f'{indicator}', f'e{indicator}')
                else:
                    error_col = f'e{data_col}'
                if error_col in error_cols:
                    activity_pairs.append(error_col)
        
        final_columns = required_columns + activity_pairs
        df = df[final_columns]
        
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        df[numerical_columns] = df[numerical_columns].round(6)
        
        # NEW CSV NAMING FORMAT (WITHOUT planet config)
        csv_filename = f"{ds_name}_{group_name}_{method_name}_{activity_cfg}_results.csv"
        csv_filepath = os.path.join(dataset_output_dir, csv_filename)
        df.to_csv(csv_filepath, index=False)
        
        print(f"  ✓ Results CSV saved: {csv_filename} ({len(df)} observations)")
        
    except Exception as e:
        print(f"  ✗ Error processing RV data: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # ============================================================================
    # PROCESS HYPERPARAMETERS
    # ============================================================================

    log_file = os.path.join(config_path, f'configuration_file_emcee_run_{subdir_name}_single.log')

    if os.path.exists(log_file):
        hyperparams = parse_log_file(log_file)
        
        if hyperparams:
            config_hyperparams = []
            
            common_prot = hyperparams.get('activity_Prot', np.nan)
            common_pdec = hyperparams.get('activity_Pdec', np.nan)
            common_oamp = hyperparams.get('activity_Oamp', np.nan)
            
            # For single instrument, extract parameters without instrument suffix
            for indicator in ['RVdata'] + activity_indicators:
                param_row = {
                    'name': indicator,
                    'Prot': common_prot,
                    'Pdec': common_pdec,
                    'Oamp': common_oamp
                }
                
                rot_amp_key = f"{indicator}_gp_multidimensional_rot_amp"
                con_amp_key = f"{indicator}_gp_multidimensional_con_amp"
                
                param_row['rot_amp'] = hyperparams.get(rot_amp_key, np.nan)
                param_row['con_amp'] = hyperparams.get(con_amp_key, np.nan)
                
                config_hyperparams.append(param_row)
            
            if config_hyperparams:
                config_hyperparams_df = pd.DataFrame(config_hyperparams)
                
                numerical_cols = ['Prot', 'Pdec', 'Oamp', 'rot_amp', 'con_amp']
                for col in numerical_cols:
                    if col in config_hyperparams_df.columns:
                        config_hyperparams_df[col] = config_hyperparams_df[col].round(6)
                
                # NEW CSV NAMING FORMAT (WITHOUT planet config)
                hyperparams_filename = f"{ds_name}_{group_name}_{method_name}_{activity_cfg}_hyperparameters.csv"
                hyperparams_filepath = os.path.join(dataset_output_dir, hyperparams_filename)
                config_hyperparams_df.to_csv(hyperparams_filepath, index=False)
                
                print(f"  ✓ Hyperparameters CSV saved: {hyperparams_filename}")
    else:
        print(f"  ✗ Log file not found: {log_file}")

print("\n" + "="*80)
print("ALL BEST SINGLE INSTRUMENT MODELS PROCESSED!")
print("="*80)
print(f"Results saved in: {output_base_dir}")
