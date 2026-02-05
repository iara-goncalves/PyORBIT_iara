#!/usr/bin/env python3
"""
PyORBIT Dynesty Log File Parser and Analyzer for HD102365

Unified analyzer for three families of dynesty runs:

1) Original HD102365 grids:
   - Log files: configuration_file_dynesty_run_HD102365_<CONFIG>.log
   - ConfigGroup: all_instr, espresso_only, no_espresso

2) UCLES-only grids:
   - Log files: configuration_file_dynesty_run_HD102365_UCLES_only_<CONFIG>.log
   - ConfigGroup: UCLES_only

3) ALL instruments, GP only on ESPRESSO:
   - Log files: configuration_file_dynesty_run_HD102365_all_instr_ESPRESSO_GP_<CONFIG>.log
   - ConfigGroup: all_instr_ESPRESSO_GP

In all cases:
- GPType: gp, no_gp   (gp and no_gp are NEVER compared directly)
- Within each (Dataset, ConfigGroup, GPType), 0p/1p/2p/3p models are compared.

Exports:
- HTML report with sections per Dataset and ConfigGroup/GPType.
- CSV with only the globally best model per (Dataset, ConfigGroup, GPType).
"""

import re
import os
import pandas as pd
from datetime import datetime
import numpy as np


def calculate_t0_from_mean_long(mean_long_deg, omega_deg, period_days, reference_epoch=0.0):
    """
    Calculate time of periastron (t0) from mean longitude.

    Formula: t0 = t_ref - (mean_long - omega) / n
    where n = 2π/P is the mean motion
    """
    mean_long_rad = np.deg2rad(mean_long_deg)
    omega_rad = np.deg2rad(omega_deg)
    mean_anomaly_rad = mean_long_rad - omega_rad
    n = 2.0 * np.pi / period_days
    dt = mean_anomaly_rad / n
    t0 = reference_epoch - dt
    return t0


def parse_dynesty_log_file(filepath):
    """
    Parses a PyORBIT dynesty log file to extract logZ, BIC, efficiency, and parameters.

    Handles three naming schemes:
      1) configuration_file_dynesty_run_HD102365_<CONFIG>.log
      2) configuration_file_dynesty_run_HD102365_UCLES_only_<CONFIG>.log
      3) configuration_file_dynesty_run_HD102365_all_instr_ESPRESSO_GP_<CONFIG>.log
    """
    logz = None
    logz_err = None
    median_bic = None
    efficiency = None
    ncall = None
    niter = None
    orbital_parameters = {}
    activity_parameters = {}

    try:
        with open(filepath, 'r') as f:
            content = f.read()

            # Extract log-evidence (logZ)
            logz_match = re.search(r'logz:\s*(-?[\d\.]+)\s*\+/-\s*([\d\.]+)', content)
            if logz_match:
                logz = float(logz_match.group(1))
                logz_err = float(logz_match.group(2))

            # Extract Median BIC
            median_bic_match = re.search(r'Median BIC\s+\(using likelihood\)\s*=\s*(-?[\d\.]+)', content)
            if median_bic_match:
                median_bic = float(median_bic_match.group(1))

            # Extract efficiency
            eff_match = re.search(r'eff\(%\):\s*([\d\.]+)', content)
            if eff_match:
                efficiency = float(eff_match.group(1))

            # Extract number of calls
            ncall_match = re.search(r'ncall:\s*(\d+)', content)
            if ncall_match:
                ncall = int(ncall_match.group(1))

            # Extract number of iterations
            niter_match = re.search(r'niter:\s*(\d+)', content)
            if niter_match:
                niter = int(niter_match.group(1))

            # Extract orbital and activity parameters from FIRST stats section
            lines = content.split('\n')

            first_stats_idx = -1
            for i, line in enumerate(lines):
                if "Statistics on the model parameters obtained from the posteriors samples" in line:
                    first_stats_idx = i
                    break

            if first_stats_idx != -1:
                current_planet = None
                in_activity_section = False

                for i in range(first_stats_idx, len(lines)):
                    line = lines[i]

                    # Stop if we hit the next major section
                    if "Statistics on the derived parameters" in line or "Parameters corresponding to" in line:
                        break

                    # Planet headers
                    planet_match = re.search(r'----- common model:\s+([a-z])\s*$', line)
                    if planet_match:
                        current_planet = planet_match.group(1)
                        in_activity_section = False
                        if current_planet not in orbital_parameters:
                            orbital_parameters[current_planet] = {}
                        continue

                    # Activity header
                    if "----- common model:  activity" in line:
                        in_activity_section = True
                        current_planet = None
                        continue

                    # Parameter lines
                    param_match = re.match(
                        r'^([A-Za-z_]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([\d\.]+).*\(15-84 p\)',
                        line.strip()
                    )
                    if param_match:
                        param_name = param_match.group(1)
                        median_str = param_match.group(2).strip()
                        lower_error_str = param_match.group(3).strip()
                        upper_error_str = param_match.group(4).strip()
                        median_value = float(median_str)
                        lower_error = float(lower_error_str)
                        upper_error = float(upper_error_str)

                        data = {
                            'value': median_value,
                            'value_str': median_str,
                            'lower_error': lower_error,
                            'lower_error_str': lower_error_str,
                            'upper_error': upper_error,
                            'upper_error_str': upper_error_str
                        }

                        if in_activity_section:
                            activity_parameters[param_name] = data
                        elif current_planet:
                            orbital_parameters[current_planet][param_name] = data

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while reading {filepath}: {e}")
        return None

    if logz is None:
        return None

    # ---------- Filename parsing ----------
    basename = os.path.basename(filepath)
    # dynesty: configuration_file_dynesty_run_HD102365[...].log
    cleaned_name = basename.replace('configuration_file_dynesty_run_', '').replace('.log', '')

    # Three possibilities now:
    #  1) HD102365_<CONFIG>
    #  2) HD102365_UCLES_only_<CONFIG>
    #  3) HD102365_all_instr_ESPRESSO_GP_<CONFIG>
    dataset = 'Unknown'
    config_name = cleaned_name

    if cleaned_name.startswith('HD102365_UCLES_only_'):
        dataset = 'HD102365_UCLES_only'
        config_name = cleaned_name[len('HD102365_UCLES_only_'):]
    elif cleaned_name.startswith('HD102365_all_instr_ESPRESSO_GP_'):
        dataset = 'HD102365_all_instr_ESPRESSO_GP'
        config_name = cleaned_name[len('HD102365_all_instr_ESPRESSO_GP_'):]
    elif cleaned_name.startswith('HD102365_'):
        dataset = 'HD102365'
        config_name = cleaned_name[len('HD102365_'):]
    else:
        # Fallback: dataset remains 'Unknown'
        pass

    # Planets from config suffix: ..._0p_dynesty, ..._1p_dynesty, ...
    pm = re.search(r'_(\dp)_dynesty', config_name)
    if pm:
        num_planets = pm.group(1)
    else:
        num_planets = 'N/A'

    directory_name = os.path.dirname(os.path.abspath(filepath))
    
    return {
        'Configuration': config_name,
        'Dataset': dataset,
        'Planets': num_planets,
        'log(Z)': logz,
        'log(Z) error': logz_err,
        'Median BIC': median_bic,
        'Efficiency %': efficiency,
        'N calls': ncall,
        'N iter': niter,
        'Orbital Parameters': orbital_parameters,
        'Activity Parameters': activity_parameters,
        'File': basename,
        'Directory': directory_name
    }


def _derive_config_group_fields(df):
    """
    Add ConfigGroup / GPType / ModelName / ModelLabel columns to df based on
    Dataset + Configuration.

    Dataset:
      - 'HD102365'                        -> original grids
      - 'HD102365_UCLES_only'            -> UCLES-only grids
      - 'HD102365_all_instr_ESPRESSO_GP' -> all-instr, ESPRESSO-only GP grids

    ConfigGroup:
      - all_instr, espresso_only, no_espresso (original)
      - UCLES_only
      - all_instr_ESPRESSO_GP
    """
    def split_row(row):
        config_name = row['Configuration']
        dataset = row['Dataset']

        group = 'other'
        gp_flag = 'unknown'
        nplanets = 'Np'

        if dataset == 'HD102365':
            # Original naming has explicit prefixes
            if config_name.startswith('all_instr_'):
                group = 'all_instr'
            elif config_name.startswith('espresso_only_'):
                group = 'espresso_only'
            elif config_name.startswith('no_espresso_'):
                group = 'no_espresso'
        elif dataset == 'HD102365_UCLES_only':
            group = 'UCLES_only'
        elif dataset == 'HD102365_all_instr_ESPRESSO_GP':
            group = 'all_instr_ESPRESSO_GP'

        # Planets: look for '_<digit>p_dynesty' OR '<digit>p_dynesty'
        m_planets = re.search(r'_(\dp)_dynesty', config_name)
        if not m_planets:
            m_planets = re.search(r'^(\dp)_dynesty', config_name)
        if m_planets:
            nplanets = m_planets.group(1)

        # GPType: original configs contain '_gp_' / '_no_gp_'
        # UCLES-only configs look like 'gp_1p_dynesty' / 'no_gp_2p_dynesty'
        if '_no_gp_' in config_name or config_name.startswith('no_gp_'):
            gp_flag = 'no_gp'
        elif '_gp_' in config_name or config_name.startswith('gp_'):
            gp_flag = 'gp'
        else:
            # For the ESPRESSO-GP-only grid, everything is GP
            if dataset == 'HD102365_all_instr_ESPRESSO_GP':
                gp_flag = 'gp'

        model_label = f"{gp_flag}_{nplanets}"
        full_model_name = f"{gp_flag}_{nplanets}_dynesty"
        return pd.Series([group, gp_flag, full_model_name, model_label])

    df[['ConfigGroup', 'GPType', 'ModelName', 'ModelLabel']] = df.apply(
        split_row, axis=1
    )

    df = df[df['ConfigGroup'].isin(
        ['all_instr', 'espresso_only', 'no_espresso', 'UCLES_only', 'all_instr_ESPRESSO_GP']
    )].copy()
    return df


def analyze_and_display_dynesty(log_files, search_directory=None):
    """
    Analyzes a list of dynesty log files and prints a formatted comparison table.
    Also exports results to HTML and a single best-model CSV.
    """
    # Create output directory based on search_directory
    
    # Always write into post_analysis/results_HD102365 next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results_HD102365")
    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    failed_files = []

    print(f"Processing {len(log_files)} dynesty log files...")
    for log_file in log_files:
        data = parse_dynesty_log_file(log_file)
        if data:
            all_data.append(data)
        else:
            failed_files.append(log_file)

    if failed_files:
        print(f"\nWarning: {len(failed_files)} files could not be parsed (missing logZ data):")
        for f in failed_files:
            print(f"  - {f}")
        print()

    if not all_data:
        print("No data could be extracted from any log files found.")
        return

    df = pd.DataFrame(all_data)
    df = _derive_config_group_fields(df)

    datasets = sorted(df['Dataset'].unique())

    print("=" * 100)
    print("PyORBIT DYNESTY MODEL COMPARISON - BAYESIAN EVIDENCE ANALYSIS")
    print("=" * 100)
    print("\nInterpretation Guide:")
    print("  • Δlog(Z) > 5.0  : Decisive evidence for better model")
    print("  • Δlog(Z) > 2.5  : Strong evidence")
    print("  • Δlog(Z) > 1.0  : Moderate evidence")
    print("  • Δlog(Z) < 1.0  : Weak/inconclusive evidence")
    print("  • Lower BIC is better (rule of thumb: ΔBIC > 10 is strong)")
    print("=" * 100)

    export_data = []

    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset]

        print(f"\n{'='*100}")
        print(f"DATASET: {dataset}")
        print(f"{'='*100}\n")

        top_groups = dataset_df.groupby('ConfigGroup')

        for cfg_group_name, cfg_group_df in top_groups:
            print(f"=== ConfigGroup: {cfg_group_name} ===\n")

            # Second layer: GPType (gp vs no_gp) – NO cross-comparison between them
            gp_groups = cfg_group_df.groupby('GPType')

            for gp_type, gp_df in gp_groups:
                print(f"--- GPType: {gp_type} ---\n")

                grouped = gp_df.groupby('Configuration')
                best_models_info = []

                for config_name, group in grouped:
                    print(f"Configuration: {config_name}\n")

                    group = group.copy()
                    group['Planets'] = pd.Categorical(group['Planets'],
                                                      categories=['0p', '1p', '2p', '3p'],
                                                      ordered=True)
                    group = group.sort_values('Planets')

                    max_logz_idx = group['log(Z)'].idxmax()
                    max_logz_value = group.loc[max_logz_idx, 'log(Z)']

                    # Handle possible all-NaN Median BIC
                    if group['Median BIC'].notna().any():
                        min_bic_idx = group['Median BIC'].idxmin()
                        min_bic_value = group.loc[min_bic_idx, 'Median BIC']
                    else:
                        min_bic_idx = None
                        min_bic_value = np.nan

                    display_group = group[['Planets', 'log(Z)', 'log(Z) error',
                                        'Median BIC', 'Efficiency %', 'N calls']].copy()

                    display_group['Δlog(Z)'] = group['log(Z)'] - max_logz_value
                    if np.isfinite(min_bic_value):
                        display_group['ΔBIC'] = group['Median BIC'] - min_bic_value
                    else:
                        display_group['ΔBIC'] = np.nan

                    display_group['Bayes Factor'] = display_group['Δlog(Z)'].apply(
                        lambda x: f"{np.exp(x):.2e}"
                    )

                    group_copy = group.copy()
                    group_copy['Preferred_logZ'] = group_copy.index == max_logz_idx
                    group_copy['Preferred_BIC'] = group_copy.index == min_bic_idx if min_bic_idx is not None else False
                    group_copy['Δlog(Z)'] = group_copy['log(Z)'] - max_logz_value
                    if np.isfinite(min_bic_value):
                        group_copy['ΔBIC'] = group_copy['Median BIC'] - min_bic_value
                    else:
                        group_copy['ΔBIC'] = np.nan

                    if min_bic_idx is not None and np.isfinite(min_bic_value):
                        print(f"Best Model by BIC (within {config_name}):    "
                            f"{group.loc[min_bic_idx, 'Planets']} "
                            f"(BIC = {min_bic_value:.2f})")
                    else:
                        print(f"Best Model by BIC (within {config_name}):    not available (Median BIC is NaN)")


                    if len(group) > 1:
                        print(f"\nEvidence Interpretation (within {config_name}):")
                        for idx, row in group.iterrows():
                            if idx != max_logz_idx:
                                delta_logz = row['log(Z)'] - max_logz_value
                                bf = np.exp(-delta_logz)

                                if delta_logz > -1.0:
                                    strength = "Weak"
                                elif delta_logz > -2.5:
                                    strength = "Moderate"
                                elif delta_logz > -5.0:
                                    strength = "Strong"
                                else:
                                    strength = "Decisive"

                                print(f"   {row['Planets']} vs {group.loc[max_logz_idx, 'Planets']}: "
                                      f"Δlog(Z) = {delta_logz:.2f}, BF = {bf:.2e} → {strength} evidence for "
                                      f"{group.loc[max_logz_idx, 'Planets']}")

                    print("\n" + "-"*100 + "\n")

                    best_model_data = group.loc[max_logz_idx]
                    best_models_info.append({
                        'Configuration': config_name,
                        'ConfigGroup': cfg_group_name,
                        'GPType': gp_type,
                        'Best Model': best_model_data['Planets'],
                        'log(Z)': best_model_data['log(Z)'],
                        'BIC': best_model_data['Median BIC'],
                        'Orbital Parameters': best_model_data['Orbital Parameters'],
                        'Activity Parameters': best_model_data['Activity Parameters']
                    })

                    # collect rows for global “best models” summary & HTML
                    export_data.append(group_copy)
                    
                if best_models_info:
                    print(f"{'-'*140}")
                    print(f"PARAMETERS SUMMARY FOR {dataset} / {cfg_group_name} / {gp_type} "
                          f"(BEST MODELS BY log(Z) per Configuration)")
                    print(f"{'-'*140}\n")

                    print("TABLE 1: ORBITAL PARAMETERS\n")
                    print(f"{'Config':<30} {'Model':<8} {'Planet':<8} "
                          f"{'P (days)':<12} {'K (m/s)':<10} {'mean_long (°)':<14} "
                          f"{'e':<10} {'ω (deg)':<10}")
                    print("-" * 140)

                    for model_info in best_models_info:
                        config_name = model_info['Configuration']
                        best_model = model_info['Best Model']
                        orbital_params = model_info['Orbital Parameters']

                        planets = sorted(orbital_params.keys()) if orbital_params else []

                        if not planets:
                            print(f"{config_name:<30} {best_model:<8} {'-':<8} {'-':<12} "
                                  f"{'-':<10} {'-':<14} {'-':<10} {'-':<10}")
                        else:
                            for planet in planets:
                                planet_params = orbital_params[planet]

                                p_val = planet_params.get('P', {}).get('value_str', '-') if 'P' in planet_params else '-'
                                k_val = planet_params.get('K', {}).get('value_str', '-') if 'K' in planet_params else '-'
                                ml_val = planet_params.get('mean_long', {}).get('value_str', '-') if 'mean_long' in planet_params else '-'
                                e_val_str = planet_params.get('e', {}).get('value_str') if 'e' in planet_params else None
                                omega_val_str = planet_params.get('omega', {}).get('value_str') if 'omega' in planet_params else None

                                e_str = e_val_str if e_val_str is not None else '-'
                                omega_str = omega_val_str if omega_val_str is not None else '-'

                                print(f"{config_name:<30} {best_model:<8} {planet:<8} "
                                      f"{p_val:<12} {k_val:<10} {ml_val:<14} {e_str:<10} {omega_str:<10}")

                    print()

                    print("TABLE 2: ACTIVITY PARAMETERS\n")
                    print(f"{'Config':<30} {'Model':<8} "
                          f"{'Prot (days)':<15} {'Pdec (days)':<15} {'Oamp':<10}")
                    print("-" * 80)

                    for model_info in best_models_info:
                        config_name = model_info['Configuration']
                        best_model = model_info['Best Model']
                        activity_params = model_info['Activity Parameters']

                        prot_val = activity_params.get('Prot', {}).get('value_str', '-') if 'Prot' in activity_params else '-'
                        pdec_val = activity_params.get('Pdec', {}).get('value_str', '-') if 'Pdec' in activity_params else '-'
                        oamp_val = activity_params.get('Oamp', {}).get('value_str', '-') if 'Oamp' in activity_params else '-'

                        print(f"{config_name:<30} {best_model:<8} "
                              f"{prot_val:<15} {pdec_val:<15} {oamp_val:<10}")

                    print("\n" + "="*140 + "\n")

    # Export to HTML and one CSV summarising best models
    if export_data:
        combined_df = pd.concat(export_data, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ---- HTML ----
        html_filename = f"dynesty_model_comparison_{timestamp}.html"
        html_filepath = os.path.join(output_dir, html_filename)

        html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PyORBIT Dynesty Model Comparison</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #fafafa;
            color: #2c3e50;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            font-weight: 600;
            margin-bottom: 10px;
            font-size: 32px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            font-size: 16px;
            margin-bottom: 30px;
        }
        .dataset-section {
            background-color: white;
            margin: 30px 0;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            overflow: hidden;
        }
        .dataset-header {
            background-color: #2c3e50;
            color: white;
            padding: 20px 30px;
            font-size: 22px;
            font-weight: 600;
        }
        .config-section {
            margin: 0;
            padding: 25px 30px;
            border-bottom: 1px solid #ecf0f1;
        }
        .config-section:last-child {
            border-bottom: none;
        }
        h2 { 
            color: #34495e;
            margin: 0 0 10px 0;
            font-size: 18px;
            font-weight: 600;
        }
        h3 {
            color: #2c3e50;
            margin: 15px 0 8px 0;
            font-size: 15px;
            font-weight: 600;
            background-color: #ecf0f1;
            padding: 6px 10px;
            border-radius: 4px;
        }
        table { 
            border-collapse: collapse; 
            margin: 10px 0 20px 0;
            width: 100%;
            background-color: white;
        }
        .params-summary-section {
            margin: 20px;
            padding: 18px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .params-table {
            border-collapse: collapse;
            margin: 10px 0;
            width: 100%;
            background-color: white;
            border: 2px solid #3498db;
            font-size: 12px;
        }
        .params-table th {
            background-color: #3498db;
            color: white;
            padding: 8px 6px;
            text-align: center;
            font-weight: 600;
            border: 1px solid #2980b9;
            font-size: 11px;
        }
        .params-table td {
            padding: 6px 6px;
            border: 1px solid #bdc3c7;
            text-align: center;
        }
        .params-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .params-table tr:hover {
            background-color: #e8f4f8;
        }
        th, td { 
            padding: 10px 12px; 
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }
        th { 
            background-color: #f8f9fa;
            color: #2c3e50;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        td {
            color: #34495e;
            font-size: 14px;
        }
        tr:last-child td {
            border-bottom: none;
        }
        tr:hover td {
            background-color: #f8f9fa;
        }
        .highlight-logz {
            background-color: #e8f5e9 !important;
            font-weight: 600;
            color: #2e7d32;
        }
        .highlight-bic {
            background-color: #fff3e0 !important;
            font-weight: 600;
            color: #e65100;
        }
        .legend {
            margin: 0 0 30px 0;
            padding: 20px 25px;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .legend h3 {
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 17px;
            font-weight: 600;
            background: none;
            padding: 0;
        }
        .legend-item {
            margin: 6px 0;
            color: #34495e;
            font-size: 14px;
            line-height: 1.5;
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge-green {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        .badge-orange {
            background-color: #fff3e0;
            color: #e65100;
        }
        .gp-label {
            font-size: 13px;
            font-weight: 600;
            color: #2c3e50;
            margin-top: 5px;
            margin-bottom: 3px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PyORBIT Dynesty Model Comparison</h1>
        <div class="subtitle">Bayesian Evidence Analysis via Nested Sampling</div>
        
        <div class="legend">
            <h3>Interpretation Guide</h3>
            <div class="legend-item"><strong>ConfigGroup:</strong> all_instr, espresso_only, no_espresso, UCLES_only, all_instr_ESPRESSO_GP</div>
            <div class="legend-item"><strong>GPType separation:</strong> gp and no_gp are NOT compared directly; all Δlog(Z), ΔBIC are computed within the same GPType only.</div>
            <div class="legend-item"><span class="badge badge-green">Green highlight</span> = Best log(Z) (highest evidence) within a given (ConfigGroup, GPType)</div>
            <div class="legend-item"><span class="badge badge-orange">Orange highlight</span> = Best BIC (lowest value) within a given (ConfigGroup, GPType)</div>
        </div>
"""
        # HTML per dataset / ConfigGroup / GPType
        for dataset in datasets:
            html_content += f'        <div class="dataset-section">\n'
            html_content += f'            <div class="dataset-header">{dataset}</div>\n'

            dataset_df = df[df['Dataset'] == dataset]
            top_groups = dataset_df.groupby('ConfigGroup')

            for cfg_group_name, cfg_group_df in top_groups:
                html_content += f'            <div class="config-section">\n'
                html_content += f'                <h2>ConfigGroup: {cfg_group_name}</h2>\n'

                gp_groups = cfg_group_df.groupby('GPType')

                for gp_type, gp_df in gp_groups:
                    html_content += f'                <div class="gp-label">GPType: {gp_type}</div>\n'

                    grouped = gp_df.groupby('Configuration')
                    best_models_info = []

                    gp_max_logz = gp_df['log(Z)'].max()
                    gp_min_bic = gp_df['Median BIC'].min()

                    html_content += '                <table>\n'
                    html_content += ('                    <tr><th>Configuration</th><th>Planets</th>'
                                     '<th>log(Z)</th><th>Δlog(Z)</th>'
                                     '<th>Median BIC</th><th>ΔBIC</th>'
                                     '<th>Efficiency %</th><th>N calls</th></tr>\n')

                    for config_name, group in grouped:
                        group = group.copy()
                        group['Planets'] = pd.Categorical(group['Planets'],
                                                          categories=['0p', '1p', '2p', '3p'],
                                                          ordered=True)
                        group = group.sort_values('Planets')

                        max_logz_idx = group['log(Z)'].idxmax()
                        row = group.loc[max_logz_idx]

                        delta_logz = row['log(Z)'] - gp_max_logz
                        delta_bic = row['Median BIC'] - gp_min_bic

                        logz_class = 'highlight-logz' if row['log(Z)'] == gp_max_logz else ''
                        bic_class = 'highlight-bic' if row['Median BIC'] == gp_min_bic else ''

                        html_content += f'                    <tr>\n'
                        html_content += f'                        <td>{config_name}</td>\n'
                        html_content += f'                        <td>{row["Planets"]}</td>\n'
                        html_content += f'                        <td class="{logz_class}">{row["log(Z)"]:.2f} ± {row["log(Z) error"]:.2f}</td>\n'
                        html_content += f'                        <td class="{logz_class}">{delta_logz:.2f}</td>\n'
                        html_content += f'                        <td class="{bic_class}">{row["Median BIC"]:.2f}</td>\n'
                        html_content += f'                        <td class="{bic_class}">{delta_bic:.2f}</td>\n'
                        html_content += f'                        <td>{row["Efficiency %"]:.2f}%</td>\n'
                        html_content += f'                        <td>{row["N calls"]:,}</td>\n'
                        html_content += f'                    </tr>\n'

                        best_models_info.append({
                            'Configuration': config_name,
                            'Best Model': row['Planets'],
                            'log(Z)': row['log(Z)'],
                            'BIC': row['Median BIC'],
                            'Orbital Parameters': row['Orbital Parameters'],
                            'Activity Parameters': row['Activity Parameters']
                        })

                    html_content += '                </table>\n'

                    if best_models_info:
                        html_content += '                <div class="params-summary-section">\n'
                        html_content += ('                    <h3>Parameters Summary '
                                         f'(Best Models by log(Z) for GPType = {gp_type})</h3>\n')

                        # Orbital parameters
                        html_content += '                    <table class="params-table">\n'
                        html_content += ('                        <tr><th>Config</th><th>Model</th><th>Planet</th>'
                                         '<th>P (days)</th><th>K (m/s)</th><th>mean_long (°)</th>'
                                         '<th>e</th><th>ω (°)</th></tr>\n')

                        for model_info in best_models_info:
                            config_name = model_info['Configuration']
                            best_model = model_info['Best Model']
                            orbital_params = model_info['Orbital Parameters']

                            planets = sorted(orbital_params.keys()) if orbital_params else []

                            if not planets:
                                html_content += ('                        <tr>\n'
                                                 f'                            <td>{config_name}</td><td>{best_model}</td><td>-</td>\n'
                                                 '                            <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>\n'
                                                 '                        </tr>\n')
                            else:
                                for i, planet in enumerate(planets):
                                    planet_params = orbital_params[planet]

                                    p_val = planet_params.get('P', {}).get('value_str', '-') if 'P' in planet_params else '-'
                                    k_val = planet_params.get('K', {}).get('value_str', '-') if 'K' in planet_params else '-'
                                    ml_val = planet_params.get('mean_long', {}).get('value_str', '-') if 'mean_long' in planet_params else '-'
                                    e_val_str = planet_params.get('e', {}).get('value_str') if 'e' in planet_params else None
                                    omega_val_str = planet_params.get('omega', {}).get('value_str') if 'omega' in planet_params else None

                                    e_str = e_val_str if e_val_str is not None else '-'
                                    omega_str = omega_val_str if omega_val_str is not None else '-'

                                    html_content += '                        <tr>\n'
                                    if i == 0:
                                        html_content += f'                            <td rowspan="{len(planets)}">{config_name}</td>\n'
                                        html_content += f'                            <td rowspan="{len(planets)}">{best_model}</td>\n'
                                    html_content += f'                            <td>{planet}</td>\n'
                                    html_content += f'                            <td>{p_val}</td><td>{k_val}</td><td>{ml_val}</td>\n'
                                    html_content += f'                            <td>{e_str}</td><td>{omega_str}</td>\n'
                                    html_content += '                        </tr>\n'

                        html_content += '                    </table>\n'

                        # Activity parameters
                        html_content += '                    <table class="params-table">\n'
                        html_content += ('                        <tr><th>Config</th><th>Model</th>'
                                         '<th>Prot (days)</th><th>Pdec (days)</th><th>Oamp</th></tr>\n')

                        for model_info in best_models_info:
                            config_name = model_info['Configuration']
                            best_model = model_info['Best Model']
                            activity_params = model_info['Activity Parameters']

                            prot_val = activity_params.get('Prot', {}).get('value_str', '-') if 'Prot' in activity_params else '-'
                            pdec_val = activity_params.get('Pdec', {}).get('value_str', '-') if 'Pdec' in activity_params else '-'
                            oamp_val = activity_params.get('Oamp', {}).get('value_str', '-') if 'Oamp' in activity_params else '-'

                            html_content += '                        <tr>\n'
                            html_content += f'                            <td>{config_name}</td><td>{best_model}</td>\n'
                            html_content += f'                            <td>{prot_val}</td><td>{pdec_val}</td><td>{oamp_val}</td>\n'
                            html_content += '                        </tr>\n'

                        html_content += '                    </table>\n'
                        html_content += '                </div>\n'  # params-summary-section

                html_content += '            </div>\n'  # config-section

            html_content += '        </div>\n'  # dataset-section

        html_content += """
    </div>
</body>
</html>
"""

        with open(html_filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # ---- single CSV with best models ----
        # Step 1: within each Configuration, keep only the row marked Preferred_logZ
        combined_df = combined_df  # just to make it explicit
        cfg_best = combined_df[combined_df['Preferred_logZ']].copy()

        # Step 2: within each (Dataset, ConfigGroup, GPType), pick the configuration
        # with the highest log(Z) – this is what the HTML highlights in green.
        idx = (
            cfg_best
            .groupby(['Dataset', 'ConfigGroup', 'GPType'])['log(Z)']
            .idxmax()
        )
        final_best = cfg_best.loc[idx].sort_values(
            ['Dataset', 'ConfigGroup', 'GPType']
        )

        best_csv_name = f"dynesty_best_models_summary_{timestamp}.csv"
        best_csv_path = os.path.join(output_dir, best_csv_name)
        final_best.to_csv(best_csv_path, index=False)

        print(f"\n('='*100)")
        print("Results exported to:")
        print(f"  • HTML: {html_filepath}")
        print(f"  • Best-model CSV: {best_csv_path} "
              "(one row per Dataset × ConfigGroup × GPType)")

        print(f"\nKey Metrics:")
        print("  • log(Z): Bayesian evidence (higher is better)")
        print("  • Δlog(Z): Difference from best model (0 = best) within same GPType")
        print("  • BIC: Bayesian Information Criterion (lower is better)")
        print("  • Efficiency: Sampling efficiency (higher is better, typically 1–5%)")
        print("  • gp and no_gp are never compared directly; all comparisons are within fixed GPType.")
        print(f"{'='*100}\n")


# --- Main Execution ---
if __name__ == "__main__":
    import sys
    import glob

    if len(sys.argv) > 1:
        search_directory = sys.argv[1]
        if not os.path.exists(search_directory):
            print(f"Error: Directory '{search_directory}' does not exist.")
            sys.exit(1)
        if not os.path.isdir(search_directory):
            print(f"Error: '{search_directory}' is not a directory.")
            sys.exit(1)
        print(f"Searching for .log files in: {search_directory}")
    else:
        search_directory = "."
        print("Searching for .log files in current directory and subdirectories...")

    all_log_files = glob.glob(os.path.join(search_directory, '**/*.log'), recursive=True)

    # HD102365-specific: only dynesty HD102365 logs (all three families)
    all_log_files = [
        f for f in all_log_files
        if os.path.basename(f).startswith('configuration_file_dynesty_run_')
        and 'HD102365' in os.path.basename(f)
    ]

    file_groups = {}
    for log_file in all_log_files:
        basename = os.path.basename(log_file)
        if basename not in file_groups:
            file_groups[basename] = []
        file_groups[basename].append(log_file)

    log_files = []
    for basename, files in file_groups.items():
        if len(files) == 1:
            log_files.append(files[0])
        else:
            best_file = min(files, key=lambda f: f.count(os.sep))
            log_files.append(best_file)
            print(f"Note: Found {len(files)} copies of '{basename}', using: {best_file}")

    if not log_files:
        print("No HD102365 dynesty .log files found matching 'configuration_file_dynesty_run_HD102365*.log'.")
    else:
        print(f"Found {len(log_files)} unique HD102365 dynesty log files to analyze (filtered from {len(all_log_files)} total).\n")
        analyze_and_display_dynesty(log_files, search_directory=search_directory)
